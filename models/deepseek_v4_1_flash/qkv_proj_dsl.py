# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 实现共享输入量化和 QA/KV MXFP8 投影。"""

import cannbotdsl as cb
import pypto.language as pl
from config import FLASH as M, T_DYN
from mxfp8_dsl import _mx_project, quantize_matrix, MX_GROUP, PADDED_TILE, CUBE_TILE, ROW_TILE


# 动态形状变量。
PAYLOAD_ROWS = pl.dynamic("PAYLOAD_ROWS")
SCALE_COLUMNS = pl.dynamic("SCALE_COLUMNS")

# 模型配置。
D = M.hidden_size
Q_LORA = M.q_lora_rank
HEAD_DIM = M.head_dim


@cb.kernel
def project_q_a(payload: cb.Tensor, weight: cb.Tensor, activation_scale: cb.Tensor,
                weight_scale: cb.Tensor, output: cb.Tensor, column: cb.ScalarValue,
                valid_rows: cb.ScalarValue):
    """将量化后的隐藏状态投影为 BF16 Query 低秩行。"""
    _mx_project(payload, weight, activation_scale, weight_scale, output, column, valid_rows)


@cb.kernel
def project_kv(payload: cb.Tensor, weight: cb.Tensor, activation_scale: cb.Tensor,
               weight_scale: cb.Tensor, output: cb.Tensor, column: cb.ScalarValue,
               num_tokens: cb.ScalarValue, row_start: cb.ScalarValue):
    """将有效行投影为 BF16 KV，并保留非有效区域的原输出。"""
    valid_rows = min(PADDED_TILE, num_tokens - row_start)
    _mx_project(payload, weight, activation_scale, weight_scale, output, column, valid_rows)


@pl.jit.inline(auto_scope=False)
def quantize_projection_input(x: pl.Tensor, num_tokens: pl.Scalar[pl.INT32]):
    """分配按 M80 填充的 Q/KV 共享输入，并通过 AIV 任务量化。"""
    capacity = pl.tensor.dim(x, 0)
    padded_rows = ((capacity + PADDED_TILE - 1) // PADDED_TILE) * PADDED_TILE
    payload = pl.create_tensor([padded_rows, D], dtype=pl.FP8E4M3FN)
    scales = pl.create_tensor([1, padded_rows * (D // MX_GROUP)], dtype=pl.FP8E8M0)
    quantize_matrix(x, payload, scales, num_tokens)
    return payload, scales


@pl.jit.inline(auto_scope=False)
def qa_projection(payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor,
                  weight_scale: pl.Tensor, output: pl.Tensor, rows: pl.Scalar[pl.INDEX]):
    """在一个 M80 激活分组上调度 QA 列块。"""
    weight_flat = pl.reshape(weight, [1, D * Q_LORA])
    scale_flat = pl.tensor.view(weight_scale, [1, D // MX_GROUP * Q_LORA], layout=pl.ND)
    for column in pl.parallel(Q_LORA // CUBE_TILE):
        dst = pl.slice(output, [PADDED_TILE, CUBE_TILE], [0, column * CUBE_TILE], valid_shape=[rows, CUBE_TILE])
        with pl.spmd(1, name_hint="dsl_q_proj_a"):
            project_q_a(payload, weight_flat, activation_scale, scale_flat, dst, column, rows)


@pl.jit.inline(auto_scope=False)
def kv_proj(payload: pl.Tensor, scales: pl.Tensor, wkv: pl.Tensor, wkv_scale: pl.Tensor,
            kv_projection: pl.Tensor, num_tokens: pl.Scalar[pl.INT32]):
    """按 M80 分组调度共享 KV 投影，确保输出列互不重叠。"""
    weight_flat = pl.reshape(wkv, [1, D * HEAD_DIM])
    scale_flat = pl.tensor.view(wkv_scale, [1, D // MX_GROUP * HEAD_DIM], layout=pl.ND)
    for group_start in pl.parallel(0, num_tokens, PADDED_TILE):
        rows = pl.min(PADDED_TILE, num_tokens - group_start)
        source = pl.slice(payload, [PADDED_TILE, D], [group_start, 0])
        scale_start = group_start * (D // MX_GROUP)
        factors = pl.slice(scales, [1, PADDED_TILE * (D // MX_GROUP)], [0, scale_start])
        for column in pl.parallel(HEAD_DIM // CUBE_TILE):
            dst = pl.slice(kv_projection, [PADDED_TILE, CUBE_TILE], [group_start, column * CUBE_TILE],
                           valid_shape=[rows, CUBE_TILE])
            with pl.spmd(1, name_hint="dsl_kv_proj", sync_start=True):
                project_kv(source, weight_flat, factors, scale_flat, dst, column, num_tokens, group_start)
    return kv_projection


@pl.jit
def input_projection_validation(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    wqa: pl.Tensor[[D, Q_LORA], pl.FP8E4M3FN],
    wqa_scale: pl.Tensor[[D // MX_GROUP, Q_LORA], pl.FP8E8M0, pl.MX_B_NN],
    wkv: pl.Tensor[[D, HEAD_DIM], pl.FP8E4M3FN],
    wkv_scale: pl.Tensor[[D // MX_GROUP, HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    payload: pl.Out[pl.Tensor[[PAYLOAD_ROWS, D], pl.FP8E4M3FN]],
    scales: pl.Out[pl.Tensor[[1, SCALE_COLUMNS], pl.FP8E8M0]],
    qa: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.BF16]],
    kv: pl.Out[pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    quantize_matrix(x, payload, scales, num_tokens)
    for group_start in pl.parallel(0, num_tokens, PADDED_TILE):
        rows = pl.min(PADDED_TILE, num_tokens - group_start)
        source = pl.slice(payload, [PADDED_TILE, D], [group_start, 0])
        scale_start = group_start * (D // MX_GROUP)
        factors = pl.slice(scales, [1, PADDED_TILE * (D // MX_GROUP)], [0, scale_start])
        dst = pl.slice(qa, [PADDED_TILE, Q_LORA], [group_start, 0], valid_shape=[rows, Q_LORA])
        qa_projection(source, wqa, factors, wqa_scale, dst, rows)
    kv = kv_proj(payload, scales, wkv, wkv_scale, kv, num_tokens)
    return payload, scales, qa, kv


def golden_input_projection(tensors):
    """量化有效输入，并按每组 32 个元素顺序累加计算 QA/KV 参考结果。"""
    import torch
    from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential
    from quant import _quantize_mxfp8_activation, encode_e8m0, pack_mx_a_scale, unpack_fp8_nz

    active = int(tensors["num_tokens"])
    tensors["payload"].zero_()
    tensors["scales"].view(torch.uint8).zero_()
    if active == 0:
        return
    x = tensors["x"][:active]
    payload, scales = _quantize_mxfp8_activation(x, scale_alg=0)
    tensors["payload"][:active].copy_(payload.flatten(-2))
    padded_codes = torch.zeros(tensors["payload"].shape[0], D // MX_GROUP, dtype=torch.uint8)
    padded_codes[:active].copy_(encode_e8m0(scales))
    tensors["scales"].view(torch.uint8).copy_(pack_mx_a_scale(padded_codes).reshape(1, -1))
    for out, weight, scale in (("qa", "wqa", "wqa_scale"), ("kv", "wkv", "wkv_scale")):
        result = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(x, unpack_fp8_nz(tensors[weight]), tensors[scale])
        tensors[out][:active].copy_(result)


def build_input_projection_specs(active=17):
    """构造 NZ 权重、非均匀 MX 缩放值和非有效区域的 NaN 哨兵值。"""
    import torch
    from golden import ScalarSpec, TensorSpec
    from quant import generate_mxfp8_weight_kn, pack_fp8_nz

    capacity = ((max(active, 1) + ROW_TILE - 1) // ROW_TILE) * ROW_TILE
    padded = ((capacity + PADDED_TILE - 1) // PADDED_TILE) * PADDED_TILE
    generator = torch.Generator().manual_seed(41111)
    x = torch.randn(capacity, D, generator=generator).bfloat16()
    x[active:] = float("nan")
    qa, qa_scale = generate_mxfp8_weight_kn(Q_LORA, D, D ** -0.5, seed=41112)
    kv, kv_scale = generate_mxfp8_weight_kn(HEAD_DIM, D, D ** -0.5, seed=41113)
    qa = pack_fp8_nz(qa)
    kv = pack_fp8_nz(kv)
    return [
        TensorSpec("x", list(x.shape), x.dtype, init_value=x),
        TensorSpec("wqa", list(qa.shape), qa.dtype, init_value=qa),
        TensorSpec("wqa_scale", list(qa_scale.shape), qa_scale.dtype, init_value=qa_scale),
        TensorSpec("wkv", list(kv.shape), kv.dtype, init_value=kv),
        TensorSpec("wkv_scale", list(kv_scale.shape), kv_scale.dtype, init_value=kv_scale),
        TensorSpec("payload", [padded, D], torch.float8_e4m3fn),
        TensorSpec("scales", [1, padded * (D // MX_GROUP)], torch.float8_e8m0fnu),
        TensorSpec("qa", [capacity, Q_LORA], torch.bfloat16, init_value=torch.full((capacity, Q_LORA), 1.75).bfloat16()),
        TensorSpec("kv", [capacity, HEAD_DIM], torch.bfloat16, init_value=torch.full((capacity, HEAD_DIM), 1.75).bfloat16()),
        ScalarSpec("num_tokens", torch.int32, active),
    ]


if __name__ == "__main__":
    import argparse
    import torch
    from golden import ratio_allclose, run
    from window_kv_dsl import compare_payload_bytes

    parser = argparse.ArgumentParser(description="Validate DSL shared input quantization and QA/KV projection on A5.")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--case", choices=["all", "empty", "tail", "multi"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    for case, active in (("empty", 0), ("tail", 17), ("multi", 81)):
        if args.case not in ("all", case):
            continue
        print(f"[QKV_INPUT] {case}: active={active}", flush=True)
        active_compare = ratio_allclose(atol=1e-2, rtol=1.0 / 128)

        def compare_projection(actual, expected, **kwargs):
            if active == 0:
                unchanged = torch.equal(actual, expected)
                return unchanged, f"inactive output unchanged={unchanged}"
            passed, detail = active_compare(actual[:active], expected[:active], **kwargs)
            unchanged = torch.equal(actual[active:], expected[active:])
            return passed and unchanged, f"{detail}; inactive output unchanged={unchanged}"

        result = run(
            fn=input_projection_validation, specs=build_input_projection_specs(active),
            golden_fn=golden_input_projection,
            compare_fn={"payload": compare_payload_bytes, "scales": compare_payload_bytes,
                        "qa": compare_projection, "kv": compare_projection},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only, save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
