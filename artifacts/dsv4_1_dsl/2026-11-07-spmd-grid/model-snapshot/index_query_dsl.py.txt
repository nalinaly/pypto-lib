# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用一个 DSL MIX 分块完成 Indexer Query 投影、RoPE 和 FP4/E8M0 量化。"""

import cannbotdsl as cb
from cannbotdsl.lang.jit import jit
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from cannbotdsl.ring_buffer import RingBuffer
import pypto.language as pl

from config import FLASH as M, INDEX_CACHE_GROUP
from index_vector_dsl import rope_quantize_row
from mxfp8_dsl import _mx_accumulate


# 模型配置
Q_LORA = M.q_lora_rank
INDEX_H = M.index_n_heads
INDEX_DIM = M.index_head_dim
ROPE_DIM = M.qk_rope_head_dim
CACHE_GROUP = INDEX_CACHE_GROUP
PACKED_DIM = INDEX_DIM // 2
SCALE_DIM = INDEX_DIM // CACHE_GROUP

# 分块参数
ROW_TILE = 80
VECTOR_TILE = 64
TASK_TILE = 32


@jit
def _project_query_mix(
    payload: cb.Tensor, weight: cb.Tensor, activation_scale: cb.Tensor, weight_scale: cb.Tensor,
    cosine: cb.Tensor, sine: cb.Tensor, output: cb.Tensor, output_scale: cb.Tensor,
    head: cb.ScalarValue, active_rows: cb.ScalarValue,
):
    """一个 AIC 和两个 AIV 协作，投影中间值只经 L0C 到 UB 传递。"""
    received = RingBuffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((ROW_TILE // 2, INDEX_DIM)), depth=1)
    accumulator = _mx_accumulate(payload, weight, activation_scale, weight_scale, head)
    cb.mem_copy(received.next_write(), accumulator.partition(dim=0))
    projected = received.next_read()
    lane = cb.get_subblock_id()
    active = max(0, min(ROW_TILE // 2, active_rows - lane * (ROW_TILE // 2)))
    cos_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
    sin_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
    values = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((INDEX_DIM + VECTOR_TILE,)))
    packed = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, PACKED_DIM)))
    factors = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, 32)))
    for row in cb.range(active):
        source_row = lane * (ROW_TILE // 2) + row
        cos_dst = cb.tile_slice(cos_ub, (1, ROPE_DIM // 2), (0, 0))
        sin_dst = cb.tile_slice(sin_ub, (1, ROPE_DIM // 2), (0, 0))
        cb.mem_copy(cos_dst, cb.tile_slice(cosine, (1, ROPE_DIM // 2), (source_row, 0)))
        cb.mem_copy(sin_dst, cb.tile_slice(sine, (1, ROPE_DIM // 2), (source_row, 0)))
        with cb.vf():
            input_mask = rr.full_mask()
            for col in range(0, INDEX_DIM, VECTOR_TILE):
                value = rr.vload(projected, row * INDEX_DIM + col)
                rounded = rr.vcast(value, cb.dtypes.bfloat16, mask=input_mask, rounding="rn")
                value = rr.vcast(rounded, cb.dtypes.float32, mask=input_mask)
                rr.vstore(values, col, value, input_mask)
            rr.vmem_bar("vst_vld")
            rope_quantize_row(values, cos_ub, sin_ub, packed, factors, 0)
        cb.mem_copy(cb.tile_slice(output, (1, PACKED_DIM), (source_row, 0)), packed)
        scale_dst = cb.tile_slice(output_scale, (1, SCALE_DIM), (source_row, 0))
        scale_src = cb.tile_slice(factors, (1, SCALE_DIM), (0, 0))
        cb.mem_copy(scale_dst.view(dtype=cb.dtypes.uint8), scale_src)


@cb.kernel
def project_query_grid(payload: cb.Tensor, weight: cb.Tensor, activation_scale: cb.Tensor,
                       weight_scale: cb.Tensor, cosine: cb.Tensor, sine: cb.Tensor,
                       output: cb.Tensor, output_scale: cb.Tensor, num_tokens: cb.ScalarValue):
    """按 M80 和 head 分配融合 Query 投影、RoPE 及量化。"""
    active = min(output.shape[0], max(num_tokens, 0))
    tiles = (active + ROW_TILE - 1) // ROW_TILE * INDEX_H
    for work in cb.range(cb.get_block_idx(), tiles, cb.get_block_num()):
        row = work // INDEX_H * ROW_TILE
        head = work % INDEX_H
        live = min(ROW_TILE, active - row)
        source = payload[row:row + ROW_TILE, None]
        offset = row * (Q_LORA // CACHE_GROUP)
        factors = activation_scale[0:1, offset:offset + ROW_TILE * (Q_LORA // CACHE_GROUP)]
        cos = cosine[row:row + live, None]
        sin = sine[row:row + live, None]
        begin = head * PACKED_DIM
        target = output[row:row + live, begin:begin + PACKED_DIM]
        scale_begin = head * SCALE_DIM
        target_scale = output_scale[row:row + live, scale_begin:scale_begin + SCALE_DIM]
        _project_query_mix(source, weight, factors, weight_scale, cos, sin, target, target_scale, head, live)


@pl.jit.inline(auto_scope=False)
def project_index_query(
    payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor, weight_scale: pl.Tensor,
    cosine: pl.Tensor, sine: pl.Tensor, output: pl.Tensor, output_scale: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], ready: pl.Scalar[pl.TASK_ID],
):
    """保留输入完成依赖，以一组 MIX SPMD 完成全部有效 Query。"""
    active = pl.min(pl.tensor.dim(output, 0), pl.max(pl.cast(num_tokens, pl.INDEX), 0))
    weight_flat = pl.reshape(weight, [1, Q_LORA * INDEX_H * INDEX_DIM])
    scale_flat = pl.tensor.view(weight_scale, [1, (Q_LORA // CACHE_GROUP) * INDEX_H * INDEX_DIM], layout=pl.ND)
    input_scale = pl.reshape(activation_scale, [1, pl.tensor.dim(payload, 0) * (Q_LORA // CACHE_GROUP)])
    target = output[0:active, :]
    factors = output_scale[0:active, :]
    cosine_view = cosine[0:active, :]
    sine_view = sine[0:active, :]
    tiles = (active + ROW_TILE - 1) // ROW_TILE * INDEX_H
    limit = pl.min(TASK_TILE, pl.system.available_cluster_count())
    workers = pl.max(1, pl.min(tiles, limit))
    with pl.spmd(workers, name_hint="dsl_idx_q", deps=[ready], sync_start=True) as done:
        project_query_grid(payload, weight_flat, input_scale, scale_flat, cosine_view, sine_view,
                           target, factors, num_tokens)
    return done


@pl.jit
def validate_query(
    payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor,
    weight_scale: pl.Tensor[[Q_LORA // CACHE_GROUP, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    cosine: pl.Tensor, sine: pl.Tensor, output: pl.InOut[pl.Tensor], output_scale: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
):
    """校验融合生产路径，以及非有效行和额外输出保护行。"""
    rows = pl.tensor.dim(cosine, 0)
    target = output[:rows, :]
    factors = output_scale[:rows, :]
    ready = pl.system.task_dummy(deps=[])
    project_index_query(payload, weight, activation_scale, weight_scale,
                        cosine, sine, target, factors, num_tokens, ready)
    return output, output_scale


def golden_query(tensors):
    """从输入的 MX 字节独立计算逐组 FP32 投影、两次 BF16 舍入及 FP4 编码。"""
    import torch
    from quant import decode_e8m0, quantize_mxfp4_cache, unpack_fp8_nz, unpack_mx_a_scale, unpack_mx_b_scale

    rows = tensors["cosine"].shape[0]
    active = max(0, min(rows, int(tensors["num_tokens"])))
    payload = tensors["payload"].float()[:active]
    weight = unpack_fp8_nz(tensors["weight"]).float()
    a_codes = tensors["activation_scale"].view(torch.uint8).reshape(-1, Q_LORA // CACHE_GROUP)
    a_scale = decode_e8m0(unpack_mx_a_scale(a_codes))[:active]
    b_scale = decode_e8m0(unpack_mx_b_scale(tensors["weight_scale"]))
    total = torch.zeros(active, INDEX_H * INDEX_DIM)
    for group in range(Q_LORA // CACHE_GROUP):
        begin = group * CACHE_GROUP
        part = payload[:, begin:begin + CACHE_GROUP] @ weight[begin:begin + CACHE_GROUP]
        part = part * a_scale[:, group:group + 1]
        part = part * b_scale[group:group + 1]
        total = total + part
    value = total.bfloat16().reshape(active, INDEX_H, INDEX_DIM)
    real, imag = value[:, :, -ROPE_DIM:].float().reshape(active, INDEX_H, ROPE_DIM // 2, 2).unbind(-1)
    cos = tensors["cosine"][:active, None, :]
    sin = tensors["sine"][:active, None, :]
    rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1)
    value[:, :, -ROPE_DIM:] = rotated.flatten(-2).bfloat16()
    packed, scales = quantize_mxfp4_cache(value, CACHE_GROUP, "e8m0", round_to_even=True)
    tensors["output"][:active] = packed.reshape(active, INDEX_H * (PACKED_DIM))
    tensors["output_scale"].view(torch.uint8)[:active] = scales.reshape(active, INDEX_H * (SCALE_DIM))


def build_query_specs(rows, active, kind="random"):
    """构造非均匀 MX 缩放、FP4 舍入中点、双 AIV 尾部和未写区域哨兵值。"""
    import torch
    from golden import ScalarSpec, TensorSpec
    from quant import pack_fp8_nz, pack_mx_a_scale, pack_mx_b_scale

    padded = (rows + ROW_TILE - 1) // ROW_TILE * ROW_TILE
    generator = torch.Generator().manual_seed(3121)
    payload = (torch.randint(-8, 9, (padded, Q_LORA), generator=generator).float() / 2).to(torch.float8_e4m3fn)
    weight_values = torch.randint(-8, 9, (Q_LORA, INDEX_H * INDEX_DIM), generator=generator).float() / 2
    weight = weight_values.to(torch.float8_e4m3fn)
    a_codes = torch.randint(121, 131, (padded, Q_LORA // CACHE_GROUP), generator=generator).to(torch.uint8)
    b_codes = torch.randint(120, 127, (Q_LORA // CACHE_GROUP, INDEX_H * INDEX_DIM), generator=generator).to(torch.uint8)
    angles = torch.randn(rows, ROPE_DIM // 2, generator=generator)
    if kind == "ties":
        payload.zero_()
        payload[:, 0] = 1.0
        weight.zero_()
        pattern = torch.tensor([6.0, -6.0, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25,
                                1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5.0, -5.0])
        weight[0] = pattern.repeat(INDEX_H * INDEX_DIM // pattern.numel()).to(weight.dtype)
        a_codes.fill_(127)
        b_codes.fill_(127)
        angles.zero_()
    payload[max(0, min(padded, active)):] = float("nan")
    activation_scale = pack_mx_a_scale(a_codes).reshape(1, -1).view(torch.float8_e8m0fnu)
    weight_scale = pack_mx_b_scale(b_codes).view(torch.float8_e8m0fnu)
    return [
        TensorSpec("payload", list(payload.shape), payload.dtype, init_value=payload),
        TensorSpec("weight", list(weight.shape), weight.dtype, init_value=pack_fp8_nz(weight)),
        TensorSpec("activation_scale", list(activation_scale.shape), activation_scale.dtype, init_value=activation_scale),
        TensorSpec("weight_scale", list(weight_scale.shape), weight_scale.dtype, init_value=weight_scale),
        TensorSpec("cosine", list(angles.shape), torch.float32, init_value=angles.cos()),
        TensorSpec("sine", list(angles.shape), torch.float32, init_value=angles.sin()),
        TensorSpec("output", [rows + 1, INDEX_H * (PACKED_DIM)], torch.uint8, init_value=0xA5),
        TensorSpec("output_scale", [rows + 1, INDEX_H * (SCALE_DIM)], torch.float8_e8m0fnu, init_value=0.125),
        ScalarSpec("num_tokens", torch.int32, active),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run
    from index_key_dsl import compare_bytes

    parser = argparse.ArgumentParser(description="在 A5 上校验单个 DSL MIX Query 投影与后处理。")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    for rows, active, kind in ((1, 1, "ties"), (17, 15, "random"), (17, 0, "random"),
                              (80, 72, "random"), (83, 81, "random")):
        print(f"[INDEX_QUERY_MIX_DSL] 行数={rows} 有效行={active} 类型={kind}", flush=True)
        result = run(
            fn=validate_query,
            specs=build_query_specs(rows, active, kind),
            golden_fn=golden_query,
            compare_fn={"output": compare_bytes, "output_scale": compare_bytes},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        print(f"[产物] {result.work_dir}", flush=True)
