# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 实现 BF16 C2A 压缩投影，并以 FP32 累加。"""

import cannbotdsl as cb
import pypto.language as pl

from bf16_projection_dsl import project_bf16_nz
from config import FLASH as M


# 模型配置
D = M.hidden_size
HEAD_DIM = M.head_dim

# 分块参数
ROW_TILE = 80
COLUMN_TILE = 64
REDUCTION_TILE = 256
TASK_TILE = 32


@cb.kernel
def project_compressor_tile(
    source: cb.Tensor, weight: cb.Tensor, output: cb.Tensor,
    column: cb.ScalarValue, active_rows: cb.ScalarValue,
):
    """连续执行 K256 BF16 Cube 累加，投影一个 M80/N64 分块。"""
    project_bf16_nz(source, weight, output, column, active_rows, ROW_TILE, COLUMN_TILE, REDUCTION_TILE)


@pl.jit.inline(auto_scope=False)
def _project_task(
    source: pl.Tensor, weight: pl.Tensor, output: pl.Tensor,
    column: pl.Scalar[pl.INDEX], active_rows: pl.Scalar[pl.INDEX], ready: pl.Scalar[pl.TASK_ID],
):
    """提交一个本地 Cube 投影，并返回其完成句柄。"""
    with pl.spmd(1, name_hint="dsl_proj_cmp", deps=[ready]) as done:
        project_compressor_tile(source, weight, output, column, active_rows)
    return done


@pl.jit.inline(auto_scope=False)
def c2a_compressor_project(
    x: pl.Tensor, wkv: pl.Tensor, wgate: pl.Tensor,
    kv_out: pl.Tensor, score_out: pl.Tensor, num_tokens: pl.Scalar[pl.INT32],
    input_ready: pl.Scalar[pl.TASK_ID],
):
    """使用原始 NZ 权重和 FP32 输出缓冲区，对 KV 和 gate 执行投影。"""
    rows = pl.tensor.dim(x, 0)
    active = pl.min(rows, pl.max(pl.cast(num_tokens, pl.INDEX), 0))
    kv_storage = pl.reshape(wkv, [D * HEAD_DIM])
    gate_storage = pl.reshape(wgate, [D * HEAD_DIM])
    kv_weight = pl.reshape(kv_storage, [HEAD_DIM // 16, D * 16])
    gate_weight = pl.reshape(gate_storage, [HEAD_DIM // 16, D * 16])
    completed = pl.array.create(TASK_TILE, pl.TASK_ID)
    for slot in pl.range(TASK_TILE):
        completed[slot] = input_ready
    for row in pl.range(0, active, ROW_TILE):
        physical = pl.min(ROW_TILE, rows - row)
        live = pl.min(ROW_TILE, active - row)
        source = pl.slice(x, [ROW_TILE, D], [row, 0], valid_shape=[physical, D])
        for column in pl.parallel(HEAD_DIM // COLUMN_TILE):
            offset = column * COLUMN_TILE
            kv = pl.slice(kv_out, [ROW_TILE, COLUMN_TILE], [row, offset], valid_shape=[physical, COLUMN_TILE])
            gate = pl.slice(score_out, [ROW_TILE, COLUMN_TILE], [row, offset], valid_shape=[physical, COLUMN_TILE])
            slot = (row // ROW_TILE * (2 * HEAD_DIM // COLUMN_TILE) + 2 * column) % TASK_TILE
            previous_kv = completed[slot]
            previous_gate = completed[slot + 1]
            kv_done = _project_task(source, kv_weight, kv, column, live, previous_kv)
            gate_done = _project_task(source, gate_weight, gate, column, live, previous_gate)
            completed[slot] = kv_done
            completed[slot + 1] = gate_done
    joined = pl.system.task_dummy(deps=[completed])
    return kv_out, score_out, joined


@pl.jit
def validate_projection(
    source: pl.Tensor,
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    kv_out: pl.InOut[pl.Tensor], score_out: pl.InOut[pl.Tensor], num_tokens: pl.Scalar[pl.INT32],
):
    """使用非有效区域哨兵值和一行保护区校验生产路径调度器。"""
    rows = pl.tensor.dim(source, 0)
    kv = pl.slice(kv_out, [rows, HEAD_DIM], [0, 0])
    gate = pl.slice(score_out, [rows, HEAD_DIM], [0, 0])
    ready = pl.system.task_dummy(deps=[])
    c2a_compressor_project(source, wkv, wgate, kv, gate, num_tokens, ready)
    return kv_out, score_out


def golden_projection(tensors):
    """将 BF16 操作数投影为 FP32，中间不对输出舍入。"""
    import torch
    from quant import unpack_bf16_nz

    rows = tensors["source"].shape[0]
    active = max(0, min(rows, int(tensors["num_tokens"])))
    source = tensors["source"][:active].float()
    for weight, output in (("wkv", "kv_out"), ("wgate", "score_out")):
        values = torch.matmul(source, unpack_bf16_nz(tensors[weight]).float())
        tensors[output][:active].copy_(values)


def build_projection_specs(rows=80, active=72, layout_case=False):
    """覆盖 NZ 坐标、行尾部、独立投影和带异常值的非有效输入。"""
    import torch
    from golden import ScalarSpec, TensorSpec
    from quant import pack_bf16_nz

    generator = torch.Generator().manual_seed(2301)
    if layout_case:
        source = ((torch.arange(rows).reshape(-1, 1) * 3 + torch.arange(D)) % 16 - 8).float() / 8
        reduction = torch.arange(D).reshape(-1, 1)
        columns = torch.arange(HEAD_DIM)
        wkv = ((reduction * 3 + columns * 5 + reduction // 16 + columns // 16) % 8 - 4).float() / 16
        wgate = ((reduction * 5 + columns * 3 + reduction // 256 + columns // 64) % 8 - 4).float() / 8
    else:
        source = torch.randn(rows, D, generator=generator)
        wkv = torch.randn(D, HEAD_DIM, generator=generator) / D**0.5
        wgate = torch.randn(D, HEAD_DIM, generator=generator) / D**0.5
    source = source.bfloat16()
    source[max(0, min(rows, active)):] = float("nan")
    return [
        TensorSpec("source", [rows, D], torch.bfloat16, init_value=source),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=pack_bf16_nz(wkv.bfloat16())),
        TensorSpec("wgate", [D, HEAD_DIM], torch.bfloat16, init_value=pack_bf16_nz(wgate.bfloat16())),
        TensorSpec("kv_out", [rows + 1, HEAD_DIM], torch.float32, init_value=-37.0),
        TensorSpec("score_out", [rows + 1, HEAD_DIM], torch.float32, init_value=53.0),
        ScalarSpec("num_tokens", torch.int32, active),
    ]


def projection_comparator(tolerance):
    """限制有效 FP32 输出的误差，并要求非有效输出逐位保持不变。"""
    from golden import ratio_allclose

    compare_active = ratio_allclose(atol=tolerance, rtol=tolerance, max_error_ratio=0.0)
    compare_tail = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)

    def compare(actual, expected, *, inputs, **kwargs):
        rows = inputs["source"].shape[0]
        active = max(0, min(rows, int(inputs["num_tokens"])))
        ok, detail = compare_active(actual[:active], expected[:active], inputs=inputs, **kwargs)
        if not ok:
            return ok, detail
        return compare_tail(actual[active:], expected[active:], inputs=inputs, **kwargs)

    return compare


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="Validate BF16 C2A compressor projections on A5.")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    for rows, active, layout_case in ((17, 15, True), (17, 0, True), (80, 72, False), (81, 81, True), (321, 321, True)):
        print(f"[C2A_PROJECTION_DSL] rows={rows} active={active} layout_case={layout_case}", flush=True)
        tolerance = 0.0 if layout_case else 1e-4
        compare = projection_comparator(tolerance)
        result = run(
            fn=validate_projection,
            specs=build_projection_specs(rows, active, layout_case),
            golden_fn=golden_projection,
            compare_fn={"kv_out": compare, "score_out": compare},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
