# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 执行 Indexer head 权重的四路投影及 FP32 顺序合并。"""

import cannbotdsl as cb
from cannbotdsl.lang.jit import jit
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
import pypto.language as pl

from bf16_projection_dsl import project_bf16_nz
from config import FLASH as M


# 模型配置
D = M.hidden_size
INDEX_H = M.index_n_heads

# 分块参数
ROW_TILE = 80
REDUCTION_TILE = 256
SPLIT_TILE = 4
MERGE_ROW_TILE = 16
VECTOR_TILE = 64
TASK_TILE = 12


@jit
def _project_index_weight_tile(
    source: cb.Tensor, weight: cb.Tensor, output: cb.Tensor, active_rows: cb.ScalarValue,
):
    """以 M80/N32/K256 计算一个 K1280 分段，并写出 FP32 部分和。"""
    project_bf16_nz(source, weight, output, 0, active_rows, ROW_TILE, INDEX_H, REDUCTION_TILE)


@jit
def _reduce_index_weight_tile(
    partials: cb.Tensor, output: cb.Tensor, active_rows: cb.ScalarValue, row: cb.ScalarValue,
):
    """按 ((分段 0 + 分段 1) + 分段 2) + 分段 3 合并，最后舍入为 BF16。"""
    tokens = partials.shape[0] // SPLIT_TILE
    first = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((MERGE_ROW_TILE, INDEX_H)))
    second = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((MERGE_ROW_TILE, INDEX_H)))
    third = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((MERGE_ROW_TILE, INDEX_H)))
    fourth = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((MERGE_ROW_TILE, INDEX_H)))
    result = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((MERGE_ROW_TILE, INDEX_H)))
    cb.mem_copy(first, partials[row:row + active_rows, None])
    cb.mem_copy(second, partials[tokens + row:tokens + row + active_rows, None])
    cb.mem_copy(third, partials[2 * tokens + row:2 * tokens + row + active_rows, None])
    cb.mem_copy(fourth, partials[3 * tokens + row:3 * tokens + row + active_rows, None])
    with cb.vf():
        full = rr.full_mask()
        for offset in range(0, MERGE_ROW_TILE * INDEX_H, VECTOR_TILE):
            value = rr.vadd(rr.vload(first, offset), rr.vload(second, offset), mask=full)
            value = rr.vadd(value, rr.vload(third, offset), mask=full)
            value = rr.vadd(value, rr.vload(fourth, offset), mask=full)
            rounded = rr.vcast(value, cb.dtypes.bfloat16, mask=full, rounding="rn")
            rr.vstore_pack(result, offset, rounded, full, pack_mode=rr.PackMode.B32_TO_B16)
    cb.mem_copy(output[0:active_rows, None], result[0:active_rows, None])


@cb.kernel
def project_index_weight_grid(source: cb.Tensor, weight: cb.Tensor, partials: cb.Tensor,
                              num_tokens: cb.ScalarValue):
    """按 M80 和四个 K1280 分段分配投影。"""
    active = min(source.shape[0], max(num_tokens, 0))
    stride = partials.shape[0] // SPLIT_TILE
    tiles = (active + ROW_TILE - 1) // ROW_TILE * SPLIT_TILE
    for work in cb.range(cb.get_block_idx(), tiles, cb.get_block_num()):
        row = work // SPLIT_TILE * ROW_TILE
        split = work % SPLIT_TILE
        live = min(ROW_TILE, active - row)
        begin = split * (D // SPLIT_TILE)
        src = source[row:row + live, begin:begin + D // SPLIT_TILE]
        matrix = weight[None, begin * 16:(begin + D // SPLIT_TILE) * 16]
        offset = split * stride + row
        dst = partials[offset:offset + live, None]
        _project_index_weight_tile(src, matrix, dst, live)


@cb.kernel
def reduce_index_weight_grid(partials: cb.Tensor, output: cb.Tensor, num_tokens: cb.ScalarValue):
    """按 M16 分配合并，四路部分和仍按固定顺序相加。"""
    active = min(output.shape[0], max(num_tokens, 0))
    tiles = (active + MERGE_ROW_TILE - 1) // MERGE_ROW_TILE
    for tile in cb.range(cb.get_block_idx(), tiles, cb.get_block_num()):
        row = tile * MERGE_ROW_TILE
        live = min(MERGE_ROW_TILE, active - row)
        dst = output[row:row + live, None]
        _reduce_index_weight_tile(partials, dst, live, row)


@pl.jit.inline(auto_scope=False)
def project_index_weights(
    source: pl.Tensor, weight: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ], output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], ready: pl.Scalar[pl.TASK_ID],
):
    """依次提交四路投影和 FP32 顺序合并，返回完整权重结果的完成依赖。"""
    tokens = pl.tensor.dim(source, 0)
    active = pl.min(tokens, pl.max(pl.cast(num_tokens, pl.INDEX), 0))
    storage = pl.reshape(weight, [D * INDEX_H])
    raw_weight = pl.reshape(storage, [INDEX_H // 16, D * 16])
    partials = pl.create_tensor([SPLIT_TILE * tokens, INDEX_H], dtype=pl.FP32)
    tiles = (active + ROW_TILE - 1) // ROW_TILE * SPLIT_TILE
    workers = pl.max(1, pl.min(TASK_TILE, tiles))
    with pl.spmd(workers, name_hint="dsl_proj_idx_w", deps=[ready]) as projected:
        project_index_weight_grid(source, raw_weight, partials, num_tokens)
    target = output[0:active, 0:INDEX_H]
    merge_workers = pl.max(1, pl.min(TASK_TILE, (active + MERGE_ROW_TILE - 1) // MERGE_ROW_TILE))
    with pl.spmd(merge_workers, name_hint="dsl_reduce_idx_w", deps=[projected]) as done:
        reduce_index_weight_grid(partials, target, num_tokens)
    return done


@pl.jit
def validate_projection(
    source: pl.Tensor, weight: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    output: pl.InOut[pl.Tensor], num_tokens: pl.Scalar[pl.INT32],
):
    """通过生产调度路径校验有效行、非有效尾部及额外保护行。"""
    rows = pl.tensor.dim(source, 0)
    target = output[:rows, :]
    ready = pl.system.task_dummy(deps=[])
    project_index_weights(source, weight, target, num_tokens, ready)
    return output


def golden_projection(tensors):
    """按 K256 累加四个 K1280 分段，FP32 顺序相加后舍入一次。"""
    import torch
    from quant import unpack_bf16_nz

    active = max(0, min(tensors["source"].shape[0], int(tensors["num_tokens"])))
    source = tensors["source"][:active].float()
    weight = unpack_bf16_nz(tensors["weight"]).float()
    partials = []
    for split in range(SPLIT_TILE):
        value = torch.zeros(active, INDEX_H, dtype=torch.float32)
        begin = split * (D // SPLIT_TILE)
        for k in range(begin, begin + D // SPLIT_TILE, REDUCTION_TILE):
            value = value + source[:, k:k + REDUCTION_TILE] @ weight[k:k + REDUCTION_TILE]
        partials.append(value)
    total = partials[0]
    for value in partials[1:]:
        total = total + value
    tensors["output"][:active].copy_(total.bfloat16())


def build_projection_specs(rows, active, kind="layout"):
    """构造 NZ 坐标、跨分段累加、BF16 中点及非有效输入哨兵值。"""
    import torch
    from golden import ScalarSpec, TensorSpec
    from quant import pack_bf16_nz

    generator = torch.Generator().manual_seed(3113)
    if kind == "random":
        source = torch.randn(rows, D, generator=generator).bfloat16()
        weight = (torch.randn(D, INDEX_H, generator=generator) / D**0.5).bfloat16()
    elif kind == "ties":
        source = torch.zeros(rows, D, dtype=torch.bfloat16)
        weight = torch.zeros(D, INDEX_H, dtype=torch.bfloat16)
        for split in range(SPLIT_TILE):
            source[:, split * (D // SPLIT_TILE)] = 1.0
            source[:, split * (D // SPLIT_TILE) + REDUCTION_TILE] = 1.0
        for column in range(INDEX_H):
            sign = 1.0 if column % 2 == 0 else -1.0
            case = column // 2 % 4
            if case < 2:
                weight[0, column] = sign * (1.0 + case / 128)
                weight[D // SPLIT_TILE, column] = sign / 256
            elif case == 2:
                # FP32 大数抵消保留左结合顺序，最后一段的小数单独留下。
                weight[0, column] = sign * 2.0**24
                weight[D // SPLIT_TILE, column] = sign
                weight[2 * D // SPLIT_TILE, column] = -sign * 2.0**24
                weight[3 * D // SPLIT_TILE, column] = sign * 0.5
            else:
                # 第一段的中点不能提前舍入为 BF16。
                weight[0, column] = sign
                weight[REDUCTION_TILE, column] = sign / 256
                weight[D // SPLIT_TILE, column] = sign / 256
    else:
        row = torch.arange(rows).reshape(-1, 1)
        k = torch.arange(D)
        source = (((row * 3 + k + k // (D // SPLIT_TILE)) % 16 - 8).float() / 8).bfloat16()
        k = k.reshape(-1, 1)
        column = torch.arange(INDEX_H)
        weight = (((k * 3 + column * 5 + k // REDUCTION_TILE + column // 16) % 8 - 4) / 16).bfloat16()
    source[max(0, min(rows, active)):] = float("nan")
    return [
        TensorSpec("source", [rows, D], torch.bfloat16, init_value=source),
        TensorSpec("weight", [D, INDEX_H], torch.bfloat16, init_value=pack_bf16_nz(weight)),
        TensorSpec("output", [rows + 1, INDEX_H], torch.bfloat16, init_value=-37.0),
        ScalarSpec("num_tokens", torch.int32, active),
    ]


def projection_comparator(actual, expected, *, inputs, **kwargs):
    """按 BF16 精度比较随机有效行，非有效尾部及保护行要求完全一致。"""
    from golden import ratio_allclose

    active = max(0, min(inputs["source"].shape[0], int(inputs["num_tokens"])))
    compare_live = ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0)
    exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    passed, detail = compare_live(actual[:active], expected[:active], inputs=inputs, **kwargs)
    if not passed:
        return passed, detail
    return exact(actual[active:], expected[active:], inputs=inputs, **kwargs)


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="在 A5 上校验 DSL Indexer head 权重投影及合并。")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    cases = ((1, 1, "ties"), (17, 15, "layout"), (17, 0, "layout"), (80, 72, "random"), (201, 199, "layout"))
    for rows, active, kind in cases:
        print(f"[INDEX_WEIGHT_DSL] 行数={rows} 有效行={active} 类型={kind}", flush=True)
        result = run(
            fn=validate_projection,
            specs=build_projection_specs(rows, active, kind),
            golden_fn=golden_projection,
            compare_fn={"output": projection_comparator if kind == "random" else exact},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        print(f"[产物] {result.work_dir}", flush=True)
