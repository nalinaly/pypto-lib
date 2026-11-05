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


@cb.kernel
def project_index_weight_tile(
    source: cb.Tensor, weight: cb.Tensor, output: cb.Tensor, active_rows: cb.ScalarValue,
):
    """以 M80/N32/K256 计算一个 K1280 分段，并写出 FP32 部分和。"""
    project_bf16_nz(source, weight, output, 0, active_rows, ROW_TILE, INDEX_H, REDUCTION_TILE)


@cb.kernel
def reduce_index_weight_tile(
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


@pl.jit.inline(auto_scope=False)
def _project_task(
    source: pl.Tensor, weight: pl.Tensor, output: pl.Tensor,
    active_rows: pl.Scalar[pl.INDEX], ready: pl.Scalar[pl.TASK_ID],
):
    """提交一个独立的本卡 AIC 投影分块。"""
    with pl.spmd(1, name_hint="dsl_proj_idx_w", deps=[ready]) as done:
        project_index_weight_tile(source, weight, output, active_rows)
    return done


@pl.jit.inline(auto_scope=False)
def _reduce_task(
    partials: pl.Tensor, output: pl.Tensor, active_rows: pl.Scalar[pl.INDEX],
    row: pl.Scalar[pl.INDEX], ready: pl.Scalar[pl.TASK_ID],
):
    """等待投影完成后提交本卡 AIV 合并任务。"""
    with pl.spmd(1, name_hint="dsl_reduce_idx_w", deps=[ready]) as done:
        reduce_index_weight_tile(partials, output, active_rows, row)
    return done


@pl.jit.inline(auto_scope=False)
def project_index_weights(
    source: pl.Tensor, weight: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ], output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], ready: pl.Scalar[pl.TASK_ID],
):
    """由 simpler 调度四路 K1280 投影、FP32 合并及下游完成依赖。"""
    tokens = pl.tensor.dim(source, 0)
    active = pl.min(tokens, pl.max(pl.cast(num_tokens, pl.INDEX), 0))
    storage = pl.reshape(weight, [D * INDEX_H])
    raw_weight = pl.reshape(storage, [INDEX_H // 16, D * 16])
    partials = pl.create_tensor([SPLIT_TILE * tokens, INDEX_H], dtype=pl.FP32)
    completed = pl.array.create(TASK_TILE, pl.TASK_ID)
    for slot in pl.range(TASK_TILE):
        completed[slot] = ready
    for row in pl.range(0, active, ROW_TILE):
        physical = pl.min(ROW_TILE, tokens - row)
        live = pl.min(ROW_TILE, active - row)
        for split in pl.parallel(SPLIT_TILE):
            begin = split * (D // SPLIT_TILE)
            tile = pl.slice(source, [ROW_TILE, D // SPLIT_TILE], [row, begin], valid_shape=[physical, D // SPLIT_TILE])
            weights = raw_weight[:, begin * 16:(begin + D // SPLIT_TILE) * 16]
            partial_row = split * tokens + row
            partial = pl.slice(partials, [ROW_TILE, INDEX_H], [partial_row, 0], valid_shape=[physical, INDEX_H])
            slot = (row // ROW_TILE * SPLIT_TILE + split) % TASK_TILE
            previous = completed[slot]
            done = _project_task(tile, weights, partial, live, previous)
            completed[slot] = done
    projected = pl.system.task_dummy(deps=[completed])
    merged = pl.array.create(TASK_TILE, pl.TASK_ID)
    for slot in pl.range(TASK_TILE):
        merged[slot] = projected
    for row in pl.parallel(0, active, MERGE_ROW_TILE):
        physical = pl.min(MERGE_ROW_TILE, tokens - row)
        live = pl.min(MERGE_ROW_TILE, active - row)
        target = pl.slice(output, [MERGE_ROW_TILE, INDEX_H], [row, 0], valid_shape=[physical, INDEX_H])
        slot = row // MERGE_ROW_TILE % TASK_TILE
        previous = merged[slot]
        done = _reduce_task(partials, target, live, row, previous)
        merged[slot] = done
    joined = pl.system.task_dummy(deps=[merged])
    return joined


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
