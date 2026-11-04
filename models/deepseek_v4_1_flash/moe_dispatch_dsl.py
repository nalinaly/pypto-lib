# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 实现本地 MoE 接收数据压紧和 MX 缩放打包。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
import pypto.language as pl

from config import FLASH as M


# 模型配置
D = M.hidden_size
MX_GROUP = 32

# 分块参数
ROW_TILE = 16
TASK_TILE = 64
BYTE_TILE = 256


@cb.kernel
def gather_rows(
    source: cb.Tensor, source_scale: cb.Tensor, source_weight: cb.Tensor, source_route: cb.Tensor,
    metadata: cb.Tensor, payload: cb.Tensor, scales: cb.Tensor, weights: cb.Tensor, routes: cb.Tensor,
    expert: cb.ScalarValue, row_begin: cb.ScalarValue, row_count: cb.ScalarValue,
):
    """按源顺序压紧一个本地 M16 分块，并逐位保留数据。"""
    width = source.shape[1]
    groups = width // MX_GROUP
    source_stride = source.shape[0] // metadata.shape[0]
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.int8, cb.make_layout((1, width)))
    scale_row = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, groups)))
    packed = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, ROW_TILE * groups)))
    with cb.vf():
        clear_mask = rr.full_mask(elem_bits=8)
        zero = rr.vdups(0, cb.dtypes.uint8, mask=clear_mask)
        for col in range(0, ROW_TILE * groups, BYTE_TILE):
            rr.vstore(packed, col, zero, clear_mask)
    source_begin = 0
    for src in cb.range(metadata.shape[0]):
        count = metadata[src, expert]
        source_end = source_begin + count
        begin = max(row_begin, source_begin)
        end = min(row_begin + row_count, source_end)
        for row in cb.range(begin, end):
            source_row = src * source_stride + row - source_begin
            target_row = row - row_begin
            cb.mem_copy(raw, cb.tile_slice(source, (1, width), (source_row, 0)))
            cb.mem_copy(cb.tile_slice(payload.view(dtype=cb.dtypes.int8), (1, width), (target_row, 0)), raw)
            cb.mem_copy(scale_row, cb.tile_slice(source_scale, (1, groups), (source_row, 0)))
            for group in cb.range(groups):
                offset = group // 2 * ROW_TILE * 2 + target_row * 2 + group % 2
                packed[0, offset] = scale_row[0, group]
            weights[target_row, 0] = source_weight[source_row, 0]
            routes[target_row, 0] = source_route[source_row, 0]
        source_begin = source_end
    cb.mem_copy(scales.view(dtype=cb.dtypes.uint8), packed)


@pl.jit.inline(auto_scope=False)
def _gather_task(
    source: pl.Tensor, source_scale: pl.Tensor, source_weight: pl.Tensor, source_route: pl.Tensor,
    metadata: pl.Tensor, payload: pl.Tensor, scales: pl.Tensor, weights: pl.Tensor, routes: pl.Tensor,
    expert: pl.Scalar[pl.INDEX], row_begin: pl.Scalar[pl.INDEX], row_count: pl.Scalar[pl.INDEX],
    ready: pl.Scalar[pl.TASK_ID],
):
    """在调用者的所有依赖完成后，提交一个本地接收分块。"""
    with pl.spmd(1, name_hint="dsl_dispatch_gather", deps=[ready], allow_early_resolve=False) as done:
        gather_rows(source, source_scale, source_weight, source_route, metadata,
                    payload, scales, weights, routes, expert, row_begin, row_count)
    return done


@pl.jit.inline(auto_scope=False)
def gather_local(
    source: pl.Tensor, source_scale: pl.Tensor, source_weight: pl.Tensor, source_route: pl.Tensor,
    metadata: pl.Tensor, counts: pl.Tensor,
    payload: pl.Tensor, scales: pl.Tensor, weights: pl.Tensor, routes: pl.Tensor,
    wait_ready: pl.Scalar[pl.TASK_ID], push_ready: pl.Scalar[pl.TASK_ID], meta_ready: pl.Scalar[pl.TASK_ID],
    aux_width: pl.constexpr, route_width: pl.constexpr,
):
    """调度互不重叠的本地接收分块，并汇合其完成句柄。

    容量按 M16 对齐，且各来源的数据通道容量相等。计数与来源元数据一致，
    每个来源的计数均不超过对应通道容量。
    输出使用现有专家缓冲区的扁平 ND 存储视图。
    """
    experts = pl.tensor.dim(counts, 0)
    capacity = pl.tensor.dim(payload, 0) // experts
    width = pl.tensor.dim(payload, 1)
    groups = width // MX_GROUP
    ready = pl.system.task_dummy(deps=[wait_ready, push_ready, meta_ready])
    completed = pl.array.create(TASK_TILE, pl.TASK_ID)
    for slot in pl.range(TASK_TILE):
        completed[slot] = ready
    for expert in pl.range(experts):
        count = pl.cast(pl.read(counts, [expert, 0]), pl.INDEX)
        base = expert * capacity
        local_x = pl.slice(source, [capacity, width], [base, 0])
        local_scale = pl.slice(source_scale, [capacity, groups], [base, 0])
        local_weight = pl.slice(source_weight, [capacity, aux_width], [base, 0])
        local_route = pl.slice(source_route, [capacity, route_width], [base, 0])
        for row in pl.parallel(0, count, ROW_TILE):
            live = pl.min(ROW_TILE, count - row)
            out_x = pl.slice(payload, [ROW_TILE, width], [base + row, 0])
            out_s = pl.slice(scales, [1, ROW_TILE * groups], [0, (base + row) * groups])
            out_w = pl.slice(weights, [ROW_TILE, 1], [base + row, 0])
            out_r = pl.slice(routes, [ROW_TILE, 1], [base + row, 0])
            slot = (expert * (capacity // ROW_TILE) + row // ROW_TILE) % TASK_TILE
            previous = completed[slot]
            done = _gather_task(
                local_x, local_scale, local_weight, local_route, metadata,
                out_x, out_s, out_w, out_r, expert, row, live, previous,
            )
            completed[slot] = done
    joined = pl.system.task_dummy(deps=[completed])
    return joined


@pl.jit
def validate_gather(
    source: pl.Tensor, source_scale: pl.Tensor, source_weight: pl.Tensor, source_route: pl.Tensor,
    metadata: pl.Tensor, counts: pl.Tensor,
    payload: pl.InOut[pl.Tensor], scales: pl.InOut[pl.Tensor],
    weights: pl.InOut[pl.Tensor], routes: pl.InOut[pl.Tensor],
):
    """使用尾部哨兵值和一个输出保护分块，验证生产路径调度器。"""
    rows = pl.tensor.dim(source, 0)
    width = pl.tensor.dim(source, 1)
    out_x = pl.slice(payload, [rows, width], [0, 0])
    out_s = pl.slice(scales, [1, rows * (width // MX_GROUP)], [0, 0])
    out_w = pl.slice(weights, [rows, 1], [0, 0])
    out_r = pl.slice(routes, [rows, 1], [0, 0])
    ready = pl.system.task_dummy(deps=[])
    gather_local(source, source_scale, source_weight, source_route, metadata, counts,
                 out_x, out_s, out_w, out_r, ready, ready, ready, 8, 8)
    return payload, scales, weights, routes


def golden_gather(tensors):
    """拼接各来源的有效数据，并打包保持不变的缩放编码。"""
    import torch
    from quant import pack_mx_a_scale

    metadata = tensors["metadata"]
    ranks, experts = metadata.shape
    capacity = tensors["source"].shape[0] // experts
    per_source = capacity // ranks
    groups = tensors["source_scale"].shape[1]
    output_x = tensors["payload"].view(torch.uint8)
    output_s = tensors["scales"].view(torch.uint8)
    for expert in range(experts):
        base = expert * capacity
        selected = []
        for src in range(ranks):
            start = base + src * per_source
            selected.extend(range(start, start + int(metadata[src, expert])))
        count = len(selected)
        if not count:
            continue
        output_x[base:base + count].copy_(tensors["source"][selected].view(torch.uint8))
        tensors["weights"][base:base + count, 0] = tensors["source_weight"][selected, 0]
        tensors["routes"][base:base + count, 0] = tensors["source_route"][selected, 0]
        padded = (count + ROW_TILE - 1) // ROW_TILE * ROW_TILE
        scale_rows = torch.zeros(padded, groups, dtype=torch.uint8)
        scale_rows[:count] = tensors["source_scale"][selected]
        output_s[0, base * groups:(base + padded) * groups] = pack_mx_a_scale(scale_rows).flatten()


def build_gather_specs(case="bounds"):
    """覆盖空来源、M16 边界、所有字节编码，以及保持不变的输出尾部。"""
    import torch
    from golden import TensorSpec

    cases = {
        "bounds": [[0, 7, 16, 9], [0, 8, 0, 8]],
        "zero": [[0, 0, 0, 0], [0, 0, 0, 0]],
        "full": [[80, 80, 80, 80], [80, 80, 80, 80]],
        "multi": [[0, 1, 9, 0], [8, 0, 0, 0], [0, 7, 7, 0], [7, 9, 1, 0]],
    }
    metadata = torch.tensor(cases[case], dtype=torch.int32)
    ranks, experts = metadata.shape
    capacity = ranks * 80
    rows = experts * capacity
    groups = D // MX_GROUP
    row_id = torch.arange(rows).reshape(-1, 1)
    raw_x = ((row_id * 29 + torch.arange(D) * 17) % 256).to(torch.uint8)
    raw_s = ((row_id * 31 + torch.arange(groups) * 13) % 256).to(torch.uint8)
    weight = (row_id * 11 + torch.arange(8)).float() / 128.0
    weight[::7, 0] = -0.0
    route = (row_id * 100003 + torch.arange(8)).to(torch.int32)
    output_rows = rows + ROW_TILE
    output_x = torch.full((output_rows, D), 0x55, dtype=torch.uint8).view(torch.float8_e4m3fn)
    output_s = torch.full((1, output_rows * groups), 0xA5, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return [
        TensorSpec("source", [rows, D], torch.int8, init_value=raw_x.view(torch.int8)),
        TensorSpec("source_scale", [rows, groups], torch.uint8, init_value=raw_s),
        TensorSpec("source_weight", [rows, 8], torch.float32, init_value=weight),
        TensorSpec("source_route", [rows, 8], torch.int32, init_value=route),
        TensorSpec("metadata", [ranks, experts], torch.int32, init_value=metadata),
        TensorSpec("counts", [experts, 1], torch.int32, init_value=metadata.sum(0).int().reshape(-1, 1)),
        TensorSpec("payload", [output_rows, D], torch.float8_e4m3fn, init_value=output_x),
        TensorSpec("scales", [1, output_rows * groups], torch.float8_e8m0fnu, init_value=output_s),
        TensorSpec("weights", [output_rows, 1], torch.float32, init_value=-37.0),
        TensorSpec("routes", [output_rows, 1], torch.int32, init_value=-37),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run
    from window_kv_dsl import compare_payload_bytes

    parser = argparse.ArgumentParser(description="Validate local DSL MoE receive gather on A5.")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--case", choices=("all", "bounds", "zero", "full", "multi"), default="all")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    cases = ("bounds", "zero", "full", "multi") if args.case == "all" else (args.case,)
    for case in cases:
        print(f"[DISPATCH_GATHER_DSL] case={case}", flush=True)
        result = run(
            fn=validate_gather,
            specs=build_gather_specs(case),
            golden_fn=golden_gather,
            compare_fn={name: compare_payload_bytes for name in ("payload", "scales", "weights", "routes")},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
