# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 实现本地 MoE 发送暂存、接收压紧和 MX 缩放布局转换。"""

import cannbotdsl as cb
from cannbotdsl.lang.jit import jit
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
import pypto.language as pl

from config import FLASH as M


# 模型配置
D = M.hidden_size
MX_GROUP = 32

# 分块参数
ROW_TILE = 16
STAGE_ROW_TILE = 8
TASK_TILE = 64
BYTE_TILE = 256


@jit
def _stage_rows(
    source: cb.Tensor, packed_scale: cb.Tensor, payload: cb.Tensor, scales: cb.Tensor,
    scale_row_begin: cb.ScalarValue, active_rows: cb.ScalarValue,
):
    """复制八行物理数据，仅解包有效行的缩放值。"""
    width = source.shape[1]
    groups = width // MX_GROUP
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.int8, cb.make_layout((STAGE_ROW_TILE, width)))
    packed = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, ROW_TILE * groups)))
    unpacked = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, groups)))
    cb.mem_copy(raw, source.view(dtype=cb.dtypes.int8))
    cb.mem_copy(payload, raw)
    if active_rows > 0:
        cb.mem_copy(packed, packed_scale.view(dtype=cb.dtypes.uint8))
        for row in cb.range(active_rows):
            for group in cb.range(groups):
                offset = group // 2 * ROW_TILE * 2 + (scale_row_begin + row) * 2 + group % 2
                unpacked[0, group] = packed[0, offset]
            cb.mem_copy(cb.tile_slice(scales, (1, groups), (row, 0)), unpacked)


@cb.kernel
def stage_grid(source: cb.Tensor, packed_scale: cb.Tensor, payload: cb.Tensor,
               scales: cb.Tensor, num_tokens: cb.ScalarValue):
    """按八行分配传输数据复制和有效缩放解包。"""
    groups = source.shape[1] // MX_GROUP
    active = min(source.shape[0], max(num_tokens, 0))
    for tile in cb.range(cb.get_block_idx(), source.shape[0] // STAGE_ROW_TILE, cb.get_block_num()):
        row = tile * STAGE_ROW_TILE
        live = min(STAGE_ROW_TILE, max(active - row, 0))
        offset = row // ROW_TILE * ROW_TILE * groups
        src = source[row:row + STAGE_ROW_TILE, None]
        factors = packed_scale[0:1, offset:offset + ROW_TILE * groups]
        dst = payload[row:row + STAGE_ROW_TILE, None]
        out_scale = scales[row:row + live, None]
        _stage_rows(src, factors, dst, out_scale, row % ROW_TILE, live)


@pl.jit.inline(auto_scope=False)
def stage_local(source: pl.Tensor, packed_scale: pl.Tensor, payload: pl.Tensor, scales: pl.Tensor,
                num_tokens: pl.Scalar[pl.INT32]):
    """以一组 AIV SPMD 准备发送数据，保留非有效缩放行。"""
    rows = pl.tensor.dim(source, 0)
    active = pl.min(rows, pl.max(pl.cast(num_tokens, pl.INDEX), 0))
    target = payload[0:rows, :]
    factors = scales[0:active, :]
    workers = pl.max(1, pl.min(TASK_TILE, rows // STAGE_ROW_TILE))
    with pl.spmd(workers, name_hint="dsl_dispatch_stage") as done:
        stage_grid(source, packed_scale, target, factors, num_tokens)
    return done


@jit
def _gather_rows(
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


@cb.kernel
def gather_grid(source: cb.Tensor, source_scale: cb.Tensor, source_weight: cb.Tensor, source_route: cb.Tensor,
                metadata: cb.Tensor, counts: cb.Tensor, payload: cb.Tensor, scales: cb.Tensor,
                weights: cb.Tensor, routes: cb.Tensor):
    """按专家的 M16 有效块分配工作，保留来源顺序和打包字节。"""
    experts = counts.shape[0]
    capacity = payload.shape[0] // experts
    groups = payload.shape[1] // MX_GROUP
    worker = cb.get_block_idx()
    workers = cb.get_block_num()
    prefix = 0
    for expert in cb.range(experts):
        count = counts[expert, 0]
        tiles = (count + ROW_TILE - 1) // ROW_TILE
        base = expert * capacity
        src = source[base:base + capacity, None]
        source_factors = source_scale[base:base + capacity, None]
        source_weights = source_weight[base:base + capacity, None]
        source_routes = source_route[base:base + capacity, None]
        first = (worker + workers - prefix % workers) % workers
        for tile in cb.range(first, tiles, workers):
            row = tile * ROW_TILE
            live = min(ROW_TILE, count - row)
            begin = base + row
            dst = payload[begin:begin + ROW_TILE, None]
            offset = begin * groups
            factors = scales[0:1, offset:offset + ROW_TILE * groups]
            out_weights = weights[begin:begin + ROW_TILE, None]
            out_routes = routes[begin:begin + ROW_TILE, None]
            _gather_rows(src, source_factors, source_weights, source_routes, metadata,
                         dst, factors, out_weights, out_routes, expert, row, live)
        prefix = prefix + tiles


@pl.jit.inline(auto_scope=False)
def gather_local(
    source: pl.Tensor, source_scale: pl.Tensor, source_weight: pl.Tensor, source_route: pl.Tensor,
    metadata: pl.Tensor, counts: pl.Tensor, payload: pl.Tensor, scales: pl.Tensor,
    weights: pl.Tensor, routes: pl.Tensor,
    wait_ready: pl.Scalar[pl.TASK_ID], publish_ready: pl.Scalar[pl.TASK_ID], meta_ready: pl.Scalar[pl.TASK_ID],
    aux_width: pl.constexpr, route_width: pl.constexpr,
):
    """保留三个协议依赖，以一组 AIV SPMD 压紧本地接收区。"""
    rows = pl.tensor.dim(payload, 0)
    width = pl.tensor.dim(payload, 1)
    local_x = pl.slice(source, [rows, width], [0, 0])
    local_scale = pl.slice(source_scale, [rows, width // MX_GROUP], [0, 0])
    local_weight = pl.slice(source_weight, [rows, aux_width], [0, 0])
    local_route = pl.slice(source_route, [rows, route_width], [0, 0])
    ready = pl.system.task_dummy(deps=[wait_ready, publish_ready, meta_ready])
    workers = pl.min(TASK_TILE, pl.system.available_aiv_count())
    with pl.spmd(workers, name_hint="dsl_dispatch_gather", deps=[ready], allow_early_resolve=False) as done:
        gather_grid(local_x, local_scale, local_weight, local_route, metadata, counts,
                    payload, scales, weights, routes)
    return done


@pl.jit
def validate_stage(
    source: pl.Tensor, packed_scale: pl.Tensor,
    payload: pl.InOut[pl.Tensor], scales: pl.InOut[pl.Tensor], num_tokens: pl.Scalar[pl.INT32],
):
    """使用生产路径的暂存逻辑，保留非有效缩放哨兵值和输出保护区。"""
    rows = pl.tensor.dim(source, 0)
    width = pl.tensor.dim(source, 1)
    out_x = pl.slice(payload, [rows, width], [0, 0])
    out_s = pl.slice(scales, [rows, width // MX_GROUP], [0, 0])
    stage_local(source, packed_scale, out_x, out_s, num_tokens)
    return payload, scales


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


def golden_stage(tensors):
    """复制所有物理数据行，并恢复有效的 ND 缩放行。"""
    import torch
    from quant import unpack_mx_a_scale

    rows, width = tensors["source"].shape
    active = max(0, min(rows, int(tensors["num_tokens"])))
    tensors["payload"][:rows].copy_(tensors["source"].view(torch.int8))
    packed = tensors["packed_scale"].view(torch.uint8).reshape(rows, width // MX_GROUP)
    tensors["scales"][:active].copy_(unpack_mx_a_scale(packed)[:active])


def build_stage_specs(rows=32, active=17):
    """覆盖所有字节编码、有效计数截断，以及保持不变的缩放尾部。"""
    import torch
    from golden import ScalarSpec, TensorSpec
    from quant import pack_mx_a_scale

    groups = D // MX_GROUP
    row_id = torch.arange(rows).reshape(-1, 1)
    raw_x = ((row_id * 43 + torch.arange(D) * 37) % 256).to(torch.uint8)
    raw_s = ((row_id * 29 + torch.arange(groups) * 13) % 256).to(torch.uint8)
    packed = pack_mx_a_scale(raw_s).view(torch.float8_e8m0fnu).reshape(1, -1)
    return [
        TensorSpec("source", [rows, D], torch.float8_e4m3fn, init_value=raw_x.view(torch.float8_e4m3fn)),
        TensorSpec("packed_scale", [1, rows * groups], torch.float8_e8m0fnu, init_value=packed),
        TensorSpec("payload", [rows + STAGE_ROW_TILE, D], torch.int8, init_value=0x5A),
        TensorSpec("scales", [rows + STAGE_ROW_TILE, groups], torch.uint8, init_value=0xA5),
        ScalarSpec("num_tokens", torch.int32, active),
    ]


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

    parser = argparse.ArgumentParser(description="Validate local DSL MoE staging and gather on A5.")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--op", choices=("all", "stage", "gather"), default="all")
    parser.add_argument("--case", choices=("all", "bounds", "zero", "full", "multi"), default="all")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    checks = []
    if args.op in ("all", "stage"):
        for rows, active in [(32, n) for n in (-1, 0, 7, 8, 9, 15, 16, 17, 35)] + [(80, 72)]:
            label = f"[DISPATCH_STAGE_DSL] rows={rows} active={active}"
            checks.append((label, validate_stage, build_stage_specs(rows, active), golden_stage, ("payload", "scales")))
    if args.op in ("all", "gather"):
        cases = ("bounds", "zero", "full", "multi") if args.case == "all" else (args.case,)
        for case in cases:
            label = f"[DISPATCH_GATHER_DSL] case={case}"
            outputs = ("payload", "scales", "weights", "routes")
            checks.append((label, validate_gather, build_gather_specs(case), golden_gather, outputs))
    for label, function, specs, golden, outputs in checks:
        print(label, flush=True)
        result = run(
            fn=function,
            specs=specs,
            golden_fn=golden,
            compare_fn={name: compare_payload_bytes for name in outputs},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
