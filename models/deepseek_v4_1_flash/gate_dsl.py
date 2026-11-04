# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 实现本地 MoE 路由器归一化、MXFP8 量化、投影和 TopK。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from cannbotdsl.ops.cube import set_fp32_mode
from cannbotdsl.ring_buffer import RingBuffer
import pypto.language as pl

from config import FLASH as M
from mxfp8_dsl import _quant_upward_row


# 模型配置
D = M.hidden_size
EPS = M.rms_norm_eps
MX_GROUP = 32
TOPK = M.num_experts_per_tok
ROUTE_SCALE = M.routed_scaling_factor

# 分块参数
NORM_ROW_TILE = 8
NORM_COLUMN_TILE = 512
QUANT_ROW_TILE = 16
QUANT_COLUMN_TILE = 1024
VECTOR_TILE = 64
PROJECT_ROW_TILE = 16
PROJECT_COLUMN_TILE = 16
PROJECT_K_TILE = 512
ROUTE_ROW_TILE = 8


@cb.kernel
def project_scores(
    source: cb.Tensor, weight: cb.Tensor, bias: cb.Tensor, scores: cb.Tensor, biased: cb.Tensor,
):
    """计算一个 FP32 M16/N16 路由器分块，并将分数分给两个 AIV。"""
    lane = cb.get_subblock_id()
    set_fp32_mode()
    a_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.float32, cb.make_layout((PROJECT_ROW_TILE, PROJECT_K_TILE)), depth=1)
    b_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.float32, cb.make_layout((PROJECT_COLUMN_TILE, PROJECT_K_TILE)), depth=1)
    a_l0 = RingBuffer(cb.MemLoc.L0A, cb.dtypes.float32, cb.make_layout((PROJECT_ROW_TILE, PROJECT_K_TILE)), depth=1)
    b_l0 = RingBuffer(cb.MemLoc.L0B, cb.dtypes.float32, cb.make_layout((PROJECT_COLUMN_TILE, PROJECT_K_TILE)), depth=1)
    c_l0 = RingBuffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((PROJECT_ROW_TILE, PROJECT_COLUMN_TILE)), depth=1)
    cv_shape = (PROJECT_ROW_TILE // 2, PROJECT_COLUMN_TILE)
    logits = RingBuffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout(cv_shape), depth=1)
    out = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout(cv_shape))
    out_bias = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout(cv_shape))
    bias_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
    cb.mem_copy(cb.tile_slice(bias_ub, (1, PROJECT_COLUMN_TILE), (0, 0)), bias)
    accumulator = c_l0.next_write()
    for k in cb.range(D // PROJECT_K_TILE):
        a = a_l1.next_write()
        b = b_l1.next_write()
        cb.mem_copy(a, cb.tile_slice(source, (PROJECT_ROW_TILE, PROJECT_K_TILE), (0, k)))
        cb.mem_copy(b, cb.tile_slice(weight, (PROJECT_COLUMN_TILE, PROJECT_K_TILE), (0, k)))
        cb.mem_copy(a_l0.next_write(), a_l1.next_read())
        cb.mem_copy(b_l0.next_write(), b_l1.next_read())
        cb.matmul(accumulator, a_l0.next_read(), b_l0.next_read(), init=k == 0)
    incoming = logits.next_write()
    cb.mem_copy(incoming, c_l0.next_read().partition(dim=0))
    values = logits.next_read()
    with cb.vf():
        full = rr.full_mask()
        offsets = rr.varange(0, cb.dtypes.uint32)
        offset_mask = rr.vdups(PROJECT_COLUMN_TILE - 1, cb.dtypes.uint32, mask=full)
        offsets = rr.vbitwise_and(offsets, offset_mask, mask=full)
        bias_value = rr.vgather(bias_ub, offsets, mask=full)
        for col in range(0, PROJECT_ROW_TILE // 2 * PROJECT_COLUMN_TILE, VECTOR_TILE):
            value = rr.vload(values, col)
            positive = rr.vmaxs(value, 0.0, mask=full)
            negative_abs = rr.vmuls(rr.vabs(value, mask=full), -1.0, mask=full)
            exponential = rr.vexp(negative_abs, mask=full)
            tail = rr.vlog(rr.vadds(exponential, 1.0, mask=full), mask=full)
            softplus = rr.vadd(positive, tail, mask=full)
            negative_shift = rr.vadds(rr.vmuls(value, -1.0, mask=full), -10.0, mask=full)
            negative_mask = rr.vmins(rr.vmaxs(negative_shift, 0.0, mask=full), 1.0, mask=full)
            negative_exp = rr.vexp(rr.vmins(value, 0.0, mask=full), mask=full)
            floor = rr.vmul(negative_mask, negative_exp, mask=full)
            score = rr.vsqrt(rr.vmax(softplus, floor, mask=full), mask=full)
            rr.vstore(out, col, score, full)
            rr.vstore(out_bias, col, rr.vadd(score, bias_value, mask=full), full)
    cb.mem_copy(scores.partition(dim=0, part_id=lane), out)
    cb.mem_copy(biased.partition(dim=0, part_id=lane), out_bias)


@cb.kernel
def normalize_rows(source: cb.Tensor, weight: cb.Tensor, output: cb.Tensor, valid_rows: cb.ScalarValue):
    """使用分块 FP32 求和、BF16 舍入和尾部补零，对八行执行归一化。"""
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, D)))
    gamma = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((D,)))
    result = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, D)))
    cb.mem_copy(gamma, weight)
    for row in cb.range(NORM_ROW_TILE):
        if row < valid_rows:
            cb.mem_copy(raw, cb.tile_slice(source, (1, D), (row, 0)))
            with cb.vf():
                full = rr.full_mask()
                total = rr.vdups(0.0, cb.dtypes.float32, mask=full)
                for begin in range(0, D, NORM_COLUMN_TILE):
                    partial = rr.vdups(0.0, cb.dtypes.float32, mask=full)
                    for col in range(begin, begin + NORM_COLUMN_TILE, VECTOR_TILE):
                        loaded = rr.vload_unpack(raw, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                        value = rr.vcast(loaded, cb.dtypes.float32, mask=full)
                        square = rr.vmul(value, value, mask=full)
                        partial = rr.vadd(partial, square, mask=full)
                    reduced = rr.vreduce_sum(partial, mask=full)
                    total = rr.vadd(total, reduced, mask=full)
                mean = rr.vmuls(total, 1.0 / D, mask=full)
                denominator = rr.vsqrt(rr.vadds(mean, EPS, mask=full), mask=full)
                one = rr.vdups(1.0, cb.dtypes.float32, mask=full)
                inverse = rr.vdiv(one, denominator, mask=full, precision_mode=True)
                inverse = rr.vdup(inverse, mask=full)
                for col in range(0, D, VECTOR_TILE):
                    loaded = rr.vload_unpack(raw, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    factor = rr.vload_unpack(gamma, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    value = rr.vcast(loaded, cb.dtypes.float32, mask=full)
                    factor = rr.vcast(factor, cb.dtypes.float32, mask=full)
                    scaled = rr.vmul(value, inverse, mask=full)
                    normalized = rr.vmul(scaled, factor, mask=full)
                    rounded = rr.vcast(normalized, cb.dtypes.bfloat16, mask=full, rounding="rn")
                    widened = rr.vcast(rounded, cb.dtypes.float32, mask=full)
                    rr.vstore(result, col, widened, full)
        else:
            with cb.vf():
                full = rr.full_mask()
                zero = rr.vdups(0.0, cb.dtypes.float32, mask=full)
                for col in range(0, D, VECTOR_TILE):
                    rr.vstore(result, col, zero, full)
        cb.mem_copy(cb.tile_slice(output, (1, D), (row, 0)), result)


@cb.kernel
def quantize_rows(source: cb.Tensor, payload: cb.Tensor, scales: cb.Tensor):
    """量化一个 M16/K1024 数据块，并使用配对的 E8M0 缩放值。"""
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, QUANT_COLUMN_TILE)))
    values = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((QUANT_COLUMN_TILE + VECTOR_TILE,)))
    packed = Buffer(cb.MemLoc.UB, cb.dtypes.float8_e4m3fn, cb.make_layout((1, QUANT_COLUMN_TILE)))
    codes_shape = (1, QUANT_ROW_TILE * (QUANT_COLUMN_TILE // MX_GROUP))
    codes = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout(codes_shape))
    for row in cb.range(QUANT_ROW_TILE):
        cb.mem_copy(raw, cb.tile_slice(source, (1, QUANT_COLUMN_TILE), (row, 0)))
        with cb.vf():
            full = rr.full_mask()
            zero = rr.vdups(0.0, cb.dtypes.float32, mask=full)
            rr.vstore(values, QUANT_COLUMN_TILE, zero, full)
            for col in range(0, QUANT_COLUMN_TILE, VECTOR_TILE):
                value = rr.vload(raw, col)
                rr.vstore(values, col, value, full)
            rr.vmem_bar("vst_vld")
            _quant_upward_row(values, packed, codes, row, QUANT_COLUMN_TILE)
        cb.mem_copy(cb.tile_slice(payload, (1, QUANT_COLUMN_TILE), (row, 0)), packed)
    cb.mem_copy(scales.view(dtype=cb.dtypes.uint8), codes)


@cb.kernel
def select_routes(
    scores: cb.Tensor, biased: cb.Tensor, indices: cb.Tensor, weights: cb.Tensor,
    valid_rows: cb.ScalarValue,
):
    """稳定选取分数 TopK，并归一化对应的不含偏置的分数。"""
    columns = scores.shape[1]
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, columns)))
    work = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, columns)))
    selected_ids = Buffer(cb.MemLoc.UB, cb.dtypes.int32, cb.make_layout((1, VECTOR_TILE)))
    selected_scores = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
    for row in cb.range(ROUTE_ROW_TILE):
        with cb.vf():
            full = rr.full_mask()
            zero_i = rr.vdups(0, cb.dtypes.int32, mask=full)
            zero_f = rr.vdups(0.0, cb.dtypes.float32, mask=full)
            rr.vstore(selected_ids, 0, zero_i, full)
            rr.vstore(selected_scores, 0, zero_f, full)
        if row < valid_rows:
            cb.mem_copy(raw, cb.tile_slice(scores, (1, columns), (row, 0)))
            cb.mem_copy(work, cb.tile_slice(biased, (1, columns), (row, 0)))
            with cb.vf():
                full = rr.full_mask()
                neg_inf = rr.vdups(-3.4028234663852886e38, cb.dtypes.float32, mask=full)
                sentinel = rr.vdups(2147483647, cb.dtypes.int32, mask=full)
                for rank in range(TOPK):
                    maximum = neg_inf
                    for col in range(0, columns, VECTOR_TILE):
                        maximum = rr.vmax(maximum, rr.vload(work, col), mask=full)
                    maximum = rr.vdup(rr.vreduce_max(maximum, mask=full), mask=full)
                    selected = sentinel
                    for col in range(0, columns, VECTOR_TILE):
                        value = rr.vload(work, col)
                        ids = rr.varange(col, cb.dtypes.int32)
                        matching = rr.veq(value, maximum, mask=full)
                        candidate = rr.vselect(ids, sentinel, cond_mask=matching)
                        selected = rr.vmin(selected, candidate, mask=full)
                    selected = rr.vdup(rr.vreduce_min(selected, mask=full), mask=full)
                    gather_ids = rr.vreinterpret(selected, cb.dtypes.uint32)
                    score = rr.vgather(raw, gather_ids, mask=full)
                    rr.vstore_first(selected_ids, rank, selected)
                    rr.vstore_first(selected_scores, rank, score)
                    for col in range(0, columns, VECTOR_TILE):
                        ids = rr.varange(col, cb.dtypes.int32)
                        matching = rr.veq(ids, selected, mask=full)
                        value = rr.vselect(neg_inf, rr.vload(work, col), cond_mask=matching)
                        rr.vstore(work, col, value, full)
                    rr.vmem_bar("vst_vld")
                active = rr.update_mask(TOPK, elem_bits=32)[0]
                values = rr.vload(selected_scores, 0)
                total = rr.vdup(rr.vreduce_sum(values, mask=active), mask=active)
                normalized = rr.vdiv(values, total, mask=active, precision_mode=True)
                scaled = rr.vmuls(normalized, ROUTE_SCALE, mask=active)
                rr.vstore(selected_scores, 0, scaled, active)
        cb.mem_copy(cb.tile_slice(indices, (1, TOPK), (row, 0)), cb.tile_slice(selected_ids, (1, TOPK), (0, 0)))
        cb.mem_copy(cb.tile_slice(weights, (1, TOPK), (row, 0)), cb.tile_slice(selected_scores, (1, TOPK), (0, 0)))


@pl.jit.inline(auto_scope=False)
def _normalize_task(source: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, valid_rows: pl.Scalar[pl.INDEX]):
    """使用显式类型的有效行数，派发一个归一化叶子任务。"""
    with pl.spmd(1, name_hint="dsl_gate_norm"):
        normalize_rows(source, weight, output, valid_rows)
    return output


@pl.jit.inline(auto_scope=False)
def normalize_gate(source: pl.Tensor, weight: pl.Tensor, output: pl.Tensor, active_tokens: pl.Scalar[pl.INDEX]):
    """调度路由器带填充的八行归一化任务。"""
    rows = pl.tensor.dim(output, 0)
    for row in pl.parallel(0, rows, NORM_ROW_TILE):
        valid = pl.cast(pl.min(NORM_ROW_TILE, pl.max(active_tokens - row, 0)), pl.INDEX)
        src = pl.slice(source, [NORM_ROW_TILE, D], [row, 0])
        dst = pl.slice(output, [NORM_ROW_TILE, D], [row, 0])
        _normalize_task(src, weight, dst, valid)
    return output


@pl.jit.inline(auto_scope=False)
def quantize_gate(source: pl.Tensor, payload: pl.Tensor, scales: pl.Tensor):
    """调度互不重叠的 M16/K1024 输入量化任务。"""
    rows = pl.tensor.dim(source, 0)
    for row in pl.parallel(0, rows, QUANT_ROW_TILE):
        for col in pl.parallel(0, D, QUANT_COLUMN_TILE):
            src = pl.slice(source, [QUANT_ROW_TILE, QUANT_COLUMN_TILE], [row, col])
            dst = pl.slice(payload, [QUANT_ROW_TILE, QUANT_COLUMN_TILE], [row, col])
            offset = row * (D // MX_GROUP) + col // MX_GROUP * QUANT_ROW_TILE
            codes = pl.slice(scales, [1, QUANT_ROW_TILE * (QUANT_COLUMN_TILE // MX_GROUP)], [0, offset])
            with pl.spmd(1, name_hint="dsl_gate_quant"):
                quantize_rows(src, dst, codes)
    return payload, scales


@pl.jit.inline(auto_scope=False)
def project_gate(
    source: pl.Tensor, weight: pl.Tensor, bias: pl.Tensor, scores: pl.Tensor, biased: pl.Tensor,
    active_tokens: pl.Scalar[pl.INDEX],
):
    """调度完整的本地 MIX 投影/打分任务。"""
    columns = pl.tensor.dim(weight, 0)
    bias_matrix = pl.reshape(bias, [1, columns])
    task_rows = (pl.max(active_tokens, 1) + PROJECT_ROW_TILE - 1) // PROJECT_ROW_TILE * PROJECT_ROW_TILE
    for row in pl.parallel(0, task_rows, PROJECT_ROW_TILE):
        for col in pl.parallel(0, columns, PROJECT_COLUMN_TILE):
            src = pl.slice(source, [PROJECT_ROW_TILE, D], [row, 0])
            matrix = pl.slice(weight, [PROJECT_COLUMN_TILE, D], [col, 0])
            offset = pl.slice(bias_matrix, [1, PROJECT_COLUMN_TILE], [0, col])
            dst = pl.slice(scores, [PROJECT_ROW_TILE, PROJECT_COLUMN_TILE], [row, col])
            dst_bias = pl.slice(biased, [PROJECT_ROW_TILE, PROJECT_COLUMN_TILE], [row, col])
            with pl.spmd(1, name_hint="dsl_gate", sync_start=True):
                project_scores(src, matrix, offset, dst, dst_bias)
    return scores, biased


@pl.jit.inline(auto_scope=False)
def _route_task(
    scores: pl.Tensor, biased: pl.Tensor, indices: pl.Tensor, weights: pl.Tensor,
    valid_rows: pl.Scalar[pl.INDEX],
):
    """使用显式类型的有效行数，派发一个本地路由任务。"""
    with pl.spmd(1, name_hint="dsl_route_sort"):
        select_routes(scores, biased, indices, weights, valid_rows)
    return indices, weights


@pl.jit.inline(auto_scope=False)
def route_gate(
    scores: pl.Tensor, biased: pl.Tensor, indices: pl.Tensor, weights: pl.Tensor,
    active_tokens: pl.Scalar[pl.INDEX],
):
    """在分数生成后调度八行本地 TopK 任务。"""
    columns = pl.tensor.dim(scores, 1)
    task_rows = (pl.max(active_tokens, 1) + ROUTE_ROW_TILE - 1) // ROUTE_ROW_TILE * ROUTE_ROW_TILE
    for row in pl.parallel(0, task_rows, ROUTE_ROW_TILE):
        valid = pl.min(ROUTE_ROW_TILE, pl.max(active_tokens - row, 0))
        src = pl.slice(scores, [ROUTE_ROW_TILE, columns], [row, 0])
        src_bias = pl.slice(biased, [ROUTE_ROW_TILE, columns], [row, 0])
        ids = pl.slice(indices, [ROUTE_ROW_TILE, TOPK], [row, 0])
        out = pl.slice(weights, [ROUTE_ROW_TILE, TOPK], [row, 0])
        _route_task(src, src_bias, ids, out, valid)
    return indices, weights


@pl.jit
def validate_gate_quant(
    source: pl.Tensor, payload: pl.Out[pl.Tensor], scales: pl.Out[pl.Tensor],
):
    quantize_gate(source, payload, scales)
    return payload, scales


@pl.jit
def validate_gate_routes(
    scores: pl.Tensor, biased: pl.Tensor, indices: pl.Out[pl.Tensor], weights: pl.Out[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
):
    active = pl.min(ROUTE_ROW_TILE, pl.max(pl.cast(num_tokens, pl.INDEX), 0))
    _route_task(scores, biased, indices, weights, active)
    return indices, weights


def golden_gate_quant(tensors):
    """使用向上取整的缩放值和 M16 缩放打包格式量化 FP32 路由器输入。"""
    import torch
    from quant import pack_mx_a_scale, quantize_mxfp8_upward_e8m0

    payload, codes = quantize_mxfp8_upward_e8m0(tensors["source"])
    tensors["payload"].copy_(payload)
    tensors["scales"].view(torch.uint8).copy_(pack_mx_a_scale(codes).reshape(1, -1))


def golden_gate_routes(tensors):
    """稳定选取带偏置分数，并归一化被选中的不含偏置的分数。"""
    import torch

    active = max(0, min(ROUTE_ROW_TILE, int(tensors["num_tokens"])))
    selected = torch.argsort(tensors["biased"], dim=-1, descending=True, stable=True)[:, :TOPK]
    values = torch.gather(tensors["scores"], -1, selected)
    weights = values / values.sum(dim=-1, keepdim=True) * ROUTE_SCALE
    selected[active:] = 0
    weights[active:] = 0
    tensors["indices"].copy_(selected.to(torch.int32))
    tensors["weights"].copy_(weights)


def build_quant_specs():
    """包含零值、FP8 舍入中点和缩放值向上取整的跳变边界。"""
    import torch
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(915)
    source = torch.randn(QUANT_ROW_TILE, D, generator=generator).bfloat16().float()
    source[0].zero_()
    boundary = torch.tensor(1.75)
    above = torch.nextafter(boundary, torch.tensor(float("inf"))).item()
    patterns = (0.0, 1e-6, 1e-4, 1.75, above, -1.75, -above, 448.0, -448.0)
    for group in range(D // MX_GROUP):
        source[1, group * MX_GROUP : (group + 1) * MX_GROUP] = patterns[group % len(patterns)]
    return [
        TensorSpec("source", [QUANT_ROW_TILE, D], torch.float32, init_value=source),
        TensorSpec("payload", [QUANT_ROW_TILE, D], torch.float8_e4m3fn),
        TensorSpec("scales", [1, QUANT_ROW_TILE * (D // MX_GROUP)], torch.float8_e8m0fnu),
    ]


def build_route_specs(active=7):
    """包含同分、跨寄存器胜出项、偏置和一个非有效行。"""
    import torch
    from golden import ScalarSpec, TensorSpec

    columns = M.n_routed_experts
    generator = torch.Generator().manual_seed(916)
    scores = torch.rand(ROUTE_ROW_TILE, columns, generator=generator) + 0.1
    bias = torch.randn(ROUTE_ROW_TILE, columns, generator=generator)
    biased = scores + bias
    biased[1].fill_(1.0)
    biased[2].fill_(-1.0)
    biased[2, [62, 63, 64, 127, 128, columns - 1]] = 2.0
    biased[3] = torch.arange(columns, dtype=torch.float32).remainder(9)
    biased[4, -1] = 100.0
    biased[5] = -scores[5]
    return [
        TensorSpec("scores", [ROUTE_ROW_TILE, columns], torch.float32, init_value=scores),
        TensorSpec("biased", [ROUTE_ROW_TILE, columns], torch.float32, init_value=biased),
        TensorSpec("indices", [ROUTE_ROW_TILE, TOPK], torch.int32),
        TensorSpec("weights", [ROUTE_ROW_TILE, TOPK], torch.float32),
        ScalarSpec("num_tokens", torch.int32, active),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run
    from window_kv_dsl import compare_payload_bytes

    parser = argparse.ArgumentParser(description="Validate local DSL gate quantization and routing on A5.")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--stage", choices=["all", "quant", "route"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    cases = []
    if args.stage in ("all", "quant"):
        cases.append(("quant", validate_gate_quant, build_quant_specs(), golden_gate_quant,
                      {"payload": compare_payload_bytes, "scales": compare_payload_bytes}))
    if args.stage in ("all", "route"):
        for active in (7, 0):
            cases.append((f"route-{active}", validate_gate_routes, build_route_specs(active), golden_gate_routes,
                          {"indices": ratio_allclose(atol=0, rtol=0, max_error_ratio=0)}))
    for name, program, specs, golden, comparisons in cases:
        print(f"[GATE_DSL] {name}", flush=True)
        result = run(
            fn=program,
            specs=specs,
            golden_fn=golden,
            compare_fn=comparisons,
            rtol=1e-5,
            atol=1e-6,
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
