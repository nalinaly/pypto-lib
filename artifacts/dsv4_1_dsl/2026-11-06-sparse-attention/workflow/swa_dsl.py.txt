# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 实现 SWA 缓存收集、BF16 Cube 注意力和 sink 归一化。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from cannbotdsl.ring_buffer import RingBuffer
import pypto.language as pl


# 模型配置。
HEADS = 64
HEAD_DIM = 512
WINDOW = 128
CACHE_GROUP = 32
SOFTMAX_SCALE = HEAD_DIM**-0.5

# 分块参数。
GATHER_TILE = 16
CUBE_TILE = 128
VECTOR_TILE = 64
TASK_TILE = 12


@cb.kernel
def swa_query(
    query: cb.Tensor,
    cache: cb.Tensor,
    scales: cb.Tensor,
    indices: cb.Tensor,
    sink: cb.Tensor,
    output: cb.Tensor,
):
    """由一个 AIC 和两个 AIV 协作计算一个 Query 的完整窗口。"""
    lane = cb.get_subblock_id()
    cache_rows = cache.shape[0]
    q_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.bfloat16, cb.make_layout((HEADS, HEAD_DIM)), depth=1)
    kv_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.bfloat16, cb.make_layout((WINDOW, HEAD_DIM)), depth=1)
    p_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.bfloat16, cb.make_layout((HEADS, WINDOW)), depth=1)
    a_l0 = RingBuffer(cb.MemLoc.L0A, cb.dtypes.bfloat16, cb.make_layout((HEADS, CUBE_TILE)), depth=1)
    b_l0 = RingBuffer(cb.MemLoc.L0B, cb.dtypes.bfloat16, cb.make_layout((CUBE_TILE, CUBE_TILE)), depth=1)
    c_l0 = RingBuffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((HEADS, CUBE_TILE)), depth=1)
    scores = RingBuffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((HEADS // 2, WINDOW)), depth=1)
    weighted = RingBuffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((HEADS // 2, CUBE_TILE)), depth=1)

    ids = Buffer(cb.MemLoc.UB, cb.dtypes.int32, cb.make_layout((1, WINDOW)))
    payload = Buffer(cb.MemLoc.UB, cb.dtypes.float8_e4m3fn, cb.make_layout((GATHER_TILE + 1, HEAD_DIM)))
    factors = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((GATHER_TILE, HEAD_DIM // CACHE_GROUP)))
    decoded = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((GATHER_TILE, HEAD_DIM)))
    kv_physical = cb.make_layout((HEAD_DIM // 16, 1, 16, 16), stride=(272, 256, 16, 1))
    kv_nz = Buffer(
        cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((GATHER_TILE, HEAD_DIM)),
        physical_layout=kv_physical, data_format="nz",
    )
    p_physical = cb.make_layout((WINDOW // 16, HEADS // 32, 16, 16), stride=(528, 256, 16, 1))
    probabilities = Buffer(
        cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((HEADS // 2, WINDOW)),
        physical_layout=p_physical, data_format="nz",
    )
    multiplier = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((HEADS // 2,)))
    sinks = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, HEADS // 2)))
    result = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((HEADS // 2, CUBE_TILE)))

    cb.mem_copy(q_l1.next_write(), query)
    cb.mem_copy(ids, indices)
    cb.mem_copy(sinks, cb.tile_slice(sink, (1, HEADS // 2), (0, lane)))
    kv_write = kv_l1.next_write()
    for chunk in cb.range(WINDOW // 2 // GATHER_TILE):
        for row in cb.range(GATHER_TILE):
            position = lane * (WINDOW // 2) + chunk * GATHER_TILE + row
            physical = indices[0, position]
            safe = max(0, min(physical, cache_rows - 1))
            cb.mem_copy(cb.tile_slice(payload, (1, HEAD_DIM), (row, 0)), cb.tile_slice(cache, (1, HEAD_DIM), (safe, 0)))
            src_scale = cb.tile_slice(scales, (1, HEAD_DIM // CACHE_GROUP), (safe, 0))
            cb.mem_copy(cb.tile_slice(factors, (1, HEAD_DIM // CACHE_GROUP), (row, 0)), src_scale)
        with cb.vf():
            half = rr.update_mask(CACHE_GROUP, elem_bits=32)[0]
            fp8_mask = rr.update_mask(CACHE_GROUP * 4, elem_bits=8)[0]
            zero = rr.vdups(0.0, cb.dtypes.float32, mask=half)
            for row in range(GATHER_TILE):
                position = lane * (WINDOW // 2) + chunk * GATHER_TILE + row
                index = rr.vload_broadcast(ids, position)
                valid = rr.mask_and(rr.vges(index, 0, mask=half), rr.vlts(index, cache_rows, mask=half), exec_mask=half)
                for group in range(HEAD_DIM // CACHE_GROUP):
                    offset = row * HEAD_DIM + group * CACHE_GROUP
                    code = rr.vload_unpack(payload, offset, unpack_mode=rr.UnpackMode.B8_TO_B32)
                    code = rr.vreinterpret_lanes(code, cb.dtypes.float8_e4m3fn)
                    value = rr.vcast(code, cb.dtypes.float32, mask=fp8_mask, reg_layout=rr.RegLayout.ZERO)
                    factor = rr.vload_broadcast(factors, row * (HEAD_DIM // CACHE_GROUP) + group)
                    factor = rr.vcast(factor, cb.dtypes.float32, mask=half)
                    value = rr.vmul(value, factor, mask=half)
                    value = rr.vselect(value, zero, cond_mask=valid)
                    value = rr.vcast(value, cb.dtypes.bfloat16, mask=half, rounding="rn")
                    rr.vstore_pack(decoded, offset, value, half, pack_mode=rr.PackMode.B32_TO_B16)
        with cb.vf():
            full16 = rr.full_mask(elem_bits=16)
            for row in range(GATHER_TILE):
                for col in range(0, HEAD_DIM, CUBE_TILE):
                    value = rr.vload(decoded, row * HEAD_DIM + col)
                    offset = row * 16 + (col // 16) * 272
                    rr.vstore_strided(kv_nz, offset, value, full16, block_stride=17, repeat_stride=0)
        target = cb.tile_slice(kv_write, (GATHER_TILE, HEAD_DIM), (lane * 4 + chunk, 0))
        cb.mem_copy(target, kv_nz)

    q_read = q_l1.next_read()
    kv_read = kv_l1.next_read()
    qk = c_l0.next_write()
    for part in cb.range(HEAD_DIM // CUBE_TILE):
        cb.mem_copy(a_l0.next_write(), cb.tile_slice(q_read, (HEADS, CUBE_TILE), (0, part)))
        cb.mem_copy(b_l0.next_write(), cb.tile_slice(kv_read, (WINDOW, CUBE_TILE), (0, part)))
        cb.matmul(qk, a_l0.next_read(), b_l0.next_read(), init=part == 0)
    cb.mem_copy(scores.next_write(), c_l0.next_read().partition(dim=0))

    qk_values = scores.next_read()
    with cb.vf():
        full = rr.full_mask()
        full16 = rr.full_mask(elem_bits=16)
        zero = rr.vdups(0.0, cb.dtypes.float32, mask=full)
        invalid = rr.vdups(-1e30, cb.dtypes.float32, mask=full)
        ids0 = rr.vload(ids, 0)
        ids1 = rr.vload(ids, VECTOR_TILE)
        valid0 = rr.mask_and(rr.vges(ids0, 0, mask=full), rr.vlts(ids0, cache_rows, mask=full), exec_mask=full)
        valid1 = rr.mask_and(rr.vges(ids1, 0, mask=full), rr.vlts(ids1, cache_rows, mask=full), exec_mask=full)
        for row in range(HEADS // 2):
            score0 = rr.vmuls(rr.vload(qk_values, row * WINDOW), SOFTMAX_SCALE, mask=full)
            score1 = rr.vmuls(rr.vload(qk_values, row * WINDOW + VECTOR_TILE), SOFTMAX_SCALE, mask=full)
            score0 = rr.vselect(score0, invalid, cond_mask=valid0)
            score1 = rr.vselect(score1, invalid, cond_mask=valid1)
            maximum = rr.vdup(rr.vreduce_max(rr.vmax(score0, score1, mask=full), mask=full), mask=full)
            p0 = rr.vexp(rr.vsub(score0, maximum, mask=full), mask=full)
            p1 = rr.vexp(rr.vsub(score1, maximum, mask=full), mask=full)
            p0 = rr.vselect(p0, zero, cond_mask=valid0)
            p1 = rr.vselect(p1, zero, cond_mask=valid1)
            total = rr.vadd(rr.vreduce_sum(p0, mask=full), rr.vreduce_sum(p1, mask=full), mask=full)
            total = rr.vdup(total, mask=full)
            sink_value = rr.vload_broadcast(sinks, row)
            final_max = rr.vmax(maximum, sink_value, mask=full)
            correction = rr.vexp(rr.vsub(maximum, final_max, mask=full), mask=full)
            sink_exp = rr.vexp(rr.vsub(sink_value, final_max, mask=full), mask=full)
            denominator = rr.vadd(rr.vmul(total, correction, mask=full), sink_exp, mask=full)
            factor = rr.vdiv(correction, denominator, mask=full, precision_mode=True)
            rr.vstore_first(multiplier, row, factor)
            p0 = rr.vcast(p0, cb.dtypes.bfloat16, mask=full, rounding="rn")
            p1 = rr.vcast(p1, cb.dtypes.bfloat16, mask=full, rounding="rn")
            packed, unused = rr.vdeinterleave(p0, p1)
            rr.vstore_strided(probabilities, row * 16, packed, full16, block_stride=33, repeat_stride=0)
    cb.mem_copy(p_l1.next_write().partition(dim=0, part_id=lane), probabilities)
    p_read = p_l1.next_read()
    for col in cb.range(HEAD_DIM // CUBE_TILE):
        cb.mem_copy(a_l0.next_write(), p_read)
        values = cb.tile_slice(kv_read, (WINDOW, CUBE_TILE), (0, col))
        cb.mem_copy(b_l0.next_write(), values.permute(1, 0))
        cb.matmul(c_l0.next_write(), a_l0.next_read(), b_l0.next_read(), init=True)
        cb.mem_copy(weighted.next_write(), c_l0.next_read().partition(dim=0))
        pv = weighted.next_read()
        with cb.vf():
            output_mask = rr.full_mask()
            for row in range(HEADS // 2):
                output_factor = rr.vload_broadcast(multiplier, row)
                for offset in range(0, CUBE_TILE, VECTOR_TILE):
                    output_value = rr.vload(pv, row * CUBE_TILE + offset)
                    output_value = rr.vmul(output_value, output_factor, mask=output_mask)
                    output_value = rr.vcast(output_value, cb.dtypes.bfloat16, mask=output_mask, rounding="rn")
                    rr.vstore_pack(result, row * CUBE_TILE + offset, output_value, output_mask, pack_mode=rr.PackMode.B32_TO_B16)
        target = cb.tile_slice(output, (HEADS, CUBE_TILE), (0, col))
        cb.mem_copy(target.partition(dim=0, part_id=lane), result)


@pl.jit.inline(auto_scope=False)
def swa_attention_dsl(
    query: pl.Tensor,
    cache: pl.Tensor,
    scales: pl.Tensor,
    indices: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    source_ready: pl.Scalar[pl.TASK_ID],
):
    """调度完整的 Query 窗口，并汇合消费缓存的任务。"""
    rows = pl.tensor.dim(cache, 0) * WINDOW
    payload = pl.reshape(cache, [rows, HEAD_DIM])
    factors = pl.reshape(scales, [rows, HEAD_DIM // CACHE_GROUP])
    sinks = pl.reshape(sink, [1, HEADS])
    completed = pl.array.create(TASK_TILE, pl.TASK_ID)
    for worker in pl.range(TASK_TILE):
        completed[worker] = source_ready
    for token in pl.parallel(num_tokens):
        q_slice = pl.slice(query, [1, HEADS * HEAD_DIM], [token, 0])
        q_row = pl.reshape(q_slice, [HEADS, HEAD_DIM])
        index_row = pl.slice(indices, [1, WINDOW], [token, 0])
        out_slice = pl.slice(output, [1, HEADS * HEAD_DIM], [token, 0])
        out_row = pl.reshape(out_slice, [HEADS, HEAD_DIM])
        previous = completed[token % TASK_TILE]
        with pl.spmd(1, name_hint="dsl_swa", deps=[previous], sync_start=True) as done:
            swa_query(q_row, payload, factors, index_row, sinks, out_row)
        completed[token % TASK_TILE] = done
    return pl.system.task_dummy(deps=[completed])
