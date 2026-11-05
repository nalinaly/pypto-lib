# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""在一个 DSL MIX 任务中完成 MXFP4 分页打分、稳定 TopK、全核同步和归并。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import UB, make_buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.constexpr import range_constexpr
from cannbotdsl.ops.memcpy import copy_options
from cannbotdsl.ops.sync import PIPE
from cannbotdsl.ring_buffer import RingBuffer, make_ring_buffer
import pypto.language as pl

from index_topk_dsl import INVALID_KEY, TOPK, decode_score, encode_score, stable_topk


# 模型契约
HEADS = 32
HEAD_DIM = 128
PAGE_SIZE = 128
WEIGHT_SCALE = 1.0 / 64.0

# 分块参数
QUERY_TILE = 6
KEY_TILE = 256
SCORE_TILE = 25600
MERGE_TILE = 16384
ARENA_BYTES = 262144


def _view(arena, dtype, shape, offset):
    tensor = cb.from_tensor(arena, dtype=dtype, layout=cb.make_layout(shape), offset=offset)
    return make_buffer(tensor)


class _Cube:
    """六行 Query 对两页 Key 的 MXFP4 点积资源。"""

    def __init__(self):
        self.query = RingBuffer(cb.MemLoc.L1, cb.dtypes.fp4x2_e2m1, cb.make_layout((192, 128)), depth=1)
        self.key = RingBuffer(cb.MemLoc.L1, cb.dtypes.fp4x2_e2m1, cb.make_layout((256, 128)), depth=1)
        self.query_scale = RingBuffer(cb.MemLoc.L1, cb.dtypes.float8_e8m0, cb.make_layout((4, 192)), depth=1)
        self.key_scale = RingBuffer(cb.MemLoc.L1, cb.dtypes.float8_e8m0, cb.make_layout((4, 256)), depth=1)
        self.left = RingBuffer(cb.MemLoc.L0A, cb.dtypes.fp4x2_e2m1, cb.make_layout((192, 128)), depth=1)
        self.right = RingBuffer(cb.MemLoc.L0B, cb.dtypes.fp4x2_e2m1, cb.make_layout((256, 128)), depth=1)
        self.acc = RingBuffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((192, 256)), depth=1)

    @jit
    def load_query(self, query, scales, begin):
        first = begin * HEADS
        last = min(first + 192, query.shape[0])
        cb.mem_copy(self.query.next_write(), query[first:last, None])
        cb.mem_copy(self.query_scale.next_write().permute(1, 0), scales[first:last, None])
        factors = self.query_scale.next_read().permute(1, 0)
        cb.mem_copy(self.left.next_write(), self.query.next_read(), options=copy_options(mx_scale=factors))

    @jit
    def compute(self, keys, scales, page0, page1, received):
        destination = self.key.next_write()
        factors = self.key_scale.next_write()
        for piece in cb.range(2):
            page = page0 if piece == 0 else page1
            if page < 0:
                page = page0 if page0 >= 0 else page1
            row = page * PAGE_SIZE
            cb.mem_copy(cb.tile_slice(destination, (128, 128), (piece, 0)), keys[row : row + 128, None])
            source_scale = scales[row : row + 128, None].permute(1, 0)
            cb.mem_copy(cb.tile_slice(factors, (4, 128), (0, piece)), source_scale)
        cb.mem_copy(
            self.right.next_write(),
            self.key.next_read(),
            options=copy_options(mx_scale=self.key_scale.next_read()),
        )
        product = cb.matmul(
            self.acc.next_write(), self.left.next_read(), self.right.next_read(), init=True, mx_mode=True
        )
        destination = received.next_write()
        # BF16 窄化不能与双目标分送合用，分别写入两个 AIV 的三行 Query。
        cb.mem_copy(
            destination, cb.tile_slice(product, (96, 256), (0, 0)), options=copy_options(dst_subblock=0)
        )
        cb.mem_copy(
            destination, cb.tile_slice(product, (96, 256), (1, 0)), options=copy_options(dst_subblock=1)
        )


@jit
def _prepare_weights(raw, normalized, source, begin, count):
    with cb.vf():
        mask = rr.update_mask(128, elem_bits=16)[0]
        zeros = rr.vdups(0.0, cb.dtypes.bfloat16, mask=mask)
        rr.vstore(raw, 0, zeros, mask)
        rr.vstore(raw, 128, zeros, mask)
        rr.vmem_bar("vst_vld")
    destination = cb.from_tensor(raw, layout=cb.make_layout((6, 32)))
    cb.mem_copy(destination, source[begin : begin + count, None])
    with cb.vf():
        mask32 = rr.update_mask(32, elem_bits=32)[0]
        mask16 = rr.update_mask(32, elem_bits=16)[0]
        for row in range(QUERY_TILE):
            packed = rr.vload_unpack(raw, row * 32, unpack_mode=rr.UnpackMode.B16_TO_B32)
            value = rr.vcast(packed, cb.dtypes.float32, mask=mask32)
            value = rr.vmuls(value, WEIGHT_SCALE, mask=mask32)
            rounded = rr.vcast(value, cb.dtypes.bfloat16, mask=mask32, rounding="rn")
            packed = rr.vpack(
                rr.vreinterpret_lanes(rounded, cb.dtypes.uint32), cb.dtypes.uint16, part="lower"
            )
            rr.vstore(normalized, row * 128, rr.vreinterpret(packed, cb.dtypes.bfloat16), mask16)
        rr.vmem_bar("vst_vld")


@jit
def _cache_panel(
    qk,
    weights,
    cache,
    valid_flags,
    lengths,
    begin,
    count,
    lane,
    logical,
    panel,
    maximum,
    width,
    page0,
    page1,
    have_dot,
    gathered,
):
    for row in cb.range(3, unroll=1):
        offset = lane * 3 + row
        live = cb.dtypes.int32(0)
        if offset < count:
            live = cb.dtypes.int32(maximum if gathered else min(lengths[begin + offset], width))
        for half in cb.range(2, unroll=1):
            valid_count = max(0, min(128, live - logical - half * 128))
            if (half == 0 and page0 < 0) or (half == 1 and page1 < 0):
                valid_count = cb.dtypes.int32(0)
            with cb.vf():
                full = rr.update_mask(128, elem_bits=16)[0]
                invalid = rr.vdups(INVALID_KEY, cb.dtypes.uint16, mask=full)
                score = rr.vdups(0.0, cb.dtypes.bfloat16, mask=full)
                if have_dot:
                    head_weights = rr.vload(weights, offset * 128)
                    for head in cb.range(HEADS, unroll=1):
                        indices = rr.vdups(head, cb.dtypes.uint16, mask=full)
                        weight = rr.vgather_reg(head_weights, indices)
                        dot = rr.vload(qk, (row * HEADS + head) * KEY_TILE + half * 128)
                        positive = rr.vmaxs(dot, 0.0, mask=full)
                        score = rr.vmadd(positive, weight, score, mask=full)
                flags = rr.vdups(1, cb.dtypes.uint16, mask=full)
                if gathered:
                    flags = rr.vload(valid_flags, half * 128)
                selected = rr.update_mask(valid_count, elem_bits=16)[0]
                selected = rr.mask_and(selected, rr.veqs(flags, 1, mask=full), exec_mask=full)
                encoded = encode_score(score, full)
                value = rr.vselect(encoded, invalid, cond_mask=selected)
                rr.vstore(cache, row * SCORE_TILE + panel * KEY_TILE + half * 128, value, full)
                rr.vmem_bar("vst_vld")


@jit
def _block_join():
    """等待两个 AIV 完成局部选择，再复用下一组的 QK/TopK 共用空间。"""
    cb.cube_sync_all()
    cb.vec_sync_all()
    cb.vec_sync_block_arrive(PIPE.MTE3, 5, mode=2)
    cb.cube_sync_block_wait(PIPE.S, 5, mode=2)
    cb.cube_sync_block_arrive(PIPE.MTE3, 6, mode=2)
    cb.vec_sync_block_wait(PIPE.S, 6, mode=2)


@jit
def _store_scores(cache, scratch, scores, token, row, first, length):
    for column in cb.range(0, length, 128):
        live = min(128, length - column)
        with cb.vf():
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            value = decode_score(rr.vload(cache, row * SCORE_TILE + column), mask16)
            for part in range_constexpr(2):
                mask32 = rr.update_mask(64, elem_bits=32)[0]
                wide = rr.vunpack(
                    rr.vreinterpret(value, cb.dtypes.uint16),
                    cb.dtypes.uint32,
                    part="lower" if part == 0 else "upper",
                )
                floats = rr.vreinterpret(rr.vshl(wide, 16, mask=mask32), cb.dtypes.float32)
                rr.vstore(scratch, part * 64, floats, mask32)
        source = scratch[0:1, 0:live]
        cb.mem_copy(scores[token : token + 1, first + column : first + column + live], source)


@jit
def _physical_indices(indices, mapping, block_table, request_ids, physical, token, width, gathered_rows):
    if gathered_rows > 0:
        count = min(width, physical.shape[1] - token * gathered_rows)
        cb.mem_copy(
            mapping[0:1, 0:count], physical[0:1, token * gathered_rows : token * gathered_rows + count]
        )
    else:
        count = (width + 127) // 128
        request = cb.dtypes.int64(request_ids[token])
        cb.mem_copy(mapping[0:1, 0:count], block_table[request : request + 1, 0:count])
    with cb.vf():
        mask = rr.update_mask(64, elem_bits=32)[0]
        for part in range(TOPK // 64):
            logical = rr.vload(indices, part * 64)
            valid = rr.vges(logical, 0, mask=mask)
            safe = rr.vmaxs(logical, 0, mask=mask)
            result = rr.vdups(-1, cb.dtypes.int32, mask=mask)
            if gathered_rows > 0:
                result = rr.vgather(mapping, rr.vreinterpret(safe, cb.dtypes.uint32), mask=mask)
            else:
                page = rr.vshr(safe, 7, mask=mask)
                page = rr.vgather(mapping, rr.vreinterpret(page, cb.dtypes.uint32), mask=mask)
                offset = rr.vbitwise_and(safe, rr.vdups(127, cb.dtypes.int32, mask=mask), mask=mask)
                result = rr.vadd(rr.vshl(page, 7, mask=mask), offset, mask=mask)
            result = rr.vselect(result, rr.vdups(-1, cb.dtypes.int32, mask=mask), cond_mask=valid)
            rr.vstore(indices, part * 64, result, mask)
        rr.vmem_bar("vst_vld")


@cb.kernel
def score_topk_mix(
    query: cb.Tensor,
    query_scale: cb.Tensor,
    weights: cb.Tensor,
    index_cache: cb.Tensor,
    index_cache_scale: cb.Tensor,
    block_table: cb.Tensor,
    request_ids: cb.Tensor,
    lengths: cb.Tensor,
    output: cb.Tensor,
    runs: cb.Tensor,
    candidate_keys: cb.Tensor,
    candidate_indices: cb.Tensor,
    valid: cb.Tensor,
    physical: cb.Tensor,
    scores: cb.Tensor,
    num_tokens: cb.ScalarValue,
    gathered_rows: cb.ScalarValue,
    emit_scores: cb.ScalarValue,
    shards: cb.ScalarValue,
    direct_short: cb.ScalarValue,
):
    block = cb.get_block_idx()
    workers = cb.get_block_num()
    lane = cb.get_subblock_id()
    width = scores.shape[1]
    panels = (width + KEY_TILE - 1) // KEY_TILE
    groups = cb.dtypes.int64(runs[3, 0])
    owners = min(shards, groups * panels)
    arena = UB.view(ARENA_BYTES)
    # QK 接收区在本组最后一个 panel 消费完后，复用于 TopK 同分位置。
    # 三行分数缓存独占 [65536, 219136)；_block_join 阻止 AIC 提前覆盖下一组。
    received_view = cb.from_tensor(arena, dtype=cb.dtypes.bfloat16, layout=cb.make_layout((96, 256)))
    received = make_ring_buffer([received_view])
    cache = _view(arena, cb.dtypes.uint16, (3, SCORE_TILE), 65536)
    equal_positions = _view(arena, cb.dtypes.uint16, (SCORE_TILE,), 0)
    selected_keys = _view(arena, cb.dtypes.uint16, (1, TOPK), 51200)
    selected_indices = _view(arena, cb.dtypes.int32, (1, TOPK), 52224)
    positions = _view(arena, cb.dtypes.int32, (TOPK,), 54272)
    histogram = _view(arena, cb.dtypes.uint16, (256,), 56320)
    raw_weights = _view(arena, cb.dtypes.bfloat16, (1, 256), 220160)
    normalized = _view(arena, cb.dtypes.bfloat16, (6, 128), 220672)
    raw_valid = _view(arena, cb.dtypes.int32, (1, KEY_TILE), 223232)
    flags = _view(arena, cb.dtypes.uint16, (KEY_TILE,), 224256)
    score_output = _view(arena, cb.dtypes.float32, (1, KEY_TILE), 225280)
    cube = _Cube()
    q = cb.from_tensor(
        query, dtype=cb.dtypes.fp4x2_e2m1, layout=cb.make_layout((query.shape[0] * HEADS, 128))
    )
    qs = cb.from_tensor(query_scale, layout=cb.make_layout((query.shape[0] * HEADS, 4)))
    k = cb.from_tensor(
        index_cache,
        dtype=cb.dtypes.fp4x2_e2m1,
        layout=cb.make_layout((index_cache.shape[0] * PAGE_SIZE, 128)),
    )
    ks = cb.from_tensor(index_cache_scale, layout=cb.make_layout((index_cache.shape[0] * PAGE_SIZE, 4)))
    for shard in cb.range(block, owners, workers):
        first = groups * panels * shard // owners
        stop = groups * panels * (shard + 1) // owners
        cursor = first
        while cursor < stop:
            group = cursor // panels
            finish = min(stop, (group + 1) * panels)
            first_panel = cursor % panels
            panel_count = finish - cursor
            begin = cb.dtypes.int64(runs[0, group])
            count = cb.dtypes.int64(runs[1, group]) - begin
            maximum = cb.dtypes.int64(runs[2, group])
            request = cb.dtypes.int64(request_ids[begin])
            cube.load_query(q, qs, begin)
            _prepare_weights(raw_weights, normalized, weights, begin, count)
            for panel in cb.range(panel_count):
                logical = (first_panel + panel) * KEY_TILE
                page0 = cb.dtypes.int64(-1)
                page1 = cb.dtypes.int64(-1)
                if gathered_rows > 0:
                    if logical < maximum:
                        page0 = (begin * gathered_rows + logical) // PAGE_SIZE
                    if logical + PAGE_SIZE < maximum:
                        page1 = (begin * gathered_rows + logical) // PAGE_SIZE + 1
                else:
                    if request >= 0 and request < block_table.shape[0]:
                        if logical < maximum and logical // PAGE_SIZE < block_table.shape[1]:
                            page0 = cb.dtypes.int64(block_table[request, logical // PAGE_SIZE])
                        if logical + PAGE_SIZE < maximum and logical // PAGE_SIZE + 1 < block_table.shape[1]:
                            page1 = cb.dtypes.int64(block_table[request, logical // PAGE_SIZE + 1])
                    if page0 >= index_cache.shape[0]:
                        page0 = cb.dtypes.int64(-1)
                    if page1 >= index_cache.shape[0]:
                        page1 = cb.dtypes.int64(-1)
                have_dot = page0 >= 0 or page1 >= 0
                if gathered_rows > 0:
                    offset = begin * gathered_rows + logical
                    copied = min(KEY_TILE, valid.shape[1] - offset)
                    cb.mem_copy(raw_valid, valid[0:1, offset : offset + copied])
                    with cb.vf():
                        mask16 = rr.update_mask(64, elem_bits=16)[0]
                        for part in range(KEY_TILE // 64):
                            value = rr.vload(raw_valid, part * 64)
                            value = rr.vpack(
                                rr.vreinterpret(value, cb.dtypes.uint32), cb.dtypes.uint16, part="lower"
                            )
                            rr.vstore(flags, part * 64, value, mask16)
                if have_dot:
                    cube.compute(k, ks, page0, page1, received)
                    _cache_panel(
                        received.next_read(),
                        normalized,
                        cache,
                        flags,
                        lengths,
                        begin,
                        count,
                        lane,
                        logical,
                        panel,
                        maximum,
                        width,
                        page0,
                        page1,
                        True,
                        gathered_rows > 0,
                    )
                else:
                    _cache_panel(
                        normalized,
                        normalized,
                        cache,
                        flags,
                        lengths,
                        begin,
                        count,
                        lane,
                        logical,
                        panel,
                        maximum,
                        width,
                        page0,
                        page1,
                        False,
                        gathered_rows > 0,
                    )
            cached_width = min(panel_count * KEY_TILE, width - first_panel * KEY_TILE)
            for row in cb.range(3):
                offset = lane * 3 + row
                if offset < count:
                    keys = cb.tile_slice(cache, (1, SCORE_TILE), (row, 0))
                    stable_topk(
                        keys,
                        selected_indices,
                        selected_keys,
                        selected_indices,
                        positions,
                        equal_positions,
                        histogram,
                        cached_width,
                        first_panel * KEY_TILE,
                        0,
                    )
                    column = shard * TOPK
                    cb.mem_copy(
                        candidate_keys[begin + offset : begin + offset + 1, column : column + TOPK],
                        selected_keys,
                    )
                    cb.mem_copy(
                        candidate_indices[begin + offset : begin + offset + 1, column : column + TOPK],
                        selected_indices,
                    )
                    if emit_scores != 0:
                        _store_scores(
                            cache,
                            score_output,
                            scores,
                            begin + offset,
                            row,
                            first_panel * KEY_TILE,
                            cached_width,
                        )
            _block_join()
            cursor = finish

    # 每个驻留 block 完成分片后，统一进入全核屏障，再读取其他 block 的候选。
    cb.global_sync_all()
    # 屏障完成后复用前一阶段的片上区域；模型候选池最多 16384 行。
    merge_keys = _view(arena, cb.dtypes.uint16, (1, MERGE_TILE), 0)
    merge_indices = _view(arena, cb.dtypes.int32, (1, MERGE_TILE), 65536)
    equal_positions = _view(arena, cb.dtypes.uint16, (MERGE_TILE,), 32768)
    histogram = _view(arena, cb.dtypes.uint16, (256,), 131072)
    selected_keys = _view(arena, cb.dtypes.uint16, (1, TOPK), 131584)
    selected_indices = _view(arena, cb.dtypes.int32, (1, TOPK), 132608)
    mapping = _view(arena, cb.dtypes.int32, (1, 24576), 134656)
    positions = _view(arena, cb.dtypes.int32, (TOPK,), 221184)
    for token in cb.range(block * 2 + lane, num_tokens, workers * 2):
        short = direct_short != 0 and lengths[token] <= TOPK
        if short:
            length = max(0, min(TOPK, cb.dtypes.int64(lengths[token])))
            request = cb.dtypes.int64(request_ids[token])
            for page_index in cb.range(TOPK // PAGE_SIZE):
                page = cb.dtypes.int32(-1)
                live = min(PAGE_SIZE, max(0, length - page_index * PAGE_SIZE))
                if live > 0 and page_index < block_table.shape[1]:
                    page = block_table[request, page_index]
                if page < 0 or page >= index_cache.shape[0]:
                    page = cb.dtypes.int32(0)
                    live = cb.dtypes.int64(0)
                with cb.vf():
                    mask = rr.update_mask(64, elem_bits=32)[0]
                    for part in range(2):
                        offsets = rr.varange(part * 64, cb.dtypes.int32)
                        physical_ids = rr.vadds(offsets, page * PAGE_SIZE, mask=mask)
                        selected = rr.vlts(offsets, live, mask=mask)
                        value = rr.vselect(
                            physical_ids, rr.vdups(-1, cb.dtypes.int32, mask=mask), cond_mask=selected
                        )
                        rr.vstore(selected_indices, page_index * PAGE_SIZE + part * 64, value, mask)
            cb.mem_copy(output[token : token + 1, None], selected_indices)
        elif owners == 0:
            with cb.vf():
                mask = rr.update_mask(64, elem_bits=32)[0]
                for part in range(TOPK // 64):
                    rr.vstore(selected_indices, part * 64, rr.vdups(-1, cb.dtypes.int32, mask=mask), mask)
            cb.mem_copy(output[token : token + 1, None], selected_indices)
        else:
            group = cb.dtypes.int64(runs[3, token + 1])
            total = groups * panels
            first = ((group * panels + 1) * owners - 1) // total
            last = ((group + 1) * panels * owners - 1) // total
            carry = cb.dtypes.int64(0)
            start = first
            while start <= last:
                batch = min(last - start + 1, MERGE_TILE // TOPK - carry // TOPK)
                loaded = batch * TOPK
                column = start * TOPK
                cb.mem_copy(
                    merge_keys[0:1, carry : carry + loaded],
                    candidate_keys[token : token + 1, column : column + loaded],
                )
                cb.mem_copy(
                    merge_indices[0:1, carry : carry + loaded],
                    candidate_indices[token : token + 1, column : column + loaded],
                )
                stable_topk(
                    merge_keys,
                    merge_indices,
                    selected_keys,
                    selected_indices,
                    positions,
                    equal_positions,
                    histogram,
                    carry + loaded,
                    0,
                    1,
                )
                if start + batch > last:
                    _physical_indices(
                        selected_indices,
                        mapping,
                        block_table,
                        request_ids,
                        physical,
                        token,
                        width,
                        gathered_rows,
                    )
                    cb.mem_copy(output[token : token + 1, None], selected_indices)
                else:
                    cb.mem_copy(merge_keys[0:1, 0:TOPK], selected_keys)
                    cb.mem_copy(merge_indices[0:1, 0:TOPK], selected_indices)
                    carry = cb.dtypes.int64(TOPK)
                start = start + batch


@pl.jit.inline(auto_scope=False)
def dispatch_score_topk(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.Tensor,
    runs: pl.Tensor,
    candidate_keys: pl.Tensor,
    candidate_indices: pl.Tensor,
    valid: pl.Tensor,
    physical: pl.Tensor,
    scores: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    gathered_rows: pl.Scalar[pl.INDEX],
    emit_scores: pl.Scalar[pl.INT32],
    shards: pl.Scalar[pl.INDEX],
    direct_short: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    plan_ready: pl.Scalar[pl.TASK_ID],
):
    """明确运行时标量类型，再由全体驻留核同步启动同一个 MIX 任务。"""
    workers = pl.system.available_cluster_count()
    with pl.spmd(
        workers,
        sync_start=True,
        name_hint="dsl_idx_score_topk",
        deps=[cache_ready, query_ready, weights_ready, plan_ready],
    ) as task:
        score_topk_mix(
            query,
            query_scale,
            weights,
            index_cache,
            index_cache_scale,
            block_table,
            request_ids,
            lengths,
            output,
            runs,
            candidate_keys,
            candidate_indices,
            valid,
            physical,
            scores,
            num_tokens,
            gathered_rows,
            emit_scores,
            shards,
            direct_short,
        )
    return task


@pl.jit
def validate_score(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    output: pl.InOut[pl.Tensor],
    runs: pl.Tensor,
    valid: pl.Tensor,
    physical: pl.Tensor,
    scores: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    gathered_rows: pl.Scalar[pl.INDEX],
    emit_scores: pl.Scalar[pl.INT32],
    shards: pl.Scalar[pl.INDEX],
    direct_short: pl.Scalar[pl.INT32],
):
    rows = pl.tensor.dim(query, 0)
    candidate_keys = pl.create_tensor([rows, shards * TOPK], dtype=pl.UINT16)
    candidate_indices = pl.create_tensor([rows, shards * TOPK], dtype=pl.INT32)
    workers = pl.system.available_cluster_count()
    with pl.spmd(workers, sync_start=True, name_hint="dsl_idx_score_topk"):
        score_topk_mix(
            query,
            query_scale,
            weights,
            index_cache,
            index_cache_scale,
            block_table,
            request_ids,
            lengths,
            output,
            runs,
            candidate_keys,
            candidate_indices,
            valid,
            physical,
            scores,
            num_tokens,
            gathered_rows,
            emit_scores,
            shards,
            direct_short,
        )
    return output, scores


def golden_score(tensors):
    """从 FP4/E8M0 字节独立计算 BF16 打分、稳定选择和物理索引映射。"""
    import torch
    from quant import dequantize_mxfp4_cache

    active = int(tensors["num_tokens"])
    width = tensors["scores"].shape[1]
    gathered = int(tensors["gathered_rows"])
    query = tensors["query"].reshape(-1, HEADS, HEAD_DIM // 2)
    query_scale = tensors["query_scale"].view(torch.uint8).reshape(-1, HEADS, 4)
    q = dequantize_mxfp4_cache(query, query_scale, 32, "e8m0")
    key = tensors["index_cache"]
    key_scale = tensors["index_cache_scale"].view(torch.uint8)
    k = dequantize_mxfp4_cache(key, key_scale, 32, "e8m0").reshape(-1, HEAD_DIM)
    weights = (tensors["weights"].float() * WEIGHT_SCALE).bfloat16().float()
    table = tensors["block_table"]
    for token in range(active):
        request = int(tensors["request_ids"][token])
        length = max(0, min(width, int(tensors["lengths"][token])))
        logical = torch.arange(width)
        if gathered:
            source_rows = token * gathered + logical
            mapping = tensors["physical"].reshape(-1)[source_rows].long()
            live = tensors["valid"].reshape(-1)[source_rows].bool()
            live = live & (logical < length)
        else:
            pages = table[request, logical // PAGE_SIZE].long()
            source_rows = pages.clamp_min(0) * PAGE_SIZE + logical % PAGE_SIZE
            live = (logical < length) & (pages >= 0) & (pages < key.shape[0])
            source_rows = source_rows.clamp_max(k.shape[0] - 1)
            mapping = pages * PAGE_SIZE + logical % PAGE_SIZE
        dots = (q[token].float() @ k[source_rows].T).bfloat16().float().clamp_min(0)
        score = torch.zeros(width)
        for head in range(HEADS):
            score = (score + dots[head] * weights[token, head]).bfloat16().float()
        score[~live] = -torch.inf
        tensors["output"][token].fill_(-1)
        if int(tensors["direct_short"]) and length <= TOPK:
            tensors["output"][token, :length] = torch.where(live[:length], mapping[:length], -1).int()
        else:
            selected = torch.argsort(score, descending=True, stable=True)[:TOPK]
            selected = selected[torch.isfinite(score[selected])].sort().values
            tensors["output"][token, : selected.numel()] = mapping[selected].int()
        if int(tensors["emit_scores"]):
            tensors["scores"][token] = score


def build_score_specs(rows=6, width=1024, active=None, shards=32, mode="paged", ties=False):
    """构造非均匀缩放、负权重、页洞、短历史和多批归并用例。"""
    import torch
    from golden import ScalarSpec, TensorSpec

    active = rows if active is None else active
    generator = torch.Generator().manual_seed(7321)
    gathered = width if mode == "pool" else 0
    table_pages = (width + PAGE_SIZE - 1) // PAGE_SIZE
    pages = rows * width // PAGE_SIZE if gathered else table_pages + 2
    query = torch.randint(0, 256, (rows, 2048), generator=generator, dtype=torch.uint8)
    keys = torch.randint(0, 256, (pages, 128, 1, 64), generator=generator, dtype=torch.uint8)
    qs = torch.randint(124, 130, (rows, 128), generator=generator, dtype=torch.uint8)
    ks = torch.randint(124, 130, (pages, 128, 1, 4), generator=generator, dtype=torch.uint8)
    weights = (torch.randint(-16, 17, (rows, 32), generator=generator).float() / 8).bfloat16()
    table = torch.randperm(pages, generator=generator)[:table_pages].reshape(1, -1).int()
    if width >= 768:
        table[0, 1] = -1
    request = torch.zeros(rows, dtype=torch.int32)
    lengths = torch.tensor([max(0, width - i * 137) for i in range(rows)], dtype=torch.int32)
    valid = torch.ones(1, max(128, rows * gathered), dtype=torch.int32)
    physical = torch.arange(valid.numel(), dtype=torch.int32).reshape_as(valid) * 3 + 7
    if gathered:
        valid[:, 19::97] = 0
    direct = int(mode == "short")
    emit = 0 if direct else 1
    if direct:
        lengths = torch.tensor([0, 129, 511, 512, 17, 400][:rows], dtype=torch.int32)
    if ties:
        query.fill_(0x11)
        keys.fill_(0x11)
        qs.fill_(127)
        ks.fill_(127)
        weights.fill_(1)
    runs = torch.zeros(4, rows + 1, dtype=torch.int32)
    groups = 0
    for begin in range(0, active, 1 if gathered else QUERY_TILE):
        count = min(1 if gathered else QUERY_TILE, active - begin)
        runs[0, groups] = begin
        runs[1, groups] = begin + count
        visible = lengths[begin : begin + count].clamp(0, width)
        if direct:
            visible = torch.where(visible <= TOPK, 0, visible)
        runs[2, groups] = int(visible.max())
        runs[3, begin + 1 : begin + count + 1] = groups
        groups += 1
    runs[3, 0] = groups
    values = {
        "query": query,
        "query_scale": qs.view(torch.float8_e8m0fnu),
        "weights": weights,
        "index_cache": keys,
        "index_cache_scale": ks.view(torch.float8_e8m0fnu),
        "block_table": table,
        "request_ids": request,
        "lengths": lengths,
        "output": torch.full((rows, TOPK), -771, dtype=torch.int32),
        "runs": runs,
        "valid": valid,
        "physical": physical,
        "scores": torch.full((rows, width), -771.0),
    }
    specs = [
        TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in values.items()
    ]
    specs += [
        ScalarSpec("num_tokens", torch.int32, active),
        ScalarSpec("gathered_rows", torch.int64, gathered),
        ScalarSpec("emit_scores", torch.int32, emit),
        ScalarSpec("shards", torch.int64, shards),
        ScalarSpec("direct_short", torch.int32, direct),
    ]
    return specs


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="校验单个 MIX+syncall Indexer 打分及 TopK。")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", choices=("basic", "all"), default="all")
    args = parser.parse_args()
    cases = [(6, 1024, 6, 32, "paged", False)]
    if args.case == "all":
        cases += [
            (7, 1024, 7, 32, "paged", True),
            (6, 512, 6, 32, "short", False),
            (3, 768, 3, 32, "pool", False),
            (6, 13056, 6, 64, "paged", True),
            (7, 2048, 7, 3, "paged", False),
            (5, 1000, 4, 2, "paged", False),
            (7, 1024, 0, 32, "paged", False),
        ]
    for case in cases:
        print(f"[INDEX_SCORE_MIX_DSL] {case}", flush=True)
        result = run(
            fn=validate_score,
            specs=build_score_specs(*case),
            golden_fn=golden_score,
            rtol=0,
            atol=0,
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        print(f"[产物] {result.work_dir}", flush=True)
