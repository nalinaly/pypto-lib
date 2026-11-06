# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 实现 C2A 双行池化、归一化、FP4 缓存发布和环形状态更新。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
import pypto.language as pl

from compressor_vector_dsl import CompressedRow, golden_normalize_latent, normalize_latent
from config import FLASH as M


# 模型配置
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
CACHE_GROUP = 16
PAGE_SIZE = 128

# 分块参数
VECTOR_TILE = 64
TASK_TILE = 16


@cb.kernel
def clear_latent(latent: cb.Tensor, num_tokens: cb.ScalarValue):
    """将有效 token 区间的 latent 置零，保留区间外的存储。"""
    zero_row = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, HEAD_DIM)))
    with cb.vf():
        clear_mask = rr.full_mask(elem_bits=16)
        zero = rr.vdups(0.0, cb.dtypes.bfloat16, mask=clear_mask)
        for col in range(0, HEAD_DIM, VECTOR_TILE * 2):
            rr.vstore(zero_row, col, zero, clear_mask)
    for token in cb.range(num_tokens):
        cb.mem_copy(cb.tile_slice(latent, (1, HEAD_DIM), (token, 0)), zero_row)


@cb.kernel
def pool_state_block(
    kv: cb.Tensor, score: cb.Tensor, state: cb.Tensor, positions: cb.Tensor,
    weight: cb.Tensor, latent: cb.Tensor, starts: cb.Tensor, requests: cb.Tensor, blocks: cb.Tensor,
    num_tokens: cb.ScalarValue, block: cb.ScalarValue,
):
    """由一个任务负责一个物理状态块，完成全部历史读取后再写回环形状态。"""
    current_kv = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, HEAD_DIM)))
    current_score = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, HEAD_DIM)))
    previous_kv = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, HEAD_DIM)))
    previous_score = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, HEAD_DIM)))
    gamma = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((HEAD_DIM,)))
    pooled = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, HEAD_DIM)))
    output = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, HEAD_DIM)))
    cb.mem_copy(gamma, weight)
    for request in cb.range(blocks.shape[0]):
        if blocks[request, 0] == block:
            start = max(starts[request], 0)
            end = min(starts[request + 1], num_tokens)
            for token in cb.range(start, end):
                position = positions[token]
                if requests[token] == request and position >= 0 and position % 2 == 1:
                    cb.mem_copy(current_kv, cb.tile_slice(kv, (1, HEAD_DIM), (token, 0)))
                    cb.mem_copy(current_score, cb.tile_slice(score, (1, HEAD_DIM), (token, 0)))
                    if token > start:
                        cb.mem_copy(previous_kv, cb.tile_slice(kv, (1, HEAD_DIM), (token - 1, 0)))
                        cb.mem_copy(previous_score, cb.tile_slice(score, (1, HEAD_DIM), (token - 1, 0)))
                    else:
                        previous = (position - 1) % state.shape[0]
                        cb.mem_copy(previous_kv, cb.tile_slice(state, (1, HEAD_DIM), (previous, 0)))
                        cb.mem_copy(previous_score, cb.tile_slice(state, (1, HEAD_DIM), (previous, 1)))
                    with cb.vf():
                        pool_mask = rr.full_mask()
                        for col in range(0, HEAD_DIM, VECTOR_TILE):
                            a = rr.vload(previous_score, col)
                            b = rr.vload(current_score, col)
                            high = rr.vmax(a, b, mask=pool_mask)
                            av = rr.vexp(rr.vsub(a, high, mask=pool_mask), mask=pool_mask)
                            bv = rr.vexp(rr.vsub(b, high, mask=pool_mask), mask=pool_mask)
                            denominator = rr.vadd(av, bv, mask=pool_mask)
                            pa = rr.vdiv(av, denominator, mask=pool_mask)
                            pb = rr.vdiv(bv, denominator, mask=pool_mask)
                            va = rr.vmul(rr.vload(previous_kv, col), pa, mask=pool_mask)
                            vb = rr.vmul(rr.vload(current_kv, col), pb, mask=pool_mask)
                            combined = rr.vadd(va, vb, mask=pool_mask)
                            rounded = rr.vcast(combined, cb.dtypes.bfloat16, mask=pool_mask, rounding="rn")
                            rr.vstore_pack(pooled, col, rounded, pool_mask, pack_mode=rr.PackMode.B32_TO_B16)
                    normalize_latent(pooled, gamma, output)
                    cb.mem_copy(cb.tile_slice(latent, (1, HEAD_DIM), (token, 0)), output)
    # 同一物理块的所有请求均读完旧状态后，按请求和 token 顺序更新最后一个环形周期。
    for request in cb.range(blocks.shape[0]):
        if blocks[request, 0] == block:
            end = min(starts[request + 1], num_tokens)
            start = max(max(starts[request], 0), end - state.shape[0])
            for token in cb.range(start, end):
                position = positions[token]
                if requests[token] == request and position >= 0:
                    cb.mem_copy(current_kv, cb.tile_slice(kv, (1, HEAD_DIM), (token, 0)))
                    cb.mem_copy(current_score, cb.tile_slice(score, (1, HEAD_DIM), (token, 0)))
                    target = position % state.shape[0]
                    cb.mem_copy(cb.tile_slice(state, (1, HEAD_DIM), (target, 0)), current_kv)
                    cb.mem_copy(cb.tile_slice(state, (1, HEAD_DIM), (target, 1)), current_score)


@cb.kernel
def publish_compressed_page(
    latent: cb.Tensor, positions: cb.Tensor, starts: cb.Tensor, requests: cb.Tensor, blocks: cb.Tensor,
    slots: cb.Tensor, cosine: cb.Tensor, sine: cb.Tensor, cache: cb.Tensor, scales: cb.Tensor,
    num_tokens: cb.ScalarValue, num_blocks: cb.ScalarValue, page: cb.ScalarValue,
):
    """按 token 顺序发布一个物理页，保留 C2A 的 E2M1 中点向零舍入规则。"""
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, HEAD_DIM)))
    publication = CompressedRow()
    for token in cb.range(num_tokens):
        slot = slots[token]
        request = requests[token]
        position = positions[token]
        if slot >= 0 and slot // PAGE_SIZE == page and request >= 0 and request < blocks.shape[0]:
            block = blocks[request, 0]
            start = starts[request]
            end = starts[request + 1]
            if block >= 0 and block < num_blocks and token >= start and token < end:
                if position >= 0 and position % 2 == 1:
                    cb.mem_copy(raw, cb.tile_slice(latent, (1, HEAD_DIM), (token, 0)))
                    cos_row = cb.tile_slice(cosine, (1, ROPE_DIM // 2), (token, 0))
                    sin_row = cb.tile_slice(sine, (1, ROPE_DIM // 2), (token, 0))
                    publication.encode(raw, cos_row, sin_row, False)
                    target = cb.tile_slice(cache, (1, HEAD_DIM // 2), (slot % PAGE_SIZE, 0))
                    factors = cb.tile_slice(scales, (1, HEAD_DIM // CACHE_GROUP), (slot % PAGE_SIZE, 0))
                    cb.mem_copy(target, publication.payload)
                    cb.mem_copy(factors, publication.scale)


@pl.jit.inline(auto_scope=False)
def _clear_task(latent: pl.Tensor, active: pl.Scalar[pl.INDEX], ready: pl.Scalar[pl.TASK_ID]):
    """使用显式类型的有效行数调度 latent 初始化。"""
    with pl.spmd(1, name_hint="dsl_c2a_clear", deps=[ready]) as done:
        clear_latent(latent, active)
    return done


@pl.jit.inline(auto_scope=False)
def _pool_task(
    kv: pl.Tensor, score: pl.Tensor, state: pl.Tensor, positions: pl.Tensor,
    weight: pl.Tensor, latent: pl.Tensor, starts: pl.Tensor, requests: pl.Tensor, blocks: pl.Tensor,
    active: pl.Scalar[pl.INDEX], block: pl.Scalar[pl.INDEX], ready: pl.Scalar[pl.TASK_ID],
):
    """提交单个物理状态块的池化和状态更新任务。"""
    with pl.spmd(1, name_hint="dsl_c2a_pool_state", deps=[ready]) as done:
        pool_state_block(kv, score, state, positions, weight, latent, starts, requests, blocks, active, block)
    return done


@pl.jit.inline(auto_scope=False)
def _publish_task(
    latent: pl.Tensor, positions: pl.Tensor, starts: pl.Tensor, requests: pl.Tensor, blocks: pl.Tensor,
    slots: pl.Tensor, cosine: pl.Tensor, sine: pl.Tensor, cache: pl.Tensor, scales: pl.Tensor,
    active: pl.Scalar[pl.INDEX], num_blocks: pl.Scalar[pl.INDEX], page: pl.Scalar[pl.INDEX],
    ready: pl.Scalar[pl.TASK_ID],
):
    """提交单个压缩缓存物理页的旋转和量化发布任务。"""
    with pl.spmd(1, name_hint="dsl_c2a_publish", deps=[ready]) as done:
        publish_compressed_page(latent, positions, starts, requests, blocks, slots, cosine, sine,
                                cache, scales, active, num_blocks, page)
    return done


@pl.jit.inline(auto_scope=False)
def c2a_compressor_publish(
    kv_proj: pl.Tensor, score_proj: pl.Tensor, query_start_loc: pl.Tensor, position_ids: pl.Tensor,
    token_to_req_indices: pl.Tensor, state_block_table: pl.Tensor, state_cache: pl.Tensor,
    norm_weight: pl.Tensor, latent: pl.Tensor, num_tokens: pl.Scalar[pl.INT32],
    state_ready: pl.Scalar[pl.TASK_ID], cosine: pl.Tensor, sine: pl.Tensor, slots: pl.Tensor,
    cache: pl.Tensor, scales: pl.Tensor,
):
    """由 simpler 调度互斥状态块和缓存页，并返回所有写入的完成依赖。"""
    rows = pl.tensor.dim(kv_proj, 0)
    active = pl.min(rows, pl.max(pl.cast(num_tokens, pl.INDEX), 0))
    cleared = _clear_task(latent, active, state_ready)
    completed = pl.array.create(TASK_TILE, pl.TASK_ID)
    for slot in pl.range(TASK_TILE):
        completed[slot] = cleared
    num_blocks = pl.tensor.dim(state_cache, 0)
    capacity = pl.tensor.dim(state_cache, 1)
    for block in pl.parallel(num_blocks):
        state_view = pl.slice(state_cache, [1, capacity, 2 * HEAD_DIM], [block, 0, 0])
        state = pl.reshape(state_view, [capacity, 2 * HEAD_DIM])
        previous = completed[block % TASK_TILE]
        done = _pool_task(kv_proj, score_proj, state, position_ids, norm_weight, latent,
                          query_start_loc, token_to_req_indices, state_block_table, active, block, previous)
        completed[block % TASK_TILE] = done
    pooled = pl.system.task_dummy(deps=[completed])
    page_completed = pl.array.create(TASK_TILE, pl.TASK_ID)
    for slot in pl.range(TASK_TILE):
        page_completed[slot] = pooled
    pages = pl.tensor.dim(cache, 0)
    payload_rows = pl.reshape(cache, [pages * PAGE_SIZE, HEAD_DIM // 2])
    scale_rows = pl.reshape(scales, [pages * PAGE_SIZE, HEAD_DIM // CACHE_GROUP])
    for page in pl.parallel(pages):
        payload = pl.slice(payload_rows, [PAGE_SIZE, HEAD_DIM // 2], [page * PAGE_SIZE, 0])
        factors = pl.slice(scale_rows, [PAGE_SIZE, HEAD_DIM // CACHE_GROUP], [page * PAGE_SIZE, 0])
        previous = page_completed[page % TASK_TILE]
        done = _publish_task(latent, position_ids, query_start_loc, token_to_req_indices, state_block_table,
                             slots, cosine, sine, payload, factors, active, num_blocks, page, previous)
        page_completed[page % TASK_TILE] = done
    published = pl.system.task_dummy(deps=[page_completed])
    return published


@pl.jit
def validate_compression(
    kv_proj: pl.Tensor, score_proj: pl.Tensor, query_start_loc: pl.Tensor, position_ids: pl.Tensor,
    token_to_req_indices: pl.Tensor, state_block_table: pl.Tensor, state_cache: pl.InOut[pl.Tensor],
    norm_weight: pl.Tensor, latent: pl.InOut[pl.Tensor], num_tokens: pl.Scalar[pl.INT32],
    cosine: pl.Tensor, sine: pl.Tensor, slots: pl.Tensor,
    cache: pl.InOut[pl.Tensor], scales: pl.InOut[pl.Tensor],
):
    """使用生产路径调度器校验状态、latent 和压缩缓存。"""
    ready = pl.system.task_dummy(deps=[])
    c2a_compressor_publish(kv_proj, score_proj, query_start_loc, position_ids, token_to_req_indices,
                           state_block_table, state_cache, norm_weight, latent, num_tokens, ready,
                           cosine, sine, slots, cache, scales)
    return state_cache, latent, cache, scales


def golden_compression(tensors):
    """由投影结果独立计算双行池化、历史状态和压缩缓存参考结果。"""
    import torch
    from quant import quantize_mxfp4_cache

    kv, score = tensors["kv_proj"], tensors["score_proj"]
    active = max(0, min(kv.shape[0], int(tensors["num_tokens"])))
    state, latent = tensors["state_cache"], tensors["latent"]
    previous_state = state.clone()
    latent[:active].zero_()
    eligible = torch.zeros(kv.shape[0], dtype=torch.bool)
    starts = tensors["query_start_loc"]
    blocks = tensors["state_block_table"].flatten()
    positions = tensors["position_ids"]
    requests = tensors["token_to_req_indices"]
    for request, mapped in enumerate(blocks):
        block = int(mapped)
        if not 0 <= block < state.shape[0]:
            continue
        start = max(int(starts[request]), 0)
        end = min(int(starts[request + 1]), active)
        for token in range(start, end):
            position = int(positions[token])
            if int(requests[token]) != request or position < 0 or position % 2 == 0:
                continue
            if token > start:
                old_kv, old_score = kv[token - 1], score[token - 1]
            else:
                previous = previous_state[block, (position - 1) % state.shape[1]]
                old_kv, old_score = previous[:HEAD_DIM], previous[HEAD_DIM:]
            maximum = torch.maximum(old_score, score[token])
            a, b = (old_score - maximum).exp(), (score[token] - maximum).exp()
            denominator = a + b
            pooled = (old_kv * (a / denominator) + kv[token] * (b / denominator)).bfloat16().float()
            latent[token] = golden_normalize_latent(pooled, tensors["norm_weight"])
            eligible[token] = True
        for token in range(max(start, end - state.shape[1]), end):
            position = int(positions[token])
            if int(requests[token]) == request and position >= 0:
                state[block, position % state.shape[1], :HEAD_DIM] = kv[token]
                state[block, position % state.shape[1], HEAD_DIM:] = score[token]
    rows = tensors["cache"].reshape(-1, HEAD_DIM // 2)
    factors = tensors["scales"].reshape(-1, HEAD_DIM // CACHE_GROUP)
    for token in range(active):
        slot = int(tensors["slots"][token])
        if eligible[token] and 0 <= slot < rows.shape[0]:
            value = latent[token].clone()
            real, imag = value[-ROPE_DIM:].float().reshape(-1, 2).unbind(-1)
            cos, sin = tensors["cosine"][token], tensors["sine"][token]
            rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1)
            value[-ROPE_DIM:] = rotated.flatten().bfloat16()
            payload, scale = quantize_mxfp4_cache(value, CACHE_GROUP, "bf16", round_to_even=False)
            rows[slot] = payload
            factors[slot] = scale


@pl.jit
def validate_replay(
    kv_proj: pl.Tensor, score_proj: pl.Tensor, query_start_loc: pl.Tensor, position_ids: pl.Tensor,
    token_to_req_indices: pl.Tensor, state_block_table: pl.Tensor, state_cache: pl.InOut[pl.Tensor],
    norm_weight: pl.Tensor, latent: pl.InOut[pl.Tensor], num_tokens: pl.Scalar[pl.INT32],
    cosine: pl.Tensor, sine: pl.Tensor, slots: pl.Tensor, cache: pl.InOut[pl.Tensor], scales: pl.InOut[pl.Tensor],
    state_snapshots: pl.Out[pl.Tensor], latent_snapshots: pl.Out[pl.Tensor],
    cache_snapshots: pl.Out[pl.Tensor], scale_snapshots: pl.Out[pl.Tensor],
):
    """在同一 program 中连续执行两个 decode 分块，逐步读取状态与缓存快照。"""
    steps = pl.tensor.dim(kv_proj, 0)
    rows = pl.tensor.dim(kv_proj, 1)
    state_rows = pl.tensor.dim(state_cache, 0) * pl.tensor.dim(state_cache, 1)
    cache_rows = pl.tensor.dim(cache, 0) * PAGE_SIZE
    latent_rows = pl.tensor.dim(latent, 0)
    flat_state = pl.reshape(state_cache, [state_rows, 2 * HEAD_DIM])
    flat_cache = pl.reshape(cache, [cache_rows, HEAD_DIM // 2])
    flat_scale = pl.reshape(scales, [cache_rows, HEAD_DIM // CACHE_GROUP])
    ready = pl.system.task_dummy(deps=[])
    for step in pl.range(steps):
        kv_view = pl.slice(kv_proj, [1, rows, HEAD_DIM], [step, 0, 0])
        kv = pl.reshape(kv_view, [rows, HEAD_DIM])
        score_view = pl.slice(score_proj, [1, rows, HEAD_DIM], [step, 0, 0])
        score = pl.reshape(score_view, [rows, HEAD_DIM])
        pos_view = pl.slice(position_ids, [1, rows], [step, 0])
        positions = pl.reshape(pos_view, [rows])
        published = c2a_compressor_publish(kv, score, query_start_loc, positions, token_to_req_indices,
                                           state_block_table, state_cache, norm_weight, latent, num_tokens, ready,
                                           cosine, sine, slots, cache, scales)
        state_view = pl.slice(state_snapshots, [1, state_rows, 2 * HEAD_DIM], [step, 0, 0])
        state_out = pl.reshape(state_view, [state_rows, 2 * HEAD_DIM])
        latent_view = pl.slice(latent_snapshots, [1, latent_rows, HEAD_DIM], [step, 0, 0])
        latent_out = pl.reshape(latent_view, [latent_rows, HEAD_DIM])
        cache_view = pl.slice(cache_snapshots, [1, cache_rows, HEAD_DIM // 2], [step, 0, 0])
        cache_out = pl.reshape(cache_view, [cache_rows, HEAD_DIM // 2])
        scale_view = pl.slice(scale_snapshots, [1, cache_rows, HEAD_DIM // CACHE_GROUP], [step, 0, 0])
        scale_out = pl.reshape(scale_view, [cache_rows, HEAD_DIM // CACHE_GROUP])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="consume_c2a_state", deps=[published]) as consumed:
            for row in pl.range(state_rows):
                state_value = pl.load(flat_state, [row, 0], [1, 2 * HEAD_DIM])
                state_out = pl.store(state_value, [row, 0], state_out)
            for row in pl.range(latent_rows):
                latent_value = pl.load(latent, [row, 0], [1, HEAD_DIM])
                latent_out = pl.store(latent_value, [row, 0], latent_out)
            for row in pl.range(cache_rows):
                cache_value = pl.load(flat_cache, [row, 0], [1, HEAD_DIM // 2])
                cache_out = pl.store(cache_value, [row, 0], cache_out)
                scale_value = pl.load(flat_scale, [row, 0], [1, HEAD_DIM // CACHE_GROUP])
                scale_out = pl.store(scale_value, [row, 0], scale_out)
        ready = consumed
    return state_cache, latent, cache, scales, state_snapshots, latent_snapshots, cache_snapshots, scale_snapshots


def golden_replay(tensors):
    """连续更新 CPU 参考状态，并保存每一步的完整输出。"""
    for step in range(tensors["kv_proj"].shape[0]):
        current = dict(tensors)
        for name in ("kv_proj", "score_proj", "position_ids"):
            current[name] = tensors[name][step]
        golden_compression(current)
        for source, target in (("state_cache", "state_snapshots"), ("latent", "latent_snapshots"),
                               ("cache", "cache_snapshots"), ("scales", "scale_snapshots")):
            tensors[target][step].copy_(tensors[source].reshape(tensors[target].shape[1:]))


def build_compression_specs(rows=17, active=15, *, random_scores=False, shared_block=False):
    """构造跨步历史、重复槽位、环形回绕、无效元数据和未写区域哨兵值。"""
    import torch
    from golden import ScalarSpec, TensorSpec

    generator = torch.Generator().manual_seed(2719)
    kv = torch.randn(rows, HEAD_DIM, generator=generator)
    score = torch.randn(rows, HEAD_DIM, generator=generator) if random_scores else torch.zeros_like(kv)
    starts = torch.tensor([0, rows // 3, 2 * rows // 3, rows], dtype=torch.int32)
    positions = torch.zeros(rows, dtype=torch.int32)
    requests = torch.zeros(rows, dtype=torch.int32)
    for request in range(3):
        start, end = int(starts[request]), int(starts[request + 1])
        positions[start:end] = torch.arange(end - start, dtype=torch.int32) + 3 + request * 2
        requests[start:end] = request
    blocks = torch.tensor([[2], [0], [2 if shared_block else 1]], dtype=torch.int32)
    state = torch.randn(4, 4, 2 * HEAD_DIM, generator=generator)
    if not random_scores:
        state[:, :, HEAD_DIM:].zero_()
    if rows >= 17:
        positions[2] = -1
        requests[7] = -1
    targets = torch.tensor([0, 127, 128, -1, 0, 255, 256, 128], dtype=torch.int64)
    slots = targets.repeat((rows + 7) // 8)[:rows]
    angle = torch.randn(rows, ROPE_DIM // 2, generator=generator)
    inactive = max(0, min(rows, active))
    kv[inactive:] = float("nan")
    score[inactive:] = float("nan")
    tensors = {
        "kv_proj": kv, "score_proj": score, "query_start_loc": starts, "position_ids": positions,
        "token_to_req_indices": requests, "state_block_table": blocks, "state_cache": state,
        "norm_weight": (torch.rand(HEAD_DIM, generator=generator) + 0.5).bfloat16(),
        "latent": torch.full((rows + 1, HEAD_DIM), -37.0, dtype=torch.bfloat16),
        "cosine": angle.cos(), "sine": angle.sin(), "slots": slots,
        "cache": torch.full((2, PAGE_SIZE, 1, HEAD_DIM // 2), 0xA5, dtype=torch.uint8),
        "scales": torch.full((2, PAGE_SIZE, 1, HEAD_DIM // CACHE_GROUP), 0.125, dtype=torch.bfloat16),
    }
    specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in tensors.items()]
    specs.insert(9, ScalarSpec("num_tokens", torch.int32, active))
    return specs


def build_replay_specs():
    """构造首块结束于偶数位置、后块从奇数位置消费历史状态的连续输入。"""
    from dataclasses import replace
    import torch
    from golden import TensorSpec

    specs = build_compression_specs(17, 17)
    indexed = {spec.name: spec for spec in specs}
    first = indexed["position_ids"].init_value.clone()
    first[first >= 0] += 1
    following = first.clone()
    starts = indexed["query_start_loc"].init_value
    for request in range(3):
        begin, end = int(starts[request]), int(starts[request + 1])
        part = following[begin:end]
        part[part >= 0] += end - begin
    for name in ("kv_proj", "score_proj", "position_ids"):
        value = indexed[name].init_value
        stacked = torch.stack((first, following)) if name == "position_ids" else torch.stack((value, -value))
        indexed[name] = replace(indexed[name], shape=list(stacked.shape), init_value=stacked)
    result = [indexed[spec.name] for spec in specs]
    for source, target, width in (("state_cache", "state_snapshots", 2 * HEAD_DIM),
                                  ("latent", "latent_snapshots", HEAD_DIM),
                                  ("cache", "cache_snapshots", HEAD_DIM // 2),
                                  ("scales", "scale_snapshots", HEAD_DIM // CACHE_GROUP)):
        value = indexed[source].init_value
        result.append(TensorSpec(target, [2, value.numel() // width, width], value.dtype))
    return result


def compare_latent(actual, expected, *, inputs, **kwargs):
    """按整层现有 BF16 数值预算校验随机池化，非有效尾部和保护行要求完全相同。"""
    from golden import ratio_allclose

    active = max(0, min(inputs["kv_proj"].shape[0], int(inputs["num_tokens"])))
    compare_live = ratio_allclose(atol=1e-2, rtol=1.0 / 128, max_error_ratio=0.0)
    compare_tail = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    passed, detail = compare_live(actual[:active], expected[:active], inputs=inputs, **kwargs)
    if not passed:
        return passed, detail
    return compare_tail(actual[active:], expected[active:], inputs=inputs, **kwargs)


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="在 A5 上校验 C2A DSL 池化和缓存发布。")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", choices=["basic", "replay", "all"], default="all")
    args = parser.parse_args()
    exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    cases = [] if args.case == "replay" else [(17, 15, False, False)]
    if args.case == "all":
        cases += [(17, 0, False, False), (1, 1, False, False), (80, 72, True, False), (17, 15, False, True)]
    for rows, active, random_scores, shared_block in cases:
        print(f"[C2A_POOL_DSL] 行数={rows} 有效行={active} 随机分数={random_scores} 共享状态块={shared_block}", flush=True)
        result = run(
            fn=validate_compression,
            specs=build_compression_specs(rows, active, random_scores=random_scores, shared_block=shared_block),
            golden_fn=golden_compression,
            compare_fn={"state_cache": exact, "latent": compare_latent if random_scores else exact,
                        "cache": exact, "scales": exact},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        print(f"[产物] {result.work_dir}", flush=True)
    if args.case in {"replay", "all"}:
        print("[C2A_POOL_DSL] 连续两步状态回放", flush=True)
        names = ("state_cache", "latent", "cache", "scales", "state_snapshots",
                 "latent_snapshots", "cache_snapshots", "scale_snapshots")
        result = run(
            fn=validate_replay,
            specs=build_replay_specs(),
            golden_fn=golden_replay,
            compare_fn={name: exact for name in names},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        print(f"[产物] {result.work_dir}", flush=True)
