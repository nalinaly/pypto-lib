# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Decode indexer query preparation, key publication, scoring, and TopK."""

from index_pool_dsl import (
    select_candidate_block_pool as select_candidate_block_pool,
    golden_select_candidate_block_pool as golden_select_candidate_block_pool,
    index_pool_block_capacity as index_pool_block_capacity,
)

from index_gather_dsl import gather_candidate_cache
from index_projection_dsl import project_index_key
from index_key_dsl import publish_index_rows
from index_weight_dsl import project_index_weights
from index_query_dsl import project_index_query
from index_score_dsl import dispatch_score_topk
from index_plan_dsl import plan_index_runs, plan_pool_runs
from index_route_dsl import classify_candidate_pool, WORKER_TILE as POOL_WORKER_TILE, FLAG_TILE as POOL_FLAG_TILE
import pypto.language as pl
from config import (
    B_DYN, FLASH, INDEX_BLOCKS_DYN, INDEX_CACHE_GROUP, INDEX_POOL_BLOCKS_DYN,
    INDEX_POOL_POSITIONS_DYN, TABLE_DYN, T_DYN,
)


# model config
D = FLASH.hidden_size
HEAD_DIM = FLASH.head_dim
INDEX_H = FLASH.index_n_heads
INDEX_DIM = FLASH.index_head_dim
INDEX_TOPK = FLASH.index_topk
Q_LORA = FLASH.q_lora_rank
ROPE_DIM = FLASH.qk_rope_head_dim
CANDIDATE_BLOCK_SIZE = FLASH.candidate_block_size
CANDIDATE_TOPK_BLOCKS = FLASH.candidate_topk_blocks

_SCALE_COUNT = INDEX_DIM // INDEX_CACHE_GROUP


@pl.jit.inline(auto_scope=False)
def score_select_reindex(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
    query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
    weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    scores: pl.Tensor[[T_DYN, INDEX_POOL_POSITIONS_DYN], pl.FP32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    """Route identity candidate pools to paged MIX and sparse pools to gathered MIX."""
    capacity = pl.tensor.dim(pool_block_ids, 1)
    width = pl.tensor.dim(scores, 1)
    table_width = pl.tensor.dim(block_table, 1)
    cache_pages = pl.tensor.dim(index_cache, 0)
    flags = pl.create_tensor([POOL_WORKER_TILE, POOL_FLAG_TILE], dtype=pl.INT32)
    plan_tid = classify_candidate_pool(
        pool_block_ids, pool_block_counts, request_ids, lengths, block_table, flags,
        num_tokens, width, cache_pages, cache_ready, pool_ready,
    )
    dense = pl.cast(1, pl.INT32)
    for worker in pl.range(POOL_WORKER_TILE):
        dense = dense * pl.read(flags, [worker, 0])
    if dense != 0:
        tokens = pl.tensor.dim(output, 0)
        dense_width = pl.min(width, table_width * 128)
        paged_scores = pl.create_tensor([tokens, dense_width], dtype=pl.FP32)
        selected, scores_tid = score_select_paged(
            query, query_scale, weights, index_cache, index_cache_scale,
            block_table, request_ids, lengths, paged_scores, output,
            num_tokens, plan_tid, query_ready, weights_ready, capacity, False,
        )
        result = selected
    else:
        result = _score_select_pool(
            query, query_scale, weights, index_cache, index_cache_scale,
            block_table, request_ids, lengths, pool_block_ids, pool_block_counts, scores, output,
            num_tokens, cache_ready, query_ready, weights_ready, plan_tid,
        )
    return result


@pl.jit.inline(auto_scope=False)
def _score_select_pool(
    query: pl.Tensor[[T_DYN, INDEX_H * INDEX_DIM // 2], pl.UINT8],
    query_scale: pl.Tensor[[T_DYN, INDEX_H * _SCALE_COUNT], pl.FP8E8M0],
    weights: pl.Tensor[[T_DYN, INDEX_H], pl.BF16],
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    pool_block_ids: pl.Tensor[[T_DYN, INDEX_POOL_BLOCKS_DYN], pl.INT32],
    pool_block_counts: pl.Tensor[[T_DYN], pl.INT32],
    scores: pl.Tensor[[T_DYN, INDEX_POOL_POSITIONS_DYN], pl.FP32],
    output: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    pool_ready: pl.Scalar[pl.TASK_ID],
):
    """Gather candidate blocks and select compact scores with physical-row mappings."""
    capacity = pl.tensor.dim(pool_block_ids, 1)
    tokens = pl.tensor.dim(output, 0)
    width = capacity * CANDIDATE_BLOCK_SIZE
    panels = (width + 255) // 256
    gathered_rows = panels * 256
    total_rows = tokens * gathered_rows
    total_scales = total_rows * _SCALE_COUNT
    payloads = pl.create_tensor([total_rows, INDEX_DIM // 2], dtype=pl.UINT8)
    scales = pl.create_tensor([1, total_scales], dtype=pl.FP8E8M0)
    valid = pl.create_tensor([1, total_rows], dtype=pl.INT32)
    physical = pl.create_tensor([1, total_rows], dtype=pl.INT32)
    runs = pl.create_tensor([4, tokens + 1], dtype=pl.INT32)
    gather_tid = gather_candidate_cache(
        index_cache, index_cache_scale, block_table, request_ids, lengths, pool_block_ids, pool_block_counts,
        payloads, scales, valid, physical, num_tokens, cache_ready, pool_ready,
    )
    runs_tid = plan_pool_runs(pool_block_counts, runs, num_tokens, capacity, pool_ready)
    workers = pl.cast(pl.system.available_cluster_count(), pl.INDEX)
    candidate_keys = pl.create_tensor([tokens, workers * INDEX_TOPK], dtype=pl.UINT16)
    candidate_indices = pl.create_tensor([tokens, workers * INDEX_TOPK], dtype=pl.INT32)
    query_rows = pl.slice(query, [tokens, INDEX_H * INDEX_DIM // 2], [0, 0])
    scale_rows = pl.slice(query_scale, [tokens, INDEX_H * _SCALE_COUNT], [0, 0])
    weight_rows = pl.slice(weights, [tokens, INDEX_H], [0, 0])
    request_rows = pl.reshape(request_ids, [tokens])
    length_rows = pl.reshape(lengths, [tokens])
    output_rows = pl.reshape(output, [tokens, INDEX_TOPK])
    score_rows = pl.slice(scores, [tokens, width], [0, 0])
    payload_pages = pl.reshape(payloads, [total_rows // 128, 128, 1, INDEX_DIM // 2])
    scale_pages = pl.reshape(scales, [total_rows // 128, 128, 1, _SCALE_COUNT])
    active_tokens = pl.cast(num_tokens, pl.INT32)
    no_scores = pl.cast(0, pl.INT32)
    no_shortcut = pl.cast(0, pl.INT32)
    mix_tid = dispatch_score_topk(
        query_rows, scale_rows, weight_rows, payload_pages, scale_pages,
        block_table, request_rows, length_rows, output_rows, runs, candidate_keys, candidate_indices,
        valid, physical, score_rows, active_tokens, gathered_rows,
        no_scores, workers, no_shortcut,
        gather_tid, query_ready, weights_ready, runs_tid,
    )
    return mix_tid


@pl.jit.inline(auto_scope=False)
def score_select_paged(
    query: pl.Tensor,
    query_scale: pl.Tensor,
    weights: pl.Tensor,
    index_cache: pl.Tensor,
    index_cache_scale: pl.Tensor,
    block_table: pl.Tensor,
    request_ids: pl.Tensor,
    lengths: pl.Tensor,
    scores: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    query_ready: pl.Scalar[pl.TASK_ID],
    weights_ready: pl.Scalar[pl.TASK_ID],
    pool_capacity: pl.Scalar[pl.INDEX],
    c2a: pl.constexpr,
):
    """由 simpler 全核同步启动一个 DSL MIX，完成分片打分、屏障和稳定 TopK 归并。"""
    width = pl.tensor.dim(scores, 1)
    capacity = pl.tensor.dim(output, 0)
    runs = pl.create_tensor([4, capacity + 1], dtype=pl.INT32)
    direct_short = pl.cast(0, pl.INT32)
    if c2a:
        direct_short = pl.cast(1, pl.INT32)
    plan_tid = plan_index_runs(request_ids, lengths, runs, num_tokens, width, direct_short)
    run_count = pl.read(runs, [3, 0])
    workers = pl.cast(pl.system.available_cluster_count(), pl.INDEX)
    panels = (width + 255) // 256
    shards = workers
    if panels > 100:
        shards = pl.max(workers, (run_count * panels + 99) // 100)
    block_count = (width + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    emit_scores = pl.cast(0, pl.INT32)
    if not c2a:
        if block_count > pl.min(pool_capacity, CANDIDATE_TOPK_BLOCKS):
            emit_scores = pl.cast(1, pl.INT32)
    active_tokens = pl.cast(num_tokens, pl.INT32)
    gathered_rows = pl.cast(0, pl.INDEX)
    query_rows = pl.slice(query, [capacity, INDEX_H * INDEX_DIM // 2], [0, 0])
    scale_rows = pl.slice(query_scale, [capacity, INDEX_H * (INDEX_DIM // INDEX_CACHE_GROUP)], [0, 0])
    weight_rows = pl.slice(weights, [capacity, INDEX_H], [0, 0])
    request_rows = pl.reshape(request_ids, [capacity])
    length_rows = pl.reshape(lengths, [capacity])
    mixed_output = pl.reshape(output, [capacity, INDEX_TOPK])
    score_rows = pl.slice(scores, [capacity, width], [0, 0])
    candidate_keys = pl.create_tensor([capacity, shards * INDEX_TOPK], dtype=pl.UINT16)
    candidate_indices = pl.create_tensor([capacity, shards * INDEX_TOPK], dtype=pl.INT32)
    mix_tid = dispatch_score_topk(
        query_rows, scale_rows, weight_rows, index_cache, index_cache_scale,
        block_table, request_rows, length_rows, mixed_output, runs, candidate_keys, candidate_indices,
        block_table, block_table, score_rows, active_tokens, gathered_rows,
        emit_scores, shards, direct_short,
        cache_ready, query_ready, weights_ready, plan_tid,
    )
    return mix_tid, mix_tid


@pl.jit.inline(auto_scope=False)
def c1a_index(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[
        [Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN
    ],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    num_tokens: pl.Scalar[pl.INT32],
    score_width: pl.Scalar[pl.INDEX],
    entry_ready: pl.Scalar[pl.TASK_ID],
    projection_payload: pl.Tensor,
    projection_scale_backing: pl.Tensor,
):
    tokens = pl.tensor.dim(x, 0)
    query = pl.create_tensor([tokens, INDEX_H * INDEX_DIM // 2], dtype=pl.UINT8)
    query_scale = pl.create_tensor([tokens, INDEX_H * (INDEX_DIM // INDEX_CACHE_GROUP)], dtype=pl.FP8E8M0)
    query_tid = project_index_query(
        projection_payload, index_wq_b, projection_scale_backing, index_wq_b_scale,
        cos, sin, query, query_scale, num_tokens, entry_ready,
    )
    weights = pl.create_tensor([tokens, INDEX_H], dtype=pl.BF16)
    weights_tid = project_index_weights(
        x, index_weights_proj, weights, num_tokens, entry_ready
    )
    scores = pl.create_tensor([tokens, score_width], dtype=pl.FP32)
    return query, query_scale, weights, scores, query_tid, weights_tid


@pl.jit.inline(auto_scope=False)
def decode_c2a_indexer(
    x: pl.Tensor[[T_DYN, D], pl.BF16],
    projection_payload: pl.Tensor,
    projection_scales: pl.Tensor,
    latent: pl.Tensor[[T_DYN, HEAD_DIM], pl.BF16],
    index_wk: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    index_norm_weight: pl.Tensor[[INDEX_DIM], pl.BF16],
    index_wq_b: pl.Tensor[[Q_LORA, INDEX_H * INDEX_DIM], pl.FP8E4M3FN],
    index_wq_b_scale: pl.Tensor[[Q_LORA // 32, INDEX_H * INDEX_DIM], pl.FP8E8M0, pl.MX_B_NN],
    index_weights_proj: pl.Tensor[[D, INDEX_H], pl.BF16, pl.NZ],
    rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_cos: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_rope_sin: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    compressed_slots: pl.Tensor[[T_DYN], pl.INT64],
    index_cache: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, (INDEX_DIM // 2)], pl.UINT8],
    index_cache_scale: pl.Tensor[[INDEX_BLOCKS_DYN, 128, 1, (INDEX_DIM // INDEX_CACHE_GROUP)], pl.FP8E8M0],
    index_block_table: pl.Tensor[[B_DYN, TABLE_DYN], pl.INT32],
    token_to_req_indices: pl.Tensor[[T_DYN], pl.INT32],
    compressed_lens: pl.Tensor[[T_DYN], pl.INT32],
    topk_indices: pl.Tensor[[T_DYN, INDEX_TOPK], pl.INT32],
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    latent_ready: pl.Scalar[pl.TASK_ID],
):
    """Publish C2A keys and select long histories through the shared MX scorer."""
    tokens = pl.tensor.dim(latent, 0)
    projected_key = pl.create_tensor([tokens, INDEX_DIM], dtype=pl.BF16)
    projected_tid = project_index_key(latent, index_wk, projected_key, num_tokens, latent_ready, max_workers=32)
    index_publish_tid = publish_index_rows(
        projected_key, index_norm_weight, compressed_rope_cos, compressed_rope_sin, compressed_slots,
        index_cache, index_cache_scale, num_tokens, cache_ready, projected_tid,
    )
    # The dense arena follows logical page-table capacity, never allocated cache pages.
    score_width = pl.tensor.dim(index_block_table, 1) * 128
    query, query_scale, raw_weights, scores, query_tid, weights_tid = c1a_index(
        x,
        index_wq_b, index_wq_b_scale, index_weights_proj,
        rope_cos, rope_sin,
        num_tokens, score_width,
        cache_ready,
        projection_payload, projection_scales,
    )
    selected_tid, scores_tid = score_select_paged(
        query, query_scale, raw_weights, index_cache, index_cache_scale,
        index_block_table, token_to_req_indices, compressed_lens, scores, topk_indices,
        num_tokens, index_publish_tid, query_tid, weights_tid, score_width // CANDIDATE_BLOCK_SIZE, True,
    )
    return selected_tid


import torch
from qkv_proj_rope import golden_rope_tail, golden_rope_interleave
from quant import dequantize_mxfp4_cache, quantize_mxfp4_cache
from gemm import golden_gemm_bf16_via_mxfp8_to_bf16_sequential
from rmsnorm import golden_rms_norm


def mapped_fp4_cache_compare(
    mapping_name: str,
    *,
    mapping_shape: tuple[int, ...],
    block_size: int,
    pool_name: str,
    leading_rank_axis: bool = False,
    max_error_ratio: float = 0.005,
):
    """Bound sparse E2M1 changes after floating-point model computation.

    The existing mapped comparator keeps every unwritten byte exact and bounds
    differing written bytes by the declared ratio. A second, unconditional gate
    rejects sign changes (including signed zero) or a jump over more than one
    magnitude level in either nibble. This is an end-to-end numerical policy,
    not an exact quantizer test or a proof of a floating-point error bound.
    """
    from golden import mapped_pool_ratio_allclose

    mapped = mapped_pool_ratio_allclose(
        mapping_name, mapping_shape=mapping_shape, block_size=block_size,
        leading_rank_axis=leading_rank_axis, pool_name=pool_name,
        atol=0.0, rtol=0.0, max_error_ratio=max_error_ratio,
    )

    def compare(actual, expected, **kwargs):
        if actual.dtype != torch.uint8 or expected.dtype != torch.uint8:
            return False, f"{pool_name}: packed E2M1 payload must have UINT8 storage"
        ok, detail = mapped(actual, expected, **kwargs)
        if not ok:
            return False, detail
        a = actual.detach().cpu().to(torch.int16)
        e = expected.detach().cpu().to(torch.int16)
        sign_changed = torch.bitwise_and(torch.bitwise_xor(a, e), 0x88) != 0
        if bool(sign_changed.any()):
            return False, f"{pool_name}: E2M1 sign changed in {int(sign_changed.sum())} byte(s)"
        low_distance = ((a & 7) - (e & 7)).abs()
        high_distance = (((a >> 4) & 7) - ((e >> 4) & 7)).abs()
        invalid = (low_distance > 1) | (high_distance > 1)
        if bool(invalid.any()):
            return False, f"{pool_name}: E2M1 magnitude skips a level in {int(invalid.sum())} byte(s)"
        return True, ""

    return compare


def golden_index_query_fp4(value: torch.Tensor) -> torch.Tensor:
    """Round group-32 index queries to E2M1 with upward E8M0 scales."""
    payload, scales = quantize_mxfp4_cache(value, 32, "e8m0", round_to_even=True)
    return dequantize_mxfp4_cache(payload, scales, 32, "e8m0").to(value.dtype)


def golden_index_key(
    latent: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Project an unrotated compressor latent into a rotated index key."""
    from quant import unpack_bf16_nz

    projected = torch.matmul(latent.float(), unpack_bf16_nz(wk).float()).to(latent.dtype)
    key = golden_rms_norm(projected, norm_weight)
    rd = cos.shape[-1] * 2
    return torch.cat((key[..., :-rd], golden_rope_interleave(key[..., -rd:], cos, sin)), dim=-1)


def golden_index_weighted_sum(dots, weights):
    """Accumulate heads in order with one BF16 rounding per fused multiply-add."""
    values = torch.zeros(dots.shape[-1], dtype=torch.bfloat16, device=dots.device)
    positive = dots.relu().double()
    head_weights = weights.double()
    for head in range(dots.shape[0]):
        values = (positive[head] * head_weights[head] + values.double()).bfloat16()
    return values.float()


def golden_c1a_index_scores(a, qr, index_payload, index_scale):
    """Rebuild the per-token index scores and the physical row behind every logical column."""
    from qkv_proj_rope import golden_rope_interleave as rope_interleave
    from quant import dequantize_mxfp4_cache, unpack_bf16_nz, unpack_fp8_nz

    x = a["x"]
    lengths = a["compressed_lens"]
    # Logical page-table score columns.
    width = a["index_block_table"].shape[-1] * 128
    heads = a["index_weights_proj"].shape[-1]
    iq = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(qr, unpack_fp8_nz(a["index_wq_b"]), a["index_wq_b_scale"])
    if iq.shape[-1] % heads:
        raise RuntimeError(f"index query width {iq.shape[-1]} is not divisible by {heads} heads")
    dim = iq.shape[-1] // heads
    rd = a["rope_cos"].shape[-1] * 2
    iq = iq.unflatten(-1, (heads, dim))
    iq = torch.cat((iq[..., :-rd], rope_interleave(iq[..., -rd:], a["rope_cos"], a["rope_sin"])), -1)
    iq = golden_index_query_fp4(iq)
    keys = dequantize_mxfp4_cache(index_payload, index_scale, 32, "e8m0").to(x.dtype).flatten(0, 1)[:, 0]
    # The device scales the projected head weight in FP32 and rounds it once to BF16.
    weight = ((x).float() @ unpack_bf16_nz(a["index_weights_proj"]).float()).to(torch.bfloat16).float() * (dim * heads) ** -0.5
    weight = weight.to(torch.bfloat16).float()
    scores = torch.full((x.shape[0], width), -torch.inf)
    physical = torch.full((x.shape[0], width), -1, dtype=torch.int64)
    for t in range(x.shape[0]):
        count = max(0, min(int(lengths[t]), width))
        pos = torch.arange(count)
        pages = a["index_block_table"][a["request_ids"][t].long(), pos // 128]
        valid = (pages >= 0) & (pages < index_payload.shape[0])
        valid_pos = pos[valid]
        rows = pages[valid] * 128 + valid_pos % 128
        physical[t, valid_pos] = rows
        if not valid_pos.numel():
            continue
        # BF16 dot products, ReLU, and sequential BF16 head FMAs.
        dots = ((iq[t]).float() @ (keys[rows.long()].T).float()).to(torch.bfloat16)
        values = golden_index_weighted_sum(dots, weight[t])
        scores[t, valid_pos] = values
    return scores, physical


def golden_pool_index_scores(scores, physical, pool_block_ids, pool_block_counts):
    """Pack dense logical scores into the Reindex block-pool score layout."""
    tokens, dense_width = scores.shape
    capacity = pool_block_ids.shape[-1]
    compact_width = capacity * FLASH.candidate_block_size
    compact_scores = torch.full((tokens, compact_width), -torch.inf, dtype=scores.dtype)
    compact_physical = torch.full((tokens, compact_width), -1, dtype=physical.dtype)
    for token in range(tokens):
        count = max(0, min(int(pool_block_counts[token]), capacity))
        for slot in range(count):
            block = int(pool_block_ids[token, slot])
            logical_begin = block * FLASH.candidate_block_size
            if block < 0 or logical_begin >= dense_width:
                continue
            logical_end = min(logical_begin + FLASH.candidate_block_size, dense_width)
            compact_begin = slot * FLASH.candidate_block_size
            compact_end = compact_begin + (logical_end - logical_begin)
            compact_scores[token, compact_begin:compact_end] = scores[token, logical_begin:logical_end]
            compact_physical[token, compact_begin:compact_end] = physical[token, logical_begin:logical_end]
    return compact_scores, compact_physical


def golden_c1a_index_topk(scores, physical, lengths, pool_block_counts=None):
    """Select finite TopK rows with stable ties in logical-position order."""
    topk = torch.full((scores.shape[0], INDEX_TOPK), -1, dtype=torch.int32)
    for token in range(scores.shape[0]):
        if pool_block_counts is None:
            count = min(int(lengths[token]), scores.shape[-1])
        else:
            count = min(int(pool_block_counts[token]) * FLASH.candidate_block_size, scores.shape[-1])
        chosen = torch.argsort(scores[token, :count], descending=True, stable=True)[:INDEX_TOPK]
        chosen = chosen[torch.isfinite(scores[token, chosen])].sort().values
        topk[token, :chosen.numel()] = physical[token, chosen].int()
    return topk


def _select_shared_c2a_topk(scores, physical_rows, lengths):
    """Select C2A rows in logical order, including every valid short-history row."""
    tokens, width = scores.shape
    output = torch.full((tokens, INDEX_TOPK), -1, dtype=torch.int32)
    for token in range(tokens):
        count = max(0, min(int(lengths[token]), width))
        if count <= INDEX_TOPK:
            output[token, :count] = physical_rows[token, :count].to(torch.int32)
            continue
        # Stable descending order makes the common logical-earliest cutoff explicit.
        logical = torch.argsort(scores[token, :count], descending=True, stable=True)[:INDEX_TOPK]
        logical = logical[torch.isfinite(scores[token, logical])].sort().values
        rows = physical_rows[token, logical]
        rows = rows[rows >= 0]
        output[token, :rows.numel()] = rows.to(torch.int32)
    return output


def golden_c2a_index(t, qr, index_cache, index_cache_scale):
    """Score and select C2A paged history with BF16 head reduction."""
    from quant import unpack_bf16_nz, unpack_fp8_nz

    x = t["x"]
    tokens = x.shape[0]
    index_dim = t["index_norm_weight"].numel()
    index_heads = t["index_weights_proj"].shape[-1]
    index_values = dequantize_mxfp4_cache(index_cache, index_cache_scale, INDEX_CACHE_GROUP, "e8m0").flatten(0, -2)
    # Accumulate the index-query projection in sequential group-32 partials.
    index_q = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(qr, unpack_fp8_nz(t["index_wq_b"]), t["index_wq_b_scale"]).unflatten(-1, (index_heads, index_dim))
    index_q = golden_rope_tail(index_q, t["rope_cos"], t["rope_sin"])
    index_q = golden_index_query_fp4(index_q)
    head_weights = ((x).float() @ unpack_bf16_nz(t["index_weights_proj"]).float()).to(torch.bfloat16).float()
    head_weights = (head_weights * (index_dim * index_heads) ** -0.5).to(torch.bfloat16).float()
    lens = t["compressed_lens"].long()
    score_width = t["index_block_table"].shape[1] * 128
    scores = torch.full((tokens, score_width), -torch.inf)
    physical_rows = torch.full((tokens, score_width), -1, dtype=torch.int64)
    for token in range(tokens):
        count = max(0, min(int(lens[token]), score_width))
        request = int(t["token_to_req_indices"][token])
        if not count or not 0 <= request < t["index_block_table"].shape[0]:
            continue
        logical = torch.arange(count)
        pages = t["index_block_table"][request, logical // 128].long()
        rows = pages * 128 + logical.remainder(128)
        valid = (pages >= 0) & (rows >= 0) & (rows < index_values.shape[0])
        valid_logical = logical[valid]
        valid_rows = rows[valid]
        physical_rows[token, valid_logical] = valid_rows
        if count <= INDEX_TOPK or not valid_rows.numel():
            continue
        # Round each fused head multiply-add to BF16.
        dots = ((index_q[token]).float() @ (index_values[valid_rows].to(torch.bfloat16).T).float()).to(torch.bfloat16)
        total = golden_index_weighted_sum(dots, head_weights[token])
        scores[token, valid_logical] = total
    return _select_shared_c2a_topk(scores, physical_rows, lens)


__all__ = [
    "c1a_index",
    "decode_c2a_indexer",
    "golden_c2a_index",
    "golden_c1a_index_scores",
    "golden_pool_index_scores",
    "golden_c1a_index_topk",
    "score_select_reindex",
]
