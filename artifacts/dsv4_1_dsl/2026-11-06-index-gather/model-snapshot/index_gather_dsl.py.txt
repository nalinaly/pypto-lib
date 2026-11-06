# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""以单个 DSL AIV 任务收集候选 Indexer 缓存字节、有效标记和物理行号。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
import pypto.language as pl

from config import FLASH as M, INDEX_CACHE_GROUP


# 模型契约
INDEX_DIM = M.index_head_dim
BLOCK_SIZE = M.candidate_block_size
PAGE_SIZE = 128
SCALE_COUNT = INDEX_DIM // INDEX_CACHE_GROUP

# 分块参数
BLOCK_TILE = 32
ROW_TILE = BLOCK_TILE * BLOCK_SIZE
WORKER_TILE = 32
BYTE_TILE = 256
VECTOR_TILE = 64


@cb.kernel
def gather_pool(
    cache: cb.Tensor, cache_scales: cb.Tensor, table: cb.Tensor,
    requests: cb.Tensor, lengths: cb.Tensor, pool_ids: cb.Tensor, pool_counts: cb.Tensor,
    payload: cb.Tensor, scales: cb.Tensor, valid: cb.Tensor, physical: cb.Tensor,
    num_tokens: cb.ScalarValue,
):
    """每个分块包含 32 个候选块，尾块完整搬运，因果范围仅控制标记和映射。"""
    worker = cb.get_block_idx()
    workers = cb.get_block_num()
    panels = (pool_ids.shape[1] + BLOCK_TILE - 1) // BLOCK_TILE
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((ROW_TILE, INDEX_DIM // 2)))
    factors = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, ROW_TILE * SCALE_COUNT)))
    flags = Buffer(cb.MemLoc.UB, cb.dtypes.int32, cb.make_layout((1, ROW_TILE)))
    mapping = Buffer(cb.MemLoc.UB, cb.dtypes.int32, cb.make_layout((1, ROW_TILE)))
    source_scales = cache_scales.view(dtype=cb.dtypes.uint8)
    target_scales = scales.view(dtype=cb.dtypes.uint8)
    for work in cb.range(worker, num_tokens * panels, workers):
        token = work // panels
        panel = work % panels
        count = pool_counts[token]
        if count < 0:
            count = cb.dtypes.int32(0)
        if count > pool_ids.shape[1]:
            count = cb.dtypes.int32(pool_ids.shape[1])
        request = requests[token]
        visible = lengths[token]
        if visible < 0:
            visible = cb.dtypes.int32(0)
        with cb.vf():
            bytes_mask = rr.full_mask(elem_bits=8)
            zero_byte = rr.vdups(0, cb.dtypes.uint8, mask=bytes_mask)
            default_scale = rr.vdups(127, cb.dtypes.uint8, mask=bytes_mask)
            for offset in range(0, ROW_TILE * INDEX_DIM // 2, BYTE_TILE):
                rr.vstore(raw, offset, zero_byte, bytes_mask)
            for offset in range(0, ROW_TILE * SCALE_COUNT, BYTE_TILE):
                rr.vstore(factors, offset, default_scale, bytes_mask)
            full = rr.full_mask()
            zero = rr.vdups(0, cb.dtypes.int32, mask=full)
            invalid = rr.vdups(-1, cb.dtypes.int32, mask=full)
            for offset in range(0, ROW_TILE, VECTOR_TILE):
                rr.vstore(flags, offset, zero, full)
                rr.vstore(mapping, offset, invalid, full)
        for block in cb.range(BLOCK_TILE):
            slot = panel * BLOCK_TILE + block
            if slot < count:
                logical = pool_ids[token, slot]
                if logical >= 0 and logical < (visible + BLOCK_SIZE - 1) // BLOCK_SIZE:
                    logical_page = logical // (PAGE_SIZE // BLOCK_SIZE)
                    if request >= 0 and request < table.shape[0] and logical_page < table.shape[1]:
                        page = table[request, logical_page]
                        if page >= 0 and page < cache.shape[0] // PAGE_SIZE:
                            source_block = page * (PAGE_SIZE // BLOCK_SIZE) + logical % (PAGE_SIZE // BLOCK_SIZE)
                            source = cb.tile_slice(cache, (BLOCK_SIZE, INDEX_DIM // 2), (source_block, 0))
                            target = cb.tile_slice(raw, (BLOCK_SIZE, INDEX_DIM // 2), (block, 0))
                            cb.mem_copy(target, source)
                            source_scale = cb.tile_slice(source_scales, (1, BLOCK_SIZE * SCALE_COUNT), (source_block, 0))
                            target_scale = cb.tile_slice(factors, (1, BLOCK_SIZE * SCALE_COUNT), (0, block))
                            cb.mem_copy(target_scale, source_scale)
                            remaining = visible - logical * BLOCK_SIZE
                            if remaining > BLOCK_SIZE:
                                remaining = cb.dtypes.int32(BLOCK_SIZE)
                            physical_row = cb.dtypes.int32(source_block * BLOCK_SIZE)
                            with cb.vf():
                                block_mask = rr.update_mask(BLOCK_SIZE, elem_bits=32)[0]
                                lanes = rr.varange(0, cb.dtypes.int32)
                                live = rr.vlts(lanes, remaining, mask=block_mask)
                                ones = rr.vdups(1, cb.dtypes.int32, mask=block_mask)
                                zeros = rr.vdups(0, cb.dtypes.int32, mask=block_mask)
                                value = rr.vselect(ones, zeros, cond_mask=live)
                                locations = rr.vadds(lanes, physical_row, mask=block_mask)
                                invalid_row = rr.vdups(-1, cb.dtypes.int32, mask=block_mask)
                                locations = rr.vselect(locations, invalid_row, cond_mask=live)
                                rr.vstore(flags, block * BLOCK_SIZE, value, block_mask)
                                rr.vstore(mapping, block * BLOCK_SIZE, locations, block_mask)
        cb.mem_copy(cb.tile_slice(payload, (ROW_TILE, INDEX_DIM // 2), (work, 0)), raw)
        cb.mem_copy(cb.tile_slice(target_scales, (1, ROW_TILE * SCALE_COUNT), (0, work)), factors)
        cb.mem_copy(cb.tile_slice(valid, (1, ROW_TILE), (0, work)), flags)
        cb.mem_copy(cb.tile_slice(physical, (1, ROW_TILE), (0, work)), mapping)


@pl.jit.inline(auto_scope=False)
def gather_candidate_cache(
    index_cache: pl.Tensor, index_cache_scale: pl.Tensor, block_table: pl.Tensor,
    request_ids: pl.Tensor, lengths: pl.Tensor, pool_block_ids: pl.Tensor, pool_block_counts: pl.Tensor,
    payloads: pl.Tensor, scales: pl.Tensor, valid: pl.Tensor, physical: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], cache_ready: pl.Scalar[pl.TASK_ID], pool_ready: pl.Scalar[pl.TASK_ID],
):
    """保留缓存和候选池依赖，以同一个 AIV SPMD 任务收集四个输出。"""
    cache_rows = pl.tensor.dim(index_cache, 0) * PAGE_SIZE
    cache_flat = pl.reshape(index_cache, [cache_rows, INDEX_DIM // 2])
    scale_flat = pl.reshape(index_cache_scale, [cache_rows // BLOCK_SIZE, BLOCK_SIZE * SCALE_COUNT])
    with pl.spmd(WORKER_TILE, name_hint="dsl_idx_gather_pool", deps=[cache_ready, pool_ready]) as done:
        gather_pool(
            cache_flat, scale_flat, block_table, request_ids, lengths, pool_block_ids, pool_block_counts,
            payloads, scales, valid, physical, num_tokens,
        )
    return done


@pl.jit
def validate_gather(
    cache: pl.Tensor, cache_scales: pl.Tensor, table: pl.Tensor,
    requests: pl.Tensor, lengths: pl.Tensor, pool_ids: pl.Tensor, pool_counts: pl.Tensor,
    payload: pl.InOut[pl.Tensor], scales: pl.InOut[pl.Tensor],
    valid: pl.InOut[pl.Tensor], physical: pl.InOut[pl.Tensor], num_tokens: pl.Scalar[pl.INT32],
):
    """通过生产入口验证实际搬运，保留非有效 token 和一个额外输出分块。"""
    ready = pl.system.task_dummy(deps=[])
    gather_candidate_cache(
        cache, cache_scales, table, requests, lengths, pool_ids, pool_counts,
        payload, scales, valid, physical, num_tokens, ready, ready,
    )
    return payload, scales, valid, physical


def golden_gather(tensors):
    """按候选顺序收集完整八行字节，单独生成因果标记和物理行号。"""
    import torch

    cache = tensors["cache"].reshape(-1, INDEX_DIM // 2)
    cache_scales = tensors["cache_scales"].view(torch.uint8).reshape(-1, SCALE_COUNT)
    factors = tensors["scales"].view(torch.uint8).reshape(-1, SCALE_COUNT)
    flags = tensors["valid"].reshape(-1)
    mapping = tensors["physical"].reshape(-1)
    table = tensors["table"]
    capacity = tensors["pool_ids"].shape[1]
    gathered_rows = (capacity + BLOCK_TILE - 1) // BLOCK_TILE * ROW_TILE
    for token in range(int(tensors["num_tokens"])):
        begin, end = token * gathered_rows, (token + 1) * gathered_rows
        tensors["payload"][begin:end].zero_()
        factors[begin:end].fill_(127)
        flags[begin:end].zero_()
        mapping[begin:end].fill_(-1)
        request = int(tensors["requests"][token])
        count = max(0, min(int(tensors["pool_counts"][token]), capacity))
        visible = max(0, int(tensors["lengths"][token]))
        for slot in range(count):
            logical = int(tensors["pool_ids"][token, slot])
            if not 0 <= logical < (visible + BLOCK_SIZE - 1) // BLOCK_SIZE:
                continue
            logical_page = logical // (PAGE_SIZE // BLOCK_SIZE)
            if not 0 <= request < table.shape[0] or logical_page >= table.shape[1]:
                continue
            page = int(table[request, logical_page])
            if not 0 <= page < tensors["cache"].shape[0]:
                continue
            source = page * PAGE_SIZE + logical % (PAGE_SIZE // BLOCK_SIZE) * BLOCK_SIZE
            target = begin + slot * BLOCK_SIZE
            tensors["payload"][target:target + BLOCK_SIZE] = cache[source:source + BLOCK_SIZE]
            factors[target:target + BLOCK_SIZE] = cache_scales[source:source + BLOCK_SIZE]
            live = min(BLOCK_SIZE, visible - logical * BLOCK_SIZE)
            flags[target:target + live] = 1
            mapping[target:target + live] = torch.arange(source, source + live, dtype=torch.int32)


def build_gather_specs(rows=7, capacity=33, active=6, kind="mixed"):
    """覆盖全部字节编码、非连续/重复块、非法页与请求、尾块和输出保护区。"""
    import torch
    from golden import ScalarSpec, TensorSpec

    cache_pages = 5
    row = torch.arange(cache_pages * PAGE_SIZE).reshape(-1, 1)
    column = torch.arange(INDEX_DIM // 2)
    block, page = row // BLOCK_SIZE, row // PAGE_SIZE
    cache = ((row * 19 + block * 37 + column * 11 + page * 5) % 256).to(torch.uint8)
    scale = ((row * 7 + block * 13 + torch.arange(SCALE_COUNT) * 61 + page * 17) % 256).to(torch.uint8)
    cache_shape = [cache_pages, PAGE_SIZE, 1, INDEX_DIM // 2]
    scale_shape = [cache_pages, PAGE_SIZE, 1, SCALE_COUNT]
    cache = cache.reshape(cache_shape)
    scale = scale.reshape(scale_shape).view(torch.float8_e8m0fnu)
    table = torch.tensor([[2, 0, -1, 5, 1], [1, 3, 0, 2, 4], [-1, 4, 2, 1, 0]], dtype=torch.int32)

    def repeated(values, size):
        return torch.tensor((values * ((size + len(values) - 1) // len(values)))[:size], dtype=torch.int32)

    requests = repeated([0, 1, -1, 3, 2, 2, 0], rows)
    lengths = repeated([515, 133, 400, 400, 1024, -7, 129], rows)
    counts = repeated([capacity + 7, capacity - 1, capacity, capacity, -3, capacity, capacity], rows)
    pattern = [0, 15, 16, 17, 31, 32, 47, 48, 63, 64, 65, -1, 80, 16, 1, 63]
    pool_ids = repeated(pattern, capacity).repeat(rows, 1)
    if kind == "bytes":
        cache = torch.arange(cache.numel()).remainder(256).to(torch.uint8).reshape(cache_shape)
        codes = torch.arange(scale.numel()).remainder(256).to(torch.uint8).reshape(scale_shape)
        scale = codes.view(torch.float8_e8m0fnu)
        table = torch.arange(cache_pages, dtype=torch.int32).repeat(3, 1)
        requests.zero_()
        lengths.fill_(cache_pages * PAGE_SIZE)
        counts.fill_(capacity)
        pool_ids = torch.arange(capacity, dtype=torch.int32).repeat(rows, 1)
    total_rows = (rows * ((capacity + BLOCK_TILE - 1) // BLOCK_TILE) + 1) * ROW_TILE
    scale_init = torch.full((1, total_rows * SCALE_COUNT), 0xA5, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    return [
        TensorSpec("cache", cache_shape, torch.uint8, init_value=cache),
        TensorSpec("cache_scales", scale_shape, torch.float8_e8m0fnu, init_value=scale),
        TensorSpec("table", list(table.shape), torch.int32, init_value=table),
        TensorSpec("requests", [rows], torch.int32, init_value=requests),
        TensorSpec("lengths", [rows], torch.int32, init_value=lengths),
        TensorSpec("pool_ids", [rows, capacity], torch.int32, init_value=pool_ids),
        TensorSpec("pool_counts", [rows], torch.int32, init_value=counts),
        TensorSpec("payload", [total_rows, INDEX_DIM // 2], torch.uint8, init_value=0xA5),
        TensorSpec("scales", [1, total_rows * SCALE_COUNT], torch.float8_e8m0fnu, init_value=scale_init),
        TensorSpec("valid", [1, total_rows], torch.int32, init_value=-71),
        TensorSpec("physical", [1, total_rows], torch.int32, init_value=-73),
        ScalarSpec("num_tokens", torch.int32, active),
    ]


def compare_bytes(actual, expected, **kwargs):
    """逐字节比较，包括 E8M0 的所有编码，不执行浮点数值转换。"""
    import torch
    from golden import ratio_allclose

    exact = ratio_allclose(atol=0, rtol=0, max_error_ratio=0)
    return exact(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8), **kwargs)


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="校验候选 Indexer 缓存收集的四个输出。")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", choices=("basic", "all"), default="all")
    args = parser.parse_args()
    cases = [(7, 33, 6)]
    if args.case == "all":
        cases += [(7, 33, 0), (3, 1, 3), (5, 32, 5), (3, 65, 3), (37, 33, 35), (1, 32, 1, "bytes")]
    for case in cases:
        print(f"[INDEX_GATHER_DSL] {case}", flush=True)
        result = run(
            fn=validate_gather,
            specs=build_gather_specs(*case),
            golden_fn=golden_gather,
            compare_fn={name: compare_bytes for name in ("payload", "scales", "valid", "physical")},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        print(f"[产物] {result.work_dir}", flush=True)
