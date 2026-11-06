# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""检查候选池的恒等块映射、可见范围和页号上界，生成稠密分支标记。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.jit import jit
import pypto.language as pl

from config import BLOCK_SIZE as PAGE_SIZE, FLASH as M


# 模型契约
BLOCK_SIZE = M.candidate_block_size

# 分块参数
WORKER_TILE = 32
FLAG_TILE = 32
ID_TILE = 1024
PAGE_TILE = 128
VECTOR_TILE = 64

# 校验存储维度
ROWS_DYN = pl.dynamic("IDX_ROUTE_ROWS")
CAPACITY_DYN = pl.dynamic("IDX_ROUTE_CAPACITY")
REQUESTS_DYN = pl.dynamic("IDX_ROUTE_REQUESTS")
PAGES_DYN = pl.dynamic("IDX_ROUTE_PAGES")


@jit
def _check_identity_panel(raw, state, live, first):
    """统计有效位置中不满足恒等块映射的数量。"""
    with cb.vf():
        full = rr.full_mask()
        zero = rr.vdups(0, cb.dtypes.int32, mask=full)
        one = rr.vdups(1, cb.dtypes.int32, mask=full)
        errors = rr.vdups(0, cb.dtypes.int32, mask=full)
        for vector in cb.range((live + VECTOR_TILE - 1) // VECTOR_TILE):
            active = rr.update_mask(live - vector * VECTOR_TILE, elem_bits=32)[0]
            value = rr.vload(raw, vector * VECTOR_TILE)
            expected = rr.varange(first + vector * VECTOR_TILE, cb.dtypes.int32)
            correct = rr.veq(value, expected, mask=active)
            invalid = rr.vselect(zero, one, cond_mask=correct)
            invalid = rr.vselect(invalid, zero, cond_mask=active)
            errors = rr.vadd(errors, invalid, mask=full)
        rr.vstore_first(state, 0, rr.vreduce_sum(errors, mask=full))
    cb.vec_sync_all()
    return state[0]


@jit
def _check_page_panel(raw, state, live, limit):
    """统计达到或超过页号上界的有效页，负页号不计入错误。"""
    with cb.vf():
        page_full = rr.full_mask()
        zero = rr.vdups(0, cb.dtypes.int32, mask=page_full)
        one = rr.vdups(1, cb.dtypes.int32, mask=page_full)
        errors = rr.vdups(0, cb.dtypes.int32, mask=page_full)
        for vector in cb.range((live + VECTOR_TILE - 1) // VECTOR_TILE):
            page_active = rr.update_mask(live - vector * VECTOR_TILE, elem_bits=32)[0]
            value = rr.vload(raw, vector * VECTOR_TILE)
            below_limit = rr.vlts(value, limit, mask=page_active)
            invalid = rr.vselect(zero, one, cond_mask=below_limit)
            invalid = rr.vselect(invalid, zero, cond_mask=page_active)
            errors = rr.vadd(errors, invalid, mask=page_full)
        rr.vstore_first(state, 0, rr.vreduce_sum(errors, mask=page_full))
    cb.vec_sync_all()
    return state[0]


@cb.kernel
def classify_pool_rows(
    pool_ids: cb.Tensor, pool_counts: cb.Tensor, request_ids: cb.Tensor, lengths: cb.Tensor,
    table: cb.Tensor, flags: cb.Tensor,
    num_tokens: cb.ScalarValue, width: cb.ScalarValue, cache_pages: cb.ScalarValue,
):
    """各 worker 累积所负责 Query 的判定，只发布一个 INT32 标记。"""
    worker = cb.get_block_idx()
    workers = cb.get_block_num()
    ids = Buffer(cb.MemLoc.UB, cb.dtypes.int32, cb.make_layout((1, ID_TILE)))
    pages = Buffer(cb.MemLoc.UB, cb.dtypes.int32, cb.make_layout((1, PAGE_TILE)))
    state = Buffer(cb.MemLoc.UB, cb.dtypes.int32, cb.make_layout((VECTOR_TILE,)))
    dense = cb.dtypes.int32(1)
    for token in cb.range(worker, num_tokens, workers):
        count = cb.dtypes.int64(pool_counts[token])
        if count > pool_ids.shape[1]:
            count = cb.dtypes.int64(pool_ids.shape[1])
        if count < 0:
            count = cb.dtypes.int64(0)
        visible = cb.dtypes.int64(lengths[token])
        if visible < 0:
            visible = cb.dtypes.int64(0)
        request = request_ids[token]
        if count != (visible + BLOCK_SIZE - 1) // BLOCK_SIZE or visible > width:
            dense = cb.dtypes.int32(0)
        if visible > table.shape[1] * PAGE_SIZE or request < 0 or request >= table.shape[0]:
            dense = cb.dtypes.int32(0)
        if dense == 1:
            for first in cb.range(0, count, ID_TILE):
                live = min(ID_TILE, count - first)
                cb.mem_copy(ids[None, 0:live], pool_ids[token:token + 1, first:first + live])
                mismatch = _check_identity_panel(ids, state, live, first)
                if mismatch != 0:
                    dense = cb.dtypes.int32(0)
            page_count = (visible + PAGE_SIZE - 1) // PAGE_SIZE
            for first_page in cb.range(0, page_count, PAGE_TILE):
                live_pages = min(PAGE_TILE, page_count - first_page)
                cb.mem_copy(pages[None, 0:live_pages], table[request:request + 1, first_page:first_page + live_pages])
                # 仅检查页号上界，负页号保留原判定。
                mismatch_pages = _check_page_panel(pages, state, live_pages, cb.dtypes.int32(cache_pages))
                if mismatch_pages != 0:
                    dense = cb.dtypes.int32(0)
    flags[worker, 0] = dense


@pl.jit.inline(auto_scope=False)
def classify_candidate_pool(
    pool_ids: pl.Tensor, pool_counts: pl.Tensor, request_ids: pl.Tensor, lengths: pl.Tensor,
    table: pl.Tensor, flags: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INDEX], cache_pages: pl.Scalar[pl.INDEX],
    cache_ready: pl.Scalar[pl.TASK_ID], pool_ready: pl.Scalar[pl.TASK_ID],
):
    """保留两个输入依赖，提交 32 worker 的本地 AIV 判定任务。"""
    with pl.spmd(WORKER_TILE, name_hint="dsl_idx_plan_pool", deps=[cache_ready, pool_ready]) as ready:
        classify_pool_rows(pool_ids, pool_counts, request_ids, lengths, table, flags, num_tokens, width, cache_pages)
    return ready


@pl.jit
def validate_route(
    pool_ids: pl.Tensor[[ROWS_DYN, CAPACITY_DYN], pl.INT32],
    pool_counts: pl.Tensor[[ROWS_DYN], pl.INT32],
    request_ids: pl.Tensor[[ROWS_DYN], pl.INT32],
    lengths: pl.Tensor[[ROWS_DYN], pl.INT32],
    table: pl.Tensor[[REQUESTS_DYN, PAGES_DYN], pl.INT32],
    flags: pl.InOut[pl.Tensor[[WORKER_TILE + 1, FLAG_TILE], pl.INT32]],
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.Scalar[pl.INDEX],
    cache_pages: pl.Scalar[pl.INDEX],
):
    """使用生产布局校验全部 worker 标记和未写入的行列保护区。"""
    pool_ids.bind_dynamic(0, ROWS_DYN)
    pool_ids.bind_dynamic(1, CAPACITY_DYN)
    pool_counts.bind_dynamic(0, ROWS_DYN)
    request_ids.bind_dynamic(0, ROWS_DYN)
    lengths.bind_dynamic(0, ROWS_DYN)
    table.bind_dynamic(0, REQUESTS_DYN)
    table.bind_dynamic(1, PAGES_DYN)
    ready = pl.system.task_dummy(deps=[])
    classify_candidate_pool(
        pool_ids, pool_counts, request_ids, lengths, table, flags,
        num_tokens, width, cache_pages, ready, ready,
    )
    return flags


def golden_route(tensors):
    """逐 Query 计算布尔条件，再对同一 worker 的结果取逻辑与。"""
    import torch

    ids, counts = tensors["pool_ids"], tensors["pool_counts"]
    requests, lengths = tensors["request_ids"], tensors["lengths"]
    table = tensors["table"]
    width, cache_pages = int(tensors["width"]), int(tensors["cache_pages"])
    active = int(tensors["num_tokens"])
    for worker in range(WORKER_TILE):
        dense = True
        for token in range(worker, active, WORKER_TILE):
            count = max(0, min(int(counts[token]), ids.shape[1]))
            visible = max(0, int(lengths[token]))
            request = int(requests[token])
            valid = count == (visible + BLOCK_SIZE - 1) // BLOCK_SIZE
            valid = valid and visible <= min(width, table.shape[1] * PAGE_SIZE)
            valid = valid and 0 <= request < table.shape[0]
            if valid:
                valid = torch.equal(ids[token, :count], torch.arange(count, dtype=torch.int32))
                page_count = (visible + PAGE_SIZE - 1) // PAGE_SIZE
                valid = valid and bool((table[request, :page_count] < cache_pages).all())
            dense = dense and valid
        tensors["flags"][worker, 0] = int(dense)


# 校验参数：Query 存储行数、有效行数、候选容量、分数宽度、页表宽度。
ROUTE_CASES = {
    "zero": (7, 0, 1, 8, 1),
    "dense": (72, 72, 33, 264, 3),
    "mixed": (72, 65, 33, 264, 3),
    "long-tail": (7, 7, 1025, 8200, 65),
    "max-pool": (72, 72, 2048, 16384, 128),
    "negative-pages": (33, 33, 33, 264, 3),
    "bounds": (17, 15, 65, 1024, 3),
}


def build_route_specs(case="mixed"):
    """覆盖判定边界、分块尾部、负页号、worker 轮转及输出保护区。"""
    import torch
    from golden import ScalarSpec, TensorSpec

    rows, active, capacity, width, page_columns = ROUTE_CASES[case]
    cache_pages = page_columns + 3
    ids = torch.arange(capacity, dtype=torch.int32).repeat(rows, 1)
    counts = torch.full((rows,), capacity, dtype=torch.int32)
    requests = torch.arange(rows, dtype=torch.int32) % 3
    lengths = torch.full((rows,), min(capacity * BLOCK_SIZE, page_columns * PAGE_SIZE), dtype=torch.int32)
    counts[:] = (lengths + BLOCK_SIZE - 1) // BLOCK_SIZE
    table = torch.arange(page_columns, dtype=torch.int32).repeat(3, 1)
    if case == "mixed":
        ids[0, 0] = 1
        counts[1] -= 1
        requests[2] = -1
        requests[3] = 3
        lengths[4] = width + 1
        counts[6], lengths[6] = -7, -4
        counts[7] = capacity + 5
        counts[8], lengths[8] = 4, 32
        requests[8] = 0
        ids[8, 4:] = -99
        counts[9], lengths[9], requests[9] = 0, 0, -1
        ids[10, 5] = ids[10, 4]
        ids[11, 0] = -2147483648
        table[2, 0] = cache_pages
    elif case == "long-tail":
        ids[1, -1] = -1
        lengths[2] -= 1
    elif case == "max-pool":
        ids[0, 1023] = -1
        ids[1, 1024] = -1
        ids[2, -1] = -1
        table[2, -1] = cache_pages
    elif case == "negative-pages":
        table[0, 0] = -1
        table[1, 0] = -2147483648
    elif case == "bounds":
        lengths[0], counts[0] = page_columns * PAGE_SIZE + 1, 49
        lengths[1], counts[1] = width + 1, capacity
        counts[2], lengths[2] = -1, 0
        counts[3], lengths[3] = 0, -1
        requests[4] = -2147483648
        requests[5] = 2147483647
        counts[6] = 2147483647
        table[2, 0] = cache_pages - 1
    counts[active:] = -1
    requests[active:] = -999
    lengths[active:] = 2147483647
    initial_flags = torch.arange((WORKER_TILE + 1) * FLAG_TILE, dtype=torch.int32) - 1000037
    flags = initial_flags.reshape(WORKER_TILE + 1, FLAG_TILE)
    return [
        TensorSpec("pool_ids", [rows, capacity], torch.int32, init_value=ids),
        TensorSpec("pool_counts", [rows], torch.int32, init_value=counts),
        TensorSpec("request_ids", [rows], torch.int32, init_value=requests),
        TensorSpec("lengths", [rows], torch.int32, init_value=lengths),
        TensorSpec("table", [3, page_columns], torch.int32, init_value=table),
        TensorSpec("flags", [WORKER_TILE + 1, FLAG_TILE], torch.int32, init_value=flags),
        ScalarSpec("num_tokens", torch.int32, active, compile_runtime=True),
        ScalarSpec("width", torch.int64, width, compile_runtime=True),
        ScalarSpec("cache_pages", torch.int64, cache_pages, compile_runtime=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="校验候选池稠密分支判定")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--case", choices=[*ROUTE_CASES, "all"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    cases = ROUTE_CASES if args.case == "all" else [args.case]
    failures = []
    for case in cases:
        print("[候选池判定] " + case, flush=True)
        result = run(
            fn=validate_route,
            specs=build_route_specs(case),
            golden_fn=golden_route,
            config={"backend_type": "a5", "execution_mode": False, "device_id": args.device},
            compare_fn={"flags": ratio_allclose(atol=0, rtol=0, max_error_ratio=0)},
            compile_only=args.compile_only,
            save_data=args.save_data,
        )
        if not result.passed:
            failures.append(f"{case}: {result.error}")
    if failures:
        raise SystemExit("; ".join(failures))
