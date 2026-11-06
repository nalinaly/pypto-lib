# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""生成 Indexer 连续请求分组和稀疏候选池的运行区间及行映射。"""

import cannbotdsl as cb
import pypto.language as pl

from config import FLASH as M


# 模型契约
INDEX_TOPK = M.index_topk
BLOCK_SIZE = M.candidate_block_size

# 分块参数
QUERY_ROW_TILE = 6

# 校验存储维度
INPUT_ROWS_DYN = pl.dynamic("IDX_PLAN_INPUT_ROWS")
RUN_COLUMNS_DYN = pl.dynamic("IDX_PLAN_RUN_COLUMNS")


@cb.kernel
def plan_runs(
    request_ids: cb.Tensor, lengths: cb.Tensor, runs: cb.Tensor,
    num_tokens: cb.ScalarValue, width: cb.ScalarValue, skip_short: cb.ScalarValue,
):
    """由一个 AIV 顺序生成区间起止、最大可见长度、Query 映射和区间总数。"""
    run_count = cb.dtypes.int32(0)
    run_start = cb.dtypes.int32(0)
    previous_request = cb.dtypes.int32(-1)
    maximum_visible = cb.dtypes.int32(0)
    for token in cb.range(num_tokens):
        request = request_ids[token]
        if token > 0:
            if request != previous_request or token - run_start == QUERY_ROW_TILE:
                runs[0, run_count] = run_start
                runs[1, run_count] = cb.dtypes.int32(token)
                runs[2, run_count] = maximum_visible
                run_count = run_count + 1
                run_start = cb.dtypes.int32(token)
                maximum_visible = cb.dtypes.int32(0)
        visible = cb.dtypes.int64(lengths[token])
        if visible > width:
            visible = cb.dtypes.int64(width)
        if visible < 0:
            visible = cb.dtypes.int64(0)
        current_visible = cb.dtypes.int32(visible)
        if skip_short != 0:
            if current_visible <= INDEX_TOPK:
                current_visible = cb.dtypes.int32(0)
        if current_visible > maximum_visible:
            maximum_visible = current_visible
        runs[3, token + 1] = run_count
        previous_request = request
    if num_tokens > 0:
        runs[0, run_count] = run_start
        runs[1, run_count] = num_tokens
        runs[2, run_count] = maximum_visible
        run_count = run_count + 1
    runs[3, 0] = run_count


@cb.kernel
def plan_pool_rows(
    pool_counts: cb.Tensor, runs: cb.Tensor, num_tokens: cb.ScalarValue, capacity: cb.ScalarValue,
):
    """每个有效 Query 独立成组，将截断后的候选块数转换为候选行数。"""
    for token in cb.range(num_tokens):
        count = cb.dtypes.int64(pool_counts[token])
        if count > capacity:
            count = cb.dtypes.int64(capacity)
        if count < 0:
            count = cb.dtypes.int64(0)
        runs[0, token] = cb.dtypes.int32(token)
        runs[1, token] = cb.dtypes.int32(token + 1)
        runs[2, token] = cb.dtypes.int32(count * BLOCK_SIZE)
        runs[3, token + 1] = cb.dtypes.int32(token)
    runs[3, 0] = num_tokens


@pl.jit.inline(auto_scope=False)
def plan_index_runs(
    request_ids: pl.Tensor, lengths: pl.Tensor, runs: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INDEX], skip_short: pl.Scalar[pl.INT32],
):
    """提交一个本地 AIV 任务，返回供 MIX 打分消费的完成依赖。"""
    with pl.spmd(1, name_hint="dsl_idx_plan_req") as ready:
        plan_runs(request_ids, lengths, runs, num_tokens, width, skip_short)
    return ready


@pl.jit.inline(auto_scope=False)
def plan_pool_runs(
    pool_counts: pl.Tensor, runs: pl.Tensor, num_tokens: pl.Scalar[pl.INT32],
    capacity: pl.Scalar[pl.INDEX], pool_ready: pl.Scalar[pl.TASK_ID],
):
    """等待候选池就绪后提交一个 AIV 规划任务，并返回下游完成依赖。"""
    with pl.spmd(1, name_hint="dsl_idx_plan_pool_runs", deps=[pool_ready]) as ready:
        plan_pool_rows(pool_counts, runs, num_tokens, capacity)
    return ready


@pl.jit
def validate_plan(
    request_ids: pl.Tensor[[INPUT_ROWS_DYN], pl.INT32],
    lengths: pl.Tensor[[INPUT_ROWS_DYN], pl.INT32],
    runs: pl.InOut[pl.Tensor[[5, RUN_COLUMNS_DYN], pl.INT32]],
    num_tokens: pl.Scalar[pl.INT32],
    width: pl.Scalar[pl.INDEX],
    skip_short: pl.Scalar[pl.INT32],
):
    """使用非零输入偏移及带保护区的非紧密输出视图验证生产入口。"""
    request_ids.bind_dynamic(0, INPUT_ROWS_DYN)
    lengths.bind_dynamic(0, INPUT_ROWS_DYN)
    runs.bind_dynamic(1, RUN_COLUMNS_DYN)
    capacity = pl.tensor.dim(request_ids, 0) - 2
    requests = request_ids[1:capacity + 1]
    visible = lengths[1:capacity + 1]
    target = runs[0:4, 1:capacity + 2]
    plan_index_runs(requests, visible, target, num_tokens, width, skip_short)
    return runs


@pl.jit
def validate_pool_plan(
    pool_counts: pl.Tensor[[INPUT_ROWS_DYN], pl.INT32],
    runs: pl.InOut[pl.Tensor[[5, RUN_COLUMNS_DYN], pl.INT32]],
    num_tokens: pl.Scalar[pl.INT32],
    capacity: pl.Scalar[pl.INDEX],
):
    """校验候选池规划的输入偏移、非连续输出视图及保护区。"""
    pool_counts.bind_dynamic(0, INPUT_ROWS_DYN)
    runs.bind_dynamic(1, RUN_COLUMNS_DYN)
    rows = pl.tensor.dim(pool_counts, 0) - 2
    counts = pool_counts[1:rows + 1]
    target = runs[0:4, 1:rows + 2]
    pool_ready = pl.system.task_dummy(deps=[])
    plan_pool_runs(counts, target, num_tokens, capacity, pool_ready)
    return runs


def golden_plan(tensors):
    """先按连续请求分段，再切成最多六行的子段，保留其余输出字节。"""
    from itertools import groupby

    active = int(tensors["num_tokens"])
    width = int(tensors["width"])
    skip_short = bool(tensors["skip_short"])
    requests = tensors["request_ids"][1:active + 1].tolist()
    lengths = tensors["lengths"][1:active + 1].clamp(min=0, max=max(0, width))
    if skip_short:
        lengths = lengths.masked_fill(lengths <= INDEX_TOPK, 0)
    output = tensors["runs"][:4, 1:-1]
    cursor = 0
    count = 0
    for _, group in groupby(requests):
        end = cursor + len(list(group))
        for begin in range(cursor, end, QUERY_ROW_TILE):
            finish = min(begin + QUERY_ROW_TILE, end)
            output[0, count] = begin
            output[1, count] = finish
            output[2, count] = lengths[begin:finish].max()
            output[3, begin + 1:finish + 1] = count
            count += 1
        cursor = end
    output[3, 0] = count


def golden_pool_plan(tensors):
    """用整型向量参考计算有效 Query 的独立区间，其余输出保持原值。"""
    import torch

    active = int(tensors["num_tokens"])
    capacity = max(0, int(tensors["capacity"]))
    counts = tensors["pool_counts"][1:active + 1].to(torch.int64).clamp(min=0, max=capacity)
    rows = torch.arange(active, dtype=torch.int32)
    output = tensors["runs"][:4, 1:-1]
    output[0, :active] = rows
    output[1, :active] = rows + 1
    output[2, :active] = (counts * BLOCK_SIZE).to(torch.int32)
    output[3, 1:active + 1] = rows
    output[3, 0] = active


# 校验参数：容量、有效行数、可见宽度、请求排列。
PLAN_CASES = {
    "zero": (7, 0, 1024, "mixed"),
    "single": (1, 1, 1024, "same"),
    "six": (6, 6, 1024, "same"),
    "seven": (7, 7, 1024, "same"),
    "short": (6, 6, 512, "same"),
    "mixed": (17, 15, 769, "mixed"),
    "decode72": (72, 72, 1024, "mixed"),
}


# 校验参数：Query 存储行数、有效行数、候选块容量。
POOL_PLAN_CASES = {
    "zero": (7, 0, 2048),
    "single": (1, 1, 1),
    "mixed": (17, 15, 33),
    "empty-capacity": (7, 7, 0),
    "decode72": (72, 72, 2048),
}


def build_plan_specs(case="mixed", skip_short=0):
    """构造请求重复、六行边界、长度截断及非有效行和输出保护区。"""
    import torch
    from golden import ScalarSpec, TensorSpec

    capacity, active, width, kind = PLAN_CASES[case]
    pattern = [0] * 7 + [2] * 2 + [0] * 6 + [-1, 3]
    requests = torch.tensor((pattern * ((capacity + 16) // 17))[:capacity], dtype=torch.int32)
    if kind == "same":
        requests.zero_()
    values = [513, -7, 0, 511, 512, 769, 4096]
    lengths = torch.tensor((values * ((capacity + 6) // 7))[:capacity], dtype=torch.int32)
    requests[active:] = 9981
    lengths[active:] = 2147483647
    inputs = torch.full((capacity + 2,), -9961, dtype=torch.int32)
    visible = torch.full_like(inputs, -9962)
    inputs[1:-1] = requests
    visible[1:-1] = lengths
    columns = capacity + 3
    initial = (torch.arange(5 * columns, dtype=torch.int32) - 1000003).reshape(5, columns)
    return [
        TensorSpec("request_ids", [capacity + 2], torch.int32, init_value=inputs),
        TensorSpec("lengths", [capacity + 2], torch.int32, init_value=visible),
        TensorSpec("runs", [5, columns], torch.int32, init_value=initial),
        ScalarSpec("num_tokens", torch.int32, active, compile_runtime=True),
        ScalarSpec("width", torch.int64, width, compile_runtime=True),
        ScalarSpec("skip_short", torch.int32, skip_short, compile_runtime=True),
    ]


def build_pool_plan_specs(case="mixed"):
    """构造负数、零、容量边界、超容量计数及非有效行和输出保护区。"""
    import torch
    from golden import ScalarSpec, TensorSpec

    rows, active, capacity = POOL_PLAN_CASES[case]
    pattern = [capacity + 9, -4, 0, 1, capacity - 1, capacity, 2147483647]
    values = torch.tensor((pattern * ((rows + 6) // 7))[:rows], dtype=torch.int32)
    values[active:] = 2147483647
    counts = torch.full((rows + 2,), -9951, dtype=torch.int32)
    counts[1:-1] = values
    columns = rows + 3
    initial = (torch.arange(5 * columns, dtype=torch.int32) - 1000033).reshape(5, columns)
    return [
        TensorSpec("pool_counts", [rows + 2], torch.int32, init_value=counts),
        TensorSpec("runs", [5, columns], torch.int32, init_value=initial),
        ScalarSpec("num_tokens", torch.int32, active, compile_runtime=True),
        ScalarSpec("capacity", torch.int64, capacity, compile_runtime=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="校验 DSL Indexer 请求分组与运行区间")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--mode", choices=["request", "pool"], default="request")
    case_names = list(dict.fromkeys([*PLAN_CASES, *POOL_PLAN_CASES]))
    parser.add_argument("--case", choices=[*case_names, "all"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    available = PLAN_CASES if args.mode == "request" else POOL_PLAN_CASES
    if args.case != "all" and args.case not in available:
        parser.error("所选模式不包含这个用例")
    cases = available if args.case == "all" else [args.case]
    failures = []
    for case in cases:
        for skip_short in ((0, 1) if args.mode == "request" else (0,)):
            if args.mode == "request":
                fn, reference = validate_plan, golden_plan
                specs = build_plan_specs(case, skip_short)
                print(f"[请求分组规划] {case}，短序列跳过={skip_short}", flush=True)
            else:
                fn, reference = validate_pool_plan, golden_pool_plan
                specs = build_pool_plan_specs(case)
                print(f"[候选池分组规划] {case}", flush=True)
            result = run(
                fn=fn,
                specs=specs,
                golden_fn=reference,
                config={"backend_type": "a5", "execution_mode": False, "device_id": args.device},
                compare_fn={"runs": ratio_allclose(atol=0, rtol=0, max_error_ratio=0)},
                compile_only=args.compile_only,
                save_data=args.save_data,
            )
            if not result.passed:
                failures.append(f"{case}/{skip_short}: {result.error}")
    if failures:
        raise SystemExit("; ".join(failures))
