# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""按连续请求和最多六行 Query 生成 Indexer 运行区间及行映射。"""

import cannbotdsl as cb
import pypto.language as pl

from config import FLASH as M


# 模型契约
INDEX_TOPK = M.index_topk

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


@pl.jit.inline(auto_scope=False)
def plan_index_runs(
    request_ids: pl.Tensor, lengths: pl.Tensor, runs: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], width: pl.Scalar[pl.INDEX], skip_short: pl.Scalar[pl.INT32],
):
    """提交一个本地 AIV 任务，返回供 MIX 打分消费的完成依赖。"""
    with pl.spmd(1, name_hint="dsl_idx_plan_req") as ready:
        plan_runs(request_ids, lengths, runs, num_tokens, width, skip_short)
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


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="校验 DSL Indexer 请求分组与运行区间")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--case", choices=[*PLAN_CASES, "all"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    cases = PLAN_CASES if args.case == "all" else [args.case]
    failures = []
    for case in cases:
        for skip_short in (0, 1):
            print(f"[分组规划] {case}，短序列跳过={skip_short}", flush=True)
            result = run(
                fn=validate_plan,
                specs=build_plan_specs(case, skip_short),
                golden_fn=golden_plan,
                config={"backend_type": "a5", "execution_mode": False, "device_id": args.device},
                compare_fn={"runs": ratio_allclose(atol=0, rtol=0, max_error_ratio=0)},
                compile_only=args.compile_only,
                save_data=args.save_data,
            )
            if not result.passed:
                failures.append(f"{case}/{skip_short}: {result.error}")
    if failures:
        raise SystemExit("; ".join(failures))
