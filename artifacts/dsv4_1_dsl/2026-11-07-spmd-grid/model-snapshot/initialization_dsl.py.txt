# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""初始化 BF16 填充行、未使用输出和 MoE 路由缓冲区。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
import pypto.language as pl

from config import FLASH as M


# 模型契约
D = M.hidden_size
TOPK = M.num_experts_per_tok
MIN_FP32 = -3.4028234663852886e38

# 分块参数
ZERO_VECTOR_TILE = 128
SCORE_COLUMN_TILE = 64
OUTPUT_ROW_TILE = 8

# 校验存储维度
ROWS_DYN = pl.dynamic("INIT_ROWS")
COLUMNS_DYN = pl.dynamic("INIT_COLUMNS")
SCORE_ROWS_DYN = pl.dynamic("INIT_SCORE_ROWS")
SCORE_COLUMNS_DYN = pl.dynamic("INIT_SCORE_COLUMNS")


@cb.kernel
def zero_bf16_rows(output: cb.Tensor):
    """按 block 分配给定行视图，将 D 列写为 BF16 正零。"""
    row = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, D)))
    bits = row.view(dtype=cb.dtypes.uint16)
    with cb.vf():
        mask16 = rr.full_mask(elem_bits=16)
        zero = rr.vdups(0, cb.dtypes.uint16, mask=mask16)
        for col in range(0, D, ZERO_VECTOR_TILE):
            rr.vstore(bits, col, zero, mask16)
    for index in cb.range(cb.get_block_idx(), output.shape[0], cb.get_block_num()):
        cb.mem_copy(cb.tile_slice(output, (1, D), (index, 0)), row)


@cb.kernel
def initialize_routes(
    indices: cb.Tensor, weights: cb.Tensor, biased: cb.Tensor,
    num_tokens: cb.ScalarValue, experts: cb.ScalarValue,
):
    """清零非有效路由，将全部分数行的补齐专家列写为最小有限 FP32。"""
    active = cb.dtypes.int64(num_tokens)
    if active < 0:
        active = cb.dtypes.int64(0)
    if active > indices.shape[0]:
        active = cb.dtypes.int64(indices.shape[0])
    for row in cb.range(active, indices.shape[0]):
        for column in cb.range(indices.shape[1]):
            indices[row, column] = cb.dtypes.int32(0)
            weights[row, column] = cb.dtypes.float32(0)
    pad = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, SCORE_COLUMN_TILE)))
    with cb.vf():
        full = rr.full_mask()
        value = rr.vdups(MIN_FP32, cb.dtypes.float32, mask=full)
        rr.vstore(pad, 0, value, full)
    for row in cb.range(biased.shape[0]):
        for column in cb.range(experts, biased.shape[1], SCORE_COLUMN_TILE):
            live = min(SCORE_COLUMN_TILE, biased.shape[1] - column)
            cb.mem_copy(biased[row:row + 1, column:column + live], pad[None, 0:live])


@pl.jit.inline(auto_scope=False)
def zero_bf16_padding(x: pl.Tensor, valid_rows: pl.Scalar[pl.INT32]):
    """只将无效尾部视图提交为多核清零任务。"""
    tokens = pl.tensor.dim(x, 0)
    first = pl.min(tokens, pl.max(0, pl.cast(valid_rows, pl.INDEX)))
    if first < tokens:
        target = x[first:tokens, 0:D]
        workers = pl.min(tokens - first, pl.system.available_aiv_count())
        with pl.spmd(workers, name_hint="dsl_decode_sp_zero_padding"):
            zero_bf16_rows(target)
    return x


@pl.jit.inline(auto_scope=False)
def zero_attention_output(output: pl.Tensor):
    """以单次 SPMD 提交未使用输出的清零。"""
    rows = pl.tensor.dim(output, 0)
    if rows > 0:
        target = output[:, 0:D]
        tiles = (rows + OUTPUT_ROW_TILE - 1) // OUTPUT_ROW_TILE
        workers = pl.min(tiles, pl.system.available_aiv_count())
        with pl.spmd(workers, name_hint="dsl_unused_attention_output"):
            zero_bf16_rows(target)
    return output


@pl.jit.inline(auto_scope=False)
def initialize_gate(
    indices: pl.Tensor, weights: pl.Tensor, biased: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], experts: pl.Scalar[pl.INDEX],
):
    """用单个 AIV 任务初始化路由结果及专家补齐列。"""
    with pl.spmd(1, name_hint="dsl_gate_pre_route"):
        initialize_routes(indices, weights, biased, num_tokens, experts)
    return indices, weights, biased


@pl.jit
def validate_padding(
    x: pl.InOut[pl.Tensor[[ROWS_DYN, COLUMNS_DYN], pl.BF16]],
    valid_rows: pl.Scalar[pl.INT32],
):
    """校验有效行和额外列保护区保持原值。"""
    x.bind_dynamic(0, ROWS_DYN)
    x.bind_dynamic(1, COLUMNS_DYN)
    zero_bf16_padding(x, valid_rows)
    return x


@pl.jit
def validate_output(x: pl.InOut[pl.Tensor[[ROWS_DYN, COLUMNS_DYN], pl.BF16]]):
    """校验八行分块的尾部和额外列保护区。"""
    x.bind_dynamic(0, ROWS_DYN)
    x.bind_dynamic(1, COLUMNS_DYN)
    zero_attention_output(x)
    return x


@pl.jit
def validate_gate_init(
    indices: pl.InOut[pl.Tensor[[ROWS_DYN, TOPK], pl.INT32]],
    weights: pl.InOut[pl.Tensor[[ROWS_DYN, TOPK], pl.FP32]],
    biased: pl.InOut[pl.Tensor[[SCORE_ROWS_DYN, SCORE_COLUMNS_DYN], pl.FP32]],
    num_tokens: pl.Scalar[pl.INT32],
    experts: pl.Scalar[pl.INDEX],
):
    """校验有效路由、实际专家列及需要初始化的区域。"""
    indices.bind_dynamic(0, ROWS_DYN)
    weights.bind_dynamic(0, ROWS_DYN)
    biased.bind_dynamic(0, SCORE_ROWS_DYN)
    biased.bind_dynamic(1, SCORE_COLUMNS_DYN)
    initialize_gate(indices, weights, biased, num_tokens, experts)
    return indices, weights, biased


def golden_padding(tensors):
    """仅清零有效行边界之后的 D 列。"""
    start = max(0, int(tensors["valid_rows"]))
    tensors["x"][start:, :D] = 0


def golden_output(tensors):
    """清零全部行的 D 列，保留额外列。"""
    tensors["x"][:, :D] = 0


def golden_gate_init(tensors):
    """使用相同的截断边界和有限 FP32 填充值生成参考。"""
    active = max(0, min(int(tensors["num_tokens"]), tensors["indices"].shape[0]))
    tensors["indices"][active:] = 0
    tensors["weights"][active:] = 0
    tensors["biased"][:, int(tensors["experts"]):] = MIN_FP32


# 校验参数：类型、行数、有效行数；路由类型另含分数行数、实际专家数和分数列数。
INIT_CASES = {
    "padding-zero": ("padding", 1, 0),
    "padding-partial": ("padding", 80, 72),
    "padding-full": ("padding", 17, 17),
    "padding-negative": ("padding", 7, -1),
    "padding-beyond": ("padding", 7, 9),
    "output-one": ("output", 1, 0),
    "output-tail": ("output", 13, 0),
    "output-full": ("output", 80, 0),
    "gate-partial": ("gate", 80, 71, 80, 16, 384),
    "gate-full": ("gate", 80, 80, 80, 384, 384),
    "gate-zero": ("gate", 16, 0, 16, 16, 32),
    "gate-negative": ("gate", 17, -3, 32, 24, 64),
    "gate-beyond": ("gate", 17, 25, 32, 64, 64),
}


def build_init_specs(case):
    """构造非零初值、正负零位模式和不应被修改的区域。"""
    import torch
    from golden import ScalarSpec, TensorSpec

    kind, rows, active, *shape = INIT_CASES[case]
    if kind != "gate":
        columns = D + 16
        values = (torch.arange(rows * columns).reshape(rows, columns) % 127 - 63).bfloat16()
        values[:, 3] = -0.0
        specs = [TensorSpec("x", [rows, columns], torch.bfloat16, init_value=values)]
        if kind == "padding":
            specs.append(ScalarSpec("valid_rows", torch.int32, active, compile_runtime=True))
        return specs
    score_rows, experts, columns = shape
    indices = torch.arange(rows * TOPK, dtype=torch.int32).reshape(rows, TOPK) + 17
    weights = (indices.float() / 8).contiguous()
    weights[:, 0] = -0.0
    biased = torch.arange(score_rows * columns, dtype=torch.float32).reshape(score_rows, columns) / 16
    biased[:, 0] = -0.0
    return [
        TensorSpec("indices", [rows, TOPK], torch.int32, init_value=indices),
        TensorSpec("weights", [rows, TOPK], torch.float32, init_value=weights),
        TensorSpec("biased", [score_rows, columns], torch.float32, init_value=biased),
        ScalarSpec("num_tokens", torch.int32, active, compile_runtime=True),
        ScalarSpec("experts", torch.int64, experts, compile_runtime=True),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run
    from index_gather_dsl import compare_bytes

    parser = argparse.ArgumentParser(description="校验三类本地初始化 DSL 任务")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--case", choices=[*INIT_CASES, "all"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--save-data", action="store_true")
    args = parser.parse_args()
    functions = {"padding": (validate_padding, golden_padding), "output": (validate_output, golden_output),
                 "gate": (validate_gate_init, golden_gate_init)}
    failures = []
    for case in (INIT_CASES if args.case == "all" else [args.case]):
        kind = INIT_CASES[case][0]
        fn, reference = functions[kind]
        outputs = ("indices", "weights", "biased") if kind == "gate" else ("x",)
        print("[本地初始化] " + case, flush=True)
        result = run(
            fn=fn,
            specs=build_init_specs(case),
            golden_fn=reference,
            config={"backend_type": "a5", "execution_mode": False, "device_id": args.device},
            compare_fn={name: compare_bytes for name in outputs},
            compile_only=args.compile_only,
            save_data=args.save_data,
        )
        if not result.passed:
            failures.append(f"{case}: {result.error}")
    if failures:
        raise SystemExit("; ".join(failures))
