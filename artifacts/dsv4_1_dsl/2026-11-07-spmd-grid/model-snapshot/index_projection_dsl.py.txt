# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 将压缩 latent 投影为 BF16 Indexer Key。"""

import cannbotdsl as cb
from cannbotdsl.lang.jit import jit
import pypto.language as pl

from bf16_projection_dsl import project_bf16_nz
from config import FLASH as M


# 模型配置
HEAD_DIM = M.head_dim
INDEX_DIM = M.index_head_dim

# 分块参数
ROW_TILE = 16
COLUMN_TILE = 128
REDUCTION_TILE = 256


@jit
def _project_index_key_tile(source: cb.Tensor, weight: cb.Tensor, output: cb.Tensor, active_rows: cb.ScalarValue):
    """以 M16/N128/K256 执行 FP32 累加，最终舍入为 BF16。"""
    project_bf16_nz(source, weight, output, 0, active_rows, ROW_TILE, COLUMN_TILE, REDUCTION_TILE)


@cb.kernel
def project_index_key_grid(source: cb.Tensor, weight: cb.Tensor, output: cb.Tensor, num_tokens: cb.ScalarValue):
    """按 M16 分配 Indexer Key 投影，仅处理有效输出行。"""
    # 静态容量不能代替动态切片的有效行数，显式携带运行时边界。
    active = min(output.shape[0], max(num_tokens, 0))
    tiles = (active + ROW_TILE - 1) // ROW_TILE
    for tile in cb.range(cb.get_block_idx(), tiles, cb.get_block_num()):
        row = tile * ROW_TILE
        live = min(ROW_TILE, active - row)
        src = source[row:row + live, 0:HEAD_DIM]
        dst = output[row:row + live, 0:INDEX_DIM]
        _project_index_key_tile(src, weight, dst, live)


@pl.jit.inline(auto_scope=False)
def project_index_key(
    source: pl.Tensor, weight: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ], output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], ready: pl.Scalar[pl.TASK_ID], max_workers: pl.constexpr,
):
    """保留输入依赖，以一组 AIC SPMD 提交有效 Key 行块。"""
    active = pl.min(pl.tensor.dim(source, 0), pl.max(pl.cast(num_tokens, pl.INDEX), 0))
    storage = pl.reshape(weight, [HEAD_DIM * INDEX_DIM])
    raw_weight = pl.reshape(storage, [INDEX_DIM // 16, HEAD_DIM * 16])
    target = output[0:active, 0:INDEX_DIM]
    source_view = source[0:active, 0:HEAD_DIM]
    workers = pl.max(1, pl.min(max_workers, (active + ROW_TILE - 1) // ROW_TILE))
    with pl.spmd(workers, name_hint="dsl_proj_idx_k", deps=[ready]) as done:
        project_index_key_grid(source_view, raw_weight, target, num_tokens)
    return done


@pl.jit
def validate_projection(
    source: pl.Tensor, weight: pl.Tensor[[HEAD_DIM, INDEX_DIM], pl.BF16, pl.NZ],
    output: pl.InOut[pl.Tensor], num_tokens: pl.Scalar[pl.INT32], max_workers: pl.constexpr = 8,
):
    """通过生产路径校验有效行、非有效尾部和额外输出保护行。"""
    rows = pl.tensor.dim(source, 0)
    target = pl.slice(output, [rows, INDEX_DIM], [0, 0])
    ready = pl.system.task_dummy(deps=[])
    project_index_key(source, weight, target, num_tokens, ready, max_workers)
    return output


def golden_projection(tensors):
    """按 K256 分块计算 FP32 部分和，最后舍入一次为 BF16。"""
    import torch
    from quant import unpack_bf16_nz

    source = tensors["source"].float()
    weight = unpack_bf16_nz(tensors["weight"]).float()
    active = max(0, min(source.shape[0], int(tensors["num_tokens"])))
    result = torch.zeros((active, INDEX_DIM), dtype=torch.float32)
    for k in range(0, HEAD_DIM, REDUCTION_TILE):
        result = result + source[:active, k:k + REDUCTION_TILE] @ weight[k:k + REDUCTION_TILE]
    tensors["output"][:active].copy_(result.bfloat16())


def build_projection_specs(rows, active, kind="layout"):
    """构造 NZ 坐标、BF16 舍入中点及带非有效输入哨兵值的投影数据。"""
    import torch
    from golden import ScalarSpec, TensorSpec
    from quant import pack_bf16_nz

    generator = torch.Generator().manual_seed(3109)
    if kind == "random":
        source = torch.randn(rows, HEAD_DIM, generator=generator).bfloat16()
        weight = (torch.randn(HEAD_DIM, INDEX_DIM, generator=generator) / HEAD_DIM**0.5).bfloat16()
    elif kind == "ties":
        source = torch.zeros(rows, HEAD_DIM, dtype=torch.bfloat16)
        source[:, 0] = 1.0
        source[:, REDUCTION_TILE] = 1.0
        weight = torch.zeros(HEAD_DIM, INDEX_DIM, dtype=torch.bfloat16)
        col = torch.arange(INDEX_DIM)
        sign = torch.where(col % 2 == 0, 1.0, -1.0)
        weight[0] = ((1.0 + (col // 2 % 4) / 128) * sign).bfloat16()
        weight[REDUCTION_TILE] = (sign / 256).bfloat16()
    else:
        row = torch.arange(rows).reshape(-1, 1)
        k = torch.arange(HEAD_DIM)
        source = (((row * 3 + k) % 16 - 8).float() / 8).bfloat16()
        k = k.reshape(-1, 1)
        col = torch.arange(INDEX_DIM)
        weight = (((k * 3 + col * 5 + k // 16 + col // 16) % 8 - 4).float() / 16).bfloat16()
    source[max(0, min(rows, active)):] = float("nan")
    return [
        TensorSpec("source", [rows, HEAD_DIM], torch.bfloat16, init_value=source),
        TensorSpec("weight", [HEAD_DIM, INDEX_DIM], torch.bfloat16, init_value=pack_bf16_nz(weight)),
        TensorSpec("output", [rows + 1, INDEX_DIM], torch.bfloat16, init_value=-37.0),
        ScalarSpec("num_tokens", torch.int32, active),
    ]


def projection_comparator(actual, expected, *, inputs, **kwargs):
    """按 BF16 精度校验随机输入的有效结果，尾部和保护行要求完全相同。"""
    from golden import ratio_allclose

    active = max(0, min(inputs["source"].shape[0], int(inputs["num_tokens"])))
    compare_live = ratio_allclose(atol=1e-4, rtol=1.0 / 128, max_error_ratio=0.0)
    exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    passed, detail = compare_live(actual[:active], expected[:active], inputs=inputs, **kwargs)
    if not passed:
        return passed, detail
    return exact(actual[active:], expected[active:], inputs=inputs, **kwargs)


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="在 A5 上校验 DSL Indexer Key 投影。")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    cases = ((1, 1, "ties"), (17, 15, "layout"), (17, 0, "layout"), (80, 72, "random"), (145, 143, "layout"))
    for rows, active, kind in cases:
        print(f"[INDEX_KEY_DSL] 行数={rows} 有效行={active} 类型={kind}", flush=True)
        compare = projection_comparator if kind == "random" else exact
        result = run(
            fn=validate_projection,
            specs=build_projection_specs(rows, active, kind),
            golden_fn=golden_projection,
            compare_fn={"output": compare},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        print(f"[产物] {result.work_dir}", flush=True)
