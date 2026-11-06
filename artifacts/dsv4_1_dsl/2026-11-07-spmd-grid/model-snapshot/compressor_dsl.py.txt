# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 实现 C1A/C2A 压缩投影，并以 FP32 累加。"""

import cannbotdsl as cb
from cannbotdsl.lang.jit import jit
import pypto.language as pl

from bf16_projection_dsl import project_bf16_nz
from config import FLASH as M


# 模型配置
D = M.hidden_size
HEAD_DIM = M.head_dim

# 分块参数
ROW_TILE = 80
COLUMN_TILE = 64
REDUCTION_TILE = 256
TASK_TILE = 32
C1A_ROW_TILE = 48
C1A_COLUMN_TILE = 64
C1A_REDUCTION_TILE = 512


@jit
def _project_compressor_tile(
    source: cb.Tensor, weight: cb.Tensor, output: cb.Tensor,
    column: cb.ScalarValue, active_rows: cb.ScalarValue,
):
    """连续执行 K256 BF16 Cube 累加，投影一个 M80/N64 分块。"""
    project_bf16_nz(source, weight, output, column, active_rows, ROW_TILE, COLUMN_TILE, REDUCTION_TILE)


@cb.kernel
def project_c1a_compressor(
    source: cb.Tensor, weight: cb.Tensor, output: cb.Tensor, num_tokens: cb.ScalarValue,
):
    """每个 worker 计算一个 M48/N64 分块，按 K512 累加后舍入为 BF16。"""
    block = cb.get_block_idx()
    row_tile = block // (HEAD_DIM // C1A_COLUMN_TILE)
    column = block % (HEAD_DIM // C1A_COLUMN_TILE)
    row = row_tile * C1A_ROW_TILE
    active = num_tokens - row
    if active > C1A_ROW_TILE:
        active = C1A_ROW_TILE
    if active > 0:
        tile = cb.tile_slice(source, (C1A_ROW_TILE, D), (row_tile, 0))
        projected = cb.tile_slice(output, (C1A_ROW_TILE, C1A_COLUMN_TILE), (row_tile, column))
        project_bf16_nz(
            tile, weight, projected, column, active,
            C1A_ROW_TILE, C1A_COLUMN_TILE, C1A_REDUCTION_TILE,
        )


@pl.jit.inline(auto_scope=False)
def c1a_compressor_project(
    source: pl.Tensor, weight: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ], output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], ready: pl.Scalar[pl.TASK_ID],
):
    """在输入依赖完成后，使用单个 AIC SPMD 任务投影所有有效 C1A 行。"""
    storage = pl.reshape(weight, [D * HEAD_DIM])
    raw_weight = pl.reshape(storage, [HEAD_DIM // 16, D * 16])
    blocks = (num_tokens + C1A_ROW_TILE - 1) // C1A_ROW_TILE * (HEAD_DIM // C1A_COLUMN_TILE)
    workers = pl.max(blocks, 1)
    with pl.spmd(workers, name_hint="dsl_c1a_proj_cmp", deps=[ready]) as completed:
        project_c1a_compressor(source, raw_weight, output, num_tokens)
    return output, completed


@cb.kernel
def project_c2a_grid(source: cb.Tensor, kv_weight: cb.Tensor, gate_weight: cb.Tensor,
                     kv_out: cb.Tensor, score_out: cb.Tensor, num_tokens: cb.ScalarValue):
    """按行块、列块和 KV/gate 分支分配压缩投影。"""
    active = min(source.shape[0], max(num_tokens, 0))
    columns = HEAD_DIM // COLUMN_TILE
    tiles = (active + ROW_TILE - 1) // ROW_TILE * columns * 2
    for work in cb.range(cb.get_block_idx(), tiles, cb.get_block_num()):
        row = work // (columns * 2) * ROW_TILE
        column = work // 2 % columns
        live = min(ROW_TILE, active - row)
        src = source[row:row + live, None]
        begin = column * COLUMN_TILE
        if work % 2 == 0:
            dst = kv_out[row:row + live, begin:begin + COLUMN_TILE]
            _project_compressor_tile(src, kv_weight, dst, column, live)
        else:
            dst_gate = score_out[row:row + live, begin:begin + COLUMN_TILE]
            _project_compressor_tile(src, gate_weight, dst_gate, column, live)


@pl.jit.inline(auto_scope=False)
def c2a_compressor_project(
    x: pl.Tensor, wkv: pl.Tensor, wgate: pl.Tensor, kv_out: pl.Tensor, score_out: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], input_ready: pl.Scalar[pl.TASK_ID],
):
    """以一组 AIC SPMD 提交 KV 和 gate 的全部压缩投影。"""
    active = pl.min(pl.tensor.dim(x, 0), pl.max(pl.cast(num_tokens, pl.INDEX), 0))
    kv_storage = pl.reshape(wkv, [D * HEAD_DIM])
    gate_storage = pl.reshape(wgate, [D * HEAD_DIM])
    kv_weight = pl.reshape(kv_storage, [HEAD_DIM // 16, D * 16])
    gate_weight = pl.reshape(gate_storage, [HEAD_DIM // 16, D * 16])
    target_kv = kv_out[0:active, 0:HEAD_DIM]
    target_gate = score_out[0:active, 0:HEAD_DIM]
    source = x[0:active, :]
    tiles = (active + ROW_TILE - 1) // ROW_TILE * (HEAD_DIM // COLUMN_TILE) * 2
    workers = pl.max(1, pl.min(TASK_TILE, tiles))
    with pl.spmd(workers, name_hint="dsl_proj_cmp", deps=[input_ready]) as done:
        project_c2a_grid(source, kv_weight, gate_weight, target_kv, target_gate, num_tokens)
    return kv_out, score_out, done


@pl.jit
def validate_projection(
    source: pl.Tensor,
    wkv: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    wgate: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    kv_out: pl.InOut[pl.Tensor], score_out: pl.InOut[pl.Tensor], num_tokens: pl.Scalar[pl.INT32],
):
    """使用非有效区域哨兵值和一行保护区校验生产路径调度器。"""
    rows = pl.tensor.dim(source, 0)
    kv = pl.slice(kv_out, [rows, HEAD_DIM], [0, 0])
    gate = pl.slice(score_out, [rows, HEAD_DIM], [0, 0])
    ready = pl.system.task_dummy(deps=[])
    c2a_compressor_project(source, wkv, wgate, kv, gate, num_tokens, ready)
    return kv_out, score_out


@pl.jit
def validate_c1a_projection(
    source: pl.Tensor, weight: pl.Tensor[[D, HEAD_DIM], pl.BF16, pl.NZ],
    output: pl.InOut[pl.Tensor], num_tokens: pl.Scalar[pl.INT32],
):
    """校验 C1A 生产投影路径，并保留一行额外输出保护区。"""
    rows = pl.tensor.dim(source, 0)
    target = pl.slice(output, [rows, HEAD_DIM], [0, 0])
    ready = pl.system.task_dummy(deps=[])
    c1a_compressor_project(source, weight, target, num_tokens, ready)
    return output


def golden_projection(tensors):
    """将 BF16 操作数投影为 FP32，中间不对输出舍入。"""
    import torch
    from quant import unpack_bf16_nz

    rows = tensors["source"].shape[0]
    active = max(0, min(rows, int(tensors["num_tokens"])))
    source = tensors["source"][:active].float()
    for weight, output in (("wkv", "kv_out"), ("wgate", "score_out")):
        values = torch.matmul(source, unpack_bf16_nz(tensors[weight]).float())
        tensors[output][:active].copy_(values)


def golden_c1a_projection(tensors):
    """按 K512 依次累加 FP32 部分和，最终一次舍入为 BF16。"""
    import torch
    from quant import unpack_bf16_nz

    active = int(tensors["num_tokens"])
    source = tensors["source"][:active].float()
    weight = unpack_bf16_nz(tensors["weight"]).float()
    value = torch.zeros((active, HEAD_DIM), dtype=torch.float32)
    for k in range(0, D, C1A_REDUCTION_TILE):
        value = value + source[:, k:k + C1A_REDUCTION_TILE] @ weight[k:k + C1A_REDUCTION_TILE]
    tensors["output"][:active].copy_(value.bfloat16())


def build_projection_specs(rows=80, active=72, layout_case=False):
    """覆盖 NZ 坐标、行尾部、独立投影和带异常值的非有效输入。"""
    import torch
    from golden import ScalarSpec, TensorSpec
    from quant import pack_bf16_nz

    generator = torch.Generator().manual_seed(2301)
    if layout_case:
        source = ((torch.arange(rows).reshape(-1, 1) * 3 + torch.arange(D)) % 16 - 8).float() / 8
        reduction = torch.arange(D).reshape(-1, 1)
        columns = torch.arange(HEAD_DIM)
        wkv = ((reduction * 3 + columns * 5 + reduction // 16 + columns // 16) % 8 - 4).float() / 16
        wgate = ((reduction * 5 + columns * 3 + reduction // 256 + columns // 64) % 8 - 4).float() / 8
    else:
        source = torch.randn(rows, D, generator=generator)
        wkv = torch.randn(D, HEAD_DIM, generator=generator) / D**0.5
        wgate = torch.randn(D, HEAD_DIM, generator=generator) / D**0.5
    source = source.bfloat16()
    source[max(0, min(rows, active)):] = float("nan")
    return [
        TensorSpec("source", [rows, D], torch.bfloat16, init_value=source),
        TensorSpec("wkv", [D, HEAD_DIM], torch.bfloat16, init_value=pack_bf16_nz(wkv.bfloat16())),
        TensorSpec("wgate", [D, HEAD_DIM], torch.bfloat16, init_value=pack_bf16_nz(wgate.bfloat16())),
        TensorSpec("kv_out", [rows + 1, HEAD_DIM], torch.float32, init_value=-37.0),
        TensorSpec("score_out", [rows + 1, HEAD_DIM], torch.float32, init_value=53.0),
        ScalarSpec("num_tokens", torch.int32, active),
    ]


def build_c1a_projection_specs(rows=80, active=72, kind="random"):
    """构造 NZ 分块坐标、BF16 舍入中点和非有效输入 NaN 哨兵。"""
    import torch
    from golden import ScalarSpec, TensorSpec
    from quant import pack_bf16_nz

    generator = torch.Generator().manual_seed(3511)
    if kind == "random":
        source = torch.randn(rows, D, generator=generator).bfloat16()
        weight = (torch.randn(D, HEAD_DIM, generator=generator) / D**0.5).bfloat16()
    elif kind == "ties":
        source = torch.zeros(rows, D, dtype=torch.bfloat16)
        source[:, 0] = 1.0
        source[:, C1A_REDUCTION_TILE] = 1.0
        weight = torch.zeros(D, HEAD_DIM, dtype=torch.bfloat16)
        column = torch.arange(HEAD_DIM)
        sign = torch.where(column % 2 == 0, 1.0, -1.0)
        weight[0] = ((1.0 + (column // 2 % 4) / 128) * sign).bfloat16()
        weight[C1A_REDUCTION_TILE] = (sign / 256).bfloat16()
    else:
        row = torch.arange(rows).reshape(-1, 1)
        k = torch.arange(D)
        source = (((row * 3 + k) % 16 - 8).float() / 8).bfloat16()
        k = k.reshape(-1, 1)
        column = torch.arange(HEAD_DIM)
        weight = (((k * 3 + column * 5 + k // 512 + column // 64) % 8 - 4).float() / 16).bfloat16()
    source[active:] = float("nan")
    return [
        TensorSpec("source", [rows, D], torch.bfloat16, init_value=source),
        TensorSpec("weight", [D, HEAD_DIM], torch.bfloat16, init_value=pack_bf16_nz(weight)),
        TensorSpec("output", [rows + 1, HEAD_DIM], torch.bfloat16, init_value=-37.0),
        ScalarSpec("num_tokens", torch.int32, active),
    ]


def projection_comparator(tolerance, rtol=None):
    """限制有效输出的误差，并要求非有效输出和保护行完全相同。"""
    from golden import ratio_allclose

    compare_active = ratio_allclose(atol=tolerance, rtol=tolerance if rtol is None else rtol, max_error_ratio=0.0)
    compare_tail = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)

    def compare(actual, expected, *, inputs, **kwargs):
        rows = inputs["source"].shape[0]
        active = max(0, min(rows, int(inputs["num_tokens"])))
        ok, detail = compare_active(actual[:active], expected[:active], inputs=inputs, **kwargs)
        if not ok:
            return ok, detail
        return compare_tail(actual[active:], expected[active:], inputs=inputs, **kwargs)

    return compare


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="校验 A5 上的 C1A/C2A BF16 压缩投影。")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--op", choices=("c1a", "c2a", "all"), default="c2a")
    args = parser.parse_args()
    c2a_cases = ((17, 15, True), (17, 0, True), (80, 72, False), (81, 81, True), (321, 321, True))
    for rows, active, layout_case in c2a_cases if args.op in ("c2a", "all") else ():
        print(f"[C2A_PROJECTION_DSL] rows={rows} active={active} layout_case={layout_case}", flush=True)
        tolerance = 0.0 if layout_case else 1e-4
        compare = projection_comparator(tolerance)
        result = run(
            fn=validate_projection,
            specs=build_projection_specs(rows, active, layout_case),
            golden_fn=golden_projection,
            compare_fn={"kv_out": compare, "score_out": compare},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        print(f"[产物] {result.work_dir}", flush=True)
    c1a_cases = ((17, 15, "layout"), (17, 0, "layout"), (1, 1, "ties"),
                 (49, 49, "layout"), (80, 72, "random"), (193, 193, "layout"))
    for rows, active, kind in c1a_cases if args.op in ("c1a", "all") else ():
        print(f"[C1A_PROJECTION_DSL] {rows=} {active=} {kind=}", flush=True)
        compare = projection_comparator(1e-4, 1 / 128) if kind == "random" else projection_comparator(0.0)
        result = run(
            fn=validate_c1a_projection,
            specs=build_c1a_projection_specs(rows, active, kind),
            golden_fn=golden_c1a_projection,
            compare_fn={"output": compare},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        print(f"[产物] {result.work_dir}", flush=True)
