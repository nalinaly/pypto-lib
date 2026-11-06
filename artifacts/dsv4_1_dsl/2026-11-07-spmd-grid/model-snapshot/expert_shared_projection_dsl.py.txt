# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 对 ND 权重执行共享专家 MXFP8 投影。"""

import cannbotdsl as cb
from cannbotdsl.lang.jit import jit
from cannbotdsl.ops.memcpy import copy_options
from cannbotdsl.ring_buffer import RingBuffer
import pypto.language as pl

from config import FLASH as M


# 动态形状变量。
PADDED_ROWS = pl.dynamic("SHARED_PROJ_PADDED_ROWS")
OUTPUT_ROWS = pl.dynamic("SHARED_PROJ_OUTPUT_ROWS")
SCALE_COLUMNS = pl.dynamic("SHARED_PROJ_SCALE_COLUMNS")

# 模型配置。
D = M.hidden_size
MOE_INTER = M.moe_intermediate_size
MX_GROUP = 32

# 分块参数。
ROW_TILE = 16
COLUMN_TILE = 128
REDUCTION_TILE = 256


@jit
def _project_tile(
    payload: cb.Tensor,
    weight: cb.Tensor,
    activation_scale: cb.Tensor,
    weight_scale: cb.Tensor,
    output: cb.Tensor,
    column: cb.ScalarValue,
    valid_rows: cb.ScalarValue,
):
    """连续执行 K256 MX 累加，投影十六行带填充的数据。"""
    reduction = payload.shape[1]
    columns = weight.shape[1]
    a_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.float8_e4m3fn, cb.make_layout((ROW_TILE, REDUCTION_TILE)), depth=1)
    b_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.float8_e4m3fn, cb.make_layout((REDUCTION_TILE, COLUMN_TILE)), depth=1)
    sa_layout = cb.make_layout((ROW_TILE // 16, REDUCTION_TILE // 64, 16, 2))
    sb_layout = cb.make_layout((COLUMN_TILE // 16, REDUCTION_TILE // 64, 16, 2))
    sa_l1 = RingBuffer(
        cb.MemLoc.L1, cb.dtypes.float8_e8m0, cb.make_layout((REDUCTION_TILE // MX_GROUP, ROW_TILE)),
        depth=1, physical_layout=sa_layout, data_format="nz",
    )
    sb_l1 = RingBuffer(
        cb.MemLoc.L1, cb.dtypes.float8_e8m0, cb.make_layout((REDUCTION_TILE // MX_GROUP, COLUMN_TILE)),
        depth=1, physical_layout=sb_layout, data_format="nz",
    )
    a_l0 = RingBuffer(cb.MemLoc.L0A, cb.dtypes.float8_e4m3fn, cb.make_layout((ROW_TILE, REDUCTION_TILE)), depth=1)
    b_l0 = RingBuffer(cb.MemLoc.L0B, cb.dtypes.float8_e4m3fn, cb.make_layout((COLUMN_TILE, REDUCTION_TILE)), depth=1)
    c_l0 = RingBuffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((ROW_TILE, COLUMN_TILE)), depth=1)
    sa_bytes = activation_scale.view(dtype=cb.dtypes.uint8).view(ROW_TILE // 16, reduction // 2)
    sb_bytes = weight_scale.view(dtype=cb.dtypes.uint8).view(columns // 16, reduction // 2)
    accumulator = c_l0.next_write()
    for k in cb.range(reduction // REDUCTION_TILE):
        a = a_l1.next_write()
        b = b_l1.next_write()
        cb.mem_copy(a, cb.tile_slice(payload, (ROW_TILE, REDUCTION_TILE), (0, k)))
        cb.mem_copy(b, cb.tile_slice(weight, (REDUCTION_TILE, COLUMN_TILE), (k, column)))
        # MX 缩放字节布局：[outer/16, K/64, 16, 2]。
        sa = sa_l1.next_write()
        sb = sb_l1.next_write()
        sa_raw_layout = cb.make_layout((ROW_TILE // 16, REDUCTION_TILE // 2))
        sb_raw_layout = cb.make_layout((COLUMN_TILE // 16, REDUCTION_TILE // 2))
        sa_raw = cb.from_tensor(sa, dtype=cb.dtypes.uint8, layout=sa_raw_layout, data_format="nd")
        sb_raw = cb.from_tensor(sb, dtype=cb.dtypes.uint8, layout=sb_raw_layout, data_format="nd")
        cb.mem_copy(sa_raw, cb.tile_slice(sa_bytes, (ROW_TILE // 16, REDUCTION_TILE // 2), (0, k)))
        cb.mem_copy(sb_raw, cb.tile_slice(sb_bytes, (COLUMN_TILE // 16, REDUCTION_TILE // 2), (column, k)))
        a_slot = a_l0.next_write()
        b_slot = b_l0.next_write()
        cb.mem_copy(a_slot, a_l1.next_read(), options=copy_options(mx_scale=sa_l1.next_read().permute(1, 0)))
        cb.mem_copy(b_slot, b_l1.next_read().permute(1, 0), options=copy_options(mx_scale=sb_l1.next_read()))
        cb.matmul(accumulator, a_l0.next_read(), b_l0.next_read(), init=k == 0, mx_mode=True)
    result = c_l0.next_read()
    valid_layout = cb.make_layout((valid_rows, COLUMN_TILE))
    valid = cb.from_tensor(result, layout=valid_layout, physical_layout=result.physical_layout)
    cb.mem_copy(output[0:valid_rows, None], valid)


@cb.kernel
def project_shared_grid(payload: cb.Tensor, weight: cb.Tensor, activation_scale: cb.Tensor,
                        weight_scale: cb.Tensor, output: cb.Tensor):
    """按 M16/N128 分配共享专家投影，尾块仅写有效行。"""
    columns = weight.shape[1] // COLUMN_TILE
    tiles = (output.shape[0] + ROW_TILE - 1) // ROW_TILE * columns
    for work in cb.range(cb.get_block_idx(), tiles, cb.get_block_num()):
        row = work // columns * ROW_TILE
        column = work % columns
        live = min(ROW_TILE, output.shape[0] - row)
        source = payload[row:row + ROW_TILE, None]
        offset = row * (payload.shape[1] // MX_GROUP)
        factors = activation_scale[0:1, offset:offset + ROW_TILE * (payload.shape[1] // MX_GROUP)]
        begin = column * COLUMN_TILE
        target = output[row:row + live, begin:begin + COLUMN_TILE]
        _project_tile(source, weight, factors, weight_scale, target, column, live)


@pl.jit.inline(auto_scope=False)
def project_shared(payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor,
                   weight_scale: pl.Tensor, output: pl.Tensor):
    """以一次 SPMD 调度共享专家的全部行列分块。"""
    rows = pl.tensor.dim(output, 0)
    reduction = pl.tensor.dim(payload, 1)
    columns = pl.tensor.dim(weight, 1)
    weight_factors = pl.tensor.view(weight_scale, [1, reduction // MX_GROUP * columns], layout=pl.ND)
    tiles = (rows + ROW_TILE - 1) // ROW_TILE * (columns // COLUMN_TILE)
    workers = pl.max(1, pl.min(tiles, pl.system.available_cluster_count()))
    with pl.spmd(workers, name_hint="dsl_shared_projection", sync_start=True):
        project_shared_grid(payload, weight, activation_scale, weight_factors, output)
    return output


@pl.jit
def validate_gate_projection(
    payload: pl.Tensor[[PADDED_ROWS, D], pl.FP8E4M3FN],
    weight: pl.Tensor[[D, MOE_INTER], pl.FP8E4M3FN],
    activation_scale: pl.Tensor[[1, SCALE_COLUMNS], pl.FP8E8M0],
    weight_scale: pl.Tensor[[D // MX_GROUP, MOE_INTER], pl.FP8E8M0, pl.MX_B_NN],
    output: pl.Out[pl.Tensor[[OUTPUT_ROWS, MOE_INTER], pl.FP32]],
):
    project_shared(payload, weight, activation_scale, weight_scale, output)
    return output


@pl.jit
def validate_down_projection(
    payload: pl.Tensor[[PADDED_ROWS, MOE_INTER], pl.FP8E4M3FN],
    weight: pl.Tensor[[MOE_INTER, D], pl.FP8E4M3FN],
    activation_scale: pl.Tensor[[1, SCALE_COLUMNS], pl.FP8E8M0],
    weight_scale: pl.Tensor[[MOE_INTER // MX_GROUP, D], pl.FP8E8M0, pl.MX_B_NN],
    output: pl.Out[pl.Tensor[[OUTPUT_ROWS, D], pl.BF16]],
):
    project_shared(payload, weight, activation_scale, weight_scale, output)
    return output


def golden_projection(tensors):
    """解量化打包的 MX 缩放值，并计算共享专家投影。"""
    from gemm import golden_gemm_mxfp8_to_fp32_dequantized
    from quant import unpack_mx_scale_codes

    rows = tensors["output"].shape[0]
    padded_rows, reduction = tensors["payload"].shape
    packed_scale = tensors["activation_scale"].reshape(padded_rows, reduction // MX_GROUP)
    scale_a = unpack_mx_scale_codes(packed_scale, side="a")
    scale_b = unpack_mx_scale_codes(tensors["weight_scale"], side="b")
    result = golden_gemm_mxfp8_to_fp32_dequantized(tensors["payload"], scale_a, tensors["weight"], scale_b)
    tensors["output"][:] = result[:rows].to(tensors["output"].dtype)


def projection_specs(rows, stage):
    """构造非均匀 E8M0 缩放值和普通 KN 权重数据。"""
    import torch
    from golden import TensorSpec
    from quant import generate_mxfp8_weight_kn, pack_mx_a_scale, quantize_mxfp8_upward_e8m0

    reduction, columns = (D, MOE_INTER) if stage == "gate" else (MOE_INTER, D)
    output_dtype = torch.float32 if stage == "gate" else torch.bfloat16
    padded_rows = (rows + ROW_TILE - 1) // ROW_TILE * ROW_TILE
    generator = torch.Generator().manual_seed(41)
    source = torch.randn(padded_rows, reduction, generator=generator, dtype=torch.float32)
    gains = torch.linspace(0.25, 4.0, reduction // MX_GROUP).repeat_interleave(MX_GROUP)
    source = (source * gains).to(torch.bfloat16)
    payload, codes = quantize_mxfp8_upward_e8m0(source)
    source_factors = pack_mx_a_scale(codes).reshape(1, -1).view(torch.float8_e8m0fnu)
    weight, weight_factors = generate_mxfp8_weight_kn(columns, reduction, 0.017, chan_cv=0.5, seed=29)
    weight_factors = weight_factors.contiguous().view(torch.float8_e8m0fnu)
    return [
        TensorSpec("payload", [padded_rows, reduction], torch.float8_e4m3fn, init_value=payload),
        TensorSpec("weight", [reduction, columns], torch.float8_e4m3fn, init_value=weight),
        TensorSpec("activation_scale", [1, padded_rows * (reduction // MX_GROUP)], torch.float8_e8m0fnu, init_value=source_factors),
        TensorSpec("weight_scale", [reduction // MX_GROUP, columns], torch.float8_e8m0fnu, init_value=weight_factors),
        TensorSpec("output", [rows, columns], output_dtype),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_reldiff, run

    parser = argparse.ArgumentParser(description="Validate shared-expert DSL MXFP8 projections on A5.")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=17)
    parser.add_argument("--stage", choices=["gate", "down", "all"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.rows < 1:
        parser.error("--rows must be positive")
    for stage in ("gate", "down"):
        if args.stage not in (stage, "all"):
            continue
        print(f"[SHARED_PROJECTION] {stage} rows={args.rows}", flush=True)
        result = run(
            fn=validate_gate_projection if stage == "gate" else validate_down_projection,
            specs=projection_specs(args.rows, stage),
            golden_fn=golden_projection,
            compare_fn={"output": ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
