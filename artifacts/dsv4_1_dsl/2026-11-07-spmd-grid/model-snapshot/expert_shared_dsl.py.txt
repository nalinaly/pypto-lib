# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 实现共享专家 SwiGLU 和缩放值向上取整的 MXFP8 量化。"""

import cannbotdsl as cb
from cannbotdsl.lang.jit import jit
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
import pypto.language as pl

from config import FLASH as M
from mxfp8_dsl import _quant_upward_row


# 模型配置
MOE_INTER = M.moe_intermediate_size
SWIGLU_LIMIT = M.swiglu_limit

# 量化契约。
MX_GROUP = 32

# 分块参数
ROW_TILE = 16
QUANT_TILE = 768
VECTOR_TILE = 64


@jit
def _shared_swiglu_quant_tile(
    gate: cb.Tensor,
    up: cb.Tensor,
    payload: cb.Tensor,
    scales: cb.Tensor,
):
    """对十六行执行激活，并发布 E4M3 数据和配对的 E8M0 缩放值。"""
    gate_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, QUANT_TILE)))
    up_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, QUANT_TILE)))
    values = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((QUANT_TILE + VECTOR_TILE,)))
    packed = Buffer(cb.MemLoc.UB, cb.dtypes.float8_e4m3fn, cb.make_layout((1, QUANT_TILE)))
    codes = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, ROW_TILE * (QUANT_TILE // MX_GROUP))))
    for row in cb.range(ROW_TILE):
        cb.mem_copy(gate_ub, cb.tile_slice(gate, (1, QUANT_TILE), (row, 0)))
        cb.mem_copy(up_ub, cb.tile_slice(up, (1, QUANT_TILE), (row, 0)))
        with cb.vf():
            full = rr.full_mask()
            one = rr.vdups(1.0, cb.dtypes.float32, mask=full)
            zero = rr.vdups(0.0, cb.dtypes.float32, mask=full)
            rr.vstore(values, QUANT_TILE, zero, full)
            for column in range(0, QUANT_TILE, VECTOR_TILE):
                gate_value = rr.vload(gate_ub, column)
                up_value = rr.vload(up_ub, column)
                gate_bf16 = rr.vcast(gate_value, cb.dtypes.bfloat16, mask=full, rounding="rn")
                up_bf16 = rr.vcast(up_value, cb.dtypes.bfloat16, mask=full, rounding="rn")
                gate_value = rr.vcast(gate_bf16, cb.dtypes.float32, mask=full)
                up_value = rr.vcast(up_bf16, cb.dtypes.float32, mask=full)
                if SWIGLU_LIMIT > 0.0:
                    gate_value = rr.vmins(gate_value, SWIGLU_LIMIT, mask=full)
                    up_value = rr.vmins(up_value, SWIGLU_LIMIT, mask=full)
                    up_value = rr.vmaxs(up_value, -SWIGLU_LIMIT, mask=full)
                gate_neg = rr.vmuls(gate_value, -1.0, mask=full)
                gate_exp = rr.vexp(gate_neg, mask=full)
                denominator = rr.vadds(gate_exp, 1.0, mask=full)
                sigmoid = rr.vdiv(one, denominator, mask=full, precision_mode=True)
                silu = rr.vmul(gate_value, sigmoid, mask=full)
                activated = rr.vmul(silu, up_value, mask=full)
                rounded = rr.vcast(activated, cb.dtypes.bfloat16, mask=full, rounding="rn")
                value = rr.vcast(rounded, cb.dtypes.float32, mask=full)
                rr.vstore(values, column, value, full)
            rr.vmem_bar("vst_vld")

            _quant_upward_row(values, packed, codes, row, QUANT_TILE)
        cb.mem_copy(cb.tile_slice(payload, (1, QUANT_TILE), (row, 0)), packed)
    cb.mem_copy(scales.view(dtype=cb.dtypes.uint8), codes)


@cb.kernel
def shared_swiglu_quant_grid(gate: cb.Tensor, up: cb.Tensor, payload: cb.Tensor, scales: cb.Tensor):
    """按 M16/K768 分工，各 block 独占打包缩放值区域。"""
    columns = gate.shape[1] // QUANT_TILE
    tiles = gate.shape[0] // ROW_TILE * columns
    for work in cb.range(cb.get_block_idx(), tiles, cb.get_block_num()):
        row = work // columns
        column = work % columns
        g = cb.tile_slice(gate, (ROW_TILE, QUANT_TILE), (row, column))
        u = cb.tile_slice(up, (ROW_TILE, QUANT_TILE), (row, column))
        out = cb.tile_slice(payload, (ROW_TILE, QUANT_TILE), (row, column))
        offset = row * ROW_TILE * (gate.shape[1] // MX_GROUP) + column * ROW_TILE * (QUANT_TILE // MX_GROUP)
        factors = scales[0:1, offset:offset + ROW_TILE * (QUANT_TILE // MX_GROUP)]
        _shared_swiglu_quant_tile(g, u, out, factors)


@pl.jit.inline(auto_scope=False)
def shared_swiglu_quant(gate: pl.Tensor, up: pl.Tensor, payload: pl.Tensor, scales: pl.Tensor):
    """以多核 SPMD 提交共享专家的激活量化阶段。"""
    tiles = pl.tensor.dim(gate, 0) // ROW_TILE * (pl.tensor.dim(gate, 1) // QUANT_TILE)
    workers = pl.max(1, pl.min(tiles, pl.system.available_aiv_count()))
    with pl.spmd(workers, name_hint="dsl_sh_act_quant"):
        shared_swiglu_quant_grid(gate, up, payload, scales)
    return payload, scales


@pl.jit
def shared_swiglu_quant_validation(
    gate: pl.Tensor,
    up: pl.Tensor,
    payload: pl.Out[pl.Tensor],
    scales: pl.Out[pl.Tensor],
):
    shared_swiglu_quant(gate, up, payload, scales)
    return payload, scales


def golden_shared_swiglu_quant(tensors):
    """执行舍入为 BF16 的 SwiGLU，并打包向上取整的 E8M0 缩放编码。"""
    import torch
    import torch.nn.functional as F

    from quant import pack_mx_a_scale, quantize_mxfp8_upward_e8m0

    gate = tensors["gate"].bfloat16().float()
    up = tensors["up"].bfloat16().float()
    if SWIGLU_LIMIT > 0:
        gate = gate.clamp(max=SWIGLU_LIMIT)
        up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    activated = (F.silu(gate) * up).bfloat16().float()
    payload, codes = quantize_mxfp8_upward_e8m0(activated)
    tensors["payload"].copy_(payload)
    tensors["scales"].view(torch.uint8).copy_(pack_mx_a_scale(codes).reshape(1, -1))


def build_swiglu_quant_specs(rows=16, case="random"):
    """构造随机行，或包含零值、截断、舍入和缩放边界的数据组。"""
    import torch

    from golden import TensorSpec

    generator = torch.Generator().manual_seed(914)
    gate = torch.randn(rows, MOE_INTER, generator=generator) * 3
    up = torch.randn(rows, MOE_INTER, generator=generator) * 3
    if case == "edges":
        patterns = (
            (0.0, 0.0), (10.0, 0.175), (10.0, 0.17578125), (12.0, 12.0), (-12.0, -12.0),
            (1.00390625, 1.01171875), (-1.00390625, -1.01171875),
        )
        for group in range(MOE_INTER // MX_GROUP):
            gate_value, up_value = patterns[group % len(patterns)]
            begin = group * MX_GROUP
            gate[:, begin : begin + MX_GROUP] = gate_value
            up[:, begin : begin + MX_GROUP] = up_value
        gate[0].zero_()
    return [
        TensorSpec("gate", [rows, MOE_INTER], torch.float32, init_value=gate),
        TensorSpec("up", [rows, MOE_INTER], torch.float32, init_value=up),
        TensorSpec("payload", [rows, MOE_INTER], torch.float8_e4m3fn),
        TensorSpec("scales", [1, rows * (MOE_INTER // MX_GROUP)], torch.float8_e8m0fnu),
    ]


if __name__ == "__main__":
    import argparse

    from golden import run
    from window_kv_dsl import compare_payload_bytes

    parser = argparse.ArgumentParser(description="Validate DSL shared-expert SwiGLU and upward MXFP8 on A5.")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--rows", type=int, choices=[16, 80], default=16)
    parser.add_argument("--case", choices=["all", "random", "edges"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    for case in ("random", "edges"):
        if args.case not in ("all", case):
            continue
        print(f"[SHARED_SWIGLU_QUANT] case={case} rows={args.rows}", flush=True)
        result = run(
            fn=shared_swiglu_quant_validation,
            specs=build_swiglu_quant_specs(args.rows, case),
            golden_fn=golden_shared_swiglu_quant,
            compare_fn={"payload": compare_payload_bytes, "scales": compare_payload_bytes},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
