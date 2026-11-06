# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 按 FP32 加法顺序执行本地 MoE 输出归约。"""

import cannbotdsl as cb
from cannbotdsl.lang.jit import jit
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
import pypto.language as pl

from config import FLASH as M


# 模型配置
D = M.hidden_size
TOPK = M.num_experts_per_tok

# 分块参数
ROW_TILE = 2
VECTOR_TILE = 64
TASK_TILE = 16


@jit
def _combine_rows(
    shared: cb.Tensor, routed: cb.Tensor, output: cb.Tensor,
    active_rows: cb.ScalarValue, store_rows: cb.ScalarValue,
):
    """先累加共享专家，再按 0..TopK-1 累加路由专家，最后舍入一次并将非有效行置零。"""
    shared_ub = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, D)))
    routed_ub = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((TOPK, D)))
    result = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, D)))
    for row in cb.range(store_rows):
        if row < active_rows:
            cb.mem_copy(shared_ub, cb.tile_slice(shared, (1, D), (row, 0)))
            cb.mem_copy(routed_ub, cb.tile_slice(routed, (TOPK, D), (row, 0)))
            with cb.vf():
                full = rr.full_mask()
                for col in range(0, D, VECTOR_TILE):
                    loaded = rr.vload_unpack(shared_ub, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    accumulator = rr.vcast(loaded, cb.dtypes.float32, mask=full)
                    for expert in range(TOPK):
                        value = rr.vload_unpack(routed_ub, expert * D + col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                        value = rr.vcast(value, cb.dtypes.float32, mask=full)
                        accumulator = rr.vadd(accumulator, value, mask=full)
                    rounded = rr.vcast(accumulator, cb.dtypes.bfloat16, mask=full, rounding="rn")
                    rr.vstore_pack(result, col, rounded, full, pack_mode=rr.PackMode.B32_TO_B16)
        else:
            with cb.vf():
                zero_mask = rr.full_mask()
                zero = rr.vdups(0.0, cb.dtypes.float32, mask=zero_mask)
                zero_bf16 = rr.vcast(zero, cb.dtypes.bfloat16, mask=zero_mask, rounding="rn")
                for col in range(0, D, VECTOR_TILE):
                    rr.vstore_pack(result, col, zero_bf16, zero_mask, pack_mode=rr.PackMode.B32_TO_B16)
        cb.mem_copy(cb.tile_slice(output, (1, D), (row, 0)), result)


@cb.kernel
def combine_grid(shared: cb.Tensor, routed: cb.Tensor, output: cb.Tensor, num_tokens: cb.ScalarValue):
    """按两行分配本地归约，保留共享专家优先的 FP32 加法顺序。"""
    active = min(shared.shape[0], max(num_tokens, 0))
    tiles = (shared.shape[0] + ROW_TILE - 1) // ROW_TILE
    for tile in cb.range(cb.get_block_idx(), tiles, cb.get_block_num()):
        row = tile * ROW_TILE
        physical = min(ROW_TILE, shared.shape[0] - row)
        live = min(physical, max(active - row, 0))
        local_shared = shared[row:row + physical, None]
        local_routed = routed[row * TOPK:(row + physical) * TOPK, None]
        target = output[row:row + physical, None]
        _combine_rows(local_shared, local_routed, target, live, physical)


@pl.jit.inline(auto_scope=False)
def combine_local(shared: pl.Tensor, routed: pl.Tensor, output: pl.Tensor,
                  num_tokens: pl.Scalar[pl.INT32], ready: pl.Scalar[pl.TASK_ID]):
    """在通信等待完成后，以一组 SPMD 归约本地输出。"""
    rows = pl.tensor.dim(shared, 0)
    target = output[0:rows, 0:D]
    workers = pl.max(1, pl.min(TASK_TILE, (rows + ROW_TILE - 1) // ROW_TILE))
    with pl.spmd(workers, name_hint="dsl_combine_reduce", deps=[ready]) as done:
        combine_grid(shared, routed, target, num_tokens)
    return done


@pl.jit
def validate_combine(
    shared: pl.Tensor, routed: pl.Tensor, output: pl.InOut[pl.Tensor], num_tokens: pl.Scalar[pl.INT32],
):
    """使用生产路径调度器，并保留一行输出保护区。"""
    rows = pl.tensor.dim(shared, 0)
    destination = pl.slice(output, [rows, D], [0, 0])
    ready = pl.system.task_dummy(deps=[])
    combine_local(shared, routed, destination, num_tokens, ready)
    return output


def golden_combine(tensors):
    """沿用共享专家优先的 FP32 加法顺序，最后转换为 BF16。"""
    import torch

    rows = tensors["shared"].shape[0]
    active = max(0, min(rows, int(tensors["num_tokens"])))
    result = torch.zeros(rows, D, dtype=torch.bfloat16)
    accumulator = tensors["shared"][:active].float()
    routed = tensors["routed"].reshape(rows, TOPK, D)
    for expert in range(TOPK):
        accumulator = accumulator + routed[:active, expert].float()
    result[:active] = accumulator.bfloat16()
    tensors["output"][:rows].copy_(result)


def build_combine_specs(rows=5, active=3):
    """包含相消、BF16 舍入中点、不同路由、异常尾部数据和写入保护区。"""
    import torch
    from golden import ScalarSpec, TensorSpec

    generator = torch.Generator().manual_seed(919)
    shared = torch.randn(rows, D, generator=generator).bfloat16()
    routed = torch.randn(rows, TOPK, D, generator=generator).bfloat16()
    shared[:, :8] = 0
    routed[:, :, :8] = 0
    shared[:, 0] = 2**25
    routed[:, 0, 0] = -(2**25)
    routed[:, 1, 0] = 1.0
    shared[:, 1] = 1.0
    routed[:, 0, 1] = 2**-8
    shared[:, 2] = 1.0 + 2**-7
    routed[:, 0, 2] = 2**-8
    shared[:, 3] = -1.0
    routed[:, 0, 3] = -(2**-8)
    shared[:, 4] = 1.0
    routed[:, 0, 4] = 2**-8
    routed[:, 1, 4] = 2**-16
    for expert in range(TOPK):
        routed[:, expert, 5] = 2**expert
    active = max(0, min(rows, active))
    shared[active:] = float("nan")
    routed[active:] = float("nan")
    return [
        TensorSpec("shared", [rows, D], torch.bfloat16, init_value=shared),
        TensorSpec("routed", [rows * TOPK, D], torch.bfloat16, init_value=routed.reshape(-1, D)),
        TensorSpec("output", [rows + 1, D], torch.bfloat16, init_value=37.0),
        ScalarSpec("num_tokens", torch.int32, active),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="Validate local DSL MoE combine on A5.")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    for rows, active in ((5, 3), (5, 0), (80, 71), (80, 80)):
        print(f"[COMBINE_DSL] rows={rows} active={active}", flush=True)
        result = run(
            fn=validate_combine,
            specs=build_combine_specs(rows, active),
            golden_fn=golden_combine,
            compare_fn={"output": ratio_allclose(atol=0, rtol=0, max_error_ratio=0)},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
