# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 实现 OCP/缩放值向上取整的 MXFP8 向量量化和 NZ 权重投影。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.ops.memcpy import copy_options
from cannbotdsl.ring_buffer import RingBuffer
import pypto.language as pl


# 量化契约。
MX_GROUP = 32

# 分块参数。
ROW_TILE = 16
VECTOR_TILE = 64
TASK_TILE = 12
PADDED_TILE = 80
CUBE_TILE = 128
QUANT_TILE = 256


@jit
def _quant_row(values, payload, codes, row, width):
    """存储 E4M3 数值，以及模型按 M16 配对排列的 E8M0 缩放字节。"""
    group_mask = rr.update_mask(MX_GROUP, elem_bits=32)[0]
    for group in range(width // MX_GROUP):
        value = rr.vload(values, group * MX_GROUP)
        maximum = rr.vreduce_max(rr.vabs(value, mask=group_mask), mask=group_mask)
        exponent = rr.vshr(rr.vreinterpret(maximum, cb.dtypes.int32), 23, mask=group_mask)
        exponent = rr.vmaxs(rr.vadds(exponent, -8, mask=group_mask), 0, mask=group_mask)
        exponent = rr.vdup(exponent, mask=group_mask)
        reciprocal = rr.vsub(rr.vdups(254, cb.dtypes.int32, mask=group_mask), exponent, mask=group_mask)
        reciprocal = rr.vmul(reciprocal, rr.vmins(exponent, 1, mask=group_mask), mask=group_mask)
        reciprocal = rr.vreinterpret(rr.vshl(reciprocal, 23, mask=group_mask), cb.dtypes.float32)
        normalized = rr.vmul(value, reciprocal, mask=group_mask)
        normalized = rr.vmins(rr.vmaxs(normalized, -448.0, mask=group_mask), 448.0, mask=group_mask)
        quantized = rr.vcast(normalized, cb.dtypes.float8_e4m3fn, mask=group_mask, rounding="rn")
        rr.vstore_pack(payload, group * MX_GROUP, quantized, group_mask, pack_mode=rr.PackMode.B32_TO_B8)
        code = rr.vcast(exponent, cb.dtypes.uint8, mask=group_mask)
        offset = (group // 2) * ROW_TILE * 2 + row * 2 + group % 2
        rr.vstore_first(codes, offset, code)


@jit
def _quant_upward_row(values, payload, codes, row, width):
    """按每组 32 个元素打包 E4M3 数据，使用向上取整到 2 的幂次的 E8M0 缩放值。"""
    group_mask = rr.update_mask(MX_GROUP, elem_bits=32)[0]
    for group in range(width // MX_GROUP):
        value = rr.vload(values, group * MX_GROUP)
        absolute = rr.vabs(value, mask=group_mask)
        maximum = rr.vreduce_max(absolute, mask=group_mask)
        maximum = rr.vmaxs(maximum, 1e-4, mask=group_mask)
        scaled = rr.vmuls(maximum, 1.0 / 448.0, mask=group_mask)
        scale_bits = rr.vreinterpret(scaled, cb.dtypes.int32)
        ceil_bits = rr.vadds(scale_bits, 8388607, mask=group_mask)
        exponent = rr.vshr(ceil_bits, 23, mask=group_mask)
        exponent = rr.vdup(exponent, mask=group_mask)
        inverse_exp = rr.vsub(rr.vdups(254, cb.dtypes.int32, mask=group_mask), exponent, mask=group_mask)
        inverse_bits = rr.vshl(inverse_exp, 23, mask=group_mask)
        inverse_scale = rr.vreinterpret(inverse_bits, cb.dtypes.float32)
        normalized = rr.vmul(value, inverse_scale, mask=group_mask)
        clipped = rr.vmins(normalized, 448.0, mask=group_mask)
        clipped = rr.vmaxs(clipped, -448.0, mask=group_mask)
        quantized = rr.vcast(clipped, cb.dtypes.float8_e4m3fn, mask=group_mask, rounding="rn")
        rr.vstore_pack(payload, group * MX_GROUP, quantized, group_mask, pack_mode=rr.PackMode.B32_TO_B8)
        code = rr.vcast(exponent, cb.dtypes.uint8, mask=group_mask)
        scale_offset = group // 2 * ROW_TILE * 2 + row * 2 + group % 2
        rr.vstore_first(codes, scale_offset, code)


@jit
def _quantize_bf16_tile(
    source: cb.Tensor, output: cb.Tensor, scales: cb.Tensor,
    num_tokens: cb.ScalarValue, row_tile: cb.ScalarValue,
):
    """量化十六行、256 列的 BF16 数据块，并将填充区域置零。"""
    live_rows = max(0, min(ROW_TILE, num_tokens - row_tile * ROW_TILE))
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, 256)))
    values = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((256 + VECTOR_TILE,)))
    packed = Buffer(cb.MemLoc.UB, cb.dtypes.float8_e4m3fn, cb.make_layout((1, 256)))
    codes = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, ROW_TILE * 8)))
    for row in cb.range(ROW_TILE):
        with cb.vf():
            zero_mask = rr.full_mask()
            zero = rr.vdups(0.0, cb.dtypes.float32, mask=zero_mask)
            for col in range(0, 256 + VECTOR_TILE, VECTOR_TILE):
                rr.vstore(values, col, zero, zero_mask)
            rr.vmem_bar("vst_vld")
        if row < live_rows:
            cb.mem_copy(raw, cb.tile_slice(source, (1, 256), (row, 0)))
            with cb.vf():
                input_mask = rr.full_mask()
                for col in range(0, 256, VECTOR_TILE):
                    value = rr.vload_unpack(raw, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    rr.vstore(values, col, rr.vcast(value, cb.dtypes.float32, mask=input_mask), input_mask)
                rr.vmem_bar("vst_vld")
                _quant_row(values, packed, codes, row, 256)
        else:
            with cb.vf():
                _quant_row(values, packed, codes, row, 256)
        cb.mem_copy(cb.tile_slice(output, (1, 256), (row, 0)), packed)
    cb.mem_copy(scales.view(dtype=cb.dtypes.uint8), codes)


@jit
def _mx_accumulate(payload, weight, activation_scale, weight_scale, column):
    """将 M80 ND 激活与原始 NZ 权重相乘，使用打包后的 MX 缩放值。"""
    reduction = payload.shape[1]
    columns = weight.shape[1] // reduction
    a_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.float8_e4m3fn, cb.make_layout((PADDED_TILE, CUBE_TILE)), depth=1)
    b_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.float8_e4m3fn, cb.make_layout((CUBE_TILE, CUBE_TILE)), depth=1)
    sa_physical = cb.make_layout((PADDED_TILE // ROW_TILE, 2, ROW_TILE, 2))
    sb_physical = cb.make_layout((CUBE_TILE // ROW_TILE, 2, ROW_TILE, 2))
    sa_l1 = RingBuffer(
        cb.MemLoc.L1, cb.dtypes.float8_e8m0, cb.make_layout((4, PADDED_TILE)),
        depth=1, physical_layout=sa_physical, data_format="nz",
    )
    sb_l1 = RingBuffer(
        cb.MemLoc.L1, cb.dtypes.float8_e8m0, cb.make_layout((4, CUBE_TILE)),
        depth=1, physical_layout=sb_physical, data_format="nz",
    )
    a_l0 = RingBuffer(cb.MemLoc.L0A, cb.dtypes.float8_e4m3fn, cb.make_layout((PADDED_TILE, CUBE_TILE)), depth=1)
    b_l0 = RingBuffer(cb.MemLoc.L0B, cb.dtypes.float8_e4m3fn, cb.make_layout((CUBE_TILE, CUBE_TILE)), depth=1)
    c_l0 = RingBuffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((PADDED_TILE, CUBE_TILE)), depth=1)
    weight_bytes = weight.view(columns // 32, reduction * 32)
    sa_bytes = activation_scale.view(dtype=cb.dtypes.uint8).view(PADDED_TILE // ROW_TILE, reduction // 2)
    sb_bytes = weight_scale.view(dtype=cb.dtypes.uint8).view(columns // ROW_TILE, reduction // 2)
    accumulator = c_l0.next_write()
    for k in cb.range(reduction // CUBE_TILE):
        a = a_l1.next_write()
        b = b_l1.next_write()
        sa = sa_l1.next_write()
        sb = sb_l1.next_write()
        cb.mem_copy(a, cb.tile_slice(payload, (PADDED_TILE, CUBE_TILE), (0, k)))
        # NZ 权重分块布局：[N/32, K/16, 16, 32]。
        b_raw = cb.from_tensor(b, layout=cb.make_layout((CUBE_TILE // 32, CUBE_TILE * 32)), data_format="nd")
        cb.mem_copy(b_raw, cb.tile_slice(weight_bytes, (CUBE_TILE // 32, CUBE_TILE * 32), (column, k)))
        # MX 缩放值布局：[outer/16, K/64, 16, 2]。
        sa_raw = cb.from_tensor(sa, dtype=cb.dtypes.uint8, layout=cb.make_layout((PADDED_TILE // ROW_TILE, 64)), data_format="nd")
        sb_raw = cb.from_tensor(sb, dtype=cb.dtypes.uint8, layout=cb.make_layout((CUBE_TILE // ROW_TILE, 64)), data_format="nd")
        cb.mem_copy(sa_raw, cb.tile_slice(sa_bytes, (PADDED_TILE // ROW_TILE, 64), (0, k)))
        cb.mem_copy(sb_raw, cb.tile_slice(sb_bytes, (CUBE_TILE // ROW_TILE, 64), (column, k)))
        a_slot = a_l0.next_write()
        b_slot = b_l0.next_write()
        cb.mem_copy(a_slot, a_l1.next_read(), options=copy_options(mx_scale=sa_l1.next_read().permute(1, 0)))
        cb.mem_copy(b_slot, b_l1.next_read().permute(1, 0), options=copy_options(mx_scale=sb_l1.next_read()))
        cb.matmul(accumulator, a_l0.next_read(), b_l0.next_read(), init=k == 0, mx_mode=True)
    return c_l0.next_read()


@jit
def _mx_project(payload, weight, activation_scale, weight_scale, output, column, valid_rows):
    """将 MXFP8 的 L0C 累加结果写入 GM 的有效行。"""
    result = _mx_accumulate(payload, weight, activation_scale, weight_scale, column)
    valid = cb.from_tensor(result, layout=cb.make_layout((valid_rows, CUBE_TILE)), physical_layout=result.physical_layout)
    cb.mem_copy(output[0:valid_rows, None], valid)


@cb.kernel
def quantize_bf16_grid(source: cb.Tensor, payload: cb.Tensor, scales: cb.Tensor, active_rows: cb.ScalarValue):
    """每个 block 循环处理互不重叠的 M16/K256 量化分块。"""
    columns = payload.shape[1] // QUANT_TILE
    tiles = payload.shape[0] // ROW_TILE * columns
    for work in cb.range(cb.get_block_idx(), tiles, cb.get_block_num()):
        row_tile = work // columns
        column = work % columns
        row = row_tile * ROW_TILE
        first = min(row, source.shape[0])
        live = max(0, min(ROW_TILE, active_rows - row))
        begin = column * QUANT_TILE
        src = source[first:first + live, begin:begin + QUANT_TILE]
        dst = cb.tile_slice(payload, (ROW_TILE, QUANT_TILE), (row_tile, column))
        offset = row * (payload.shape[1] // MX_GROUP) + column * ROW_TILE * (QUANT_TILE // MX_GROUP)
        factors = scales[0:1, offset:offset + ROW_TILE * (QUANT_TILE // MX_GROUP)]
        _quantize_bf16_tile(src, dst, factors, active_rows, row_tile)


@pl.jit.inline(auto_scope=False)
def quantize_matrix(source: pl.Tensor, payload: pl.Tensor, scales: pl.Tensor,
                    active_rows: pl.Scalar[pl.INT32]):
    """由一组 AIV SPMD 量化所有 M16/K256 分块，并返回整组完成依赖。"""
    tiles = pl.tensor.dim(payload, 0) // ROW_TILE * (pl.tensor.dim(payload, 1) // QUANT_TILE)
    workers = pl.max(1, pl.min(TASK_TILE, tiles))
    with pl.spmd(workers, name_hint="dsl_mxfp8_quant") as done:
        quantize_bf16_grid(source, payload, scales, active_rows)
    return done
