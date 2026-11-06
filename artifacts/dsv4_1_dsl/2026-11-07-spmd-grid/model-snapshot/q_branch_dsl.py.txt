# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 DSL 实现 Q 低秩 RMSNorm、OCP 量化、QB 投影和相邻元素对 RoPE。"""

import cannbotdsl as cb
from cannbotdsl.lang.jit import jit
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
import pypto.language as pl
from config import FLASH as M
from mxfp8_dsl import _mx_project, _quant_row, MX_GROUP, ROW_TILE, VECTOR_TILE, TASK_TILE, PADDED_TILE, CUBE_TILE


# 模型配置。
Q_LORA = M.q_lora_rank
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
EPS = M.rms_norm_eps


@jit
def _normalize_quant_rows(source: cb.Tensor, weight: cb.Tensor, latent: cb.Tensor,
                         payload: cb.Tensor, scales: cb.Tensor,
                         num_tokens: cb.ScalarValue, row_tile: cb.ScalarValue):
    """归一化十六行 Q 低秩数据，输出 BF16 和打包后的 MXFP8 结果。"""
    active = max(0, min(ROW_TILE, num_tokens - row_tile * ROW_TILE))
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, Q_LORA)))
    gamma = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((Q_LORA,)))
    rounded_row = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, Q_LORA)))
    values = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((Q_LORA + VECTOR_TILE,)))
    packed = Buffer(cb.MemLoc.UB, cb.dtypes.float8_e4m3fn, cb.make_layout((1, Q_LORA)))
    codes = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, ROW_TILE * (Q_LORA // MX_GROUP))))
    cb.mem_copy(gamma, weight)
    for row in cb.range(ROW_TILE):
        with cb.vf():
            zero_mask = rr.full_mask()
            zero = rr.vdups(0.0, cb.dtypes.float32, mask=zero_mask)
            for col in range(0, Q_LORA + VECTOR_TILE, VECTOR_TILE):
                rr.vstore(values, col, zero, zero_mask)
            rr.vmem_bar("vst_vld")
        if row < active:
            cb.mem_copy(raw, cb.tile_slice(source, (1, Q_LORA), (row, 0)))
            with cb.vf():
                full = rr.full_mask()
                accum = rr.vdups(0.0, cb.dtypes.float32, mask=full)
                for col in range(0, Q_LORA, VECTOR_TILE):
                    loaded = rr.vload_unpack(raw, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    value = rr.vcast(loaded, cb.dtypes.float32, mask=full)
                    accum = rr.vadd(accum, rr.vmul(value, value, mask=full), mask=full)
                total = rr.vreduce_sum(accum, mask=full)
                mean = rr.vmuls(total, 1.0 / Q_LORA, mask=full)
                denominator = rr.vsqrt(rr.vadds(mean, EPS, mask=full), mask=full)
                one = rr.vdups(1.0, cb.dtypes.float32, mask=full)
                inverse = rr.vdup(rr.vdiv(one, denominator, mask=full, precision_mode=True), mask=full)
                for col in range(0, Q_LORA, VECTOR_TILE):
                    x16 = rr.vload_unpack(raw, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    g16 = rr.vload_unpack(gamma, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    value = rr.vcast(x16, cb.dtypes.float32, mask=full)
                    factor = rr.vcast(g16, cb.dtypes.float32, mask=full)
                    normalized = rr.vmul(rr.vmul(value, inverse, mask=full), factor, mask=full)
                    rounded = rr.vcast(normalized, cb.dtypes.bfloat16, mask=full, rounding="rn")
                    rr.vstore_pack(rounded_row, col, rounded, full, pack_mode=rr.PackMode.B32_TO_B16)
                    rr.vstore(values, col, rr.vcast(rounded, cb.dtypes.float32, mask=full), full)
                rr.vmem_bar("vst_vld")
                _quant_row(values, packed, codes, row, Q_LORA)
            cb.mem_copy(cb.tile_slice(latent, (1, Q_LORA), (row, 0)), rounded_row)
        else:
            with cb.vf():
                _quant_row(values, packed, codes, row, Q_LORA)
        cb.mem_copy(cb.tile_slice(payload, (1, Q_LORA), (row, 0)), packed)
    cb.mem_copy(scales.view(dtype=cb.dtypes.uint8), codes)


@cb.kernel
def project_q_b(payload: cb.Tensor, weight: cb.Tensor, activation_scale: cb.Tensor,
                weight_scale: cb.Tensor, output: cb.Tensor, valid_rows: cb.ScalarValue):
    """按 QB 列块分工，投影结果在 RoPE 前舍入为 BF16。"""
    columns = weight.shape[1] // payload.shape[1] // CUBE_TILE
    for column in cb.range(cb.get_block_idx(), columns, cb.get_block_num()):
        begin = column * CUBE_TILE
        target = output[0:valid_rows, begin:begin + CUBE_TILE]
        _mx_project(payload, weight, activation_scale, weight_scale, target, column, valid_rows)


@jit
def _query_rope_head(source: cb.Tensor, cosine: cb.Tensor, sine: cb.Tensor, output: cb.Tensor,
                    num_tokens: cb.ScalarValue, row_tile: cb.ScalarValue):
    """旋转一个 Query 头最后 64 个 BF16 元素，保持相邻元素配对关系。"""
    active = max(0, min(ROW_TILE, num_tokens - row_tile * ROW_TILE))
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, HEAD_DIM)))
    result = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, HEAD_DIM)))
    cosine_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
    sine_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
    values = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((HEAD_DIM + VECTOR_TILE,)))
    for row in cb.range(active):
        cb.mem_copy(raw, cb.tile_slice(source, (1, HEAD_DIM), (row, 0)))
        cb.mem_copy(cb.tile_slice(cosine_ub, (1, ROPE_DIM // 2), (0, 0)), cb.tile_slice(cosine, (1, ROPE_DIM // 2), (row, 0)))
        cb.mem_copy(cb.tile_slice(sine_ub, (1, ROPE_DIM // 2), (0, 0)), cb.tile_slice(sine, (1, ROPE_DIM // 2), (row, 0)))
        with cb.vf():
            full = rr.full_mask()
            half = rr.update_mask(ROPE_DIM // 2, elem_bits=32)[0]
            zero = rr.vdups(0.0, cb.dtypes.float32, mask=full)
            rr.vstore(values, HEAD_DIM, zero, full)
            for col in range(0, HEAD_DIM, VECTOR_TILE):
                loaded = rr.vload_unpack(raw, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                rr.vstore(values, col, rr.vcast(loaded, cb.dtypes.float32, mask=full), full)
            rr.vmem_bar("vst_vld")
            real, imag = rr.vload_deinterleave(values, HEAD_DIM - ROPE_DIM, width="b32")
            cos_value = rr.vload(cosine_ub)
            sin_value = rr.vload(sine_ub)
            real_cos = rr.vmul(real, cos_value, mask=half)
            imag_sin = rr.vmul(imag, sin_value, mask=half)
            real_sin = rr.vmul(real, sin_value, mask=half)
            imag_cos = rr.vmul(imag, cos_value, mask=half)
            rotated_real = rr.vsub(real_cos, imag_sin, mask=half)
            rotated_imag = rr.vadd(real_sin, imag_cos, mask=half)
            rr.vstore_interleave(values, HEAD_DIM - ROPE_DIM, rotated_real, rotated_imag, width="b32")
            rr.vmem_bar("vst_vld")
            for col in range(0, HEAD_DIM, VECTOR_TILE):
                value = rr.vload(values, col)
                rounded = rr.vcast(value, cb.dtypes.bfloat16, mask=full, rounding="rn")
                rr.vstore_pack(result, col, rounded, full, pack_mode=rr.PackMode.B32_TO_B16)
        cb.mem_copy(cb.tile_slice(output, (1, HEAD_DIM), (row, 0)), result)


@cb.kernel
def normalize_quant_grid(source: cb.Tensor, weight: cb.Tensor, latent: cb.Tensor,
                         payload: cb.Tensor, scales: cb.Tensor, num_tokens: cb.ScalarValue):
    """按 M16 分工完成整行归一化和 MXFP8 打包。"""
    for tile in cb.range(cb.get_block_idx(), payload.shape[0] // ROW_TILE, cb.get_block_num()):
        row = tile * ROW_TILE
        live = max(0, min(ROW_TILE, num_tokens - row))
        first = min(row, num_tokens)
        src = source[first:first + live, None]
        norm = latent[first:first + live, None]
        dst = payload[row:row + ROW_TILE, None]
        offset = row * (Q_LORA // MX_GROUP)
        factors = scales[0:1, offset:offset + ROW_TILE * (Q_LORA // MX_GROUP)]
        _normalize_quant_rows(src, weight, norm, dst, factors, num_tokens, tile)


@pl.jit.inline(auto_scope=False)
def normalize_quant(source: pl.Tensor, weight: pl.Tensor, latent: pl.Tensor,
                    payload: pl.Tensor, scales: pl.Tensor, num_tokens: pl.Scalar[pl.INT32]):
    """以一组 AIV 任务完成 Q 归一化和量化。"""
    workers = pl.max(1, pl.min(TASK_TILE, pl.tensor.dim(payload, 0) // ROW_TILE))
    with pl.spmd(workers, name_hint="dsl_q_norm_quant"):
        normalize_quant_grid(source, weight, latent, payload, scales, num_tokens)
    return latent, payload, scales


@pl.jit.inline(auto_scope=False)
def qb_projection(payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor,
                  weight_scale: pl.Tensor, output: pl.Tensor, rows: pl.Scalar[pl.INDEX]):
    """以一组 AIC SPMD 提交全部 QB 列块。"""
    width = pl.tensor.dim(weight, 1)
    weight_flat = pl.reshape(weight, [1, Q_LORA * width])
    scale_flat = pl.tensor.view(weight_scale, [1, Q_LORA // MX_GROUP * width], layout=pl.ND)
    workers = pl.max(1, pl.min(width // CUBE_TILE, pl.system.available_cluster_count()))
    with pl.spmd(workers, name_hint="dsl_q_proj_b"):
        project_q_b(payload, weight_flat, activation_scale, scale_flat, output, rows)
    return output


@cb.kernel
def query_rope_grid(source: cb.Tensor, cosine: cb.Tensor, sine: cb.Tensor,
                    output: cb.Tensor, num_tokens: cb.ScalarValue):
    """每个 block 处理若干行块和头，保留未使用的输出行。"""
    heads = output.shape[1] // HEAD_DIM
    tiles = PADDED_TILE // ROW_TILE * heads
    for work in cb.range(cb.get_block_idx(), tiles, cb.get_block_num()):
        tile = work // heads
        head = work % heads
        row = min(tile * ROW_TILE, num_tokens)
        live = max(0, min(ROW_TILE, num_tokens - tile * ROW_TILE))
        begin = head * HEAD_DIM
        src = source[row:row + live, begin:begin + HEAD_DIM]
        dst = output[row:row + live, begin:begin + HEAD_DIM]
        cos = cosine[row:row + live, None]
        sin = sine[row:row + live, None]
        _query_rope_head(src, cos, sin, dst, num_tokens, tile)


@pl.jit.inline(auto_scope=False)
def query_rope(source: pl.Tensor, cosine: pl.Tensor, sine: pl.Tensor, output: pl.Tensor,
               num_tokens: pl.Scalar[pl.INT32]):
    """合并 Query 各头的 RoPE 提交，并返回整组完成依赖。"""
    heads = pl.tensor.dim(output, 1) // HEAD_DIM
    workers = pl.max(1, pl.min(TASK_TILE, PADDED_TILE // ROW_TILE * heads))
    with pl.spmd(workers, name_hint="dsl_q_rope") as done:
        query_rope_grid(source, cosine, sine, output, num_tokens)
    return done
