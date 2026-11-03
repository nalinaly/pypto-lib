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


@cb.kernel
def normalize_quant_rows(source: cb.Tensor, weight: cb.Tensor, latent: cb.Tensor,
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
                weight_scale: cb.Tensor, output: cb.Tensor, column: cb.ScalarValue,
                valid_rows: cb.ScalarValue):
    """投影一个 QB 列块，并在 RoPE 前将输出舍入为 BF16。"""
    _mx_project(payload, weight, activation_scale, weight_scale, output, column, valid_rows)


@cb.kernel
def query_rope_head(source: cb.Tensor, cosine: cb.Tensor, sine: cb.Tensor, output: cb.Tensor,
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


@pl.jit.inline(auto_scope=False)
def normalize_quant(source: pl.Tensor, weight: pl.Tensor, latent: pl.Tensor,
                    payload: pl.Tensor, scales: pl.Tensor, num_tokens: pl.Scalar[pl.INT32]):
    """调度整行 Q RMSNorm，输出 BF16 和打包后的 MXFP8 结果。"""
    row_tiles = pl.tensor.dim(payload, 0) // ROW_TILE
    for row_tile in pl.parallel(row_tiles):
        t0 = row_tile * ROW_TILE
        rows = pl.max(pl.min(ROW_TILE, num_tokens - t0), 0)
        start = pl.min(t0, num_tokens)
        src = pl.slice(source, [ROW_TILE, Q_LORA], [start, 0], valid_shape=[rows, Q_LORA])
        dst = pl.slice(latent, [ROW_TILE, Q_LORA], [start, 0], valid_shape=[rows, Q_LORA])
        data = pl.slice(payload, [ROW_TILE, Q_LORA], [t0, 0])
        factors = pl.slice(scales, [1, ROW_TILE * (Q_LORA // MX_GROUP)], [0, t0 * (Q_LORA // MX_GROUP)])
        with pl.spmd(1, name_hint="dsl_q_norm_quant"):
            normalize_quant_rows(src, weight, dst, data, factors, num_tokens, row_tile)


@pl.jit.inline(auto_scope=False)
def qb_projection(payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor,
                  weight_scale: pl.Tensor, output: pl.Tensor, rows: pl.Scalar[pl.INDEX]):
    """调度 QB 并写入 BF16 存储，显式指定有效输出行数。"""
    width = pl.tensor.dim(weight, 1)
    weight_flat = pl.reshape(weight, [1, Q_LORA * width])
    scale_flat = pl.tensor.view(weight_scale, [1, Q_LORA // MX_GROUP * width], layout=pl.ND)
    for column in pl.parallel(width // CUBE_TILE):
        dst = pl.slice(output, [PADDED_TILE, CUBE_TILE], [0, column * CUBE_TILE], valid_shape=[rows, CUBE_TILE])
        with pl.spmd(1, name_hint="dsl_q_proj_b"):
            project_q_b(payload, weight_flat, activation_scale, scale_flat, dst, column, rows)


@pl.jit.inline(auto_scope=False)
def query_rope(source: pl.Tensor, cosine: pl.Tensor, sine: pl.Tensor, output: pl.Tensor,
               num_tokens: pl.Scalar[pl.INT32]):
    """在 BF16 QB 投影之后，按头和行调度 RoPE 分块。"""
    heads = pl.tensor.dim(output, 1) // HEAD_DIM
    ready = pl.system.task_dummy(deps=[])
    completed = pl.array.create(TASK_TILE, pl.TASK_ID)
    for i in pl.range(TASK_TILE):
        completed[i] = ready
    for row_tile in pl.range(PADDED_TILE // ROW_TILE):
        t0 = pl.min(row_tile * ROW_TILE, num_tokens)
        rows = pl.max(pl.min(ROW_TILE, num_tokens - row_tile * ROW_TILE), 0)
        cos_tile = pl.slice(cosine, [ROW_TILE, ROPE_DIM // 2], [t0, 0], valid_shape=[rows, ROPE_DIM // 2])
        sin_tile = pl.slice(sine, [ROW_TILE, ROPE_DIM // 2], [t0, 0], valid_shape=[rows, ROPE_DIM // 2])
        for head in pl.parallel(heads):
            src = pl.slice(source, [ROW_TILE, HEAD_DIM], [t0, head * HEAD_DIM], valid_shape=[rows, HEAD_DIM])
            dst = pl.slice(output, [ROW_TILE, HEAD_DIM], [t0, head * HEAD_DIM], valid_shape=[rows, HEAD_DIM])
            work = row_tile * heads + head
            previous = completed[work % TASK_TILE]
            with pl.spmd(1, name_hint="dsl_q_rope", deps=[previous]) as done:
                query_rope_head(src, cos_tile, sin_tile, dst, num_tokens, row_tile)
            completed[work % TASK_TILE] = done
    joined = pl.system.task_dummy(deps=[completed])
    return joined
