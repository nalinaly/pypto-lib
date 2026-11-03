# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""实现延迟 mHC 数据流合并/RMSNorm 和 FP32 残差展开的 AIV 叶子任务。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from config import FLASH as M


# 模型配置
D = M.hidden_size
HC_MULT = M.hc_mult
HC_DIM = M.hc_dim
EPS = M.rms_norm_eps

# 分块参数
VECTOR_TILE = 64
TASK_TILE = 12


@cb.kernel
def collapse_norm_row(source: cb.Tensor, pre_mix: cb.Tensor, weight: cb.Tensor, output: cb.Tensor):
    """对一行执行 FP32 两两求和、BF16 舍入，再执行 RMSNorm。"""
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, HC_DIM)))
    gamma = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((D,)))
    values = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((D,)))
    result = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, D)))
    cb.mem_copy(raw, source)
    cb.mem_copy(gamma, weight)
    p0 = pre_mix[0, 0]
    p1 = pre_mix[0, 1]
    p2 = pre_mix[0, 2]
    p3 = pre_mix[0, 3]
    with cb.vf():
        full = rr.full_mask()
        accum = rr.vdups(0.0, cb.dtypes.float32, mask=full)
        for col in range(0, D, VECTOR_TILE):
            y0 = rr.vmuls(rr.vload(raw, col), p0, mask=full)
            y1 = rr.vmuls(rr.vload(raw, D + col), p1, mask=full)
            y2 = rr.vmuls(rr.vload(raw, 2 * D + col), p2, mask=full)
            y3 = rr.vmuls(rr.vload(raw, 3 * D + col), p3, mask=full)
            pair01 = rr.vadd(y0, y1, mask=full)
            pair23 = rr.vadd(y2, y3, mask=full)
            mixed = rr.vadd(pair01, pair23, mask=full)
            rounded = rr.vcast(mixed, cb.dtypes.bfloat16, mask=full, rounding="rn")
            value = rr.vcast(rounded, cb.dtypes.float32, mask=full)
            rr.vstore(values, col, value, full)
            accum = rr.vadd(accum, rr.vmul(value, value, mask=full), mask=full)
        rr.vmem_bar("vst_vld")
        total = rr.vreduce_sum(accum, mask=full)
        mean = rr.vmuls(total, 1.0 / D, mask=full)
        denominator = rr.vsqrt(rr.vadds(mean, EPS, mask=full), mask=full)
        one = rr.vdups(1.0, cb.dtypes.float32, mask=full)
        inverse = rr.vdup(rr.vdiv(one, denominator, mask=full, precision_mode=True), mask=full)
        for col in range(0, D, VECTOR_TILE):
            g16 = rr.vload_unpack(gamma, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
            factor = rr.vcast(g16, cb.dtypes.float32, mask=full)
            scaled = rr.vmul(rr.vload(values, col), inverse, mask=full)
            normalized = rr.vmul(scaled, factor, mask=full)
            rounded_out = rr.vcast(normalized, cb.dtypes.bfloat16, mask=full, rounding="rn")
            rr.vstore_pack(result, col, rounded_out, full, pack_mode=rr.PackMode.B32_TO_B16)
    cb.mem_copy(output, result)


@cb.kernel
def expand_row(sublayer: cb.Tensor, residual: cb.Tensor, post_mix: cb.Tensor,
               residual_mix: cb.Tensor, output: cb.Tensor):
    """按顺序执行不融合的加法，将一行展开为四路 FP32 数据流。"""
    hidden = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, D)))
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, HC_DIM)))
    result = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, D)))
    cb.mem_copy(hidden, sublayer)
    cb.mem_copy(raw, residual)
    for out_h in cb.range(HC_MULT):
        post = post_mix[0, out_h]
        r0 = residual_mix[0, out_h]
        r1 = residual_mix[0, HC_MULT + out_h]
        r2 = residual_mix[0, 2 * HC_MULT + out_h]
        r3 = residual_mix[0, 3 * HC_MULT + out_h]
        with cb.vf():
            full = rr.full_mask()
            for col in range(0, D, VECTOR_TILE):
                x16 = rr.vload_unpack(hidden, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                value = rr.vmuls(rr.vcast(x16, cb.dtypes.float32, mask=full), post, mask=full)
                term0 = rr.vmuls(rr.vload(raw, col), r0, mask=full)
                value = rr.vadd(value, term0, mask=full)
                term1 = rr.vmuls(rr.vload(raw, D + col), r1, mask=full)
                value = rr.vadd(value, term1, mask=full)
                term2 = rr.vmuls(rr.vload(raw, 2 * D + col), r2, mask=full)
                value = rr.vadd(value, term2, mask=full)
                term3 = rr.vmuls(rr.vload(raw, 3 * D + col), r3, mask=full)
                value = rr.vadd(value, term3, mask=full)
                rr.vstore(result, col, value, full)
        cb.mem_copy(cb.tile_slice(output, (1, D), (0, out_h)), result)
