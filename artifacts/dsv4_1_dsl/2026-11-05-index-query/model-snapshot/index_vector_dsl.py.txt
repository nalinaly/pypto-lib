# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""计算 Indexer 的相邻对 RoPE、BF16 舍入及 group-32 E8M0/FP4 量化。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.lang.jit import jit

from config import FLASH as M, INDEX_CACHE_GROUP


# 模型配置
INDEX_DIM = M.index_head_dim
ROPE_DIM = M.qk_rope_head_dim
CACHE_GROUP = INDEX_CACHE_GROUP

# 分块参数
VECTOR_TILE = 64


@jit
def rope_quantize_row(values, cos_ub, sin_ub, packed, factors, scale_offset):
    """对已舍入为 BF16 的 FP32 行进行旋转和最近偶数 FP4 编码。"""
    full = rr.full_mask()
    group_mask = rr.update_mask(CACHE_GROUP, elem_bits=32)[0]
    pair_mask = rr.update_mask(VECTOR_TILE // 2, elem_bits=32)[0]
    # 后 64 个元素按相邻实虚对旋转，并再次舍入为 BF16。
    real, imag = rr.vload_deinterleave(values, INDEX_DIM - ROPE_DIM, width="b32")
    cos = rr.vload(cos_ub)
    sin = rr.vload(sin_ub)
    real_cos = rr.vmul(real, cos, mask=group_mask)
    imag_sin = rr.vmul(imag, sin, mask=group_mask)
    real_sin = rr.vmul(real, sin, mask=group_mask)
    imag_cos = rr.vmul(imag, cos, mask=group_mask)
    real_rotated = rr.vsub(real_cos, imag_sin, mask=group_mask)
    imag_rotated = rr.vadd(real_sin, imag_cos, mask=group_mask)
    real_bf16 = rr.vcast(real_rotated, cb.dtypes.bfloat16, mask=group_mask, rounding="rn")
    imag_bf16 = rr.vcast(imag_rotated, cb.dtypes.bfloat16, mask=group_mask, rounding="rn")
    real_rounded = rr.vcast(real_bf16, cb.dtypes.float32, mask=group_mask)
    imag_rounded = rr.vcast(imag_bf16, cb.dtypes.float32, mask=group_mask)
    rr.vstore_interleave(values, INDEX_DIM - ROPE_DIM, real_rounded, imag_rounded, width="b32")
    rr.vmem_bar("vst_vld")
    for group in range(INDEX_DIM // CACHE_GROUP):
        value = rr.vload(values, group * CACHE_GROUP)
        maximum = rr.vreduce_max(rr.vabs(value, mask=group_mask), mask=group_mask)
        maximum = rr.vmaxs(maximum, 6.0 * 2.0**-126, mask=group_mask)
        raw_scale = rr.vmuls(maximum, 1.0 / 6.0, mask=group_mask)
        bits = rr.vreinterpret(raw_scale, cb.dtypes.int32)
        ceil_bits = rr.vadds(bits, 0x7FFFFF, mask=group_mask)
        exponent = rr.vdup(rr.vshr(ceil_bits, 23, mask=group_mask), mask=group_mask)
        scale = rr.vreinterpret(rr.vshl(exponent, 23, mask=group_mask), cb.dtypes.float32)
        code = rr.vcast(exponent, cb.dtypes.uint8, mask=group_mask)
        offset = scale_offset + group
        rr.vstore_first(factors, offset, code)
        normalized = rr.vdiv(value, scale, mask=group_mask)
        rr.vstore(values, group * CACHE_GROUP, normalized, group_mask)
    rr.vmem_bar("vst_vld")
    # 每次打包 64 个元素，确保 UB 存储地址按 32 字节对齐。
    for col in range(0, INDEX_DIM, VECTOR_TILE):
        value = rr.vload(values, col)
        magnitude = rr.vabs(value, mask=full)
        codes = rr.vdups(0, cb.dtypes.uint32, mask=full)
        # 奇数编码的下边界取严格大于，偶数编码的下边界包含中点。
        for level, threshold in ((1, 0.25), (2, 0.75), (3, 1.25), (4, 1.75), (5, 2.5), (6, 3.5), (7, 5.0)):
            selected = rr.vgts(magnitude, threshold, mask=full)
            if level % 2 == 0:
                selected = rr.vges(magnitude, threshold, mask=full)
            encoded = rr.vdups(level, cb.dtypes.uint32, mask=full)
            codes = rr.vselect(encoded, codes, cond_mask=selected)
        sign_bits = rr.vreinterpret(value, cb.dtypes.uint32)
        sign = rr.vshr(sign_bits, 28, mask=full)
        sign = rr.vbitwise_and(sign, rr.vdups(8, cb.dtypes.uint32), mask=full)
        codes = rr.vbitwise_or(codes, sign, mask=full)
        even, odd = rr.vdeinterleave(codes, codes)
        high = rr.vshl(odd, 4, mask=pair_mask)
        byte = rr.vbitwise_or(even, high, mask=pair_mask)
        rr.vstore_pack(packed, col // 2, byte, pair_mask, pack_mode=rr.PackMode.B32_TO_B8)
