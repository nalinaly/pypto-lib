# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""计算压缩 latent 的 RMSNorm、相邻对 RoPE 和 group-16 BF16-scale FP4 量化。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import const_expr, range_constexpr
from cannbotdsl.lang.jit import jit

from config import FLASH as M


# 模型契约
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
EPS = M.rms_norm_eps
CACHE_GROUP = 16
FP4_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)

# 分块参数
VECTOR_TILE = 64


@jit
def normalize_latent(source, gamma, output):
    """按 64 元素分块顺序规约，再以 sqrt 和倒数计算 BF16 RMSNorm。"""
    with cb.vf():
        full = rr.full_mask()
        total = rr.vdups(0.0, cb.dtypes.float32, mask=full)
        for column in range(0, HEAD_DIM, VECTOR_TILE):
            raw = rr.vload_unpack(source, column, unpack_mode=rr.UnpackMode.B16_TO_B32)
            value = rr.vcast(raw, cb.dtypes.float32, mask=full)
            square = rr.vmul(value, value, mask=full)
            total = rr.vadd(total, square, mask=full)
        total = rr.vreduce_sum(total, mask=full)
        mean = rr.vmuls(total, 1.0 / HEAD_DIM, mask=full)
        variance = rr.vadds(mean, EPS, mask=full)
        root = rr.vsqrt(variance, mask=full)
        one = rr.vdups(1.0, cb.dtypes.float32, mask=full)
        inverse = rr.vdiv(one, root, mask=full)
        inverse = rr.vdup(inverse, mask=full)
        for column in range(0, HEAD_DIM, VECTOR_TILE):
            raw = rr.vload_unpack(source, column, unpack_mode=rr.UnpackMode.B16_TO_B32)
            value = rr.vcast(raw, cb.dtypes.float32, mask=full)
            gain = rr.vload_unpack(gamma, column, unpack_mode=rr.UnpackMode.B16_TO_B32)
            gain = rr.vcast(gain, cb.dtypes.float32, mask=full)
            value = rr.vmul(value, inverse, mask=full)
            value = rr.vmul(value, gain, mask=full)
            rounded = rr.vcast(value, cb.dtypes.bfloat16, mask=full, rounding="rn")
            rr.vstore_pack(output, column, rounded, full, pack_mode=rr.PackMode.B32_TO_B16)


class CompressedRow:
    """一行 latent 的 RoPE、FP4 payload 和 BF16 scale 片上工作区。"""

    def __init__(self):
        self.values = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((HEAD_DIM + VECTOR_TILE,)))
        self.normalized = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((HEAD_DIM,)))
        self.cosine = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
        self.sine = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
        self.payload = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, HEAD_DIM // 2)))
        self.scale = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, HEAD_DIM // CACHE_GROUP)))

    @jit
    def encode(self, latent, cosine, sine, round_even):
        """保持 BF16 RoPE 舍入；C1A 中点取偶数，C2A 中点取较小幅值。"""
        cos_dst = cb.tile_slice(self.cosine, (1, ROPE_DIM // 2), (0, 0))
        sin_dst = cb.tile_slice(self.sine, (1, ROPE_DIM // 2), (0, 0))
        cb.mem_copy(cos_dst, cosine)
        cb.mem_copy(sin_dst, sine)
        with cb.vf():
            full = rr.full_mask()
            pair_mask = rr.update_mask(ROPE_DIM // 2, elem_bits=32)[0]
            group_mask = rr.update_mask(CACHE_GROUP, elem_bits=32)[0]
            for column in range(0, HEAD_DIM, VECTOR_TILE):
                raw = rr.vload_unpack(latent, column, unpack_mode=rr.UnpackMode.B16_TO_B32)
                value = rr.vcast(raw, cb.dtypes.float32, mask=full)
                rr.vstore(self.values, column, value, full)
            rr.vmem_bar("vst_vld")
            real, imag = rr.vload_deinterleave(self.values, HEAD_DIM - ROPE_DIM, width="b32")
            cos = rr.vload(self.cosine)
            sin = rr.vload(self.sine)
            real_cos = rr.vmul(real, cos, mask=pair_mask)
            imag_sin = rr.vmul(imag, sin, mask=pair_mask)
            real_sin = rr.vmul(real, sin, mask=pair_mask)
            imag_cos = rr.vmul(imag, cos, mask=pair_mask)
            rotated_real = rr.vsub(real_cos, imag_sin, mask=pair_mask)
            rotated_imag = rr.vadd(real_sin, imag_cos, mask=pair_mask)
            bf_real = rr.vcast(rotated_real, cb.dtypes.bfloat16, mask=pair_mask, rounding="rn")
            bf_imag = rr.vcast(rotated_imag, cb.dtypes.bfloat16, mask=pair_mask, rounding="rn")
            real = rr.vcast(bf_real, cb.dtypes.float32, mask=pair_mask)
            imag = rr.vcast(bf_imag, cb.dtypes.float32, mask=pair_mask)
            rr.vstore_interleave(self.values, HEAD_DIM - ROPE_DIM, real, imag, width="b32")
            rr.vmem_bar("vst_vld")
            for group in range(HEAD_DIM // CACHE_GROUP):
                value = rr.vload(self.values, group * CACHE_GROUP)
                absolute = rr.vabs(value, mask=group_mask)
                maximum = rr.vreduce_max(absolute, mask=group_mask)
                scale = rr.vmuls(maximum, 1.0 / 6.0, mask=group_mask)
                scale = rr.vmaxs(scale, 2.0**-9, mask=group_mask)
                scale16 = rr.vcast(scale, cb.dtypes.bfloat16, mask=group_mask, rounding="rn")
                rr.vstore_first(self.scale, group, scale16)
                scale32 = rr.vcast(scale16, cb.dtypes.float32, mask=group_mask)
                scale32 = rr.vdup(scale32, mask=group_mask)
                value = rr.vdiv(value, scale32, mask=group_mask)
                rr.vstore(self.normalized, group * CACHE_GROUP, value, group_mask)
            rr.vmem_bar("vst_vld")
            for column in range(0, HEAD_DIM, VECTOR_TILE):
                value = rr.vload(self.normalized, column)
                magnitude = rr.vabs(value, mask=full)
                codes = rr.vdups(0, cb.dtypes.uint32, mask=full)
                for code in range_constexpr(1, 8):
                    threshold = FP4_THRESHOLDS[code - 1]
                    if const_expr(round_even and code % 2 == 0):
                        selected = rr.vges(magnitude, threshold, mask=full)
                    else:
                        selected = rr.vgts(magnitude, threshold, mask=full)
                    level = rr.vdups(code, cb.dtypes.uint32, mask=full)
                    codes = rr.vselect(level, codes, cond_mask=selected)
                bits = rr.vreinterpret(value, cb.dtypes.uint32)
                sign = rr.vshr(bits, 28, mask=full)
                sign_mask = rr.vdups(8, cb.dtypes.uint32, mask=full)
                sign = rr.vbitwise_and(sign, sign_mask, mask=full)
                codes = rr.vbitwise_or(codes, sign, mask=full)
                even, odd = rr.vdeinterleave(codes, codes)
                high = rr.vshl(odd, 4, mask=pair_mask)
                byte = rr.vbitwise_or(even, high, mask=pair_mask)
                rr.vstore_pack(self.payload, column // 2, byte, pair_mask, pack_mode=rr.PackMode.B32_TO_B8)


def golden_normalize_latent(source, gamma):
    """按相同分块和规约顺序计算 BF16 latent 的 CPU 参考。"""
    import torch

    value = source.float()
    squared = value.square().reshape(-1, HEAD_DIM // VECTOR_TILE, VECTOR_TILE)
    total = squared[:, 0].clone()
    for chunk in range(1, HEAD_DIM // VECTOR_TILE):
        total = total + squared[:, chunk]
    while total.shape[-1] > 1:
        total = total[:, 0::2] + total[:, 1::2]
    inverse = torch.sqrt(total / HEAD_DIM + EPS).reciprocal()
    result = value.reshape(-1, HEAD_DIM) * inverse * gamma.float()
    return result.reshape_as(source).bfloat16()
