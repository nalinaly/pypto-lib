# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""反量化 Attention 的 FP8/FP4 缓存，并写入 BF16 NZ 分块。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.lang.jit import jit


# 模型契约
HEAD_DIM = 512
FP8_GROUP = 32
FP4_GROUP = 16

# 分块参数
GATHER_TILE = 16
VECTOR_TILE = 128
NZ_STRIDE = 272


@jit
def dequantize_fp8(payload, factors, ids, decoded, first, cache_rows):
    """将 group-32 FP8 和 BF16 scale 反量化到 BF16，屏蔽无效物理行。"""
    with cb.vf():
        half = rr.update_mask(FP8_GROUP, elem_bits=32)[0]
        fp8_mask = rr.update_mask(FP8_GROUP * 4, elem_bits=8)[0]
        zero = rr.vdups(0.0, cb.dtypes.float32, mask=half)
        for row in range(GATHER_TILE):
            index = rr.vload_broadcast(ids, first + row)
            lower = rr.vges(index, 0, mask=half)
            upper = rr.vlts(index, cache_rows, mask=half)
            valid = rr.mask_and(lower, upper, exec_mask=half)
            for group in range(HEAD_DIM // FP8_GROUP):
                offset = row * HEAD_DIM + group * FP8_GROUP
                code = rr.vload_unpack(payload, offset, unpack_mode=rr.UnpackMode.B8_TO_B32)
                code = rr.vreinterpret_lanes(code, cb.dtypes.float8_e4m3fn)
                value = rr.vcast(code, cb.dtypes.float32, mask=fp8_mask, reg_layout=rr.RegLayout.ZERO)
                factor = rr.vload_broadcast(factors, row * (HEAD_DIM // FP8_GROUP) + group)
                factor = rr.vcast(factor, cb.dtypes.float32, mask=half)
                value = rr.vmul(value, factor, mask=half)
                value = rr.vselect(value, zero, cond_mask=valid)
                value = rr.vcast(value, cb.dtypes.bfloat16, mask=half, rounding="rn")
                rr.vstore_pack(decoded, offset, value, half, pack_mode=rr.PackMode.B32_TO_B16)


@jit
def store_nz(decoded, nz):
    """将 16×512 BF16 行块转换为带一行 padding 的 NZ 格式。"""
    with cb.vf():
        full = rr.full_mask(elem_bits=16)
        for row in range(GATHER_TILE):
            for column in range(0, HEAD_DIM, VECTOR_TILE):
                value = rr.vload(decoded, row * HEAD_DIM + column)
                offset = row * 16 + column // 16 * NZ_STRIDE
                rr.vstore_strided(nz, offset, value, full, block_stride=17, repeat_stride=0)


@jit
def dequantize_fp4(payload, factors, ids, nz, first, cache_rows):
    """展开低半字节在前的 E2M1，乘 group-16 BF16 scale 后直接写 NZ。"""
    with cb.vf():
        full8 = rr.full_mask(elem_bits=8)
        full16 = rr.full_mask(elem_bits=16)
        full32 = rr.full_mask(elem_bits=32)
        zero = rr.vdups(0.0, cb.dtypes.bfloat16, mask=full16)
        for row in range(GATHER_TILE):
            index = rr.vload_broadcast(ids, first + row)
            lower = rr.vges(index, 0, mask=full32)
            upper = rr.vlts(index, cache_rows, mask=full32)
            valid = rr.mask_and(lower, upper, exec_mask=full32)
            flags = rr.vselect(rr.vdups(1, cb.dtypes.uint32), rr.vdups(0, cb.dtypes.uint32), cond_mask=valid)
            flags16 = rr.vreinterpret_lanes(flags, cb.dtypes.uint16)
            # 广播索引对应整行，任意一个有效位可扩展到全部 BF16 元素。
            flags16 = rr.vdup(flags16, mask=full16)
            valid16 = rr.veqs(flags16, 1, mask=full16)
            for column in range(0, HEAD_DIM, VECTOR_TILE):
                packed = rr.vload_unpack(payload, row * 256 + column // 2, unpack_mode=rr.UnpackMode.B8_TO_B32)
                codes = rr.vreinterpret_lanes(packed, cb.dtypes.fp4x2_e2m1)
                value = rr.vcast(codes, cb.dtypes.bfloat16, mask=full8, reg_layout=rr.RegLayout.ZERO)
                factor = rr.vload_broadcast(factors, row * 32 + column // FP4_GROUP, mode="elem2datablock")
                factor = rr.vreinterpret_lanes(factor, cb.dtypes.bfloat16)
                value = rr.vmul(value, factor, mask=full16)
                value = rr.vselect(value, zero, cond_mask=valid16)
                offset = row * 16 + column // 16 * NZ_STRIDE
                rr.vstore_strided(nz, offset, value, full16, block_stride=17, repeat_stride=0)
