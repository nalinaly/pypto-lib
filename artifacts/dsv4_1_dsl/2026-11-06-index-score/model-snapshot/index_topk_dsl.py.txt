# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""对 BF16 排序键执行稳定的基数 TopK，并按原逻辑顺序输出候选。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.lang.jit import jit


# 模型契约
TOPK = 512
INVALID_KEY = 0x007F

@jit
def encode_score(value, mask):
    value = rr.vadds(value, 0.0, mask=mask)
    bits = rr.vreinterpret(value, cb.dtypes.uint16)
    positive = rr.vbitwise_xor(bits, rr.vdups(0x8000, cb.dtypes.uint16, mask=mask), mask=mask)
    negative = rr.vbitwise_xor(bits, rr.vdups(0xFFFF, cb.dtypes.uint16, mask=mask), mask=mask)
    sign = rr.vlts(rr.vreinterpret(bits, cb.dtypes.int16), 0, mask=mask)
    return rr.vselect(negative, positive, cond_mask=sign)


@jit
def decode_score(keys, mask):
    positive = rr.vbitwise_xor(keys, rr.vdups(0x8000, cb.dtypes.uint16, mask=mask), mask=mask)
    negative = rr.vbitwise_xor(keys, rr.vdups(0xFFFF, cb.dtypes.uint16, mask=mask), mask=mask)
    bits = rr.vselect(positive, negative, cond_mask=rr.vges(keys, 0x8000, mask=mask))
    return rr.vreinterpret(bits, cb.dtypes.bfloat16)


@jit
def _histogram(keys, histogram, width, prefix):
    with cb.vf():
        full = rr.update_mask(128, elem_bits=16)[0]
        h0 = rr.vdups(0, cb.dtypes.uint16, mask=full)
        h1 = rr.vdups(0, cb.dtypes.uint16, mask=full)
        for chunk in cb.range((width + 255) // 256, unroll=1):
            mask = rr.update_mask(width - chunk * 256, elem_bits=8)[0]
            low, high = rr.vload_deinterleave(keys, chunk * 256, width="b8")
            if prefix < 0:
                h0 = rr.vhistogram_accumulate(h0, high, mask=mask, bin=0)
                h1 = rr.vhistogram_accumulate(h1, high, mask=mask, bin=1)
            else:
                selected = rr.veqs(high, prefix, mask=mask)
                h0 = rr.vhistogram_accumulate(h0, low, mask=selected, bin=0)
                h1 = rr.vhistogram_accumulate(h1, low, mask=selected, bin=1)
        rr.vstore(histogram, 0, h0, full)
        rr.vstore(histogram, 128, h1, full)
        rr.vmem_bar("vst_vld")
    cb.vec_sync_all()


@jit
def _select_bin(histogram, rank):
    low = cb.dtypes.int32(0)
    high = cb.dtypes.int32(255)
    for step in cb.range(8):
        middle = (low + high) // 2
        if cb.dtypes.int32(histogram[middle]) >= rank:
            high = middle
        else:
            low = middle + 1
    previous = cb.dtypes.int32(0)
    if low > 0:
        previous = cb.dtypes.int32(histogram[low - 1])
    count = cb.dtypes.int32(histogram[low]) - previous
    return low, rank - previous, count


@jit
def stable_topk(
    keys,
    input_indices,
    selected_keys,
    selected_indices,
    positions,
    equal_positions,
    histogram,
    width,
    logical_begin,
    merge,
):
    """同分取较早位置；无效键不进入结果，未填满的位置写 -1。"""
    cutoff = cb.dtypes.int32(INVALID_KEY)
    boundary = cb.dtypes.int32(-1)
    if width > TOPK:
        _histogram(keys, histogram, width, cb.dtypes.int32(-1))
        high, rank, unused_count = _select_bin(histogram, cb.dtypes.int32(width - TOPK + 1))
        _histogram(keys, histogram, width, high)
        low, rank, count = _select_bin(histogram, rank)
        cutoff = high * 256 + low
        with cb.vf():
            cursor = rr.vstore_unalign_begin(equal_positions)
            for chunk in cb.range((width + 127) // 128, unroll=1):
                mask = rr.update_mask(width - chunk * 128, elem_bits=16)[0]
                value = rr.vload(keys, chunk * 128)
                indices = rr.varange(chunk * 128, cb.dtypes.uint16)
                equal = rr.veqs(value, cutoff, mask=mask)
                packed = rr.vsqueeze_and_storeunalign_init(indices, mask=equal)
                rr.vsqueeze_and_storeunalign(equal_positions, 0, packed, cursor)
            rr.vstore_unalign_post(equal_positions, 0, cursor)
            rr.vmem_bar("vst_vld")
        cb.vec_sync_all()
        boundary = cb.dtypes.int32(equal_positions[count - rank])

    with cb.vf():
        mask32 = rr.update_mask(64, elem_bits=32)[0]
        invalid = rr.vdups(-1, cb.dtypes.int32, mask=mask32)
        for chunk in range(TOPK // 64):
            rr.vstore(positions, chunk * 64, invalid, mask32)
        rr.vmem_bar("vst_vst")
        cursor = rr.vstore_unalign_begin(positions)
        for chunk in cb.range((width + 63) // 64, unroll=1):
            mask = rr.update_mask(width - chunk * 64, elem_bits=32)[0]
            value16 = rr.vload(keys, chunk * 64)
            value = rr.vunpack(value16, cb.dtypes.uint32, part="lower")
            index = rr.varange(chunk * 64, cb.dtypes.int32)
            strict = rr.vgts(value, cutoff, mask=mask)
            equal = rr.veqs(value, cutoff, mask=mask)
            early = rr.vles(index, boundary, mask=mask)
            equal = rr.mask_and(equal, early, exec_mask=mask)
            keep = rr.mask_or(strict, equal, exec_mask=mask)
            keep = rr.mask_and(keep, rr.vgts(value, INVALID_KEY, mask=mask), exec_mask=mask)
            packed = rr.vsqueeze_and_storeunalign_init(index, mask=keep)
            rr.vsqueeze_and_storeunalign(positions, 0, packed, cursor)
        rr.vstore_unalign_post(positions, 0, cursor)
        rr.vmem_bar("vst_vld")
    with cb.vf():
        mask32 = rr.update_mask(64, elem_bits=32)[0]
        mask16 = rr.update_mask(64, elem_bits=16)[0]
        for chunk in range(TOPK // 64):
            position = rr.vload(positions, chunk * 64)
            valid = rr.vges(position, 0, mask=mask32)
            safe = rr.vmaxs(position, 0, mask=mask32)
            safe16 = rr.vpack(rr.vreinterpret(safe, cb.dtypes.uint32), cb.dtypes.uint16, part="lower")
            selected = rr.vgather(keys, safe16, mask=mask16)
            zero = rr.vdups(0, cb.dtypes.int32, mask=mask32)
            flags = rr.vselect(rr.vdups(1, cb.dtypes.int32, mask=mask32), zero, cond_mask=valid)
            flags16 = rr.vpack(rr.vreinterpret(flags, cb.dtypes.uint32), cb.dtypes.uint16, part="lower")
            valid16 = rr.veqs(flags16, 1, mask=mask16)
            selected = rr.vselect(selected, rr.vdups(0, cb.dtypes.uint16, mask=mask16), cond_mask=valid16)
            rr.vstore(selected_keys, chunk * 64, selected, mask16)
            index = rr.vadds(position, logical_begin, mask=mask32)
            if merge != 0:
                index = rr.vgather(input_indices, rr.vreinterpret(safe, cb.dtypes.uint32), mask=mask32)
            index = rr.vselect(index, rr.vdups(-1, cb.dtypes.int32, mask=mask32), cond_mask=valid)
            rr.vstore(selected_indices, chunk * 64, index, mask32)
        rr.vmem_bar("vst_vld")
