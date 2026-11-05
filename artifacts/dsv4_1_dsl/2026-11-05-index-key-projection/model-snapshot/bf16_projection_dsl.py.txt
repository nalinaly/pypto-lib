# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 对 BF16 ND 输入和 NZ 权重执行分块 FP32 累加投影。"""

import cannbotdsl as cb
from cannbotdsl.lang.jit import jit
from cannbotdsl.ring_buffer import RingBuffer


@jit
def project_bf16_nz(
    source, weight, output, column, active_rows,
    row_tile: cb.Constexpr[int], column_tile: cb.Constexpr[int], reduction_tile: cb.Constexpr[int],
):
    """按 K 分块顺序累加，并通过 Fixpipe 写入目标类型的有效行。"""
    a_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.bfloat16, cb.make_layout((row_tile, reduction_tile)), depth=1)
    b_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.bfloat16, cb.make_layout((reduction_tile, column_tile)), depth=1)
    a_l0 = RingBuffer(cb.MemLoc.L0A, cb.dtypes.bfloat16, cb.make_layout((row_tile, reduction_tile)), depth=1)
    b_l0 = RingBuffer(cb.MemLoc.L0B, cb.dtypes.bfloat16, cb.make_layout((column_tile, reduction_tile)), depth=1)
    c_l0 = RingBuffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((row_tile, column_tile)), depth=1)
    accumulator = c_l0.next_write()
    for k in cb.range(source.shape[1] // reduction_tile):
        a = a_l1.next_write()
        b = b_l1.next_write()
        cb.mem_copy(a, cb.tile_slice(source[0:active_rows, None], (row_tile, reduction_tile), (0, k)))
        # NZ BF16 分块布局：[N/16, K/16, 16, 16]。
        raw_layout = cb.make_layout((column_tile // 16, reduction_tile * 16))
        b_raw = cb.from_tensor(b, layout=raw_layout, data_format="nd")
        cb.mem_copy(b_raw, cb.tile_slice(weight, (column_tile // 16, reduction_tile * 16), (column, k)))
        cb.mem_copy(a_l0.next_write(), a_l1.next_read())
        cb.mem_copy(b_l0.next_write(), b_l1.next_read().permute(1, 0))
        cb.matmul(accumulator, a_l0.next_read(), b_l0.next_read(), init=k == 0)
    result = c_l0.next_read()
    valid_layout = cb.make_layout((active_rows, column_tile))
    valid = cb.from_tensor(result, layout=valid_layout, physical_layout=result.physical_layout)
    cb.mem_copy(output[0:active_rows, None], valid)
