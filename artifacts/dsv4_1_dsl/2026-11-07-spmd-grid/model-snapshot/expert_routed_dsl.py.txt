# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 实现本地 MXFP4 路由专家投影和带权激活。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.ops.memcpy import copy_options
from cannbotdsl.ring_buffer import RingBuffer
import pypto.language as pl

from config import FLASH as M
from mxfp8_dsl import _quant_upward_row


# 模型配置
MX_GROUP = 32
SWIGLU_LIMIT = M.swiglu_limit

# 分块参数
ROW_TILE = 16
COLUMN_TILE = 256
K_TILE = 128
VECTOR_TILE = 64
WORKER_TILE = 32


@jit
def _unpack_weight(packed, decoded):
    """精确扩展所有 E2M1 编码，并按标准 FP8 NZ 顺序存储 K64/N256 数据。"""
    with cb.vf():
        full = rr.full_mask(elem_bits=8)
        lane = rr.varange(0, cb.dtypes.uint8)
        magnitude = rr.vbitwise_and(lane, rr.vdups(7, cb.dtypes.uint8, mask=full), mask=full)
        codes = rr.vadds(rr.vshl(magnitude, 2, mask=full), 0x30, mask=full)
        zero = rr.vdups(0, cb.dtypes.uint8, mask=full)
        half = rr.vdups(0x30, cb.dtypes.uint8, mask=full)
        codes = rr.vselect(zero, codes, cond_mask=rr.veqs(magnitude, 0, mask=full))
        codes = rr.vselect(half, codes, cond_mask=rr.veqs(magnitude, 1, mask=full))
        sign = rr.vbitwise_and(lane, rr.vdups(8, cb.dtypes.uint8, mask=full), mask=full)
        table = rr.vbitwise_or(codes, rr.vshl(sign, 4, mask=full), mask=full)
        for row in cb.range(0, K_TILE // 2, 2):
            value = rr.vload(packed, row * (COLUMN_TILE // 2))
            high_index = rr.vshr(value, 4, mask=full)
            low = rr.vgather_reg(table, value)
            high = rr.vgather_reg(table, high_index)
            even_row, odd_row = rr.vinterleave(low, high)
            rr.vstore_strided(decoded, row * 32, even_row, full, block_stride=K_TILE // 2)
            rr.vstore_strided(decoded, (row + 1) * 32, odd_row, full, block_stride=K_TILE // 2)


@jit
def _project_tile(source, source_scale, packed, weight_scale, output,
                  expert, source_row, output_row, column, valid_rows, k_begin, k_end):
    """连续累加 K128 分块，计算一个 M16/N256 MXA8W4 投影。"""
    reduction = source.shape[1]
    experts = packed.shape[0]
    columns = packed.shape[1] * COLUMN_TILE // reduction
    lane = cb.get_subblock_id()
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((K_TILE // 2, COLUMN_TILE // 2)))
    nz_shape = cb.make_layout((COLUMN_TILE // 32, K_TILE // 32, 16, 32))
    decoded = Buffer(cb.MemLoc.UB, cb.dtypes.float8_e4m3fn, cb.make_layout((K_TILE // 2, COLUMN_TILE)),
                     physical_layout=nz_shape, data_format="nz")
    a_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.float8_e4m3fn, cb.make_layout((ROW_TILE, K_TILE)), depth=1)
    b_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.float8_e4m3fn, cb.make_layout((K_TILE, COLUMN_TILE)), depth=1)
    sa_physical = cb.make_layout((ROW_TILE // 16, K_TILE // 64, 16, 2))
    sb_physical = cb.make_layout((COLUMN_TILE // 16, K_TILE // 64, 16, 2))
    sa_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.float8_e8m0, cb.make_layout((K_TILE // MX_GROUP, ROW_TILE)),
                       depth=1, physical_layout=sa_physical, data_format="nz")
    sb_l1 = RingBuffer(cb.MemLoc.L1, cb.dtypes.float8_e8m0, cb.make_layout((K_TILE // MX_GROUP, COLUMN_TILE)),
                       depth=1, physical_layout=sb_physical, data_format="nz")
    a_l0 = RingBuffer(cb.MemLoc.L0A, cb.dtypes.float8_e4m3fn, cb.make_layout((ROW_TILE, K_TILE)), depth=1)
    b_l0 = RingBuffer(cb.MemLoc.L0B, cb.dtypes.float8_e4m3fn, cb.make_layout((COLUMN_TILE, K_TILE)), depth=1)
    c_l0 = RingBuffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((ROW_TILE, COLUMN_TILE)), depth=1)
    # 打包字节布局：[expert, N/256, K/256, 256, N_tile/2]；偶数 N 存于低半字节。
    packed_rows = packed.view(experts * packed.shape[1], COLUMN_TILE // 2)
    sa_bytes = source_scale.view(dtype=cb.dtypes.uint8).view(source.shape[0] // ROW_TILE, reduction // 2)
    sb_bytes = weight_scale.view(dtype=cb.dtypes.uint8).view(columns // ROW_TILE, experts * reduction // 2)
    accumulator = c_l0.next_write()
    for k in cb.range(k_begin // K_TILE, k_end // K_TILE):
        packed_row = expert * packed.shape[1] + column * reduction + k * K_TILE + lane * (K_TILE // 2)
        cb.mem_copy(raw, cb.tile_slice(packed_rows, (K_TILE // 2, COLUMN_TILE // 2), (packed_row // (K_TILE // 2), 0)))
        _unpack_weight(raw, decoded)
        b = b_l1.next_write()
        cb.mem_copy(b.partition(dim=0, part_id=lane), decoded)
        a = a_l1.next_write()
        cb.mem_copy(a, cb.tile_slice(source, (ROW_TILE, K_TILE), (source_row // ROW_TILE, k)))
        sa = sa_l1.next_write()
        sb = sb_l1.next_write()
        sa_raw = cb.from_tensor(sa, dtype=cb.dtypes.uint8, layout=cb.make_layout((1, K_TILE // 2)), data_format="nd")
        sb_raw = cb.from_tensor(sb, dtype=cb.dtypes.uint8, layout=cb.make_layout((COLUMN_TILE // 16, K_TILE // 2)), data_format="nd")
        cb.mem_copy(sa_raw, cb.tile_slice(sa_bytes, (1, K_TILE // 2), (source_row // ROW_TILE, k)))
        sb_index = expert * (reduction // K_TILE) + k
        cb.mem_copy(sb_raw, cb.tile_slice(sb_bytes, (COLUMN_TILE // 16, K_TILE // 2), (column, sb_index)))
        cb.mem_copy(a_l0.next_write(), a_l1.next_read(), options=copy_options(mx_scale=sa_l1.next_read().permute(1, 0)))
        cb.mem_copy(b_l0.next_write(), b_l1.next_read().permute(1, 0), options=copy_options(mx_scale=sb_l1.next_read()))
        cb.matmul(accumulator, a_l0.next_read(), b_l0.next_read(), init=k == k_begin // K_TILE, mx_mode=True)
    result = c_l0.next_read()
    valid = cb.from_tensor(result, layout=cb.make_layout((valid_rows, COLUMN_TILE)), physical_layout=result.physical_layout)
    start = column * COLUMN_TILE
    cb.mem_copy(output[output_row:output_row + valid_rows, start:start + COLUMN_TILE], valid)


@cb.kernel
def unpack_weight_probe(source: cb.Tensor, output: cb.Tensor):
    """提供与路由专家投影相同的精确 FP4 到 NZ FP8 转换。"""
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((K_TILE // 2, COLUMN_TILE // 2)))
    nz_shape = cb.make_layout((COLUMN_TILE // 32, K_TILE // 32, 16, 32))
    decoded = Buffer(cb.MemLoc.UB, cb.dtypes.float8_e4m3fn, cb.make_layout((K_TILE // 2, COLUMN_TILE)),
                     physical_layout=nz_shape, data_format="nz")
    cb.mem_copy(raw, source)
    _unpack_weight(raw, decoded)
    flat = cb.from_tensor(decoded, dtype=cb.dtypes.uint8, layout=cb.make_layout((1, K_TILE // 2 * COLUMN_TILE)), data_format="nd")
    cb.mem_copy(output, flat)


@cb.kernel
def project_up_worker(
    source: cb.Tensor, source_scale: cb.Tensor, counts: cb.Tensor,
    w1: cb.Tensor, w1_scale: cb.Tensor, w3: cb.Tensor, w3_scale: cb.Tensor,
    gate: cb.Tensor, up: cb.Tensor,
):
    """将专家的有效行投影到对齐的 FP32 gate/up 数据块。"""
    worker = cb.get_block_idx()
    workers = cb.get_block_num()
    experts = counts.shape[0]
    stride = source.shape[0] // experts
    hidden = gate.shape[1]
    padded_base = 0
    work_base = 0
    for expert in cb.range(experts):
        count = counts[expert, 0]
        row_blocks = (count + ROW_TILE - 1) // ROW_TILE
        work_count = row_blocks * (hidden // COLUMN_TILE)
        first = (worker + workers - work_base % workers) % workers
        for item in cb.range(first, work_count, workers):
            row = item // (hidden // COLUMN_TILE) * ROW_TILE
            column = item % (hidden // COLUMN_TILE)
            valid = min(ROW_TILE, count - row)
            _project_tile(source, source_scale, w1, w1_scale, gate,
                          expert, expert * stride + row, padded_base + row, column, valid, 0, source.shape[1])
            _project_tile(source, source_scale, w3, w3_scale, up,
                          expert, expert * stride + row, padded_base + row, column, valid, 0, source.shape[1])
        padded_base = padded_base + row_blocks * ROW_TILE
        work_base = work_base + work_count


@cb.kernel
def activate_worker(
    gate: cb.Tensor, up: cb.Tensor, weights: cb.Tensor, counts: cb.Tensor,
    payload: cb.Tensor, scales: cb.Tensor,
):
    """对对齐的专家数据块执行 FP32 带权 SwiGLU 和缩放值向上取整的 MXFP8 量化。"""
    worker = cb.get_block_idx()
    workers = cb.get_block_num()
    experts = counts.shape[0]
    stride = weights.shape[0] // experts
    hidden = gate.shape[1]
    gate_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, hidden)))
    up_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, hidden)))
    values = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((hidden + VECTOR_TILE,)))
    packed = Buffer(cb.MemLoc.UB, cb.dtypes.float8_e4m3fn, cb.make_layout((1, hidden)))
    codes = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, ROW_TILE * (hidden // MX_GROUP))))
    padded_base = 0
    work_base = 0
    for expert in cb.range(experts):
        count = counts[expert, 0]
        row_blocks = (count + ROW_TILE - 1) // ROW_TILE
        first = (worker + workers - work_base % workers) % workers
        for block in cb.range(first, row_blocks, workers):
            local_row = block * ROW_TILE
            begin = padded_base + local_row
            for row in cb.range(ROW_TILE):
                with cb.vf():
                    clear_mask = rr.full_mask()
                    zero = rr.vdups(0.0, cb.dtypes.float32, mask=clear_mask)
                    for col in range(0, hidden + VECTOR_TILE, VECTOR_TILE):
                        rr.vstore(values, col, zero, clear_mask)
                if local_row + row < count:
                    cb.mem_copy(gate_ub, cb.tile_slice(gate, (1, hidden), (begin + row, 0)))
                    cb.mem_copy(up_ub, cb.tile_slice(up, (1, hidden), (begin + row, 0)))
                    weight = weights[expert * stride + local_row + row, 0]
                    with cb.vf():
                        full = rr.full_mask()
                        one = rr.vdups(1.0, cb.dtypes.float32, mask=full)
                        for col in range(0, hidden, VECTOR_TILE):
                            g = rr.vload(gate_ub, col)
                            u = rr.vload(up_ub, col)
                            if SWIGLU_LIMIT > 0.0:
                                g = rr.vmins(g, SWIGLU_LIMIT, mask=full)
                                u = rr.vmins(rr.vmaxs(u, -SWIGLU_LIMIT, mask=full), SWIGLU_LIMIT, mask=full)
                            exponential = rr.vexp(rr.vmuls(g, -1.0, mask=full), mask=full)
                            sigmoid = rr.vdiv(one, rr.vadds(exponential, 1.0, mask=full), mask=full, precision_mode=True)
                            activated = rr.vmul(rr.vmul(g, sigmoid, mask=full), u, mask=full)
                            weighted = rr.vmuls(activated, weight, mask=full)
                            rr.vstore(values, col, weighted, full)
                with cb.vf():
                    rr.vmem_bar("vst_vld")
                    _quant_upward_row(values, packed, codes, row, hidden)
                cb.mem_copy(cb.tile_slice(payload, (1, hidden), (begin + row, 0)), packed)
            code_begin = begin * (hidden // MX_GROUP)
            cb.mem_copy(scales.view(dtype=cb.dtypes.uint8)[0:1, code_begin:code_begin + ROW_TILE * (hidden // MX_GROUP)], codes)
        padded_base = padded_base + row_blocks * ROW_TILE
        work_base = work_base + row_blocks


@cb.kernel
def project_down_worker(
    source: cb.Tensor, source_scale: cb.Tensor, counts: cb.Tensor,
    weight: cb.Tensor, weight_scale: cb.Tensor, partial: cb.Tensor,
):
    """使用紧凑、无填充的专家行前缀，写入两份 FP32 split-K 部分和。"""
    worker = cb.get_block_idx()
    workers = cb.get_block_num()
    hidden = source.shape[1]
    width = partial.shape[1]
    partial_stride = partial.shape[0] // 2
    padded_base = 0
    compact_base = 0
    work_base = 0
    for expert in cb.range(counts.shape[0]):
        count = counts[expert, 0]
        row_blocks = (count + ROW_TILE - 1) // ROW_TILE
        work_count = row_blocks * 2 * (width // COLUMN_TILE)
        first = (worker + workers - work_base % workers) % workers
        for item in cb.range(first, work_count, workers):
            column = item % (width // COLUMN_TILE)
            part = item // (width // COLUMN_TILE) % 2
            row = item // ((width // COLUMN_TILE) * 2) * ROW_TILE
            valid = min(ROW_TILE, count - row)
            k_begin = (hidden // COLUMN_TILE) * part // 2 * COLUMN_TILE
            k_end = (hidden // COLUMN_TILE) * (part + 1) // 2 * COLUMN_TILE
            destination_row = part * partial_stride + compact_base + row
            _project_tile(source, source_scale, weight, weight_scale, partial,
                          expert, padded_base + row, destination_row, column, valid, k_begin, k_end)
        padded_base = padded_base + row_blocks * ROW_TILE
        compact_base = compact_base + count
        work_base = work_base + work_count


@pl.jit.inline(auto_scope=False)
def routed_ffn(
    source: pl.Tensor, source_scale: pl.Tensor, weights: pl.Tensor, counts: pl.Tensor,
    w1: pl.Tensor, w1_scale: pl.Tensor, w3: pl.Tensor, w3_scale: pl.Tensor,
    w2: pl.Tensor, w2_scale: pl.Tensor, partial: pl.Tensor,
    input_ready: pl.Scalar[pl.TASK_ID], experts: pl.constexpr, width: pl.constexpr,
    hidden: pl.constexpr, padded_rows: pl.constexpr,
):
    """由三组 SPMD 依次完成投影、带权量化和下投影。"""
    gate = pl.create_tensor([padded_rows, hidden], dtype=pl.FP32, manual_dep=True)
    up = pl.create_tensor([padded_rows, hidden], dtype=pl.FP32, manual_dep=True)
    s1 = pl.tensor.view(w1_scale, [1, experts * (width // MX_GROUP) * hidden], layout=pl.ND)
    s3 = pl.tensor.view(w3_scale, [1, experts * (width // MX_GROUP) * hidden], layout=pl.ND)
    workers = pl.min(WORKER_TILE, pl.system.available_cluster_count())
    with pl.spmd(workers, name_hint="dsl_routed_up", deps=[input_ready], sync_start=True) as up_ready:
        project_up_worker(source, source_scale, counts, w1, s1, w3, s3, gate, up)

    payload = pl.create_tensor([padded_rows, hidden], dtype=pl.FP8E4M3FN, manual_dep=True)
    scales = pl.create_tensor([1, padded_rows * (hidden // MX_GROUP)], dtype=pl.FP8E8M0, manual_dep=True)
    with pl.spmd(WORKER_TILE, name_hint="dsl_routed_act_quant", deps=[up_ready]) as activation_ready:
        activate_worker(gate, up, weights, counts, payload, scales)

    s2 = pl.tensor.view(w2_scale, [1, experts * (hidden // MX_GROUP) * width], layout=pl.ND)
    with pl.spmd(workers, name_hint="dsl_routed_down", deps=[activation_ready], sync_start=True) as done:
        project_down_worker(payload, scales, counts, w2, s2, partial)
    return partial, done


@cb.kernel
def projection_probe(source: cb.Tensor, source_scale: cb.Tensor, packed: cb.Tensor,
                     weight_scale: cb.Tensor, output: cb.Tensor):
    """使用生产路径的 MXA8W4 辅助函数投影一个完整权重块。"""
    _project_tile(source, source_scale, packed, weight_scale, output,
                  0, 0, 0, 0, ROW_TILE, 0, source.shape[1])


@pl.jit
def validate_unpack(source: pl.Tensor, output: pl.Out[pl.Tensor]):
    unpack_weight_probe(source, output)
    return output


@pl.jit
def validate_projection(source: pl.Tensor, source_scale: pl.Tensor, packed: pl.Tensor,
                        weight_scale: pl.Tensor, output: pl.Out[pl.Tensor]):
    with pl.spmd(1, name_hint="dsl_routed_projection", sync_start=True):
        projection_probe(source, source_scale, packed, weight_scale, output)
    return output


@pl.jit
def validate_activation(
    gate: pl.Tensor, up: pl.Tensor, weights: pl.Tensor, counts: pl.Tensor,
    payload: pl.Out[pl.Tensor], scales: pl.Out[pl.Tensor],
):
    with pl.spmd(WORKER_TILE, name_hint="dsl_routed_act_quant"):
        activate_worker(gate, up, weights, counts, payload, scales)
    return payload, scales


def golden_activation(tensors):
    """量化带权 FP32 SwiGLU，中间不经过 BF16 舍入。"""
    import torch
    import torch.nn.functional as F
    from quant import pack_mx_a_scale, quantize_mxfp8_upward_e8m0

    active = int(tensors["counts"][0, 0])
    gate = tensors["gate"][:active].float()
    up = tensors["up"][:active].float()
    if SWIGLU_LIMIT > 0.0:
        gate = gate.clamp(max=SWIGLU_LIMIT)
        up = up.clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
    values = torch.zeros_like(tensors["gate"])
    values[:active] = F.silu(gate) * up * tensors["weights"][:active]
    payload, codes = quantize_mxfp8_upward_e8m0(values)
    tensors["payload"].copy_(payload)
    tensors["scales"].view(torch.uint8).copy_(pack_mx_a_scale(codes).reshape(1, -1))


def build_activation_specs():
    import torch
    from golden import TensorSpec

    generator = torch.Generator().manual_seed(918)
    gate = torch.randn(ROW_TILE, COLUMN_TILE, generator=generator) * 8
    up = torch.randn(ROW_TILE, COLUMN_TILE, generator=generator) * 8
    weights = torch.linspace(0.0, 1.5, ROW_TILE).reshape(ROW_TILE, 1)
    gate[1].fill_(1.00390625)
    up[1].fill_(1.01171875)
    gate[2, :64] = 12.0
    up[2, :32] = -12.0
    up[2, 32:64] = 12.0
    gate[9:] = float("nan")
    up[9:] = float("nan")
    weights[9:] = float("nan")
    return [
        TensorSpec("gate", [ROW_TILE, COLUMN_TILE], torch.float32, init_value=gate),
        TensorSpec("up", [ROW_TILE, COLUMN_TILE], torch.float32, init_value=up),
        TensorSpec("weights", [ROW_TILE, 1], torch.float32, init_value=weights),
        TensorSpec("counts", [1, 1], torch.int32, init_value=9),
        TensorSpec("payload", [ROW_TILE, COLUMN_TILE], torch.float8_e4m3fn),
        TensorSpec("scales", [1, ROW_TILE * COLUMN_TILE // MX_GROUP], torch.float8_e8m0fnu),
    ]


def golden_unpack(tensors):
    """解码 E2M1 字节，并以标准 K64/N256 FP8 NZ 字节布局输出。"""
    import torch
    from quant import FP4_E4M3_CODES

    packed = tensors["source"].to(torch.int64)
    nibbles = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(K_TILE // 2, COLUMN_TILE)
    values = FP4_E4M3_CODES[nibbles].to(torch.uint8)
    nz = values.reshape(K_TILE // 32, 16, COLUMN_TILE // 32, 32).permute(2, 0, 1, 3)
    tensors["output"].copy_(nz.contiguous().reshape(1, -1))


def build_unpack_specs():
    import torch
    from golden import TensorSpec

    packed = torch.arange(K_TILE // 2 * COLUMN_TILE // 2, dtype=torch.int32).remainder(256).to(torch.uint8)
    packed = packed.reshape(K_TILE // 2, COLUMN_TILE // 2)
    return [TensorSpec("source", list(packed.shape), torch.uint8, init_value=packed),
            TensorSpec("output", [1, K_TILE // 2 * COLUMN_TILE], torch.uint8)]


def golden_projection(tensors):
    from quant import unpack_mxfp4_weight_tiles, unpack_mx_scale_codes
    from gemm import golden_gemm_mxfp8_to_fp32_dequantized

    reduction = tensors["source"].shape[1]
    weight = unpack_mxfp4_weight_tiles(tensors["packed"].reshape(-1, 256 * 256 // 4), reduction, COLUMN_TILE, 256, 256)
    a_scale = unpack_mx_scale_codes(tensors["source_scale"].reshape(ROW_TILE, reduction // MX_GROUP))
    b_scale = unpack_mx_scale_codes(tensors["weight_scale"].reshape(reduction // MX_GROUP, COLUMN_TILE), side="b")
    output = golden_gemm_mxfp8_to_fp32_dequantized(tensors["source"], a_scale, weight, b_scale)
    tensors["output"].copy_(output)


def build_projection_specs():
    import torch
    from golden import TensorSpec
    from quant import pack_mx_a_scale, pack_mx_b_scale, pack_mxfp4_weight_tiles
    from quant import quantize_mxfp4_weight, quantize_mxfp8_upward_e8m0

    reduction = 512
    generator = torch.Generator().manual_seed(917)
    source, a_scale = quantize_mxfp8_upward_e8m0(torch.randn(ROW_TILE, reduction, generator=generator))
    weight, b_scale = quantize_mxfp4_weight(torch.randn(COLUMN_TILE, reduction, generator=generator) * 0.1)
    packed = pack_mxfp4_weight_tiles(weight, 256, 256).reshape(1, reduction, COLUMN_TILE // 2)
    a_scale = pack_mx_a_scale(a_scale).view(torch.float8_e8m0fnu).reshape(1, -1)
    b_scale = pack_mx_b_scale(b_scale.transpose(0, 1).contiguous()).view(torch.float8_e8m0fnu).reshape(1, -1)
    return [
        TensorSpec("source", list(source.shape), torch.float8_e4m3fn, init_value=source),
        TensorSpec("source_scale", list(a_scale.shape), torch.float8_e8m0fnu, init_value=a_scale),
        TensorSpec("packed", list(packed.shape), torch.uint8, init_value=packed),
        TensorSpec("weight_scale", list(b_scale.shape), torch.float8_e8m0fnu, init_value=b_scale),
        TensorSpec("output", [ROW_TILE, COLUMN_TILE], torch.float32),
    ]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, ratio_reldiff, run

    parser = argparse.ArgumentParser(description="Validate local MXFP4 routed-expert leaves on A5.")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--stage", choices=["all", "unpack", "projection", "activation"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    cases = []
    if args.stage in ("all", "unpack"):
        cases.append(("unpack", validate_unpack, build_unpack_specs(), golden_unpack,
                      ratio_allclose(atol=0, rtol=0, max_error_ratio=0)))
    if args.stage in ("all", "projection"):
        cases.append(("projection", validate_projection, build_projection_specs(), golden_projection,
                      ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)))
    if args.stage in ("all", "activation"):
        from window_kv_dsl import compare_payload_bytes
        cases.append(("activation", validate_activation, build_activation_specs(), golden_activation,
                      {"payload": compare_payload_bytes, "scales": compare_payload_bytes}))
    for name, program, specs, golden, compare in cases:
        print(f"[ROUTED_DSL] {name}", flush=True)
        result = run(
            fn=program,
            specs=specs,
            golden_fn=golden,
            compare_fn=compare if isinstance(compare, dict) else {"output": compare},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
