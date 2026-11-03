# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 实现逆 RoPE 和 OCP MXFP8 输出投影。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.ops.memcpy import copy_options
from cannbotdsl.ring_buffer import RingBuffer
import pypto.language as pl
from config import FLASH as M


# 动态形状变量。
O_ROWS = pl.dynamic("O_ROWS")

# 模型配置。
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
GROUP_WIDTH = M.num_attention_heads * HEAD_DIM // M.o_groups
O_GROUPS = M.o_groups
O_RANK = M.o_lora_rank
O_WIDTH = O_GROUPS * O_RANK
HIDDEN = M.hidden_size
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

@cb.kernel
def rope_quant_head(
    source: cb.Tensor,
    cosine: cb.Tensor,
    sine: cb.Tensor,
    output: cb.Tensor,
    scales: cb.Tensor,
    num_tokens: cb.ScalarValue,
    work: cb.ScalarValue,
):
    """对一个头的十六行 token 做旋转和量化，并将填充区域置零。"""
    active = max(0, min(ROW_TILE, num_tokens - (work // M.num_attention_heads) * ROW_TILE))
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, HEAD_DIM)))
    cos_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
    sin_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
    values = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((HEAD_DIM + VECTOR_TILE,)))
    packed = Buffer(cb.MemLoc.UB, cb.dtypes.float8_e4m3fn, cb.make_layout((1, HEAD_DIM)))
    codes = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, ROW_TILE * (HEAD_DIM // MX_GROUP))))
    for row in cb.range(ROW_TILE):
        with cb.vf():
            zero_mask = rr.full_mask()
            zero = rr.vdups(0.0, cb.dtypes.float32, mask=zero_mask)
            for col in range(0, HEAD_DIM + VECTOR_TILE, VECTOR_TILE):
                rr.vstore(values, col, zero, zero_mask)
            rr.vmem_bar("vst_vld")
        if row < active:
            cb.mem_copy(raw, cb.tile_slice(source, (1, HEAD_DIM), (row, 0)))
            cb.mem_copy(cb.tile_slice(cos_ub, (1, ROPE_DIM // 2), (0, 0)), cb.tile_slice(cosine, (1, ROPE_DIM // 2), (row, 0)))
            cb.mem_copy(cb.tile_slice(sin_ub, (1, ROPE_DIM // 2), (0, 0)), cb.tile_slice(sine, (1, ROPE_DIM // 2), (row, 0)))
            with cb.vf():
                input_mask = rr.full_mask()
                rope_mask = rr.update_mask(ROPE_DIM // 2, elem_bits=32)[0]
                for col in range(0, HEAD_DIM, VECTOR_TILE):
                    value = rr.vload_unpack(raw, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    value = rr.vcast(value, cb.dtypes.float32, mask=input_mask)
                    rr.vstore(values, col, value, input_mask)
                rr.vmem_bar("vst_vld")
                real, imag = rr.vload_deinterleave(values, HEAD_DIM - ROPE_DIM, width="b32")
                cos_value = rr.vload(cos_ub)
                sin_value = rr.vload(sin_ub)
                real_cos = rr.vmul(real, cos_value, mask=rope_mask)
                imag_sin = rr.vmul(imag, sin_value, mask=rope_mask)
                imag_cos = rr.vmul(imag, cos_value, mask=rope_mask)
                real_sin = rr.vmul(real, sin_value, mask=rope_mask)
                rotated_real = rr.vadd(real_cos, imag_sin, mask=rope_mask)
                rotated_imag = rr.vsub(imag_cos, real_sin, mask=rope_mask)
                rotated_real = rr.vcast(rotated_real, cb.dtypes.bfloat16, mask=rope_mask, rounding="rn")
                rotated_imag = rr.vcast(rotated_imag, cb.dtypes.bfloat16, mask=rope_mask, rounding="rn")
                rotated_real = rr.vcast(rotated_real, cb.dtypes.float32, mask=rope_mask)
                rotated_imag = rr.vcast(rotated_imag, cb.dtypes.float32, mask=rope_mask)
                rr.vstore_interleave(values, HEAD_DIM - ROPE_DIM, rotated_real, rotated_imag, width="b32")
                rr.vmem_bar("vst_vld")
                _quant_row(values, packed, codes, row, HEAD_DIM)
        else:
            with cb.vf():
                _quant_row(values, packed, codes, row, HEAD_DIM)
        cb.mem_copy(cb.tile_slice(output, (1, HEAD_DIM), (row, 0)), packed)
    scale_bytes = scales.view(dtype=cb.dtypes.uint8)
    cb.mem_copy(scale_bytes, codes)


@cb.kernel
def quant_projection_tile(
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
def _mx_project(payload, weight, activation_scale, weight_scale, output, column, valid_rows):
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
        b_raw = b.reinterpret(layout=cb.make_layout((CUBE_TILE // 32, CUBE_TILE * 32)), data_format="nd")
        cb.mem_copy(b_raw, cb.tile_slice(weight_bytes, (CUBE_TILE // 32, CUBE_TILE * 32), (column, k)))
        # MX 缩放值布局：[outer/16, K/64, 16, 2]。
        sa_raw = sa.reinterpret(dtype=cb.dtypes.uint8, layout=cb.make_layout((PADDED_TILE // ROW_TILE, 64)), data_format="nd")
        sb_raw = sb.reinterpret(dtype=cb.dtypes.uint8, layout=cb.make_layout((CUBE_TILE // ROW_TILE, 64)), data_format="nd")
        cb.mem_copy(sa_raw, cb.tile_slice(sa_bytes, (PADDED_TILE // ROW_TILE, 64), (0, k)))
        cb.mem_copy(sb_raw, cb.tile_slice(sb_bytes, (CUBE_TILE // ROW_TILE, 64), (column, k)))
        a_slot = a_l0.next_write()
        b_slot = b_l0.next_write()
        cb.mem_copy(a_slot, a_l1.next_read(), options=copy_options(mx_scale=sa_l1.next_read().permute(1, 0)))
        cb.mem_copy(b_slot, b_l1.next_read().permute(1, 0), options=copy_options(mx_scale=sb_l1.next_read()))
        cb.matmul(accumulator, a_l0.next_read(), b_l0.next_read(), init=k == 0, mx_mode=True)
    result = c_l0.next_read()
    valid = result.reinterpret(layout=cb.make_layout((valid_rows, CUBE_TILE)), physical_layout=result.physical_layout)
    cb.mem_copy(output[0:valid_rows, None], valid)


@cb.kernel
def project_o_a(
    payload: cb.Tensor, weight: cb.Tensor, activation_scale: cb.Tensor,
    weight_scale: cb.Tensor, output: cb.Tensor, column: cb.ScalarValue,
):
    _mx_project(payload, weight, activation_scale, weight_scale, output, column, PADDED_TILE)


@cb.kernel
def project_o_b(
    payload: cb.Tensor, weight: cb.Tensor, activation_scale: cb.Tensor,
    weight_scale: cb.Tensor, output: cb.Tensor, column: cb.ScalarValue,
    valid_rows: cb.ScalarValue,
):
    _mx_project(payload, weight, activation_scale, weight_scale, output, column, valid_rows)


@pl.jit.inline(auto_scope=False)
def rope_quant(
    attended: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    payload: pl.Tensor,
    scales: pl.Tensor,
    active_rows: pl.Scalar[pl.INT32],
):
    """按头和 token 分块调度，确保打包缩放值的写入区域互不重叠。"""
    heads = pl.tensor.dim(attended, 1) // HEAD_DIM
    ready = pl.system.task_dummy(deps=[])
    completed = pl.array.create(TASK_TILE, pl.TASK_ID)
    for i in pl.range(TASK_TILE):
        completed[i] = ready
    for tile in pl.parallel((PADDED_TILE // ROW_TILE) * heads):
        head = tile % heads
        t0 = (tile // heads) * ROW_TILE
        active = pl.max(pl.min(ROW_TILE, active_rows - t0), 0)
        source_start = pl.min(t0, pl.tensor.dim(attended, 0))
        source = pl.slice(attended, [active, HEAD_DIM], [source_start, head * HEAD_DIM])
        cos_tile = pl.slice(cosine, [active, ROPE_DIM // 2], [source_start, 0])
        sin_tile = pl.slice(sine, [active, ROPE_DIM // 2], [source_start, 0])
        group = head // (GROUP_WIDTH // HEAD_DIM)
        local_head = head % (GROUP_WIDTH // HEAD_DIM)
        dst = pl.slice(payload, [ROW_TILE, HEAD_DIM], [group * PADDED_TILE + t0, local_head * HEAD_DIM])
        scale_offset = t0 * (GROUP_WIDTH // MX_GROUP) + local_head * ROW_TILE * (HEAD_DIM // MX_GROUP)
        factors = pl.slice(scales, [1, ROW_TILE * (HEAD_DIM // MX_GROUP)], [group, scale_offset])
        previous = completed[tile % TASK_TILE]
        with pl.spmd(1, name_hint="dsl_o_rope_quant", deps=[previous]) as done:
            rope_quant_head(source, cos_tile, sin_tile, dst, factors, active_rows, tile)
        completed[tile % TASK_TILE] = done
    joined = pl.system.task_dummy(deps=[completed])
    return joined


@pl.jit.inline(auto_scope=False)
def grouped_projection(payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor,
                       weight_scale: pl.Tensor, output: pl.Tensor):
    """在八个独立分组中调度互不重叠的 WoA 列块。"""
    weight_flat = pl.reshape(weight, [O_GROUPS, GROUP_WIDTH * O_RANK])
    scale_flat = pl.reshape(weight_scale, [O_GROUPS, GROUP_WIDTH // MX_GROUP * O_RANK])
    for group in pl.parallel(O_GROUPS):
        a = pl.slice(payload, [PADDED_TILE, GROUP_WIDTH], [group * PADDED_TILE, 0])
        w = pl.slice(weight_flat, [1, GROUP_WIDTH * O_RANK], [group, 0])
        sa = pl.slice(activation_scale, [1, PADDED_TILE * (GROUP_WIDTH // MX_GROUP)], [group, 0])
        sb = pl.slice(scale_flat, [1, GROUP_WIDTH // MX_GROUP * O_RANK], [group, 0])
        for column in pl.parallel(O_RANK // CUBE_TILE):
            dst = pl.slice(output, [PADDED_TILE, CUBE_TILE], [0, group * O_RANK + column * CUBE_TILE])
            with pl.spmd(1, name_hint="dsl_o_proj_a"):
                project_o_a(a, w, sa, sb, dst, column)


@pl.jit.inline(auto_scope=False)
def quantize_projection(source: pl.Tensor, payload: pl.Tensor, scales: pl.Tensor,
                        active_rows: pl.Scalar[pl.INT32]):
    """调度 BF16 再量化，确保 M16 配对缩放值的区域互不重叠。"""
    ready = pl.system.task_dummy(deps=[])
    completed = pl.array.create(TASK_TILE, pl.TASK_ID)
    for i in pl.range(TASK_TILE):
        completed[i] = ready
    for row_tile in pl.range(PADDED_TILE // ROW_TILE):
        for column in pl.parallel(O_WIDTH // QUANT_TILE):
            t0 = row_tile * ROW_TILE
            k0 = column * QUANT_TILE
            src = pl.slice(source, [ROW_TILE, QUANT_TILE], [t0, k0])
            dst = pl.slice(payload, [ROW_TILE, QUANT_TILE], [t0, k0])
            scale_offset = t0 * (O_WIDTH // MX_GROUP) + column * ROW_TILE * (QUANT_TILE // MX_GROUP)
            factors = pl.slice(scales, [1, ROW_TILE * (QUANT_TILE // MX_GROUP)], [0, scale_offset])
            work = row_tile * (O_WIDTH // QUANT_TILE) + column
            previous = completed[work % TASK_TILE]
            with pl.spmd(1, name_hint="dsl_o_requant", deps=[previous]) as done:
                quant_projection_tile(src, dst, factors, active_rows, row_tile)
            completed[work % TASK_TILE] = done
    joined = pl.system.task_dummy(deps=[completed])
    return joined


@pl.jit.inline(auto_scope=False)
def output_projection(payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor,
                      weight_scale: pl.Tensor, output: pl.Tensor,
                      rows: pl.Scalar[pl.INDEX]):
    """保留 WoB 的打包字节格式，并调度最终的列块。"""
    weight_flat = pl.reshape(weight, [1, O_WIDTH * HIDDEN])
    scale_flat = pl.tensor.view(weight_scale, [1, O_WIDTH // MX_GROUP * HIDDEN], layout=pl.ND)
    for column in pl.parallel(HIDDEN // CUBE_TILE):
        dst = pl.slice(output, [PADDED_TILE, CUBE_TILE], [0, column * CUBE_TILE],
                       valid_shape=[rows, CUBE_TILE])
        with pl.spmd(1, name_hint="dsl_o_proj_b"):
            project_o_b(payload, weight_flat, activation_scale, scale_flat, dst, column, rows)


@pl.jit
def rope_quant_validation(
    attended: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    payload: pl.Out[pl.Tensor],
    scales: pl.Out[pl.Tensor],
    active_rows: pl.Scalar[pl.INT32],
):
    rope_quant(attended, cosine, sine, payload, scales, active_rows)
    return payload, scales


def golden_rope_quant(tensors):
    """分别量化旋转后的 BF16 行，并打包其 E8M0 字节。"""
    import torch
    from qkv_proj_rope import golden_rope_interleave
    from quant import _quantize_mxfp8_activation, encode_e8m0, pack_mx_a_scale

    active = int(tensors["active_rows"])
    groups = tensors["payload"].shape[0] // PADDED_TILE
    tensors["payload"].zero_()
    tensors["scales"].view(torch.uint8).zero_()
    if active == 0:
        return
    values = tensors["attended"][:active].reshape(active, groups * 8, HEAD_DIM)
    tail = golden_rope_interleave(
        values[..., -ROPE_DIM:], tensors["cosine"][:active], tensors["sine"][:active], inverse=True,
    )
    rotated = torch.cat((values[..., :-ROPE_DIM], tail), dim=-1).reshape(active, groups, GROUP_WIDTH)
    quantized, scales = _quantize_mxfp8_activation(rotated, scale_alg=0)
    quantized = quantized.flatten(-2)
    codes = encode_e8m0(scales)
    for group in range(groups):
        tensors["payload"][group * PADDED_TILE:group * PADDED_TILE + active].copy_(quantized[:, group])
        padded = torch.zeros(PADDED_TILE, GROUP_WIDTH // MX_GROUP, dtype=torch.uint8)
        padded[:active].copy_(codes[:, group])
        tensors["scales"].view(torch.uint8)[group].copy_(pack_mx_a_scale(padded).flatten())


def build_rope_specs(active=17):
    import torch
    from golden import ScalarSpec, TensorSpec

    capacity = max(active, 1)
    generator = torch.Generator().manual_seed(41103)
    source = torch.randn(capacity, 64 * HEAD_DIM, generator=generator).bfloat16()
    source[active:] = float("nan")
    angles = torch.randn(capacity, ROPE_DIM // 2, generator=generator)
    return [
        TensorSpec("attended", list(source.shape), source.dtype, init_value=source),
        TensorSpec("cosine", list(angles.shape), torch.float32, init_value=angles.cos()),
        TensorSpec("sine", list(angles.shape), torch.float32, init_value=angles.sin()),
        TensorSpec("payload", [8 * PADDED_TILE, GROUP_WIDTH], torch.float8_e4m3fn),
        TensorSpec("scales", [8, PADDED_TILE * (GROUP_WIDTH // MX_GROUP)], torch.float8_e8m0fnu),
        ScalarSpec("active_rows", torch.int32, active),
    ]


@pl.jit
def mx_projection_validation(
    payload: pl.Tensor,
    weight: pl.Tensor,
    activation_scale: pl.Tensor,
    weight_scale: pl.Tensor,
    output: pl.Out[pl.Tensor[[O_ROWS, 256], pl.BF16]],
    rows: pl.Scalar[pl.INDEX],
):
    columns = pl.tensor.dim(output, 1) // CUBE_TILE
    for column in pl.parallel(columns):
        tile = pl.slice(output, [O_ROWS, CUBE_TILE], [0, column * CUBE_TILE])
        with pl.spmd(1, name_hint="dsl_mx_projection"):
            project_o_b(payload, weight, activation_scale, weight_scale, tile, column, rows)
    return output


def build_mx_specs(rows=17):
    import torch
    from golden import ScalarSpec, TensorSpec
    from quant import pack_fp8_nz, pack_mx_a_scale, pack_mx_b_scale

    generator = torch.Generator().manual_seed(41104)
    payload = torch.randint(-4, 5, (PADDED_TILE, 256), generator=generator).to(torch.float8_e4m3fn)
    weight = torch.randint(-4, 5, (256, 256), generator=generator).to(torch.float8_e4m3fn)
    a_codes = torch.randint(125, 130, (PADDED_TILE, 8), generator=generator, dtype=torch.uint8)
    b_codes = torch.randint(125, 130, (8, 256), generator=generator, dtype=torch.uint8)
    a_scale = pack_mx_a_scale(a_codes).reshape(1, -1).view(torch.float8_e8m0fnu)
    b_scale = pack_mx_b_scale(b_codes).reshape(1, -1).view(torch.float8_e8m0fnu)
    weight = pack_fp8_nz(weight).reshape(1, -1)
    return [
        TensorSpec("payload", list(payload.shape), payload.dtype, init_value=payload),
        TensorSpec("weight", list(weight.shape), weight.dtype, init_value=weight),
        TensorSpec("activation_scale", list(a_scale.shape), a_scale.dtype, init_value=a_scale),
        TensorSpec("weight_scale", list(b_scale.shape), b_scale.dtype, init_value=b_scale),
        TensorSpec("output", [rows, 256], torch.bfloat16),
        ScalarSpec("rows", torch.int64, rows),
    ]


def golden_mx_projection(tensors):
    import torch
    from quant import unpack_fp8_nz, unpack_mx_a_scale, unpack_mx_b_scale

    a = tensors["payload"].float().reshape(PADDED_TILE, 8, 32)
    b = unpack_fp8_nz(tensors["weight"].reshape(256, 256)).float().reshape(8, 32, 256)
    sa = unpack_mx_a_scale(tensors["activation_scale"].view(torch.uint8).reshape(PADDED_TILE, 8))
    sb = unpack_mx_b_scale(tensors["weight_scale"].view(torch.uint8).reshape(8, 256))
    partials = torch.einsum("mgk,gkn->mgn", a, b)
    partials *= torch.exp2(sa.float() - 127).unsqueeze(-1)
    partials *= torch.exp2(sb.float() - 127).unsqueeze(0)
    tensors["output"].copy_(partials.sum(dim=1)[:int(tensors["rows"])].bfloat16())


if __name__ == "__main__":
    import argparse
    from golden import run
    from window_kv_dsl import compare_payload_bytes

    parser = argparse.ArgumentParser(description="Validate DSL output-projection stages on A5.")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--active", type=int, default=17, choices=range(81))
    parser.add_argument("--stage", choices=["rope", "mx"], default="rope")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.stage == "mx" and args.active == 0:
        parser.error("the MX stage needs 1..80 output rows; use --stage rope to test zero active rows")
    result = run(
        fn=rope_quant_validation if args.stage == "rope" else mx_projection_validation,
        specs=build_rope_specs(args.active) if args.stage == "rope" else build_mx_specs(args.active),
        golden_fn=golden_rope_quant if args.stage == "rope" else golden_mx_projection,
        compare_fn={"payload": compare_payload_bytes, "scales": compare_payload_bytes} if args.stage == "rope" else None,
        config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
        compile_only=args.compile_only,
        save_data=True,
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
