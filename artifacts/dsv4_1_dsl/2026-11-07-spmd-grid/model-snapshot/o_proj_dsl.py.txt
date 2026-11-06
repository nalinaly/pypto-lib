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
from cannbotdsl.lang.jit import jit
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
import pypto.language as pl
from config import FLASH as M
from mxfp8_dsl import _mx_project, _quant_row, MX_GROUP, ROW_TILE, VECTOR_TILE, TASK_TILE, PADDED_TILE, CUBE_TILE


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


@jit
def _rope_quant_head(
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
def project_o_a(payload: cb.Tensor, weight: cb.Tensor, activation_scale: cb.Tensor,
                weight_scale: cb.Tensor, output: cb.Tensor):
    """按分组和列块分配 WoA 投影。"""
    columns = O_RANK // CUBE_TILE
    for work in cb.range(cb.get_block_idx(), O_GROUPS * columns, cb.get_block_num()):
        group = work // columns
        column = work % columns
        row = group * PADDED_TILE
        source = payload[row:row + PADDED_TILE, None]
        matrix = weight[group:group + 1, None]
        factors = activation_scale[group:group + 1, None]
        weight_factors = weight_scale[group:group + 1, None]
        begin = group * O_RANK + column * CUBE_TILE
        target = output[0:PADDED_TILE, begin:begin + CUBE_TILE]
        _mx_project(source, matrix, factors, weight_factors, target, column, PADDED_TILE)


@cb.kernel
def project_o_b(payload: cb.Tensor, weight: cb.Tensor, activation_scale: cb.Tensor,
                weight_scale: cb.Tensor, output: cb.Tensor, valid_rows: cb.ScalarValue):
    """按 WoB 列块分工，最终只写有效行。"""
    columns = weight.shape[1] // payload.shape[1] // CUBE_TILE
    for column in cb.range(cb.get_block_idx(), columns, cb.get_block_num()):
        begin = column * CUBE_TILE
        target = output[0:valid_rows, begin:begin + CUBE_TILE]
        _mx_project(payload, weight, activation_scale, weight_scale, target, column, valid_rows)


@cb.kernel
def rope_quant_grid(attended: cb.Tensor, cosine: cb.Tensor, sine: cb.Tensor,
                    payload: cb.Tensor, scales: cb.Tensor, active_rows: cb.ScalarValue):
    """按行块和头分工，独占对应分组中的 MXFP8 缩放存储。"""
    heads = attended.shape[1] // HEAD_DIM
    tiles = PADDED_TILE // ROW_TILE * heads
    for work in cb.range(cb.get_block_idx(), tiles, cb.get_block_num()):
        head = work % heads
        row = work // heads * ROW_TILE
        live = max(0, min(ROW_TILE, active_rows - row))
        first = min(row, attended.shape[0])
        begin = head * HEAD_DIM
        src = attended[first:first + live, begin:begin + HEAD_DIM]
        cos = cosine[first:first + live, None]
        sin = sine[first:first + live, None]
        group = head // (GROUP_WIDTH // HEAD_DIM)
        local = head % (GROUP_WIDTH // HEAD_DIM)
        dst_row = group * PADDED_TILE + row
        dst_col = local * HEAD_DIM
        dst = payload[dst_row:dst_row + ROW_TILE, dst_col:dst_col + HEAD_DIM]
        offset = row * (GROUP_WIDTH // MX_GROUP) + local * ROW_TILE * (HEAD_DIM // MX_GROUP)
        factors = scales[group:group + 1, offset:offset + ROW_TILE * (HEAD_DIM // MX_GROUP)]
        _rope_quant_head(src, cos, sin, dst, factors, active_rows, work)


@pl.jit.inline(auto_scope=False)
def rope_quant(attended: pl.Tensor, cosine: pl.Tensor, sine: pl.Tensor,
               payload: pl.Tensor, scales: pl.Tensor, active_rows: pl.Scalar[pl.INT32]):
    """以一组 AIV SPMD 完成输出 RoPE 和分组量化。"""
    heads = pl.tensor.dim(attended, 1) // HEAD_DIM
    workers = pl.max(1, pl.min(TASK_TILE, PADDED_TILE // ROW_TILE * heads))
    with pl.spmd(workers, name_hint="dsl_o_rope_quant") as done:
        rope_quant_grid(attended, cosine, sine, payload, scales, active_rows)
    return done


@pl.jit.inline(auto_scope=False)
def grouped_projection(payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor,
                       weight_scale: pl.Tensor, output: pl.Tensor):
    """合并八组 WoA 的独立列块提交。"""
    weight_flat = pl.reshape(weight, [O_GROUPS, GROUP_WIDTH * O_RANK])
    scale_flat = pl.reshape(weight_scale, [O_GROUPS, GROUP_WIDTH // MX_GROUP * O_RANK])
    workers = pl.min(O_GROUPS * (O_RANK // CUBE_TILE), pl.system.available_cluster_count())
    with pl.spmd(workers, name_hint="dsl_o_proj_a"):
        project_o_a(payload, weight_flat, activation_scale, scale_flat, output)
    return output


@pl.jit.inline(auto_scope=False)
def output_projection(payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor,
                      weight_scale: pl.Tensor, output: pl.Tensor, rows: pl.Scalar[pl.INDEX]):
    """以一组 AIC SPMD 完成 WoB 输出投影。"""
    weight_flat = pl.reshape(weight, [1, O_WIDTH * HIDDEN])
    scale_flat = pl.tensor.view(weight_scale, [1, O_WIDTH // MX_GROUP * HIDDEN], layout=pl.ND)
    workers = pl.min(HIDDEN // CUBE_TILE, pl.system.available_cluster_count())
    with pl.spmd(workers, name_hint="dsl_o_proj_b"):
        project_o_b(payload, weight_flat, activation_scale, scale_flat, output, rows)
    return output


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
    payload: pl.Tensor, weight: pl.Tensor, activation_scale: pl.Tensor, weight_scale: pl.Tensor,
    output: pl.Out[pl.Tensor[[O_ROWS, 256], pl.BF16]], rows: pl.Scalar[pl.INDEX],
):
    with pl.spmd(256 // CUBE_TILE, name_hint="dsl_mx_projection"):
        project_o_b(payload, weight, activation_scale, weight_scale, output, rows)
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
