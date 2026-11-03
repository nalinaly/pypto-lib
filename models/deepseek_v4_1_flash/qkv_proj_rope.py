# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared V4.1 Q/KV projection, normalization, RoPE, and window cache publication."""


import pypto.language as pl
import torch

from config import FLASH as M, TP_SIZE, T_DYN
from gemm import (
    golden_gemm_bf16_via_mxfp8_to_bf16_sequential, golden_mxfp8_linear_grouped_sum,
)
from rmsnorm import golden_rms_norm
from window_kv_dsl import window_publish as window_publish
from qkv_proj_dsl import quantize_projection_input as quantize_projection_input, kv_proj as kv_proj, qa_projection
from q_branch_dsl import normalize_quant, qb_projection, query_rope
from mxfp8_dsl import MX_GROUP, ROW_TILE, PADDED_TILE, quantize_matrix


# 动态校验形状。
Q_PAYLOAD_ROWS = pl.dynamic("Q_PAYLOAD_ROWS")
Q_SCALE_COLUMNS = pl.dynamic("Q_SCALE_COLUMNS")


# 模型配置
LOCAL_H = M.num_attention_heads // TP_SIZE
D = M.hidden_size
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
Q_LORA = M.q_lora_rank

# 分块参数
PROJ_M_TILE = 80


@pl.jit.inline(auto_scope=False)
def q_proj_rope(
    payload: pl.Tensor,
    scales: pl.Tensor,
    wq_a: pl.Tensor,
    wq_a_scale: pl.Tensor,
    q_norm_weight: pl.Tensor,
    wq_b: pl.Tensor,
    wq_b_scale: pl.Tensor,
    rope_cos: pl.Tensor,
    rope_sin: pl.Tensor,
    query_latent: pl.Tensor,
    query: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
):
    """Project NZ QA/QB weights, rotate Q, and retain M80 QB input for Index-Q."""
    payload_rows = pl.tensor.dim(payload, 0)
    qb_payload = pl.create_tensor([payload_rows, Q_LORA], dtype=pl.FP8E4M3FN)
    qb_scales = pl.create_tensor([1, payload_rows * (Q_LORA // 32)], dtype=pl.FP8E8M0)
    for group_start in pl.parallel(0, payload_rows, PROJ_M_TILE):
        group_rows = pl.max(pl.min(PROJ_M_TILE, num_tokens - group_start), 0)
        group_payload = pl.slice(payload, [PROJ_M_TILE, D], [group_start, 0])
        scale_start = group_start * (D // MX_GROUP)
        group_scales = pl.slice(scales, [1, PROJ_M_TILE * (D // MX_GROUP)], [0, scale_start])
        projected = pl.create_tensor([PROJ_M_TILE, Q_LORA], dtype=pl.BF16)
        if group_rows > 0:
            qa_projection(group_payload, wq_a, group_scales, wq_a_scale, projected, group_rows)
        qb_payload_group = pl.slice(qb_payload, [PROJ_M_TILE, Q_LORA], [group_start, 0])
        qb_scale_start = group_start * (Q_LORA // MX_GROUP)
        qb_scales_group = pl.slice(qb_scales, [1, PROJ_M_TILE * (Q_LORA // MX_GROUP)], [0, qb_scale_start])
        latent_start = pl.min(group_start, num_tokens)
        latent_group = pl.slice(query_latent, [PROJ_M_TILE, Q_LORA], [latent_start, 0], valid_shape=[group_rows, Q_LORA])
        normalize_quant(projected, q_norm_weight, latent_group, qb_payload_group, qb_scales_group, group_rows)
        if group_rows > 0:
            raw_query = pl.create_tensor([PROJ_M_TILE, LOCAL_H * HEAD_DIM], dtype=pl.BF16)
            qb_projection(qb_payload_group, wq_b, qb_scales_group, wq_b_scale, raw_query, group_rows)
            cos_group = pl.slice(rope_cos, [PROJ_M_TILE, ROPE_DIM // 2], [group_start, 0],
                                 valid_shape=[group_rows, ROPE_DIM // 2])
            sin_group = pl.slice(rope_sin, [PROJ_M_TILE, ROPE_DIM // 2], [group_start, 0],
                                 valid_shape=[group_rows, ROPE_DIM // 2])
            query_group = pl.slice(query, [PROJ_M_TILE, LOCAL_H * HEAD_DIM], [group_start, 0],
                                   valid_shape=[group_rows, LOCAL_H * HEAD_DIM])
            query_rope(raw_query, cos_group, sin_group, query_group, group_rows)
    return query_latent, query, qb_payload, qb_scales


@pl.jit
def q_norm_validation(
    source: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    weight: pl.Tensor[[Q_LORA], pl.BF16],
    latent: pl.Out[pl.Tensor[[T_DYN, Q_LORA], pl.BF16]],
    payload: pl.Out[pl.Tensor[[Q_PAYLOAD_ROWS, Q_LORA], pl.FP8E4M3FN]],
    scales: pl.Out[pl.Tensor[[1, Q_SCALE_COLUMNS], pl.FP8E8M0]],
    num_tokens: pl.Scalar[pl.INT32],
):
    normalize_quant(source, weight, latent, payload, scales, num_tokens)
    return latent, payload, scales


@pl.jit
def qb_rope_validation(
    latent: pl.Tensor[[T_DYN, Q_LORA], pl.BF16],
    weight: pl.Tensor[[Q_LORA, LOCAL_H * HEAD_DIM], pl.FP8E4M3FN],
    weight_scale: pl.Tensor[[Q_LORA // MX_GROUP, LOCAL_H * HEAD_DIM], pl.FP8E8M0, pl.MX_B_NN],
    cosine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    sine: pl.Tensor[[T_DYN, ROPE_DIM // 2], pl.FP32],
    payload: pl.Out[pl.Tensor[[Q_PAYLOAD_ROWS, Q_LORA], pl.FP8E4M3FN]],
    scales: pl.Out[pl.Tensor[[1, Q_SCALE_COLUMNS], pl.FP8E8M0]],
    projected: pl.Out[pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16]],
    query: pl.Out[pl.Tensor[[T_DYN, LOCAL_H * HEAD_DIM], pl.BF16]],
    num_tokens: pl.Scalar[pl.INT32],
):
    quantize_matrix(latent, payload, scales, num_tokens)
    for start in pl.parallel(0, num_tokens, PROJ_M_TILE):
        rows = pl.min(PROJ_M_TILE, num_tokens - start)
        source = pl.slice(payload, [PROJ_M_TILE, Q_LORA], [start, 0])
        factors = pl.slice(scales, [1, PROJ_M_TILE * (Q_LORA // MX_GROUP)], [0, start * (Q_LORA // MX_GROUP)])
        raw = pl.slice(projected, [PROJ_M_TILE, LOCAL_H * HEAD_DIM], [start, 0],
                       valid_shape=[rows, LOCAL_H * HEAD_DIM])
        qb_projection(source, weight, factors, weight_scale, raw, rows)
        cos_tile = pl.slice(cosine, [PROJ_M_TILE, ROPE_DIM // 2], [start, 0], valid_shape=[rows, ROPE_DIM // 2])
        sin_tile = pl.slice(sine, [PROJ_M_TILE, ROPE_DIM // 2], [start, 0], valid_shape=[rows, ROPE_DIM // 2])
        dst = pl.slice(query, [PROJ_M_TILE, LOCAL_H * HEAD_DIM], [start, 0], valid_shape=[rows, LOCAL_H * HEAD_DIM])
        query_rope(raw, cos_tile, sin_tile, dst, rows)
    return payload, scales, projected, query


def golden_rope_interleave(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply adjacent-pair rotary embedding to the final dimension."""
    value = x.float().unflatten(-1, (-1, 2))
    real, imag = value.unbind(dim=-1)
    if inverse:
        sin = -sin
    while cos.ndim < real.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
    rotated = torch.stack((real * cos - imag * sin, imag * cos + real * sin), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def golden_rope_tail(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    """Rotate the trailing 2*cos.shape[-1] lanes as adjacent (real, imag) pairs."""
    rope_dim = cos.shape[-1] * 2
    rotated = golden_rope_interleave(x[..., -rope_dim:], cos, sin, inverse)
    return torch.cat((x[..., :-rope_dim], rotated), dim=-1)


def golden_qkv_proj_rope(
    x: torch.Tensor,
    wq_a: torch.Tensor,
    wq_a_scale: torch.Tensor | None,
    q_norm_weight: torch.Tensor,
    wq_b: torch.Tensor,
    wq_b_scale: torch.Tensor | None,
    wkv: torch.Tensor,
    wkv_scale: torch.Tensor | None,
    kv_norm_weight: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    sequential_query: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project normalized query latent and shared KV, then rotate their RoPE tails."""
    from quant import unpack_fp8_nz

    if wq_a.dtype == torch.float8_e4m3fn:
        wq_a = unpack_fp8_nz(wq_a)
    if wq_b.dtype == torch.float8_e4m3fn:
        wq_b = unpack_fp8_nz(wq_b)
    if wkv.dtype == torch.float8_e4m3fn:
        wkv = unpack_fp8_nz(wkv)

    def project(value, weight, scale):
        if sequential_query and scale is not None:
            return golden_gemm_bf16_via_mxfp8_to_bf16_sequential(value, weight, scale)
        return golden_mxfp8_linear_grouped_sum(value, weight, scale)

    if wq_a_scale is not None:
        projected = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(x, wq_a, wq_a_scale)
    else:
        projected = golden_mxfp8_linear_grouped_sum(x, wq_a, wq_a_scale)
    qr = golden_rms_norm(projected, q_norm_weight)
    head_dim = wkv.shape[-1]
    num_heads = wq_b.shape[-1] // head_dim
    q = project(qr, wq_b, wq_b_scale).unflatten(-1, (num_heads, head_dim))
    kv = golden_rms_norm(project(x, wkv, wkv_scale), kv_norm_weight)
    q = golden_rope_tail(q, cos, sin)
    kv = golden_rope_tail(kv, cos, sin)
    return q, kv, qr


def _golden_q_payload(tensors, values):
    from quant import _quantize_mxfp8_activation, encode_e8m0, pack_mx_a_scale

    tensors["payload"].zero_()
    tensors["scales"].view(torch.uint8).zero_()
    active = values.shape[0]
    if active == 0:
        return
    payload, scales = _quantize_mxfp8_activation(values, scale_alg=0)
    tensors["payload"][:active].copy_(payload.flatten(-2))
    codes = torch.zeros(tensors["payload"].shape[0], Q_LORA // MX_GROUP, dtype=torch.uint8)
    codes[:active].copy_(encode_e8m0(scales))
    tensors["scales"].view(torch.uint8).copy_(pack_mx_a_scale(codes).reshape(1, -1))


def golden_q_norm(tensors):
    active = int(tensors["num_tokens"])
    rounded = golden_rms_norm(tensors["source"][:active], tensors["weight"])
    tensors["latent"][:active].copy_(rounded)
    _golden_q_payload(tensors, rounded)


def golden_qb_rope(tensors):
    from quant import unpack_fp8_nz

    active = int(tensors["num_tokens"])
    latent = tensors["latent"][:active]
    _golden_q_payload(tensors, latent)
    if active == 0:
        return
    raw = golden_gemm_bf16_via_mxfp8_to_bf16_sequential(
        latent, unpack_fp8_nz(tensors["weight"]), tensors["weight_scale"],
    )
    tensors["projected"][:active].copy_(raw)
    query = golden_rope_tail(raw.unflatten(-1, (LOCAL_H, HEAD_DIM)), tensors["cosine"][:active], tensors["sine"][:active])
    tensors["query"][:active].copy_(query.flatten(-2))


def build_q_branch_specs(stage, active):
    from golden import ScalarSpec, TensorSpec
    from quant import generate_mxfp8_weight_kn, pack_fp8_nz

    capacity = ((max(active, 1) + ROW_TILE - 1) // ROW_TILE) * ROW_TILE
    padded = ((capacity + PADDED_TILE - 1) // PADDED_TILE) * PADDED_TILE
    generator = torch.Generator().manual_seed(41121)
    values = torch.randn(capacity, Q_LORA, generator=generator).bfloat16()
    values[active:] = float("nan")
    if active > 0:
        values[0] = 0
    if active > 1:
        values[1] = 1.0e-5
    shared = [
        TensorSpec("payload", [padded, Q_LORA], torch.float8_e4m3fn),
        TensorSpec("scales", [1, padded * (Q_LORA // MX_GROUP)], torch.float8_e8m0fnu),
    ]
    if stage == "norm":
        gamma = (1 + 0.1 * torch.randn(Q_LORA, generator=generator)).bfloat16()
        latent = torch.full((capacity, Q_LORA), 1.75).bfloat16()
        specs = [
            TensorSpec("source", list(values.shape), values.dtype, init_value=values),
            TensorSpec("weight", [Q_LORA], gamma.dtype, init_value=gamma),
            TensorSpec("latent", list(latent.shape), latent.dtype, init_value=latent),
            *shared,
        ]
    else:
        width = LOCAL_H * HEAD_DIM
        weight, scales = generate_mxfp8_weight_kn(width, Q_LORA, Q_LORA ** -0.5, seed=41122)
        weight = pack_fp8_nz(weight)
        angles = torch.randn(capacity, ROPE_DIM // 2, generator=generator)
        raw = torch.full((capacity, width), 1.75).bfloat16()
        query = torch.full((capacity, width), -1.25).bfloat16()
        specs = [
            TensorSpec("latent", list(values.shape), values.dtype, init_value=values),
            TensorSpec("weight", list(weight.shape), weight.dtype, init_value=weight),
            TensorSpec("weight_scale", list(scales.shape), scales.dtype, init_value=scales),
            TensorSpec("cosine", list(angles.shape), torch.float32, init_value=angles.cos()),
            TensorSpec("sine", list(angles.shape), torch.float32, init_value=angles.sin()),
            *shared,
            TensorSpec("projected", list(raw.shape), raw.dtype, init_value=raw),
            TensorSpec("query", list(query.shape), query.dtype, init_value=query),
        ]
    return [*specs, ScalarSpec("num_tokens", torch.int32, active)]


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run
    from window_kv_dsl import compare_payload_bytes

    parser = argparse.ArgumentParser(description="Validate DSL Q RMSNorm/quantization and QB/RoPE stages on A5.")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--stage", choices=["all", "norm", "qb"], default="all")
    parser.add_argument("--case", choices=["all", "empty", "tail", "multi"], default="all")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    for stage in ("norm", "qb"):
        if args.stage not in ("all", stage):
            continue
        for case, active in (("empty", 0), ("tail", 17), ("multi", 81)):
            if args.case not in ("all", case):
                continue
            print(f"[Q_BRANCH] {stage} {case}: active={active}", flush=True)
            if stage == "norm":
                active_compare = ratio_allclose(atol=1e-6, rtol=1.0 / 128, max_error_ratio=0.0)
            else:
                active_compare = ratio_allclose(atol=1e-2, rtol=1.0 / 128)

            def compare_output(actual, expected, **kwargs):
                if active == 0:
                    unchanged = torch.equal(actual, expected)
                    return unchanged, f"inactive output unchanged={unchanged}"
                passed, detail = active_compare(actual[:active], expected[:active], **kwargs)
                unchanged = torch.equal(actual[active:], expected[active:])
                return passed and unchanged, f"{detail}; inactive output unchanged={unchanged}"

            comparators = {"payload": compare_payload_bytes, "scales": compare_payload_bytes}
            for name in ("latent",) if stage == "norm" else ("projected", "query"):
                comparators[name] = compare_output
            result = run(
                fn=q_norm_validation if stage == "norm" else qb_rope_validation,
                specs=build_q_branch_specs(stage, active),
                golden_fn=golden_q_norm if stage == "norm" else golden_qb_rope,
                compare_fn=comparators, config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
                compile_only=args.compile_only, save_data=True,
            )
            if not result.passed:
                raise SystemExit(result.error or 1)
