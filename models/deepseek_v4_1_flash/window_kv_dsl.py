# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 实现窗口 KV 的 RMSNorm、相邻元素对 RoPE、FP8 量化和缓存发布。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
import pypto.language as pl
import torch

from config import FLASH as M


# 模型配置
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
EPS = M.rms_norm_eps
CACHE_GROUP = 32
PAGE_SIZE = 128

# 分块参数
VECTOR_TILE = 64
PUBLISH_TASK_TILE = 12


@cb.kernel
def window_kv_page(
    source: cb.Tensor,
    weight: cb.Tensor,
    cosine: cb.Tensor,
    sine: cb.Tensor,
    slots: cb.Tensor,
    cache: cb.Tensor,
    scales: cb.Tensor,
    num_tokens: cb.ScalarValue,
    page: cb.ScalarValue,
):
    """发布一个物理页；同一槽位的重复写入按 token 序号递增执行。"""
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, HEAD_DIM)))
    gamma = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((HEAD_DIM,)))
    cos_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
    sin_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
    values = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((HEAD_DIM + VECTOR_TILE,)))
    packed = Buffer(cb.MemLoc.UB, cb.dtypes.float8_e4m3fn, cb.make_layout((1, HEAD_DIM)))
    factors = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, HEAD_DIM // CACHE_GROUP)))
    cb.mem_copy(gamma, weight)
    for token in cb.range(min(num_tokens, min(source.shape[0], slots.shape[0]))):
        slot = slots[token]
        if slot >= 0 and slot // PAGE_SIZE == page:
            cb.mem_copy(raw, cb.tile_slice(source, (1, HEAD_DIM), (token, 0)))
            cos_dst = cb.tile_slice(cos_ub, (1, ROPE_DIM // 2), (0, 0))
            sin_dst = cb.tile_slice(sin_ub, (1, ROPE_DIM // 2), (0, 0))
            cb.mem_copy(cos_dst, cb.tile_slice(cosine, (1, ROPE_DIM // 2), (token, 0)))
            cb.mem_copy(sin_dst, cb.tile_slice(sine, (1, ROPE_DIM // 2), (token, 0)))
            with cb.vf():
                full = rr.full_mask()
                half = rr.update_mask(CACHE_GROUP, elem_bits=32)[0]
                accum = rr.vdups(0.0, cb.dtypes.float32, mask=full)
                for col in range(0, HEAD_DIM, VECTOR_TILE):
                    loaded = rr.vload_unpack(raw, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    x = rr.vcast(loaded, cb.dtypes.float32, mask=full)
                    squared = rr.vmul(x, x, mask=full)
                    accum = rr.vadd(accum, squared, mask=full)
                total = rr.vreduce_sum(accum, mask=full)
                mean = rr.vmuls(total, 1.0 / HEAD_DIM, mask=full)
                denominator = rr.vsqrt(rr.vadds(mean, EPS, mask=full), mask=full)
                one = rr.vdups(1.0, cb.dtypes.float32, mask=full)
                # 倒数的 1 ULP 误差可能同时跨过 BF16 和 FP8 的舍入边界。
                inverse = rr.vdup(rr.vdiv(one, denominator, mask=full, precision_mode=True), mask=full)
                for col in range(0, HEAD_DIM, VECTOR_TILE):
                    x16 = rr.vload_unpack(raw, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    g16 = rr.vload_unpack(gamma, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    x = rr.vcast(x16, cb.dtypes.float32, mask=full)
                    g = rr.vcast(g16, cb.dtypes.float32, mask=full)
                    normalized = rr.vmul(rr.vmul(x, inverse, mask=full), g, mask=full)
                    rounded = rr.vcast(normalized, cb.dtypes.bfloat16, mask=full, rounding="rn")
                    value = rr.vcast(rounded, cb.dtypes.float32, mask=full)
                    rr.vstore(values, col, value, full)
                rr.vmem_bar("vst_vld")
                real, imag = rr.vload_deinterleave(values, HEAD_DIM - ROPE_DIM, width="b32")
                cos = rr.vload(cos_ub)
                sin = rr.vload(sin_ub)
                real_cos = rr.vmul(real, cos, mask=half)
                imag_sin = rr.vmul(imag, sin, mask=half)
                real_sin = rr.vmul(real, sin, mask=half)
                imag_cos = rr.vmul(imag, cos, mask=half)
                real_rotated = rr.vsub(real_cos, imag_sin, mask=half)
                imag_rotated = rr.vadd(real_sin, imag_cos, mask=half)
                real_bf16 = rr.vcast(real_rotated, cb.dtypes.bfloat16, mask=half, rounding="rn")
                imag_bf16 = rr.vcast(imag_rotated, cb.dtypes.bfloat16, mask=half, rounding="rn")
                real_rounded = rr.vcast(real_bf16, cb.dtypes.float32, mask=half)
                imag_rounded = rr.vcast(imag_bf16, cb.dtypes.float32, mask=half)
                rr.vstore_interleave(values, HEAD_DIM - ROPE_DIM, real_rounded, imag_rounded, width="b32")
                rr.vmem_bar("vst_vld")
                for group in range(HEAD_DIM // CACHE_GROUP):
                    value = rr.vload(values, group * CACHE_GROUP)
                    maximum = rr.vreduce_max(rr.vabs(value, mask=half), mask=half)
                    scale = rr.vmuls(rr.vmaxs(maximum, 1.0e-4, mask=half), 1.0 / 448.0, mask=half)
                    scale_bits = rr.vreinterpret(scale, cb.dtypes.uint32)
                    rounded_bits = rr.vadds(scale_bits, 0x7FFFFF, mask=half)
                    exponent_mask = rr.vdups(0x7F800000, cb.dtypes.uint32, mask=half)
                    power_bits = rr.vbitwise_and(rounded_bits, exponent_mask, mask=half)
                    scale = rr.vdup(rr.vreinterpret(power_bits, cb.dtypes.float32), mask=half)
                    scale_bf16 = rr.vcast(scale, cb.dtypes.bfloat16, mask=half, rounding="rn")
                    rr.vstore_first(factors, group, scale_bf16)
                    quantized = rr.vdiv(value, scale, mask=half, precision_mode=True)
                    quantized = rr.vmins(rr.vmaxs(quantized, -448.0, mask=half), 448.0, mask=half)
                    code = rr.vcast(quantized, cb.dtypes.float8_e4m3fn, mask=half, rounding="rn")
                    rr.vstore_pack(packed, group * CACHE_GROUP, code, half, pack_mode=rr.PackMode.B32_TO_B8)
            cb.mem_copy(cb.tile_slice(cache, (1, HEAD_DIM), (slot % PAGE_SIZE, 0)), packed)
            cb.mem_copy(cb.tile_slice(scales, (1, HEAD_DIM // CACHE_GROUP), (slot % PAGE_SIZE, 0)), factors)


@pl.jit.inline(auto_scope=False)
def window_publish(
    source: pl.Tensor,
    weight: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.Tensor,
    scales: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
):
    """调度互不重叠的物理缓存页，并汇合各页的发布任务。"""
    pages = pl.tensor.dim(cache, 0)
    cache_rows = pl.reshape(cache, [pages, PAGE_SIZE, HEAD_DIM])
    scale_rows = pl.reshape(scales, [pages, PAGE_SIZE, HEAD_DIM // CACHE_GROUP])
    completed = pl.array.create(PUBLISH_TASK_TILE, pl.TASK_ID)
    for worker in pl.range(PUBLISH_TASK_TILE):
        completed[worker] = cache_ready
    for page in pl.parallel(pages):
        payload_page = pl.slice(cache_rows, [1, PAGE_SIZE, HEAD_DIM], [page, 0, 0])
        payload = pl.reshape(payload_page, [PAGE_SIZE, HEAD_DIM])
        factor_page = pl.slice(scale_rows, [1, PAGE_SIZE, HEAD_DIM // CACHE_GROUP], [page, 0, 0])
        factors = pl.reshape(factor_page, [PAGE_SIZE, HEAD_DIM // CACHE_GROUP])
        previous = completed[page % PUBLISH_TASK_TILE]
        with pl.spmd(1, name_hint="dsl_window_kv", deps=[previous], sync_start=True) as done:
            window_kv_page(source, weight, cosine, sine, slots, payload, factors, num_tokens, page)
        completed[page % PUBLISH_TASK_TILE] = done
    published = pl.system.task_dummy(deps=[completed])
    return published


@pl.jit
def window_kv_program(
    source: pl.Tensor,
    weight: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.InOut[pl.Tensor],
    scales: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
):
    ready = pl.system.task_dummy(deps=[])
    window_publish(source, weight, cosine, sine, slots, cache, scales, num_tokens, ready)
    return cache, scales


@pl.jit
def window_kv_replay_program(
    source: pl.Tensor,
    weight: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.InOut[pl.Tensor],
    scales: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
    snapshots: pl.Out[pl.Tensor],
    scale_snapshots: pl.Out[pl.Tensor],
):
    """在下一次 decode 覆盖缓存前，等待所有缓存更新被消费。"""
    steps = pl.tensor.dim(source, 0)
    tokens = pl.tensor.dim(source, 1)
    rows = pl.tensor.dim(cache, 0) * PAGE_SIZE
    cache_rows = pl.reshape(cache, [rows, HEAD_DIM])
    factor_rows = pl.reshape(scales, [rows, HEAD_DIM // CACHE_GROUP])
    ready = pl.system.task_dummy(deps=[])
    for step in pl.range(steps):
        source_slice = pl.slice(source, [1, tokens, HEAD_DIM], [step, 0, 0])
        source_step = pl.reshape(source_slice, [tokens, HEAD_DIM])
        cos_slice = pl.slice(cosine, [1, tokens, ROPE_DIM // 2], [step, 0, 0])
        cos_step = pl.reshape(cos_slice, [tokens, ROPE_DIM // 2])
        sin_slice = pl.slice(sine, [1, tokens, ROPE_DIM // 2], [step, 0, 0])
        sin_step = pl.reshape(sin_slice, [tokens, ROPE_DIM // 2])
        slots_slice = pl.slice(slots, [1, tokens], [step, 0])
        slots_step = pl.reshape(slots_slice, [tokens])
        published = window_publish(
            source_step, weight, cos_step, sin_step,
            slots_step, cache, scales, num_tokens, ready,
        )
        snapshot_slice = pl.slice(snapshots, [1, rows, HEAD_DIM], [step, 0, 0])
        snapshot = pl.reshape(snapshot_slice, [rows, HEAD_DIM])
        scales_slice = pl.slice(scale_snapshots, [1, rows, HEAD_DIM // CACHE_GROUP], [step, 0, 0])
        scale_snapshot = pl.reshape(scales_slice, [rows, HEAD_DIM // CACHE_GROUP])
        with pl.at(level=pl.Level.CORE_GROUP, name_hint="consume_window_kv", deps=[published]) as consumed:
            for row in pl.range(rows):
                payload = pl.load(cache_rows, [row, 0], [1, HEAD_DIM])
                snapshot = pl.store(payload, [row, 0], snapshot)
                factor = pl.load(factor_rows, [row, 0], [1, HEAD_DIM // CACHE_GROUP])
                scale_snapshot = pl.store(factor, [row, 0], scale_snapshot)
        ready = consumed
    return cache, scales, snapshots, scale_snapshots


def quantize_window_kv(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """按每组 32 个元素编码 E4M3，使用以 BF16 存储的 2 的幂次缩放值。"""
    grouped = value.float().unflatten(-1, (-1, CACHE_GROUP))
    amax = grouped.abs().amax(dim=-1).clamp_min(1.0e-4)
    raw_scale = amax * (1.0 / 448.0)
    scale = torch.exp2(torch.ceil(torch.log2(raw_scale.double()))).bfloat16()
    normalized = (grouped / scale.float().unsqueeze(-1)).clamp(-448.0, 448.0)
    return normalized.flatten(-2).to(torch.float8_e4m3fn), scale


def golden_window_kv(tensors):
    source = tensors["source"].float()
    squared = source.square().unflatten(-1, (HEAD_DIM // VECTOR_TILE, VECTOR_TILE))
    partial = squared[:, 0].clone()
    for chunk in range(1, HEAD_DIM // VECTOR_TILE):
        partial = partial + squared[:, chunk]
    while partial.shape[-1] > 1:
        partial = partial[..., 0::2] + partial[..., 1::2]
    inverse = torch.sqrt(partial / HEAD_DIM + EPS).reciprocal()
    normalized = (source * inverse * tensors["weight"].float()).bfloat16()
    tail = normalized[:, -ROPE_DIM:].float().unflatten(-1, (ROPE_DIM // 2, 2))
    real, imag = tail.unbind(dim=-1)
    cosine, sine = tensors["cosine"], tensors["sine"]
    rotated = torch.stack((real * cosine - imag * sine, real * sine + imag * cosine), dim=-1)
    normalized[:, -ROPE_DIM:] = rotated.flatten(-2).bfloat16()
    payload, scale = quantize_window_kv(normalized)
    count = max(0, min(int(tensors["num_tokens"]), source.shape[0], tensors["slots"].numel()))
    cache_rows = tensors["cache"].view(torch.uint8).reshape(-1, HEAD_DIM)
    scale_rows = tensors["scales"].reshape(-1, HEAD_DIM // CACHE_GROUP)
    for token in range(count):
        slot = int(tensors["slots"][token])
        if 0 <= slot < cache_rows.shape[0]:
            cache_rows[slot] = payload[token].view(torch.uint8)
            scale_rows[slot] = scale[token]


def golden_window_kv_replay(tensors):
    for step in range(tensors["source"].shape[0]):
        current = dict(tensors)
        for name in ("source", "cosine", "sine", "slots"):
            current[name] = tensors[name][step]
        golden_window_kv(current)
        tensors["snapshots"][step].copy_(tensors["cache"].reshape(-1, HEAD_DIM))
        tensors["scale_snapshots"][step].copy_(tensors["scales"].reshape(-1, HEAD_DIM // CACHE_GROUP))


def compare_payload_bytes(actual, expected, **context):
    from golden import ratio_allclose

    exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    return exact(actual.view(torch.uint8), expected.view(torch.uint8), **context)


def build_tensor_specs(tokens=8, pages=3, num_tokens=None, seed=0):
    from golden import ScalarSpec, TensorSpec

    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(tokens, HEAD_DIM, generator=generator).bfloat16()
    source[0].zero_()
    weight = (torch.rand(HEAD_DIM, generator=generator) + 0.5).bfloat16()
    angles = torch.randn(tokens, ROPE_DIM // 2, generator=generator)
    targets = torch.tensor([0, 127, 128, -1, 0, pages * PAGE_SIZE, pages * PAGE_SIZE - 1, 128])
    slots = targets.repeat((tokens + targets.numel() - 1) // targets.numel())[:tokens]
    cache = torch.full((pages, PAGE_SIZE, 1, HEAD_DIM), 0.5).to(torch.float8_e4m3fn)
    scales = torch.full((pages, PAGE_SIZE, 1, HEAD_DIM // CACHE_GROUP), 0.125, dtype=torch.bfloat16)
    return [
        TensorSpec("source", list(source.shape), source.dtype, init_value=source),
        TensorSpec("weight", list(weight.shape), weight.dtype, init_value=weight),
        TensorSpec("cosine", list(angles.shape), torch.float32, init_value=angles.cos()),
        TensorSpec("sine", list(angles.shape), torch.float32, init_value=angles.sin()),
        TensorSpec("slots", [tokens], torch.int64, init_value=slots),
        TensorSpec("cache", list(cache.shape), cache.dtype, init_value=cache),
        TensorSpec("scales", list(scales.shape), scales.dtype, init_value=scales),
        ScalarSpec("num_tokens", torch.int32, value=tokens if num_tokens is None else num_tokens),
    ]


def build_replay_specs(tokens=8, pages=3, num_tokens=None, seed=0):
    from dataclasses import replace
    from golden import TensorSpec

    specs = build_tensor_specs(tokens, pages, num_tokens, seed)
    for index, spec in enumerate(specs):
        if spec.name == "source":
            value = torch.stack((spec.init_value, (spec.init_value.float() * -0.5 + 0.25).bfloat16()))
        elif spec.name in ("cosine", "sine"):
            value = torch.stack((spec.init_value, spec.init_value.flip(0)))
        elif spec.name == "slots":
            next_slots = spec.init_value.clone()
            valid = (next_slots >= 0) & (next_slots < pages * PAGE_SIZE)
            next_slots[valid] = (next_slots[valid] + 1) % (pages * PAGE_SIZE)
            value = torch.stack((spec.init_value, next_slots))
        else:
            continue
        specs[index] = replace(spec, shape=list(value.shape), init_value=value)
    specs.extend([
        TensorSpec("snapshots", [2, pages * PAGE_SIZE, HEAD_DIM], torch.float8_e4m3fn),
        TensorSpec("scale_snapshots", [2, pages * PAGE_SIZE, HEAD_DIM // CACHE_GROUP], torch.bfloat16),
    ])
    return specs


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="Validate DSL window KV publication on Ascend 950.")
    parser.add_argument("-p", "--platform", choices=["a5"], default="a5")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--num-tokens", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    if args.tokens < 1 or args.pages < 1:
        parser.error("--tokens and --pages must be positive")
    build_specs = build_replay_specs if args.replay else build_tensor_specs
    comparators = {"cache": compare_payload_bytes}
    if args.replay:
        comparators["snapshots"] = compare_payload_bytes
    result = run(
        fn=window_kv_replay_program if args.replay else window_kv_program,
        specs=build_specs(args.tokens, args.pages, args.num_tokens, args.seed),
        golden_fn=golden_window_kv_replay if args.replay else golden_window_kv,
        compare_fn=comparators,
        compile_only=args.compile_only,
        config=dict(backend_type=args.platform, execution_mode=False, device_id=args.device),
        rtol=0.0,
        atol=0.0,
    )
    if not result.passed:
        raise SystemExit(result.error or 1)
