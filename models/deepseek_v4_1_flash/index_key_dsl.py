# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""使用 CANNBotDSL 归一化、旋转并发布带 group-32 E8M0 缩放值的 FP4 Indexer Key。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
import pypto.language as pl

from config import FLASH as M, INDEX_CACHE_GROUP


# 模型配置
INDEX_DIM = M.index_head_dim
ROPE_DIM = M.qk_rope_head_dim
EPS = M.rms_norm_eps
CACHE_GROUP = INDEX_CACHE_GROUP
PAGE_SIZE = 128

# 分块参数
VECTOR_TILE = 64
TASK_TILE = 12


@cb.kernel
def publish_index_page(
    source: cb.Tensor, weight: cb.Tensor, cosine: cb.Tensor, sine: cb.Tensor, slots: cb.Tensor,
    cache: cb.Tensor, scales: cb.Tensor, num_tokens: cb.ScalarValue, page: cb.ScalarValue,
):
    """按 token 顺序更新一个物理页，最后整页写回 E8M0 缩放字节。"""
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, INDEX_DIM)))
    gamma = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((INDEX_DIM,)))
    cos_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
    sin_ub = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, VECTOR_TILE)))
    values = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((INDEX_DIM + VECTOR_TILE,)))
    packed = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, INDEX_DIM // 2)))
    factors = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((1, PAGE_SIZE * (INDEX_DIM // CACHE_GROUP))))
    scale_bytes = scales.view(dtype=cb.dtypes.uint8).view(1, PAGE_SIZE * (INDEX_DIM // CACHE_GROUP))
    cb.mem_copy(gamma, weight)
    cb.mem_copy(factors, scale_bytes)
    for token in cb.range(num_tokens):
        slot = slots[token]
        if slot >= 0 and slot // PAGE_SIZE == page:
            cb.mem_copy(raw, cb.tile_slice(source, (1, INDEX_DIM), (token, 0)))
            cos_dst = cb.tile_slice(cos_ub, (1, ROPE_DIM // 2), (0, 0))
            sin_dst = cb.tile_slice(sin_ub, (1, ROPE_DIM // 2), (0, 0))
            cb.mem_copy(cos_dst, cb.tile_slice(cosine, (1, ROPE_DIM // 2), (token, 0)))
            cb.mem_copy(sin_dst, cb.tile_slice(sine, (1, ROPE_DIM // 2), (token, 0)))
            with cb.vf():
                full = rr.full_mask()
                group_mask = rr.update_mask(CACHE_GROUP, elem_bits=32)[0]
                pair_mask = rr.update_mask(VECTOR_TILE // 2, elem_bits=32)[0]
                # 两个 64 元素段先分别归约，再相加。
                first16 = rr.vload_unpack(raw, 0, unpack_mode=rr.UnpackMode.B16_TO_B32)
                second16 = rr.vload_unpack(raw, VECTOR_TILE, unpack_mode=rr.UnpackMode.B16_TO_B32)
                first = rr.vcast(first16, cb.dtypes.float32, mask=full)
                second = rr.vcast(second16, cb.dtypes.float32, mask=full)
                square0 = rr.vmul(first, first, mask=full)
                square1 = rr.vmul(second, second, mask=full)
                sum0 = rr.vreduce_sum(square0, mask=full)
                sum1 = rr.vreduce_sum(square1, mask=full)
                total = rr.vadd(sum0, sum1, mask=full)
                mean = rr.vmuls(total, 1.0 / INDEX_DIM, mask=full)
                denominator = rr.vsqrt(rr.vadds(mean, EPS, mask=full), mask=full)
                one = rr.vdups(1.0, cb.dtypes.float32, mask=full)
                inverse = rr.vdup(rr.vdiv(one, denominator, mask=full), mask=full)
                for col in range(0, INDEX_DIM, VECTOR_TILE):
                    x16 = rr.vload_unpack(raw, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    g16 = rr.vload_unpack(gamma, col, unpack_mode=rr.UnpackMode.B16_TO_B32)
                    x = rr.vcast(x16, cb.dtypes.float32, mask=full)
                    g = rr.vcast(g16, cb.dtypes.float32, mask=full)
                    normalized = rr.vmul(rr.vmul(x, inverse, mask=full), g, mask=full)
                    rounded = rr.vcast(normalized, cb.dtypes.bfloat16, mask=full, rounding="rn")
                    rr.vstore(values, col, rr.vcast(rounded, cb.dtypes.float32, mask=full), full)
                rr.vmem_bar("vst_vld")
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
                    offset = (slot % PAGE_SIZE) * (INDEX_DIM // CACHE_GROUP) + group
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
            cb.mem_copy(cb.tile_slice(cache, (1, INDEX_DIM // 2), (slot % PAGE_SIZE, 0)), packed)
    # 页内保留未选中的缩放字节，一次对齐写回避免短行写入互相覆盖。
    cb.mem_copy(scale_bytes, factors)


@pl.jit.inline(auto_scope=False)
def _publish_task(
    source: pl.Tensor, weight: pl.Tensor, cosine: pl.Tensor, sine: pl.Tensor, slots: pl.Tensor,
    cache: pl.Tensor, scales: pl.Tensor, active: pl.Scalar[pl.INDEX], page: pl.Scalar[pl.INDEX],
    ready: pl.Scalar[pl.TASK_ID],
):
    """在输入和缓存依赖完成后，提交一个物理页的本卡 AIV 任务。"""
    with pl.spmd(1, name_hint="dsl_idx_k_norm_rope_publish", deps=[ready]) as done:
        publish_index_page(source, weight, cosine, sine, slots, cache, scales, active, page)
    return done


@pl.jit.inline(auto_scope=False)
def publish_index_rows(
    source: pl.Tensor, weight: pl.Tensor, cosine: pl.Tensor, sine: pl.Tensor, slots: pl.Tensor,
    cache: pl.Tensor, scales: pl.Tensor, num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID], kv_ready: pl.Scalar[pl.TASK_ID],
):
    """由 simpler 调度互不重叠的物理页，并汇合全部缓存发布任务。"""
    active = pl.min(pl.tensor.dim(source, 0), pl.max(pl.cast(num_tokens, pl.INDEX), 0))
    active = pl.min(active, pl.tensor.dim(slots, 0))
    ready = pl.system.task_dummy(deps=[cache_ready, kv_ready])
    completed = pl.array.create(TASK_TILE, pl.TASK_ID)
    for slot in pl.range(TASK_TILE):
        completed[slot] = ready
    pages = pl.tensor.dim(cache, 0)
    payload_rows = pl.reshape(cache, [pages * PAGE_SIZE, INDEX_DIM // 2])
    scale_rows = pl.reshape(scales, [pages * PAGE_SIZE, INDEX_DIM // CACHE_GROUP])
    for page in pl.parallel(pages):
        payload = payload_rows[page * PAGE_SIZE:(page + 1) * PAGE_SIZE, :]
        factors = scale_rows[page * PAGE_SIZE:(page + 1) * PAGE_SIZE, :]
        previous = completed[page % TASK_TILE]
        done = _publish_task(source, weight, cosine, sine, slots, payload, factors, active, page, previous)
        completed[page % TASK_TILE] = done
    published = pl.system.task_dummy(deps=[completed])
    return published


@pl.jit
def validate_publication(
    source: pl.Tensor, weight: pl.Tensor, cosine: pl.Tensor, sine: pl.Tensor, slots: pl.Tensor,
    cache: pl.InOut[pl.Tensor], scales: pl.InOut[pl.Tensor], num_tokens: pl.Scalar[pl.INT32],
):
    """通过生产调度入口校验有效缓存和未选中区域。"""
    ready = pl.system.task_dummy(deps=[])
    publish_index_rows(source, weight, cosine, sine, slots, cache, scales, num_tokens, ready, ready)
    return cache, scales


def golden_publication(tensors):
    """独立计算两段 RMSNorm、相邻对 RoPE 及最近偶数 FP4 量化。"""
    import torch
    from quant import quantize_mxfp4_cache

    source = tensors["source"].float()
    partial = source.square().unflatten(-1, (INDEX_DIM // VECTOR_TILE, VECTOR_TILE))
    while partial.shape[-1] > 1:
        partial = partial[..., 0::2] + partial[..., 1::2]
    total = partial[:, 0, :] + partial[:, 1, :]
    inverse = torch.sqrt(total / INDEX_DIM + EPS).reciprocal()
    normalized = (source * inverse * tensors["weight"].float()).bfloat16()
    real, imag = normalized[:, -ROPE_DIM:].float().unflatten(-1, (ROPE_DIM // 2, 2)).unbind(-1)
    cos, sin = tensors["cosine"], tensors["sine"]
    rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1)
    normalized[:, -ROPE_DIM:] = rotated.flatten(-2).bfloat16()
    payload, codes = quantize_mxfp4_cache(normalized, CACHE_GROUP, "e8m0", round_to_even=True)
    cache = tensors["cache"].reshape(-1, INDEX_DIM // 2)
    scales = tensors["scales"].view(torch.uint8).reshape(-1, INDEX_DIM // CACHE_GROUP)
    active = max(0, min(int(tensors["num_tokens"]), source.shape[0], tensors["slots"].numel()))
    for token in range(active):
        slot = int(tensors["slots"][token])
        if 0 <= slot < cache.shape[0]:
            cache[slot] = payload[token]
            scales[slot] = codes[token]


def compare_bytes(actual, expected, **context):
    """按原始字节比较 FP4 数据和 E8M0 缩放值。"""
    import torch
    from golden import ratio_allclose

    exact = ratio_allclose(atol=0.0, rtol=0.0, max_error_ratio=0.0)
    return exact(actual.view(torch.uint8), expected.view(torch.uint8), **context)


def build_publication_specs(rows, active, pages=3, kind="random"):
    """构造量化中点、跨页和重复槽位、无效尾部及未选中缓存哨兵值。"""
    import torch
    from golden import ScalarSpec, TensorSpec

    generator = torch.Generator().manual_seed(3111)
    source = torch.randn(rows, INDEX_DIM, generator=generator).bfloat16()
    weight = (torch.rand(INDEX_DIM, generator=generator) + 0.5).bfloat16()
    angles = torch.randn(rows, ROPE_DIM // 2, generator=generator)
    if kind == "ties":
        source.fill_(1.0)
        boundaries = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.bfloat16)
        neighbors = torch.cat((boundaries, torch.nextafter(boundaries, torch.zeros_like(boundaries))))
        group = torch.cat((torch.tensor([6.0, 0.0, -0.0, -6.0]), neighbors, -neighbors))
        weight = (group.repeat(INDEX_DIM // CACHE_GROUP).reshape(-1, CACHE_GROUP)
                  * torch.tensor([2.0**-20, 0.5, 1.0, 2.0**10]).reshape(-1, 1)).flatten().bfloat16()
        angles.zero_()
    elif kind == "zero":
        source.zero_()
        source[:, 1::2] = -0.0
        angles.zero_()
    source[max(0, min(rows, active)):] = float("nan")
    targets = torch.tensor([0, 127, 128, -1, 0, pages * PAGE_SIZE, pages * PAGE_SIZE - 1, 129, 128])
    slots = targets.repeat((rows + targets.numel() - 1) // targets.numel())[:rows]
    if pages > TASK_TILE:
        slots = (torch.arange(rows) * 127) % (pages * PAGE_SIZE)
        slots[-1] = slots[0]
    cache = torch.arange(pages * PAGE_SIZE * (INDEX_DIM // 2)).to(torch.uint8)
    cache = cache.reshape(pages, PAGE_SIZE, 1, INDEX_DIM // 2)
    codes = (torch.arange(pages * PAGE_SIZE * (INDEX_DIM // CACHE_GROUP)) % 253 + 1).to(torch.uint8)
    scales = codes.reshape(pages, PAGE_SIZE, 1, INDEX_DIM // CACHE_GROUP).view(torch.float8_e8m0fnu)
    return [
        TensorSpec("source", list(source.shape), source.dtype, init_value=source),
        TensorSpec("weight", list(weight.shape), weight.dtype, init_value=weight),
        TensorSpec("cosine", list(angles.shape), torch.float32, init_value=angles.cos()),
        TensorSpec("sine", list(angles.shape), torch.float32, init_value=angles.sin()),
        TensorSpec("slots", [rows], torch.int64, init_value=slots),
        TensorSpec("cache", list(cache.shape), cache.dtype, init_value=cache),
        TensorSpec("scales", list(scales.shape), scales.dtype, init_value=scales),
        ScalarSpec("num_tokens", torch.int32, active),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="在 A5 上校验 DSL Indexer Key 归一化、旋转及缓存发布。")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    cases = ((1, 1, 1, "ties"), (17, 15, 3, "zero"), (17, 0, 3, "random"),
             (80, 72, 3, "random"), (17, 17, 13, "random"))
    for rows, active, pages, kind in cases:
        print(f"[INDEX_KEY_PUBLISH_DSL] 行数={rows} 有效行={active} 页数={pages} 类型={kind}", flush=True)
        result = run(
            fn=validate_publication,
            specs=build_publication_specs(rows, active, pages, kind),
            golden_fn=golden_publication,
            compare_fn={"cache": compare_bytes, "scales": compare_bytes},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        print(f"[产物] {result.work_dir}", flush=True)
