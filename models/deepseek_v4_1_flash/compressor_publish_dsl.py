# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""在单个 DSL AIV 任务中生成 C1A latent，并发布 RoPE 后的 FP4 缓存。"""

import cannbotdsl as cb
from cannbotdsl.buffer import Buffer
import pypto.language as pl

from compressor_vector_dsl import CompressedRow, golden_normalize_latent, normalize_latent
from config import FLASH as M


# 模型契约
HEAD_DIM = M.head_dim
ROPE_DIM = M.qk_rope_head_dim
CACHE_GROUP = 16
PAGE_SIZE = 128

# 分块参数
OWNER_TILE = 8


@cb.kernel
def publish_c1a(
    source: cb.Tensor,
    slots: cb.Tensor,
    cache: cb.Tensor,
    scales: cb.Tensor,
    cosine: cb.Tensor,
    sine: cb.Tensor,
    weight: cb.Tensor,
    latent: cb.Tensor,
    num_tokens: cb.ScalarValue,
):
    """同一物理页固定归属一个 worker，重复槽位按 token 顺序覆盖。"""
    worker = cb.get_block_idx()
    workers = cb.get_block_num()
    raw = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, HEAD_DIM)))
    normalized = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((1, HEAD_DIM)))
    gamma = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((HEAD_DIM,)))
    publication = CompressedRow()
    cb.mem_copy(gamma, weight)
    for token in cb.range(num_tokens):
        slot = slots[token]
        owner = (token // OWNER_TILE) % workers
        if slot >= 0:
            owner = (slot // PAGE_SIZE) % workers
        if owner == worker:
            cb.mem_copy(raw, cb.tile_slice(source, (1, HEAD_DIM), (token, 0)))
            normalize_latent(raw, gamma, normalized)
            cb.mem_copy(cb.tile_slice(latent, (1, HEAD_DIM), (token, 0)), normalized)
            # 负槽位仍生成未旋转 latent，供 Indexer 使用。
            if slot >= 0 and slot < cache.shape[0]:
                cos_row = cb.tile_slice(cosine, (1, ROPE_DIM // 2), (token, 0))
                sin_row = cb.tile_slice(sine, (1, ROPE_DIM // 2), (token, 0))
                publication.encode(normalized, cos_row, sin_row, True)
                target = cb.tile_slice(cache, (1, HEAD_DIM // 2), (slot, 0))
                factors = cb.tile_slice(scales, (1, HEAD_DIM // CACHE_GROUP), (slot, 0))
                cb.mem_copy(target, publication.payload)
                cb.mem_copy(factors, publication.scale)


@pl.jit.inline(auto_scope=False)
def c1a_compressor_publish(
    source: pl.Tensor,
    norm_weight: pl.Tensor,
    latent: pl.Tensor,
    cosine: pl.Tensor,
    sine: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.Tensor,
    scales: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    cache_ready: pl.Scalar[pl.TASK_ID],
    source_ready: pl.Scalar[pl.TASK_ID],
):
    """保留页归属和两个前置依赖，调度完整的归一化、旋转及缓存发布任务。"""
    rows = pl.tensor.dim(cache, 0) * PAGE_SIZE
    payload = pl.reshape(cache, [rows, HEAD_DIM // 2])
    factors = pl.reshape(scales, [rows, HEAD_DIM // CACHE_GROUP])
    workers = pl.max(pl.min(num_tokens, OWNER_TILE), 1)
    with pl.spmd(workers, name_hint="dsl_c1a_compress", deps=[cache_ready, source_ready]) as published:
        publish_c1a(source, slots, payload, factors, cosine, sine, norm_weight, latent, num_tokens)
    return published


@pl.jit
def validate_publication(
    source: pl.Tensor,
    slots: pl.Tensor,
    cache: pl.InOut[pl.Tensor],
    scales: pl.InOut[pl.Tensor],
    cosine: pl.Tensor,
    sine: pl.Tensor,
    norm_weight: pl.Tensor,
    latent: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
):
    ready = pl.system.task_dummy(deps=[])
    _published = c1a_compressor_publish(
        source, norm_weight, latent, cosine, sine, slots, cache, scales, num_tokens, ready, ready,
    )
    return cache, scales, latent


def golden_publication(tensors):
    """按 token 顺序计算 RMSNorm、RoPE 和最近偶数 FP4，保留所有未写入区域。"""
    import torch
    from quant import quantize_mxfp4_cache

    active = int(tensors["num_tokens"])
    if not active:
        return
    latent = golden_normalize_latent(tensors["source"][:active], tensors["norm_weight"])
    tensors["latent"][:active] = latent
    rows = tensors["cache"].reshape(-1, HEAD_DIM // 2)
    factors = tensors["scales"].reshape(-1, HEAD_DIM // CACHE_GROUP)
    for token in range(active):
        slot = int(tensors["slots"][token])
        if 0 <= slot < rows.shape[0]:
            value = latent[token].clone()
            real, imag = value[-ROPE_DIM:].float().reshape(-1, 2).unbind(-1)
            cos, sin = tensors["cosine"][token], tensors["sine"][token]
            rotated = torch.stack((real * cos - imag * sin, real * sin + imag * cos), dim=-1)
            value[-ROPE_DIM:] = rotated.flatten().bfloat16()
            payload, scale = quantize_mxfp4_cache(value, CACHE_GROUP, "bf16", round_to_even=True)
            rows[slot] = payload
            factors[slot] = scale


def build_publication_specs(rows=17, active=15, kind="random"):
    """覆盖跨页、重复/负槽位、非有效行、FP4 中点和零幅值缩放下限。"""
    import torch
    from golden import ScalarSpec, TensorSpec

    generator = torch.Generator().manual_seed(8142)
    source = (torch.randint(-16, 17, (rows, HEAD_DIM), generator=generator).float() / 16).bfloat16()
    weight = (torch.randint(-8, 9, (HEAD_DIM,), generator=generator).float() / 8).bfloat16()
    angles = torch.randn(rows, ROPE_DIM // 2, generator=generator)
    slots = torch.tensor([127, 128, 3, 259, -1, 3, 384, 510, -1], dtype=torch.int64)
    slots = slots.repeat((rows + slots.numel() - 1) // slots.numel())[:rows].clone()
    if kind == "ties":
        source.fill_(1)
        pattern = torch.tensor([6, -6, 0.25, -0.25, 0.75, -0.75, 1.25, -1.25,
                                1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 5, -5])
        weight = pattern.repeat(HEAD_DIM // pattern.numel()).bfloat16()
        angles.zero_()
    elif kind == "zero":
        source.zero_()
        angles.zero_()
    source[active:] = float("nan")
    values = {
        "source": source, "slots": slots,
        "cache": torch.full((4, PAGE_SIZE, 1, HEAD_DIM // 2), 0xA5, dtype=torch.uint8),
        "scales": torch.full((4, PAGE_SIZE, 1, HEAD_DIM // CACHE_GROUP), -7, dtype=torch.bfloat16),
        "cosine": angles.cos(), "sine": angles.sin(), "norm_weight": weight,
        "latent": torch.full_like(source, -77),
    }
    specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in values.items()]
    specs.append(ScalarSpec("num_tokens", torch.int32, active))
    return specs


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="校验 C1A 单个 DSL AIV 缓存发布任务。")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", choices=("basic", "all"), default="all")
    args = parser.parse_args()
    cases = [(17, 15, "random")]
    if args.case == "all":
        cases += [(17, 0, "random"), (1, 1, "ties"), (9, 9, "zero"), (81, 73, "random")]
    for case in cases:
        print(f"[C1A_PUBLISH_DSL] {case}", flush=True)
        result = run(
            fn=validate_publication,
            specs=build_publication_specs(*case),
            golden_fn=golden_publication,
            rtol=0,
            atol=0,
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        print(f"[产物] {result.work_dir}", flush=True)
