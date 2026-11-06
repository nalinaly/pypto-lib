# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""以单个 DSL AIV 任务生成 FP32 分数对应的稳定候选块池。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.constexpr import range_constexpr
from cannbotdsl.lang.jit import jit
import pypto.language as pl
import torch
import torch.nn.functional as F

from config import FLASH as M


# 模型契约
BLOCK_SIZE = M.candidate_block_size
TOPK = M.candidate_topk_blocks
SCORE_FLOOR = -1e30
LATEST_SCORE = 1e30

# 分块参数
BLOCK_TILE = 1024
VECTOR_TILE = 64
WORKER_TILE = 12


class PoolSelector:
    """逐块扫描 FP32 分数，用四轮字节直方图定位稳定 TopK 阈值。"""

    def __init__(self):
        self.raw = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, BLOCK_TILE * BLOCK_SIZE)))
        self.keys = Buffer(cb.MemLoc.UB, cb.dtypes.uint32, cb.make_layout((BLOCK_TILE,)))
        self.histogram = Buffer(cb.MemLoc.UB, cb.dtypes.uint32, cb.make_layout((256,)))
        self.histogram16 = Buffer(cb.MemLoc.UB, cb.dtypes.uint16, cb.make_layout((384,)))
        self.output = Buffer(cb.MemLoc.UB, cb.dtypes.int32, cb.make_layout((1, TOPK)))
        self.state = Buffer(cb.MemLoc.UB, cb.dtypes.int32, cb.make_layout((16,)))

    @jit
    def load_keys(self, scores, token, chunk, blocks):
        """每八个 FP32 位置取最大值，再编码为保序 UINT32；最新块使用固定高分。"""
        cb.mem_copy(self.raw, cb.tile_slice(scores, (1, BLOCK_TILE * BLOCK_SIZE), (token, chunk)))
        live = blocks - chunk * BLOCK_TILE
        if live > BLOCK_TILE:
            live = BLOCK_TILE
        with cb.vf():
            for group in cb.range((live + VECTOR_TILE - 1) // VECTOR_TILE):
                mask = rr.update_mask(live - group * VECTOR_TILE, elem_bits=32)[0]
                lane = rr.varange(group * VECTOR_TILE, cb.dtypes.uint32)
                offsets = rr.vshl(lane, 3, mask=mask)
                value = rr.vgather(self.raw, offsets, mask=mask)
                for element in range(1, BLOCK_SIZE):
                    location = rr.vadds(offsets, element, mask=mask)
                    other = rr.vgather(self.raw, location, mask=mask)
                    value = rr.vmax(value, other, mask=mask)
                value = rr.vmaxs(value, SCORE_FLOOR, mask=mask)
                logical = rr.vadds(lane, chunk * BLOCK_TILE, mask=mask)
                latest = rr.veqs(logical, blocks - 1, mask=mask)
                high_score = rr.vdups(LATEST_SCORE, cb.dtypes.float32, mask=mask)
                value = rr.vselect(high_score, value, cond_mask=latest)
                # 将正负零归为同分，保留其他 FP32 位。
                value = rr.vadds(value, 0.0, mask=mask)
                bits = rr.vreinterpret(value, cb.dtypes.uint32)
                positive = rr.vbitwise_xor(bits, rr.vdups(0x80000000, cb.dtypes.uint32), mask=mask)
                negative = rr.vbitwise_xor(bits, rr.vdups(0xFFFFFFFF, cb.dtypes.uint32), mask=mask)
                sign = rr.vlts(rr.vreinterpret(bits, cb.dtypes.int32), 0, mask=mask)
                key = rr.vselect(negative, positive, cond_mask=sign)
                rr.vstore(self.keys, group * VECTOR_TILE, key, mask)

    @jit
    def accumulate_histogram(self, live, prefix, shift: cb.Constexpr[int], prefix_mask: cb.Constexpr[int]):
        """分块累计 16 位直方图，再扩展到 32 位，避免长历史计数溢出。"""
        with cb.vf():
            mask16 = rr.update_mask(128, elem_bits=16)[0]
            full = rr.full_mask()
            h0 = rr.vdups(0, cb.dtypes.uint16, mask=mask16)
            h1 = rr.vdups(0, cb.dtypes.uint16, mask=mask16)
            for group in cb.range((live + VECTOR_TILE - 1) // VECTOR_TILE):
                mask = rr.update_mask(live - group * VECTOR_TILE, elem_bits=32)[0]
                mask8 = rr.update_mask(live - group * VECTOR_TILE, elem_bits=8)[0]
                # 每轮只压缩当前 64 个 32 位排序键。
                lanes8 = rr.varange(0, cb.dtypes.uint8)
                mask8 = rr.vlts(lanes8, VECTOR_TILE, mask=mask8)
                keys = rr.vload(self.keys, group * VECTOR_TILE)
                upper = rr.vbitwise_and(keys, rr.vdups(prefix_mask, cb.dtypes.uint32), mask=mask)
                matching = rr.veqs(upper, prefix, mask=mask)
                flags = rr.vselect(rr.vdups(1, cb.dtypes.uint32), rr.vdups(0, cb.dtypes.uint32), cond_mask=matching)
                flags16 = rr.vpack(flags, cb.dtypes.uint16)
                flags8 = rr.vpack(flags16, cb.dtypes.uint8)
                selected = rr.veqs(flags8, 1, mask=mask8)
                byte = rr.vshr(keys, shift, mask=mask)
                byte = rr.vbitwise_and(byte, rr.vdups(255, cb.dtypes.uint32), mask=mask)
                byte16 = rr.vpack(byte, cb.dtypes.uint16)
                byte8 = rr.vpack(byte16, cb.dtypes.uint8)
                h0 = rr.vhistogram_accumulate(h0, byte8, mask=selected, bin=0)
                h1 = rr.vhistogram_accumulate(h1, byte8, mask=selected, bin=1)
            rr.vstore(self.histogram16, 0, h0, mask16)
            rr.vstore(self.histogram16, 128, h1, mask16)
            rr.vmem_bar("vst_vld")
            for part in range(4):
                value16 = rr.vload(self.histogram16, part * VECTOR_TILE)
                value = rr.vunpack(value16, cb.dtypes.uint32)
                total = rr.vadd(rr.vload(self.histogram, part * VECTOR_TILE), value, mask=full)
                rr.vstore(self.histogram, part * VECTOR_TILE, total, full)

    @jit
    def emit(self, logical_begin, live, cutoff, equal_left, append: cb.Constexpr[bool]):
        """按逻辑块号递增压缩输出，阈值同分仅保留最早的所需数量。"""
        with cb.vf():
            full = rr.full_mask()
            remaining = rr.vdups(equal_left, cb.dtypes.int32, mask=full)
            cursor = rr.vstore_unalign_begin(self.output, no_clear_ar=append)
            for group in cb.range((live + VECTOR_TILE - 1) // VECTOR_TILE):
                mask = rr.update_mask(live - group * VECTOR_TILE, elem_bits=32)[0]
                keys = rr.vload(self.keys, group * VECTOR_TILE)
                equal = rr.veqs(keys, cutoff, mask=mask)
                rank = rr.vunsqueeze(equal, cb.dtypes.int32)
                early = rr.vlt(rank, remaining, mask=mask)
                take_equal = rr.mask_and(equal, early, exec_mask=mask)
                strict = rr.vgts(keys, cutoff, mask=mask)
                selected = rr.mask_or(strict, take_equal, exec_mask=mask)
                indices = rr.varange(logical_begin + group * VECTOR_TILE, cb.dtypes.int32)
                packed = rr.vsqueeze_and_storeunalign_init(indices, mask=selected)
                rr.vsqueeze_and_storeunalign(self.output, 0, packed, cursor)
                flags = rr.vselect(rr.vdups(1, cb.dtypes.int32), rr.vdups(0, cb.dtypes.int32), cond_mask=equal)
                count = rr.vreduce_sum(flags, mask=mask)
                count = rr.vdup(count, mask=full)
                remaining = rr.vsub(remaining, count, mask=full)
            rr.vstore_unalign_post(self.output, 0, cursor)
            rr.vstore_first(self.state, 0, remaining)
            rr.vmem_bar("vst_vld")
        cb.vec_sync_all()
        return self.state[0]

    @jit
    def select(self, scores, token, blocks, selected):
        """无筛选时直接生成块号，否则保持 FP32 的四字节排序精度。"""
        with cb.vf():
            full = rr.full_mask()
            for group in range(TOPK // VECTOR_TILE):
                ids = rr.varange(group * VECTOR_TILE, cb.dtypes.int32)
                valid = rr.vlts(ids, selected, mask=full)
                value = rr.vselect(ids, rr.vdups(-1, cb.dtypes.int32), cond_mask=valid)
                rr.vstore(self.output, group * VECTOR_TILE, value, full)
        if blocks > selected:
            prefix = cb.dtypes.uint32(0)
            rank = cb.dtypes.int32(blocks - selected + 1)
            population = cb.dtypes.int32(0)
            for byte in range_constexpr(4):
                shift = (3 - byte) * 8
                prefix_mask = (0xFFFFFFFF << (shift + 8)) & 0xFFFFFFFF
                with cb.vf():
                    full = rr.full_mask()
                    zero = rr.vdups(0, cb.dtypes.uint32, mask=full)
                    for group in range(4):
                        rr.vstore(self.histogram, group * VECTOR_TILE, zero, full)
                for chunk in cb.range((blocks + BLOCK_TILE - 1) // BLOCK_TILE):
                    self.load_keys(scores, token, chunk, blocks)
                    live = blocks - chunk * BLOCK_TILE
                    if live > BLOCK_TILE:
                        live = BLOCK_TILE
                    self.accumulate_histogram(live, prefix, shift, prefix_mask)
                cb.vec_sync_all()
                low = cb.dtypes.int32(0)
                high = cb.dtypes.int32(255)
                for step in cb.range(8):
                    middle = (low + high) // 2
                    if cb.dtypes.int32(self.histogram[middle]) >= rank:
                        high = middle
                    else:
                        low = middle + 1
                previous = cb.dtypes.int32(0)
                if low > 0:
                    previous = cb.dtypes.int32(self.histogram[low - 1])
                population = cb.dtypes.int32(self.histogram[low]) - previous
                rank = rank - previous
                prefix = prefix + cb.dtypes.uint32(low) * (1 << shift)
            equal_left = population - rank + 1
            self.load_keys(scores, token, 0, blocks)
            first_live = blocks
            if first_live > BLOCK_TILE:
                first_live = BLOCK_TILE
            equal_left = self.emit(0, first_live, prefix, equal_left, False)
            for chunk in cb.range(1, (blocks + BLOCK_TILE - 1) // BLOCK_TILE):
                self.load_keys(scores, token, chunk, blocks)
                live = blocks - chunk * BLOCK_TILE
                if live > BLOCK_TILE:
                    live = BLOCK_TILE
                equal_left = self.emit(chunk * BLOCK_TILE, live, prefix, equal_left, True)


@jit
def visible_block_count(lengths, token, num_tokens, width):
    """将非有效行和因果长度规整为当前分数宽度内的候选块数。"""
    visible = cb.dtypes.int32(0)
    if token < num_tokens:
        visible = lengths[token]
        if visible < 0:
            visible = cb.dtypes.int32(0)
        if visible > width:
            visible = cb.dtypes.int32(width)
    return (visible + BLOCK_SIZE - 1) // BLOCK_SIZE


@cb.kernel
def generate_pool(scores: cb.Tensor, lengths: cb.Tensor, ids: cb.Tensor, counts: cb.Tensor,
                  num_tokens: cb.ScalarValue):
    """同一 Query 的全部归约和筛选归属同一 worker，计数向量由 worker 0 发布。"""
    worker = cb.get_block_idx()
    workers = cb.get_block_num()
    selection = PoolSelector()
    for token in cb.range(worker, ids.shape[0], workers):
        blocks = visible_block_count(lengths, token, num_tokens, scores.shape[1])
        selected = blocks
        if selected > ids.shape[1]:
            selected = cb.dtypes.int32(ids.shape[1])
        if selected > TOPK:
            selected = cb.dtypes.int32(TOPK)
        selection.select(scores, token, blocks, selected)
        target = cb.tile_slice(ids, (1, TOPK), (token, 0))
        cb.mem_copy(target, selection.output[None, 0:ids.shape[1]])
    if worker == 0:
        for token in cb.range(ids.shape[0]):
            count = visible_block_count(lengths, token, num_tokens, scores.shape[1])
            if count > ids.shape[1]:
                count = cb.dtypes.int32(ids.shape[1])
            if count > TOPK:
                count = cb.dtypes.int32(TOPK)
            counts[token] = count


@pl.jit.inline(auto_scope=False)
def select_candidate_block_pool(
    index_scores: pl.Tensor, compressed_lens: pl.Tensor, pool_block_ids: pl.Tensor, pool_block_counts: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32], scores_ready: pl.Scalar[pl.TASK_ID],
):
    """调度单个本地 AIV SPMD 任务，并返回完整候选池的完成句柄。"""
    tokens = pl.tensor.dim(pool_block_ids, 0)
    workers = pl.max(1, pl.min(tokens, WORKER_TILE))
    with pl.spmd(workers, name_hint="dsl_index_pool", deps=[scores_ready]) as done:
        generate_pool(index_scores, compressed_lens, pool_block_ids, pool_block_counts, num_tokens)
    return done


@pl.jit
def validate_pool(
    scores: pl.Tensor, lengths: pl.Tensor, ids: pl.InOut[pl.Tensor], counts: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
):
    """通过生产入口验证候选池，并在两个输出尾部保留保护区。"""
    tokens = pl.tensor.dim(scores, 0)
    capacity = pl.tensor.dim(ids, 1)
    output_ids = pl.slice(ids, [tokens, capacity], [0, 0])
    output_counts = pl.slice(counts, [tokens], [0])
    ready = pl.system.task_dummy(deps=[])
    select_candidate_block_pool(scores, lengths, output_ids, output_counts, num_tokens, ready)
    return ids, counts


def index_pool_block_capacity(score_width: int) -> int:
    """返回按 32 个块号对齐的候选池容量。"""
    if score_width < 0:
        raise ValueError(f"score_width must be nonnegative, got {score_width}")
    block_size = M.candidate_block_size
    blocks = max(1, (score_width + block_size - 1) // block_size)
    aligned = (blocks + 31) // 32 * 32
    return min(M.candidate_topk_blocks, aligned)


def golden_select_candidate_block_pool(
    logits: torch.Tensor,
    compressed_lens: torch.Tensor | int,
    topk_blocks: int = M.candidate_topk_blocks,
    block_size: int = M.candidate_block_size,
    *,
    num_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """保持原参考：最新因果块加分、同分稳定选择、逻辑块号升序输出。"""
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank-2, got shape {tuple(logits.shape)}")
    if block_size != BLOCK_SIZE:
        raise ValueError(f"block_size must be {BLOCK_SIZE}, got {block_size}")
    if topk_blocks != TOPK:
        raise ValueError(f"topk_blocks must be {TOPK}, got {topk_blocks}")

    tokens, width = logits.shape
    # 非有效行发布空候选池。
    active_tokens = tokens if num_tokens is None else max(0, min(tokens, int(num_tokens)))
    capacity = index_pool_block_capacity(width)
    ids = torch.full((tokens, capacity), -1, dtype=torch.int32, device=logits.device)
    counts = torch.zeros((tokens,), dtype=torch.int32, device=logits.device)
    if width == 0:
        return ids, counts

    lengths = torch.as_tensor(compressed_lens, device=logits.device, dtype=torch.int64)
    if lengths.ndim == 0:
        lengths = lengths.expand(tokens)
    if lengths.ndim != 1 or lengths.numel() != tokens:
        raise ValueError("compressed_lens must provide one causal length per score row")

    padded = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    block_scores = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    for token in range(active_tokens):
        visible = max(0, min(int(lengths[token]), width))
        visible_blocks = (visible + block_size - 1) // block_size
        if visible_blocks == 0:
            continue
        selected_count = min(visible_blocks, topk_blocks, capacity)
        # 将块分数截断到设备排序下限。
        scores = block_scores[token, :visible_blocks].clone().clamp_min(-1e30)
        # 最新可见块使用固定高分。
        scores[visible_blocks - 1] = 1e30
        # 同分按较早逻辑块号选择。
        selected = torch.argsort(scores, descending=True, stable=True)[:selected_count].to(torch.int32)
        ids[token, :selected_count] = selected.sort().values
        counts[token] = selected_count
    return ids, counts



def golden_pool(tensors):
    """将共用 CPU 参考写入有效输出区域，保留额外保护行。"""
    ids, counts = golden_select_candidate_block_pool(
        tensors["scores"], tensors["lengths"], num_tokens=int(tensors["num_tokens"]),
    )
    rows = tensors["scores"].shape[0]
    tensors["ids"][:rows] = ids
    tensors["counts"][:rows] = counts


def build_pool_specs(rows=5, width=257, active=4, kind="random"):
    """构造短/长历史、因果尾块、FP32 近邻值、同分和直方图计数边界。"""
    from golden import ScalarSpec, TensorSpec

    generator = torch.Generator().manual_seed(4173)
    scores = torch.randn(rows, width, generator=generator)
    if kind in ("ties", "overflow"):
        scores.zero_()
        scores[:, 1::2] = -0.0
    elif kind in ("fp32", "late"):
        block = torch.arange((width + BLOCK_SIZE - 1) // BLOCK_SIZE)
        ordered = block if kind == "late" else (block * 37) % 4099
        values = 1.0 + ordered.float() * 2**-22
        scores[:] = values.repeat_interleave(BLOCK_SIZE)[:width]
    elif kind == "floor":
        scores.fill_(-float("inf"))
        scores[:, 16::32] = -3e30
        scores[:, 24::32] = -2e30
    pattern = [width, max(0, width - 11), 0, -3, width + 29, 7, 8, 9]
    lengths = torch.tensor((pattern * ((rows + len(pattern) - 1) // len(pattern)))[:rows], dtype=torch.int32)
    scores[max(0, min(rows, active)):] = float("nan")
    capacity = index_pool_block_capacity(width)
    return [
        TensorSpec("scores", [rows, width], torch.float32, init_value=scores),
        TensorSpec("lengths", [rows], torch.int32, init_value=lengths),
        TensorSpec("ids", [rows + 1, capacity], torch.int32, init_value=-777),
        TensorSpec("counts", [rows + 16], torch.int32, init_value=-888),
        ScalarSpec("num_tokens", torch.int32, active),
    ]


if __name__ == "__main__":
    import argparse
    from golden import run

    parser = argparse.ArgumentParser(description="校验单个 DSL AIV 候选块池任务。")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--case", choices=("basic", "all"), default="all")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    cases = [(5, 257, 4, "random"), (3, 16385, 2, "random")]
    if args.case == "all":
        cases += [(5, 257, 0, "random"), (3, 16384, 2, "random"), (2, 32779, 2, "fp32"),
                  (2, 65549, 2, "ties"), (3, 32776, 0, "floor"), (2, 32776, 2, "floor"),
                  (1, 524317, 1, "overflow"), (1, 32779, 1, "late")]
    for case in cases:
        print(f"[INDEX_POOL_DSL] {case}", flush=True)
        result = run(
            fn=validate_pool,
            specs=build_pool_specs(*case),
            golden_fn=golden_pool,
            rtol=0,
            atol=0,
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        print(f"[产物] {result.work_dir}", flush=True)
