# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""在一个 DSL MIX 任务中完成混合量化 KV 收集、稀疏 Attention 和 sink 归一化。"""

import cannbotdsl as cb
from cannbotdsl import reg as rr
from cannbotdsl.buffer import Buffer
from cannbotdsl.lang.jit import jit
from cannbotdsl.lang.constexpr import const_expr
from cannbotdsl.ring_buffer import RingBuffer
import pypto.language as pl
import torch

from attention_cache_dsl import dequantize_fp4, dequantize_fp8, store_nz


# 模型契约
HEADS = 64
HEAD_DIM = 512
WINDOW = 128
TOPK = 512
SOFTMAX_SCALE = HEAD_DIM**-0.5

# 分块参数
GATHER_TILE = 16
KEY_TILE = 128
CUBE_TILE = 128
VECTOR_TILE = 64
WORKER_TILE = 32


class _Attention:
    """单个 AIC 和两个 AIV 共享的 QK/PV 流水与在线 Softmax 状态。"""

    def __init__(self):
        self.q = RingBuffer(cb.MemLoc.L1, cb.dtypes.bfloat16, cb.make_layout((HEADS, HEAD_DIM)), depth=1)
        self.kv = RingBuffer(cb.MemLoc.L1, cb.dtypes.bfloat16, cb.make_layout((KEY_TILE, HEAD_DIM)), depth=1)
        self.p = RingBuffer(cb.MemLoc.L1, cb.dtypes.bfloat16, cb.make_layout((HEADS, KEY_TILE)), depth=1)
        self.left = RingBuffer(cb.MemLoc.L0A, cb.dtypes.bfloat16, cb.make_layout((HEADS, CUBE_TILE)), depth=1)
        self.right = RingBuffer(cb.MemLoc.L0B, cb.dtypes.bfloat16, cb.make_layout((KEY_TILE, CUBE_TILE)), depth=1)
        self.acc = RingBuffer(cb.MemLoc.L0C, cb.dtypes.float32, cb.make_layout((HEADS, CUBE_TILE)), depth=1)
        self.scores = RingBuffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((32, KEY_TILE)), depth=1)
        self.weighted = RingBuffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((32, CUBE_TILE)), depth=1)
        self.ids = Buffer(cb.MemLoc.UB, cb.dtypes.int32, cb.make_layout((1, KEY_TILE)))
        self.fp8 = Buffer(cb.MemLoc.UB, cb.dtypes.float8_e4m3fn, cb.make_layout((GATHER_TILE + 1, HEAD_DIM)))
        self.fp4 = Buffer(cb.MemLoc.UB, cb.dtypes.uint8, cb.make_layout((GATHER_TILE + 1, HEAD_DIM // 2)))
        self.scale8 = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((GATHER_TILE, 16)))
        self.scale4 = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((GATHER_TILE, 32)))
        self.decoded = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((GATHER_TILE, HEAD_DIM)))
        kv_physical = cb.make_layout((32, 1, 16, 16), stride=(272, 256, 16, 1))
        self.nz = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((GATHER_TILE, HEAD_DIM)),
                         physical_layout=kv_physical, data_format="nz")
        p_physical = cb.make_layout((8, 2, 16, 16), stride=(528, 256, 16, 1))
        self.probabilities = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((32, KEY_TILE)),
                                    physical_layout=p_physical, data_format="nz")
        self.state = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((3, 32)))
        self.numerator = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((32, HEAD_DIM)))
        self.sinks = Buffer(cb.MemLoc.UB, cb.dtypes.float32, cb.make_layout((1, 32)))
        self.result = Buffer(cb.MemLoc.UB, cb.dtypes.bfloat16, cb.make_layout((32, CUBE_TILE)))

    @jit
    def initialize(self, query, sink, lane):
        cb.mem_copy(self.q.next_write(), query)
        cb.mem_copy(self.sinks, cb.tile_slice(sink, (1, 32), (0, lane)))
        with cb.vf():
            heads = rr.update_mask(32, elem_bits=32)[0]
            full = rr.full_mask()
            zero = rr.vdups(0.0, cb.dtypes.float32, mask=full)
            rr.vstore(self.state, 0, rr.vdups(-1e30, cb.dtypes.float32, mask=heads), heads)
            rr.vstore(self.state, 32, zero, heads)
            for chunk in range(32 * HEAD_DIM // VECTOR_TILE):
                rr.vstore(self.numerator, chunk * VECTOR_TILE, zero, full)
            rr.vmem_bar("vst_vld")

    @jit
    def gather(self, cache, scales, indices, lane, fp4):
        cb.mem_copy(self.ids, indices)
        destination = self.kv.next_write()
        for chunk in cb.range(KEY_TILE // 2 // GATHER_TILE):
            first = lane * (KEY_TILE // 2) + chunk * GATHER_TILE
            for row in cb.range(GATHER_TILE):
                physical = cb.dtypes.int64(indices[0, first + row])
                safe = max(0, min(physical, cache.shape[0] - 1))
                if const_expr(fp4):
                    source = cb.tile_slice(cache, (1, HEAD_DIM // 2), (safe, 0))
                    cb.mem_copy(cb.tile_slice(self.fp4, (1, HEAD_DIM // 2), (row, 0)), source)
                    factors = cb.tile_slice(scales, (1, 32), (safe, 0))
                    cb.mem_copy(cb.tile_slice(self.scale4, (1, 32), (row, 0)), factors)
                else:
                    source = cb.tile_slice(cache, (1, HEAD_DIM), (safe, 0))
                    cb.mem_copy(cb.tile_slice(self.fp8, (1, HEAD_DIM), (row, 0)), source)
                    factors = cb.tile_slice(scales, (1, 16), (safe, 0))
                    cb.mem_copy(cb.tile_slice(self.scale8, (1, 16), (row, 0)), factors)
            if const_expr(fp4):
                dequantize_fp4(self.fp4, self.scale4, self.ids, self.nz, first, cache.shape[0])
            else:
                dequantize_fp8(self.fp8, self.scale8, self.ids, self.decoded, first, cache.shape[0])
                store_nz(self.decoded, self.nz)
            target = cb.tile_slice(destination, (GATHER_TILE, HEAD_DIM), (lane * 4 + chunk, 0))
            cb.mem_copy(target, self.nz)

    @jit
    def qk(self, query, keys):
        product = self.acc.next_write()
        for chunk in cb.range(HEAD_DIM // CUBE_TILE):
            cb.mem_copy(self.left.next_write(), cb.tile_slice(query, (HEADS, CUBE_TILE), (0, chunk)))
            cb.mem_copy(self.right.next_write(), cb.tile_slice(keys, (KEY_TILE, CUBE_TILE), (0, chunk)))
            cb.matmul(product, self.left.next_read(), self.right.next_read(), init=chunk == 0)
        cb.mem_copy(self.scores.next_write(), self.acc.next_read().partition(dim=0))

    @jit
    def softmax(self, values, cache_rows):
        with cb.vf():
            full = rr.full_mask()
            full16 = rr.full_mask(elem_bits=16)
            zero = rr.vdups(0.0, cb.dtypes.float32, mask=full)
            invalid = rr.vdups(-1e30, cb.dtypes.float32, mask=full)
            ids0 = rr.vload(self.ids, 0)
            ids1 = rr.vload(self.ids, VECTOR_TILE)
            valid0 = rr.mask_and(rr.vges(ids0, 0, mask=full), rr.vlts(ids0, cache_rows, mask=full), exec_mask=full)
            valid1 = rr.mask_and(rr.vges(ids1, 0, mask=full), rr.vlts(ids1, cache_rows, mask=full), exec_mask=full)
            for row in range(32):
                score0 = rr.vmuls(rr.vload(values, row * KEY_TILE), SOFTMAX_SCALE, mask=full)
                score1 = rr.vmuls(rr.vload(values, row * KEY_TILE + VECTOR_TILE), SOFTMAX_SCALE, mask=full)
                score0 = rr.vselect(score0, invalid, cond_mask=valid0)
                score1 = rr.vselect(score1, invalid, cond_mask=valid1)
                maximum = rr.vdup(rr.vreduce_max(rr.vmax(score0, score1, mask=full), mask=full), mask=full)
                previous_max = rr.vload_broadcast(self.state, row)
                maximum = rr.vmax(maximum, previous_max, mask=full)
                correction = rr.vexp(rr.vsub(previous_max, maximum, mask=full), mask=full)
                p0 = rr.vexp(rr.vsub(score0, maximum, mask=full), mask=full)
                p1 = rr.vexp(rr.vsub(score1, maximum, mask=full), mask=full)
                p0 = rr.vselect(p0, zero, cond_mask=valid0)
                p1 = rr.vselect(p1, zero, cond_mask=valid1)
                total = rr.vadd(rr.vreduce_sum(p0, mask=full), rr.vreduce_sum(p1, mask=full), mask=full)
                total = rr.vdup(total, mask=full)
                previous_denominator = rr.vload_broadcast(self.state, 32 + row)
                denominator = rr.vmul(previous_denominator, correction, mask=full)
                denominator = rr.vadd(denominator, total, mask=full)
                rr.vstore_first(self.state, row, maximum)
                rr.vstore_first(self.state, 32 + row, denominator)
                rr.vstore_first(self.state, 64 + row, correction)
                p0 = rr.vcast(p0, cb.dtypes.bfloat16, mask=full, rounding="rn")
                p1 = rr.vcast(p1, cb.dtypes.bfloat16, mask=full, rounding="rn")
                packed, unused = rr.vdeinterleave(p0, p1)
                rr.vstore_strided(self.probabilities, row * 16, packed, full16, block_stride=33, repeat_stride=0)
            rr.vmem_bar("vst_vld")

    @jit
    def pv(self, keys, lane):
        cb.mem_copy(self.p.next_write().partition(dim=0, part_id=lane), self.probabilities)
        probability = self.p.next_read()
        for column in cb.range(HEAD_DIM // CUBE_TILE):
            cb.mem_copy(self.left.next_write(), probability)
            values = cb.tile_slice(keys, (KEY_TILE, CUBE_TILE), (0, column))
            cb.mem_copy(self.right.next_write(), values.permute(1, 0))
            cb.matmul(self.acc.next_write(), self.left.next_read(), self.right.next_read(), init=True)
            cb.mem_copy(self.weighted.next_write(), self.acc.next_read().partition(dim=0))
            result = self.weighted.next_read()
            with cb.vf():
                full = rr.full_mask()
                for row in range(32):
                    correction = rr.vload_broadcast(self.state, 64 + row)
                    for offset in range(0, CUBE_TILE, VECTOR_TILE):
                        position = row * HEAD_DIM + column * CUBE_TILE + offset
                        old = rr.vload(self.numerator, position)
                        new = rr.vload(result, row * CUBE_TILE + offset)
                        old = rr.vmul(old, correction, mask=full)
                        total = rr.vadd(old, new, mask=full)
                        rr.vstore(self.numerator, position, total, full)
                rr.vmem_bar("vst_vld")

    @jit
    def finish(self, output, lane):
        # sink 只在最终归一化时加入分母，不参与中间 BF16 P 的舍入。
        for column in cb.range(HEAD_DIM // CUBE_TILE):
            with cb.vf():
                full = rr.full_mask()
                for row in range(32):
                    maximum = rr.vload_broadcast(self.state, row)
                    denominator = rr.vload_broadcast(self.state, 32 + row)
                    sink = rr.vload_broadcast(self.sinks, row)
                    final_max = rr.vmax(maximum, sink, mask=full)
                    correction = rr.vexp(rr.vsub(maximum, final_max, mask=full), mask=full)
                    sink_exp = rr.vexp(rr.vsub(sink, final_max, mask=full), mask=full)
                    denominator = rr.vadd(rr.vmul(denominator, correction, mask=full), sink_exp, mask=full)
                    factor = rr.vdiv(correction, denominator, mask=full, precision_mode=True)
                    for offset in range(0, CUBE_TILE, VECTOR_TILE):
                        value = rr.vload(self.numerator, row * HEAD_DIM + column * CUBE_TILE + offset)
                        value = rr.vmul(value, factor, mask=full)
                        value = rr.vcast(value, cb.dtypes.bfloat16, mask=full, rounding="rn")
                        rr.vstore_pack(self.result, row * CUBE_TILE + offset, value, full, pack_mode=rr.PackMode.B32_TO_B16)
            target = cb.tile_slice(output, (HEADS, CUBE_TILE), (0, column))
            cb.mem_copy(target.partition(dim=0, part_id=lane), self.result)


@cb.kernel
def sparse_attention_mix(
    query: cb.Tensor,
    window: cb.Tensor,
    window_scale: cb.Tensor,
    compressed: cb.Tensor,
    compressed_scale: cb.Tensor,
    window_indices: cb.Tensor,
    compressed_indices: cb.Tensor,
    sink: cb.Tensor,
    output: cb.Tensor,
    num_tokens: cb.ScalarValue,
):
    """每个 block 独立处理完整 Query，片上完成 128 窗口加 512 压缩行的 Attention。"""
    lane = cb.get_subblock_id()
    attention = _Attention()
    for token in cb.range(cb.get_block_idx(), num_tokens, cb.get_block_num()):
        query_row = cb.tile_slice(query, (HEADS, HEAD_DIM), (token, 0))
        attention.initialize(query_row, sink, lane)
        q = attention.q.next_read()
        for part in cb.range((WINDOW + TOPK) // KEY_TILE):
            cache_rows = cb.dtypes.int64(window.shape[0])
            if part == 0:
                attention.gather(window, window_scale, window_indices[token:token + 1, None], lane, False)
            else:
                first = (part - 1) * KEY_TILE
                indices = compressed_indices[token:token + 1, first:first + KEY_TILE]
                attention.gather(compressed, compressed_scale, indices, lane, True)
                cache_rows = cb.dtypes.int64(compressed.shape[0])
            keys = attention.kv.next_read()
            attention.qk(q, keys)
            attention.softmax(attention.scores.next_read(), cache_rows)
            attention.pv(keys, lane)
        output_row = cb.tile_slice(output, (HEADS, HEAD_DIM), (token, 0))
        attention.finish(output_row, lane)


@pl.jit.inline(auto_scope=False)
def sparse_attention_dsl(
    query: pl.Tensor,
    window: pl.Tensor,
    window_scale: pl.Tensor,
    compressed: pl.Tensor,
    compressed_scale: pl.Tensor,
    window_indices: pl.Tensor,
    compressed_indices: pl.Tensor,
    sink: pl.Tensor,
    output: pl.Tensor,
    num_tokens: pl.Scalar[pl.INT32],
    source_ready: pl.Scalar[pl.TASK_ID],
):
    """保留本卡缓存和物理行索引，由 simpler 提交一个完整 MIX SPMD 任务。"""
    rows = pl.tensor.dim(query, 0)
    window_rows = pl.tensor.dim(window, 0) * WINDOW
    compressed_rows = pl.tensor.dim(compressed, 0) * WINDOW
    queries = pl.reshape(query, [rows * HEADS, HEAD_DIM])
    result = pl.reshape(output, [rows * HEADS, HEAD_DIM])
    wp = pl.reshape(window, [window_rows, HEAD_DIM])
    ws = pl.reshape(window_scale, [window_rows, 16])
    cp = pl.reshape(compressed, [compressed_rows, HEAD_DIM // 2])
    cs = pl.reshape(compressed_scale, [compressed_rows, 32])
    sinks = pl.reshape(sink, [1, HEADS])
    workers = pl.max(1, pl.min(num_tokens, WORKER_TILE))
    with pl.spmd(workers, name_hint="dsl_sparse_attn", deps=[source_ready], sync_start=True) as attended:
        sparse_attention_mix(queries, wp, ws, cp, cs, window_indices, compressed_indices, sinks, result, num_tokens)
    return attended


@pl.jit
def validate_sparse_attention(
    query: pl.Tensor,
    window: pl.Tensor,
    window_scale: pl.Tensor,
    compressed: pl.Tensor,
    compressed_scale: pl.Tensor,
    window_indices: pl.Tensor,
    compressed_indices: pl.Tensor,
    sink: pl.Tensor,
    output: pl.InOut[pl.Tensor],
    num_tokens: pl.Scalar[pl.INT32],
):
    ready = pl.system.task_dummy(deps=[])
    _done = sparse_attention_dsl(query, window, window_scale, compressed, compressed_scale,
                                 window_indices, compressed_indices, sink, output, num_tokens, ready)
    return output


def golden_selected_attention(
    q, window_value, window_indices, compressed_value, compressed_indices, sink,
    *, key_tile=64,
):
    """以 BF16 概率分块和最终 sink 分母计算所选 KV 的 CPU 参考。"""
    tokens, heads, head_dim = q.shape
    gathered = torch.cat(
        (
            window_value.flatten(0, 1).squeeze(-2)[window_indices.clamp_min(0).long()],
            compressed_value.flatten(0, 1).squeeze(-2)[compressed_indices.clamp_min(0).long()],
        ),
        dim=-2,
    ).to(torch.bfloat16).float()
    visible = torch.cat((window_indices >= 0, compressed_indices >= 0), dim=-1)
    gathered = gathered.masked_fill(~visible.unsqueeze(-1), 0.0)
    maximum = torch.full((tokens, heads), -1e30)
    denominator = torch.zeros_like(maximum)
    numerator = torch.zeros(tokens, heads, head_dim, dtype=torch.float32)
    for start in range(0, gathered.shape[-2], key_tile):
        tile = gathered[:, start : start + key_tile]
        tile_visible = visible[:, None, start : start + key_tile]
        logits = torch.einsum("thd,tkd->thk", q.float(), tile) * head_dim**-0.5
        logits = logits.masked_fill(~tile_visible, -1e30)
        next_maximum = torch.maximum(maximum, logits.amax(-1))
        correction = torch.exp(maximum - next_maximum)
        probabilities = torch.exp(logits - next_maximum.unsqueeze(-1))
        probabilities = probabilities * tile_visible
        denominator = denominator * correction + probabilities.sum(-1)
        weights = probabilities.to(torch.bfloat16).float()
        numerator = numerator * correction.unsqueeze(-1) + torch.einsum("thk,tkd->thd", weights, tile)
        maximum = next_maximum
    sink_value = sink.float().view(1, -1)
    final_maximum = torch.maximum(maximum, sink_value)
    correction = torch.exp(maximum - final_maximum)
    denominator = denominator * correction + torch.exp(sink_value - final_maximum)
    return (numerator * (correction / denominator).unsqueeze(-1)).to(q.dtype)


def golden_sparse(tensors):
    """使用现有缓存字节解码和共享 Attention 参考，保留非有效输出。"""
    from quant import dequantize_mxfp4_cache, dequantize_mxfp8_cache

    count = int(tensors["num_tokens"])
    if not count:
        return
    window = dequantize_mxfp8_cache(tensors["window"], tensors["window_scale"])
    compressed = dequantize_mxfp4_cache(tensors["compressed"], tensors["compressed_scale"], 16, "bf16")
    q = tensors["query"][:count].reshape(count, HEADS, HEAD_DIM)
    result = golden_selected_attention(
        q, window, tensors["window_indices"][:count], compressed,
        tensors["compressed_indices"][:count], tensors["sink"], key_tile=KEY_TILE,
    )
    tensors["output"][:count].copy_(result.flatten(1))


def build_sparse_specs(rows=4, active=3, kind="mixed"):
    """构造分组缩放、跨页/重复索引、全屏蔽、sink 和非有效行哨兵。"""
    from golden import ScalarSpec, TensorSpec

    generator = torch.Generator().manual_seed(6381)
    query = (torch.randint(-16, 17, (rows, HEADS * HEAD_DIM), generator=generator).float() / 16).bfloat16()
    raw = torch.randint(-32, 33, (5, 128, 1, HEAD_DIM), generator=generator).float() / 4
    window = raw.to(torch.float8_e4m3fn)
    window_scale = (torch.randint(4, 33, (5, 128, 1, 16), generator=generator).float() / 127).bfloat16()
    compressed = torch.randint(0, 256, (7, 128, 1, HEAD_DIM // 2), generator=generator, dtype=torch.uint8)
    compressed_scale = (torch.randint(4, 33, (7, 128, 1, 32), generator=generator).float() / 63).bfloat16()
    wi = torch.randint(0, 5 * 128, (rows, WINDOW), generator=generator, dtype=torch.int32)
    ci = torch.randint(0, 7 * 128, (rows, TOPK), generator=generator, dtype=torch.int32)
    wi[:, 13::17] = -1
    ci[:, 7::19] = -1
    wi[:, 1] = wi[:, 0]
    ci[:, 1] = ci[:, 0]
    sinks = torch.linspace(-3, 3, HEADS)
    if kind == "masked":
        wi.fill_(-1)
        ci.fill_(-1)
    elif kind == "window":
        ci.fill_(-1)
    elif kind == "compressed":
        wi.fill_(-1)
    elif kind == "sink":
        sinks.fill_(40)
    query[active:] = float("nan")
    values = {
        "query": query, "window": window, "window_scale": window_scale,
        "compressed": compressed, "compressed_scale": compressed_scale,
        "window_indices": wi, "compressed_indices": ci, "sink": sinks,
        "output": torch.full_like(query, -777),
    }
    specs = [TensorSpec(name, list(value.shape), value.dtype, init_value=value) for name, value in values.items()]
    specs.append(ScalarSpec("num_tokens", torch.int32, active))
    return specs


if __name__ == "__main__":
    import argparse
    from golden import ratio_allclose, run

    parser = argparse.ArgumentParser(description="校验单个 MIX 任务的混合量化稀疏 Attention。")
    parser.add_argument("-d", "--device", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--case", choices=("basic", "all"), default="all")
    args = parser.parse_args()
    cases = [(4, 3, "mixed")]
    if args.case == "all":
        cases += [(2, 0, "mixed"), (1, 1, "mixed"), (7, 5, "masked"), (3, 2, "window"),
                  (3, 2, "compressed"), (3, 2, "sink"), (35, 33, "mixed")]
    for rows, active, kind in cases:
        print(f"[SPARSE_ATTENTION_MIX_DSL] {rows=} {active=} {kind=}", flush=True)
        result = run(
            fn=validate_sparse_attention,
            specs=build_sparse_specs(rows, active, kind),
            golden_fn=golden_sparse,
            compare_fn={"output": ratio_allclose(atol=0.003, rtol=0.03, max_error_ratio=0.0)},
            config=dict(backend_type="a5", execution_mode=False, device_id=args.device),
            compile_only=args.compile_only,
            save_data=True,
        )
        if not result.passed:
            raise SystemExit(result.error or 1)
        if not args.compile_only:
            actual = torch.load(result.work_dir / "data/out/output.pt", weights_only=True)
            initial = torch.load(result.work_dir / "data/in/output.pt", weights_only=True)
            torch.testing.assert_close(actual[active:], initial[active:], rtol=0, atol=0)
            if kind == "masked":
                torch.testing.assert_close(actual[:active], torch.zeros_like(actual[:active]), rtol=0, atol=0)
        print(f"[产物] {result.work_dir}", flush=True)

