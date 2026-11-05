# DeepSeek V4.1 Flash

> **Temporary architecture documentation freeze:** Defer architecture documentation changes until the model architecture is finalized. Development priorities may be updated as work progresses.

`models/deepseek_v4_1_flash/` contains the DeepSeek V4.1 Flash implementation.

## Development priorities

| Field | Current priority |
| --- | --- |
| Performance target | 72-token decode. Apply optimizations to the existing shared implementation and adapt all affected callers. |
| Deferred performance work | Prefill and decode with other token counts. Their performance need not be preserved or optimized at this stage; reconsider performance-driven branches or file splits when these become active optimization targets. |
| Compatibility requirements | Preserve correctness and interface contracts for already supported paths, including prefill and other token counts. Update affected callers to use the revised implementation. The DSL routed FFN requires A5 device execution; the CPU simulator is not a supported validation path. |

## Window KV validation

The shared window-cache publisher uses
[`window_kv_dsl.py`](../../../models/deepseek_v4_1_flash/window_kv_dsl.py).
It requires CANNBotDSL SourceBundle/device ABI generation and the matching
PyPTO foreign-kernel JIT integration, including BF16, FP8 E4M3FN and INDEX argument
support. These capabilities are required even when running CPU goldens that
import the model. See the dependency setup in
[Installation and Environment](../../get-started/installation.md).

The publisher consumes BF16 KV projections, accumulates RMSNorm in FP32,
rounds to BF16, applies adjacent-pair RoPE, rounds to BF16 again, and stores
E4M3FN values with one BF16 scale per 32 elements. The scale is
`2 ** ceil(log2(max(amax, 1e-4) / 448))`; FP8 conversion uses nearest-even
rounding. Payload and scales retain the model's separate paged buffers.
Previous cache rows are interpreted using their stored scales.
RMS reciprocal and FP8 normalization use precise FP32 division: a one-ULP
reciprocal error near a BF16 tie can otherwise change an FP8 cache value.

The common CPU cache initializer and DSL golden now share
`quant.quantize_mxfp8_cache`, using the FP32 bit-rounding rule from
[upstream #1667](https://github.com/hw-native-sys/pypto-lib/pull/1667), which
closed [#1666](https://github.com/hw-native-sys/pypto-lib/issues/1666).
This aligns newly prepared historical cache rows with the publisher's
existing upward power-of-two rule. It does not change the OCP E8M0
activation quantizer used by matrix projections. Layer validation retains
this branch's 1.5% attention-output outlier budget; the upstream change to
2% is not part of this backport.

PyPTO schedules one AIV task per physical page. Each DSL task processes its
matching slots in token order: the last active token wins a repeated slot,
negative/out-of-range slots are skipped, and unselected rows retain their
input bytes. A bounded completion array joins the tasks and carries explicit
dependencies from prior consumers.
Each page is a one-block SPMD task with `sync_start=True`; this flag does not
introduce a collective start barrier across the separate page tasks.

From the repository root, with the selected environment active and an
allocated A5 device in `TASK_DEVICE`:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python models/deepseek_v4_1_flash/window_kv_dsl.py -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/window_kv_dsl.py --replay --tokens 72 --num-tokens 71 --pages 13 -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/window_kv_dsl.py --num-tokens 0 -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/window_kv_dsl.py --tokens 1 --pages 1 -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/decode_swa.py -d "$TASK_DEVICE"
```

The replay case consumes a full cache snapshot after each of two updates
before allowing the next overwrite. Payload comparisons are byte-exact;
scale comparisons use zero tolerance. The composed SWA entry retains its
existing numerical tolerances and requires 72 query tokens. `--compile-only`
checks code generation without executing on a device.

For a complete SWA decode layer, allocate two A5 devices and pass their
comma-separated IDs through `TASK_DEVICE`:

```bash
python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d "$TASK_DEVICE"
```

This composes the DSL publisher with the existing Attention and EP MoE and
checks 72 active tokens per rank using the layer entry's existing comparators.
It validates one decode layer with generated weights; it is not a whole-model
checkpoint validation.

## SWA Attention in DSL

The split-cache SWA entry uses
[`swa_dsl.py`](../../../models/deepseek_v4_1_flash/swa_dsl.py) when the local
head count is 64 (TP1). One mixed task consumes one Query's complete 128-entry
window, with one AIC and two AIV participants. PyPTO/simpler owns Query task
dispatch, the bounded completion pool, and cache dependencies. The leaf has
no cross-task Softmax reduction or launch-wide barrier.

The leaf reads the existing separate FP8 E4M3 payload and group-32 BF16 scale
buffers using physical cache indices. It dequantizes to BF16, computes QK in
FP32, masks invalid columns, rounds unnormalized probabilities to BF16 for
PV, and applies the sink contribution during final FP32 normalization before
storing BF16 output. It consumes the stored scale values without selecting a
new quantizer. Inactive output rows retain their caller-provided contents.

The packed544 entry, other TP configurations, and compressed-KV Attention
retain their existing implementations during this staged migration. Extending
the DSL reader to compressed KV requires a separate FP4 contract review.

Run the existing standalone numerical checks with the selected environment
active and an allocated A5 device in `TASK_DEVICE`:

```bash
python models/deepseek_v4_1_flash/sparse_attention.py --mode swa --swa-case all -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/decode_swa.py -d "$TASK_DEVICE"
```

The core cases cover zero active Queries, masked windows, cross-page and wrap
indices, irregular/repeated indices, and inactive output sentinels. The
existing core and composed Attention error thresholds are unchanged. Use the
layer-0 command above to validate composition with EP MoE.

## Shared input and QA/KV projection in DSL

[`qkv_proj_dsl.py`](../../../models/deepseek_v4_1_flash/qkv_proj_dsl.py)
quantizes the shared BF16 input to OCP MXFP8, projects QA to BF16 query
latents, and projects KV to BF16 rows. The existing `qkv_proj_rope.py`
interface uses these leaves together with the Q-branch DSL stages below.

Input and output projection share the vector quantizer and NZ-weight
matrix primitive in
[`mxfp8_dsl.py`](../../../models/deepseek_v4_1_flash/mxfp8_dsl.py).
PyPTO/simpler schedules 16x256 quantization tiles and M80/N128/K128
projection tiles. NZ weights, paired E8M0 scale packing, BF16 output
rounding and inactive-row preservation retain the model's contract.
The shared input quantizer fills its padded rows with zero payload and
zero E8M0 codes. The existing Q/KV callers keep the same interface.

Run the standalone stages with an allocated A5 device:

```bash
python models/deepseek_v4_1_flash/qkv_proj_dsl.py --case all -d "$TASK_DEVICE"
```

The default cases cover 0, 17 and 81 active tokens, including an M80
boundary and inactive NaN input sentinels. FP8/E8M0 output bytes are
compared exactly; QA/KV use the existing projection tolerance with exact
inactive-output checks. The SWA and layer-0 commands exercise composition
with Q normalization, QB, RoPE and the DSL output projection.

## Q normalization, QB and RoPE in DSL

[`q_branch_dsl.py`](../../../models/deepseek_v4_1_flash/q_branch_dsl.py)
completes the Q branch with three task types: AIV RMSNorm/quantization,
AIC QB projection, and AIV adjacent-pair RoPE. PyPTO/simpler owns their
dependencies. `q_proj_rope` keeps its four outputs: BF16 query latent,
BF16 Query, MXFP8 QB payload and packed E8M0 scales for Indexer reuse.

RMSNorm accumulates in FP32, uses square root and precise FP32 division,
rounds to BF16, then applies the shared OCP quantizer. QB writes an explicit
BF16 intermediate before RoPE. RoPE keeps the non-rotary prefix and stores
the final 64 lanes as adjacent real/imaginary pairs with BF16 rounding.
Padded payload/scales are initialized to zero, including inactive M80
groups; inactive latent and Query output rows retain their input contents.

The intermediate QB buffer makes each stage independently inspectable;
this implementation does not depend on a hand-written Cube/Vector
handshake. The obsolete native QB entry and its private header are removed.
This completes the TP1 SWA projection/cache/Attention/output-projection
compute path in DSL. The mHC boundary and coefficient leaves are described
below; the MoE kernels retain their separate implementations.

With an allocated A5 device, run:

```bash
python models/deepseek_v4_1_flash/qkv_proj_rope.py --stage all --case all -d "$TASK_DEVICE"
```

The default stages cover 0/17/81 active tokens, zero and near-zero rows,
inactive NaN input sentinels, and an M80 boundary. They check normalized
BF16 values, exact FP8/E8M0 bytes, QB output before RoPE, final Q, and exact
inactive-output preservation. The composed SWA and layer entries keep
their original golden functions and comparison thresholds.

## mHC boundary operations in DSL

[`hc_dsl.py`](../../../models/deepseek_v4_1_flash/hc_dsl.py) provides the
AIV leaves used by the shared `hc_head_norm` and `hc_post` interfaces in
[`hc.py`](../../../models/deepseek_v4_1_flash/hc.py). PyPTO/simpler schedules
one task per physical row with a bounded pool of completion dependencies.
The collapse helper returns a joined TaskId; residual expansion honors the
existing caller-supplied dependencies.

Collapse uses the incoming delayed pre-mix, multiplies four FP32 streams,
sums them as `(y0 + y1) + (y2 + y3)`, and rounds to BF16 before computing
RMSNorm in FP32. The inverse uses square root and precise FP32 division;
the weighted normalized output is BF16. Residual expansion casts the BF16
sublayer output to FP32, then adds residual streams 0 through 3 in order
using `residual_mix[input_stream, output_stream]`. Its output stays FP32.
Both helpers process every physical row; existing callers own padding clears.

The same shared residual expansion serves Attention and MoE callers. The
collapse-only `hc_head` and prefill's padding-aware `prefill_sp_post` remain
on their existing paths. Their migration requires separate validation of
their distinct contracts.

With an allocated A5 device, run the standalone boundary checks and the
SWA/layer entries documented above:

```bash
python models/deepseek_v4_1_flash/hc.py --op boundary --batch 17 -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/hc.py --op boundary --batch 1 -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/hc.py --op boundary --batch 72 -d "$TASK_DEVICE"
```

The fused-norm entry checks zero and near-zero rows with no allowed
outliers. The post entry checks the existing model golden and exact
elementwise equality against sequential FP32 accumulation. Composed model
goldens and acceptance thresholds are unchanged.

## mHC coefficient generation in DSL

The shared `hc_project` and `hc_sinkhorn` interfaces use three DSL leaves
from `hc_dsl.py`: AIV inverse RMS, AIC HF32 projection, and AIV coefficient
generation. Their FP32 scratch layout and the joined coefficient TaskId
remain compatible with the Attention and MoE callers. The obsolete
`hc_linear.cpp` and `hc_sinkhorn.cpp` leaves are removed.

RMS statistics come from the original four FP32 residual streams, before
collapse or BF16 rounding. Inverse RMS uses square root and precise FP32
division. Decode projection explicitly enables nearest-away HF32 rounding
for both operands, accumulates K64 steps in FP32, and retains four split-K
segments. PyPTO schedules M16 projection tiles with explicit runtime row
counts; each task writes only its active rows and 24 coefficient columns.

The coefficient leaf merges partials in order `((p0 + p1) + p2) + p3`,
applies inverse RMS, learned scales and biases, then computes Sigmoid and
the 4x4 residual Sinkhorn matrix. Epsilon placement, 20 iterations, FP32
storage and delayed `next_pre_mix` semantics are preserved. Full-FP32
projection (`hf32=False`) retains its PyPTO implementation and uses the
shared DSL RMS and coefficient leaves.

With an allocated A5 device:

```bash
python models/deepseek_v4_1_flash/hc.py --op coefficients --batch 17 -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/hc.py --op coefficients --batch 1 -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/hc.py --op coefficients --batch 72 -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/hc.py --op project --full-fp32 --batch 17 -d "$TASK_DEVICE"
```

The coefficient cases include independent Sinkhorn, RMS/projection and
composed mHC-pre checks. They cover split-K cancellation order, Sigmoid
saturation, zero/near-zero rows, NaN input padding, and exact positive and
negative midpoint probes for both HF32 operands. The HF32 stage also
checks untouched output padding. The legacy FP32 branch is checked over
its defined coefficient columns; its unused scratch columns have no
preservation contract. The SWA and layer-0 commands above validate the
same implementation with the original model goldens and thresholds.

## Output projection in DSL

For TP1, [`o_proj_dsl.py`](../../../models/deepseek_v4_1_flash/o_proj_dsl.py)
implements inverse RoPE with OCP MXFP8 quantization, grouped WoA, BF16
requantization, and WoB. PyPTO/simpler schedules head/row AIV tasks and
M80/N128 AIC projection tiles. The other TP configurations retain the
existing implementation until their local head/group dimensions are migrated
and validated.

The leaves consume the existing NZ weight bytes and paired E8M0 scale
packing directly. Inverse RoPE rounds to BF16 before quantization; WoA
rounds to BF16 before requantization; WoB stores BF16. Explicit runtime
valid-row counts protect partial output tiles and zero inactive rows.
The existing numerical thresholds remain unchanged.

The companion PyPTO adapter recognizes `cb.kernel` and owns dtype conversion
(including E8M0), compiler invocation, runtime ABI wrappers and external task configuration.
PyPTO JIT also propagates shaped `tensor.view` metadata so packed MX scales
can cross the DSL boundary as an explicit ND storage view. CANNBotDSL
provides generic compilation, SourceBundle output and the device ABI.

With an allocated A5 device in `TASK_DEVICE`, run:

```bash
python models/deepseek_v4_1_flash/o_proj_dsl.py --stage rope --active 17 -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/o_proj_dsl.py --stage mx --active 17 -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/o_proj.py --case all -d "$TASK_DEVICE"
```

The stage checks compare FP8/E8M0 bytes exactly and test MX projection with
nonuniform scales at zero tolerance. The composed checks cover empty and
tail inputs, 72-token output, and a second 80-row chunk with FP32 HC post.
The SWA and layer-0 entries above exercise the same projection implementation.

## Shared-expert FFN in DSL

[`expert_shared.py`](../../../models/deepseek_v4_1_flash/expert_shared.py)
schedules W1/W3 projection, SwiGLU with MXFP8 requantization, and W2 projection
through DSL leaves. PyPTO/simpler owns the task graph and tensor dependencies.
The existing decode and prefill callers use the same shared-expert interface.

[`expert_shared_projection_dsl.py`](../../../models/deepseek_v4_1_flash/expert_shared_projection_dsl.py)
consumes ordinary KN weight payloads and the existing MX_A_ZZ/MX_B_NN scale
bytes. M16/N128 AIC tasks perform consecutive K256 accumulations, including
the final K256 block of the 2304-wide W2 reduction. W1/W3 retain FP32 outputs;
W2 writes BF16 and limits stores to valid output rows.

[`expert_shared_dsl.py`](../../../models/deepseek_v4_1_flash/expert_shared_dsl.py)
rounds gate/up to BF16, applies the existing asymmetric clipping and SwiGLU,
and rounds the activation to BF16 before requantization. Its group-32 scale
is `ceil_pow2(max(amax, 1e-4) / 448)`, encoded as E8M0. This retains the
shared expert's upward scale policy; the attention OCP quantizer is separate.

With an allocated A5 device in `TASK_DEVICE`, run:

```bash
python models/deepseek_v4_1_flash/expert_shared_dsl.py --rows 16 --case all -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/expert_shared_dsl.py --rows 80 --case all -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/expert_shared_projection_dsl.py --rows 17 --stage all -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/expert_shared.py -d "$TASK_DEVICE"
```

The vector checks compare FP8 payload and E8M0 scale bytes exactly, including
zero groups, clipping, BF16 midpoint and scale-transition probes. Projection
checks cover nonuniform scales, multiple M16 tiles and a partial output row.
The full FFN retains its existing golden and `ratio_reldiff` thresholds.
The layer-0 EP2 command above validates its integration with the decode layer.


## Local MoE router tasks in DSL

[`gate.py`](../../../models/deepseek_v4_1_flash/gate.py) calls the local leaves in
[`gate_dsl.py`](../../../models/deepseek_v4_1_flash/gate_dsl.py). The task mapping is:

| Original task | DSL task | Computation |
| --- | --- | --- |
| `rms_norm` | `dsl_gate_norm` | FP32 chunked RMSNorm, BF16 rounding, padded FP32 output |
| `x_norm_mx_quant` | `dsl_gate_quant` | Group-32 upward MXFP8 and paired E8M0 scales |
| `gate` | `dsl_gate` | FP32 M16/N16/K512 projection and sqrt-softplus scores |
| `route_sort` | `dsl_route_sort` | Stable TopK and normalization of unbiased scores |

Only tasks without cross-rank operations are replaced. Dispatch, remote stores,
notifications, deferred waits and the mixed compute/communication
`ffn_reduce_scatter` task retain their existing implementations. The local
`gate_pre_route` initialization also remains in PyPTO. PyPTO/simpler owns the
task graph; each foreign MIX invocation reserves its AIC/AIV participants with
`sync_start=True`.

The input quantizer and shared expert reuse `_quant_upward_row` in
[`mxfp8_dsl.py`](../../../models/deepseek_v4_1_flash/mxfp8_dsl.py). They retain
`ceil_pow2(max(amax, 1e-4) / 448)`, nearest-even E4M3 conversion, and scale storage
`[M/16, K/64, 16, 2]`. A zero group has scale code 105. The attention OCP helper
keeps its own scale rule. Router projection uses full FP32 mode. TopK uses
repeated maximum reductions with the lower expert index winning exact ties;
correction bias affects selection only. Inactive rows retain zero route ids
and weights.

With an allocated A5 device in `TASK_DEVICE`, run:

```bash
python models/deepseek_v4_1_flash/gate_dsl.py -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/gate.py --num-tokens 71 -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/gate.py --num-tokens 0 -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/gate.py --num-tokens 80 --save-data -d "$TASK_DEVICE"
```

Leaf checks compare quantized bytes and stable route ids exactly, with separate
unbiased-weight checks. The composed gate retains its original goldens and
comparators. The shared `gate_normalized` interface uses the same quantization,
projection and routing leaves for pre-normalized callers.

The official `cann/cannbot-dsl` samples were refreshed to `48fa202` for this work.
Their `dynamic_mx_quant_sr` sample uses stochastic rounding and a distinct
all-zero scale rule; `engram_gate` implements residual gating. Neither is a
drop-in replacement for this router. The execution environment continues to
use the generic DSL compiler at `91feb22` with PyPTO integration at `4a65a9c`.

## Local routed-expert FFN in DSL

[`expert_routed.py`](../../../models/deepseek_v4_1_flash/expert_routed.py)
replaces the former `fused_expert_ffn` C++ task through
[`expert_routed_dsl.py`](../../../models/deepseek_v4_1_flash/expert_routed_dsl.py).
The replacement has three explicitly joined local compute phases:

| Task | Computation | Engine |
| --- | --- | --- |
| `dsl_routed_up` | MXFP4 W1/W3 projections with FP32 outputs | MIX |
| `dsl_routed_act_quant` | FP32 SwiGLU, route-weight multiplication and upward MXFP8 | AIV |
| `dsl_routed_down` | MXFP4 W2 projection into two FP32 partial planes | MIX |

All phases consume local tensors. The first phase depends on `input_ready`;
the returned TaskId joins every final projection worker. The existing
`ffn_reduce_scatter`, its remote stores and all notification/wait tasks are
unchanged. The final sum of the two FP32 partials and BF16 rounding remain
in that existing consumer. The former C++ implementation is removed.

Weights retain the model's 256-by-256 panel order with adjacent N elements
packed into each byte, unlike the official A8W4 sample's K-adjacent packing.
Each MIX projection uses two AIVs to widen all 16 E2M1 codes exactly to E4M3
and publish NZ weight tiles to the AIC. It does not introduce the sample's
temporary 1/64 weight scaling. Packed MX_A_ZZ/MX_B_NN scale storage is retained;
the PyPTO bridge receives explicit ND storage views.

The activation remains FP32 through clipping, SwiGLU and route-weight
multiplication. There is no intermediate BF16 rounding. Quantization reuses
the shared upward-scale helper and keeps zero padded rows finite. W2 splits
the 2304-wide hidden reduction at 1024, yielding 1024- and 1280-column
partials. Their row positions use the original unpadded expert-count prefix
and the original plane stride.

The compute phases use private M16-aligned expert slabs and explicit
completion joins. Internal padding does not change receive or return buffers.
For the current EP2 configuration, the declared intermediate gate/up/payload/
scale capacity is about 76.2 MiB per rank. This is a correctness-first staged
implementation; on-chip fusion and performance tuning are deferred.

With an allocated A5 device in `TASK_DEVICE`, run:

```bash
python models/deepseek_v4_1_flash/expert_routed_dsl.py -d "$TASK_DEVICE"
python models/deepseek_v4_1_flash/expert_routed.py --case all -d "$TASK_DEVICE"
```

The leaf entry checks every FP4 encoding and NZ position, A8W4 projection,
weighted activation bytes, zero route weights, clipping and poisoned inactive
rows. The composed entry retains its original goldens and
`ratio_reldiff(diff_thd=2e-3, pct_thd=0.01)` threshold, with exact preservation
of inactive output sentinels. Its cases cover empty experts, receive capacity,
15/16/17 and 63/64/65 rows, and repeated invocations with changed counts.

## Local MoE output reduction in DSL

The decode `combine_reduce` computation in
[`decode_moe.py`](../../../models/deepseek_v4_1_flash/decode_moe.py) calls
[`moe_combine_dsl.py`](../../../models/deepseek_v4_1_flash/moe_combine_dsl.py).
The existing `combine_notify` and `combine_wait` scopes remain unchanged.
The routed tensor is a view of the current rank's communication window;
the DSL leaf receives its local address and performs no cross-rank operation.

Each `dsl_combine_reduce` AIV leaf handles up to two rows. It widens the shared
expert output to FP32, adds routed lanes 0 through 5 in order, and rounds once
to BF16 with nearest-even rounding. Inactive rows become zero without reading
their inputs. An explicit physical-row bound protects an odd final row.

PyPTO owns slicing and completion tracking. A bounded completion array is
initialized from the original `combine_wait` TaskId; each slot chains its
subsequent leaf tasks, and the returned join covers every output row before
mHC post-processing. No DSL or PyPTO framework changes are required.

With an allocated A5 device in `TASK_DEVICE`, run:

```bash
python models/deepseek_v4_1_flash/moe_combine_dsl.py -d "$TASK_DEVICE"
```

The bounded cases cover 5 physical rows with 3 or 0 active rows, and 80 rows
with 71 or 80 active rows. They compare BF16 outputs at zero tolerance,
including FP32 cancellation order, BF16 rounding midpoints, distinct routed
lanes, NaN inactive inputs and an untouched output guard row. The layer-0
EP2 entry validates the local communication-window view and the preserved
notification/wait boundary using the existing model comparators.

## Local MoE send staging in DSL

The decode `dispatch_stage` task uses `stage_local` in
[`moe_dispatch_dsl.py`](../../../models/deepseek_v4_1_flash/moe_dispatch_dsl.py).
Each `dsl_dispatch_stage` AIV task copies eight physical FP8 rows as unchanged
INT8 transport bytes and unpacks their E8M0 scale codes from `MX_A_ZZ` into ND
token rows. This performs no requantization or floating-point arithmetic.

The original write contract is preserved: payload staging includes inactive
physical rows, while scale staging writes only the active prefix after clamping
`num_tokens` to the physical capacity. Inactive scale rows retain their input
contents. Physical capacity is M16-aligned, and each eight-row scale slab owns
complete 64-byte cache lines. PyPTO slices the buffers, joins every staging task,
and passes that completion handle to the unchanged `dispatch_push` scope.
Metadata, sends, notifications, waits and receive gather retain their code.
Prefill retains its separate staging implementation.

With an allocated A5 device in `TASK_DEVICE`:

```bash
python models/deepseek_v4_1_flash/moe_dispatch_dsl.py --op stage -d "$TASK_DEVICE"
```

The ten staging cases cover negative and zero active counts, 7/8/9 and 15/16/17
rows, an active count above capacity, and the 72-active/80-physical decode shape.
All 256 byte codes occur in the fixtures. Payload and scale comparisons are
byte-exact, including inactive scale sentinels and an extra output guard slab.
The default `--op all` additionally runs the four receive-gather cases below.

## Local MoE receive gather in DSL

The decode `dispatch_gather` task in
[`decode_moe.py`](../../../models/deepseek_v4_1_flash/decode_moe.py) calls
[`moe_dispatch_dsl.py`](../../../models/deepseek_v4_1_flash/moe_dispatch_dsl.py).
Each `dsl_dispatch_gather` AIV task compacts up to 16 rows for one local expert
in source-rank order. FP8 payload bytes, E8M0 scale codes, FP32 route weights
and INT32 route indices retain their original bits. The scales are rearranged
from transport ND rows into the existing M16-paired `MX_A_ZZ` storage; no
quantization or floating-point arithmetic is performed.

The DSL receives local views of the current rank's receive windows. The
existing metadata publication/finalization, remote payload stores,
notification and deferred-wait scopes remain unchanged. PyPTO/simpler retains
the original push, wait and metadata dependencies and joins every local gather
before routed-expert compute. A bounded completion pool also retains those
dependencies when every expert is empty.

The final active scale tile has zero codes in unused row positions. Other
inactive payload, scale, weight and route regions remain untouched. Expert
capacity stays M16-aligned with equal source lanes. The intermediate ND scale
buffer is removed; each DSL task packs its scale tile in UB.

With an allocated A5 device in `TASK_DEVICE`:

```bash
python models/deepseek_v4_1_flash/moe_dispatch_dsl.py --op gather -d "$TASK_DEVICE"
```

The four bounded cases cover empty experts and sources, 15/16/17-row boundaries,
full receive capacity, and local windows containing four source lanes. All four
output buffers are compared byte-for-byte, including sentinel tails and an
extra output guard tile. Fixtures exercise all 256 payload and scale byte codes
and signed-zero route weights. The four-source case is a local data-layout
test; distributed integration is validated by the existing EP2 layer entry.

## C2A compressor projections in DSL

[`compressor_dsl.py`](../../../models/deepseek_v4_1_flash/compressor_dsl.py)
provides the shared C2A KV and gate projections used by decode and prefill.
Each `dsl_proj_cmp` AIC task retains the original M80/N64 tile and consecutive
K256 BF16 multiplications with FP32 accumulation and output. NZ weights retain
their original bytes; whole-tensor flattening exposes their physical storage
before the DSL loads the existing `[N/16, K/16, 16, 16]` faces.

KV and gate tasks are independent. PyPTO preserves the caller's readiness
dependency, bounds the completion pool, and joins both projections before
pooling or cache publication. C1A projections, C2A pooling, ring state,
quantization and communication kernels retain their implementations. The
obsolete PyPTO C2A projection body and tiling constants are removed. Neither
PyPTO nor CANNBotDSL requires a framework change.

With one A5 device allocated through the site's task queue:

```bash
python models/deepseek_v4_1_flash/compressor_dsl.py -d "$TASK_DEVICE"
```

The bounded cases cover NZ face coordinates, zero tokens, partial row tiles,
72 active rows, and completion-pool reuse across 321 rows. Structured cases
compare exactly with the CPU reference; the random BF16 case uses FP32
`atol=rtol=1e-4` with no allowed outliers. Every inactive output and guard row
must remain exact. The evidence archive additionally compares the original
PyPTO projection with the production DSL projection on identical A5 inputs
at zero numerical tolerance.

The complete-layer fixture now supports C2A full source layers, including
`--layer-id 2`, using the existing C2A fixture builder and acceptance policy
independently for each EP rank. The SWA acceptance policy is unchanged. The
common interface's SWA-only `attention_output` port is explicitly zeroed on
other attention branches; C2A's numerical attention boundary remains
`attention_hidden`, which is validated before the EP MoE comparison.

With two allocated A5 devices:

```bash
python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 2 -d "$TASK_DEVICE"
```

The shared prefill caller is compiled with the new interface. Its complete
device execution is outside this decode validation scope.

## DSL development branch and evidence

Continue DSL implementation and evidence collection on
[`nalinaly/pypto-lib:dsv4.1-dsl`](https://github.com/nalinaly/pypto-lib/tree/dsv4.1-dsl).
The [dated evidence archive](../../../artifacts/dsv4_1_dsl/2026-11-03/README.txt)
contains validation logs, frozen numerical data, L1 swimlanes, generated device
source, and companion toolchain patches. Preserve earlier captures when adding
new runs under `artifacts/dsv4_1_dsl/`.
The [SWA Attention archive](../../../artifacts/dsv4_1_dsl/2026-11-03-swa-attention/README.txt)
contains the subsequent MIX leaf's validation and generated device code.
The [output projection archive](../../../artifacts/dsv4_1_dsl/2026-11-03-o-proj/README.txt)
contains its staged validation, generated kernels and cumulative companion
patches, including the adapter migration to PyPTO.
The [QA/KV input archive](../../../artifacts/dsv4_1_dsl/2026-11-03-qkv-input/README.txt)
records the next projection stage and the #1666/#1667 cache-quantizer alignment.
The [Q-branch archive](../../../artifacts/dsv4_1_dsl/2026-11-03-q-branch/README.txt)
contains normalization/QB/RoPE stage evidence, frozen data and generated kernels.
The [mHC boundary archive](../../../artifacts/dsv4_1_dsl/2026-11-04-mhc-boundary/README.txt)
contains collapse/RMSNorm and FP32 residual-expansion validation and generated kernels.
The [mHC coefficient archive](../../../artifacts/dsv4_1_dsl/2026-11-04-mhc-coefficients/README.txt)
contains RMS/HF32/Sinkhorn validation, midpoint probes, frozen fixtures and generated kernels.
The [shared-expert archive](../../../artifacts/dsv4_1_dsl/2026-11-04-shared-expert/README.txt)
contains the new-DSL baseline, SwiGLU/quantization and projection checks, full FFN
and layer-0 EP2 validation, and the companion PyPTO call-specialization correction.
The [local router archive](../../../artifacts/dsv4_1_dsl/2026-11-04-gate/README.txt)
records the official sample refresh, input and router checks, generated leaves,
CPU fixtures, and the layer-0 EP2 regression with communication tasks retained.
The [routed-expert archive](../../../artifacts/dsv4_1_dsl/2026-11-04-routed-expert/README.txt)
contains the original baseline, leaf checks, full local FFN cases, generated
compute stages, CPU leaf fixtures, and layer-0 EP2 integration evidence.
The [combine archive](../../../artifacts/dsv4_1_dsl/2026-11-04-combine/README.txt)
contains the local reduction checks, communication-scope review, generated
leaf/wrapper/orchestration, CPU fixtures and the layer-0 EP2 regression.
The [receive-gather archive](../../../artifacts/dsv4_1_dsl/2026-11-04-dispatch-gather/README.txt)
contains byte-exact layout/compaction checks, unchanged communication-scope
review, generated code, CPU fixtures and the layer-0 EP2 regression.
The [send-staging archive](../../../artifacts/dsv4_1_dsl/2026-11-05-dispatch-stage/README.txt)
contains byte-exact staging and gather regressions, unchanged communication
review, generated code, CPU fixtures and the layer-0 EP2 regression.
The [C2A projection archive](../../../artifacts/dsv4_1_dsl/2026-11-05-c2a-projection/README.txt)
contains the original projection comparison, complete-layer fixture support,
generated sources, CPU references and the layer-2 EP2 regression.
