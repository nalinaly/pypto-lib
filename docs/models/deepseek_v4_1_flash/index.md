# DeepSeek V4.1 Flash

> **Temporary architecture documentation freeze:** Defer architecture documentation changes until the model architecture is finalized. Development priorities may be updated as work progresses.

`models/deepseek_v4_1_flash/` contains the DeepSeek V4.1 Flash implementation.

## Development priorities

| Field | Current priority |
| --- | --- |
| Performance target | 72-token decode. Apply optimizations to the existing shared implementation and adapt all affected callers. |
| Deferred performance work | Prefill and decode with other token counts. Their performance need not be preserved or optimized at this stage; reconsider performance-driven branches or file splits when these become active optimization targets. |
| Compatibility requirements | Preserve correctness and interface contracts for already supported paths, including prefill and other token counts. Update affected callers to use the revised implementation. The optimized C++ routed FFN requires A5 device execution: the pinned CPU simulator does not model its intra-block events, shared L1 access, or dual-AIV Fixpipe transfers. Simulation requests fail explicitly; they do not select a different numerical implementation. |

## Window KV validation

The shared window-cache publisher uses
[`window_kv_dsl.py`](../../../models/deepseek_v4_1_flash/window_kv_dsl.py).
It requires CANNBotDSL device-callable export and the matching PyPTO
foreign-kernel JIT integration, including BF16, FP8 E4M3FN and INDEX argument
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

The companion PyPTO adapter owns dtype conversion (including E8M0), device
export invocation, runtime ABI wrappers and external task configuration.
PyPTO JIT also propagates shaped `tensor.view` metadata so packed MX scales
can cross the DSL boundary as an explicit ND storage view. CANNBotDSL
retains generic device export and a compatibility forwarder.

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
