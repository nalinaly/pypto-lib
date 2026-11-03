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
