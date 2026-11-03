Shared input quantization and QA/KV DSL projection -- 2026-11-03
==============================================================

Scope and source
----------------
qkv_proj_dsl.py implements shared BF16 input quantization, QA projection
and KV projection. mxfp8_dsl.py owns the shared OCP MXFP8 vector quantizer
and M80/N128/K128 matrix primitive, extracted from the preceding o_proj
implementation. Both input projection and o_proj requantization consume
the same quantization kernel; the composed SWA manifest registers it once.

PyPTO/simpler schedules 16x256 AIV quantization tasks and disjoint AIC
projection tiles. Existing NZ weight bytes and paired E8M0 scales are
consumed directly. Matrix outputs retain BF16 rounding, padded quantizer
rows contain zero payload and scale codes, and projection tasks preserve
inactive output rows. Explicit runtime counts distinguish active rows from
the static M80/M16 capacity of the physical views.

The public qkv_proj_rope interfaces and existing callers are preserved.
Q RMSNorm, QB projection and Q RoPE remain their existing implementation;
the next migration stage can replace them separately. The now-unused
proj_q_a.cpp and proj_kv.cpp entries were removed. No C++ implementation
or PyPTO/CANNBotDSL framework code was added or modified in this step.

Issue #1666 alignment
---------------------
Upstream PR #1667, commit e04d7b670d6c99db227b8a514fd7a799f5711cba,
closed issue #1666 by aligning window-cache scale rounding with recipes:
  raw_scale = FP32(max(amax, 1e-4) * (1/448))
  scale_bits = (bits(raw_scale) + 0x007fffff) & 0xff800000
  scale = BF16(float_from_bits(scale_bits))

The DSL publisher already used this upward power-of-two rule. The common
quant.quantize_mxfp8_cache still had the old direct BF16 scale, so that
helper now follows the upstream bit rule. window_kv_dsl re-exports this
common quantizer for its golden and existing consumers, removing its
separate float64 log2 reference. Historical-cache initialization and new
publication now use the same rule. For amax=1 the scale is 0.00390625.
The OCP E8M0 activation quantizer used by GEMMs is unchanged.

upstream-1666-fix.json records the inspected commit and its three patches.
The upstream layer attention-output outlier budget increase from 1.5% to
2% was not imported. This branch's original 1.5% budget and pointwise
tolerances were retained and passed the complete layer0 EP2 check.

Validation
----------
validation-summary.json records task allocations and exact pass lines:
- Standalone 0/17/81 active-token cases pass, including an M80 boundary.
  FP8 payload and E8M0 scales compare byte-for-byte. QA/KV use the existing
  projection tolerance with exact inactive-output sentinel preservation.
- CPU cache check passes 2069 groups, including zero/clamp inputs and
  adjacent FP32 values around power-of-two boundaries.
- Two-update cache replay passes with 72 physical / 71 active tokens and
  13 pages, including exact payload snapshots and zero-tolerance scales.
- Complete 72-token SWA and layer0 EP2 pass all existing comparators.
  These are generated-weight checks, not whole-network checkpoint accuracy.

The first stage compile attempt had an incorrect golden comparator import;
it failed before NPU execution and was corrected. Its log is diagnostic
history. No numerical thresholds were relaxed and no performance A/B claim
is made from these correctness-run timings.

Artifacts and reproduction
--------------------------
fixtures/ freezes the first 17-active and final 81-active stage captures,
including generated NZ weights. .pt.gz files are lossless gzip archives.
data/out files hold CPU goldens; device comparison results are in the logs.
The multi-case harness reused one build directory, overwriting earlier
0/17-active data there. Those cases remain reproducible from the entry.
Cache replay, SWA and layer runs did not request save_data; their inputs
remain reproducible through the existing deterministic model constructors.

generated/ contains device exports, ABI manifests, entry wrappers, kernel
configs, orchestration source and available runtime incore_*.bin text
images. The composed SWA configuration shows the shared quantizer and both
new projections. ELF files and installed toolchains are excluded. Text
workspace paths are normalized to ${WORKSPACE}. artifact-manifest.json
lists source paths, stored file sizes and model-source files.

The paired framework requirements are unchanged from the preceding archive.
Use ../2026-11-03-o-proj/companion-patches against its recorded PyPTO and
CANNBotDSL bases. Do not stack cumulative patches from different captures.
Toolchain: Python 3.12.15; CANN version.info 9.2.0 under cann-9.3.0;
PTOAS 0.70; simpler 6b6b88f2353cca12cfafe809750b50b59931b0d1;
PTO ISA 217bb98952a8f9004b95fe938da97d8d5f68f702.

From the model repository root with the paired environment active and an
A5 device supplied through task-submit in TASK_DEVICE:
  python models/deepseek_v4_1_flash/qkv_proj_dsl.py --case all -d "$TASK_DEVICE"
  python models/deepseek_v4_1_flash/window_kv_dsl.py --replay --tokens 72 --num-tokens 71 --pages 13 -d "$TASK_DEVICE"
  python models/deepseek_v4_1_flash/decode_swa.py -d "$TASK_DEVICE"
For two allocated devices, pass their comma-separated IDs:
  python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d "$TASK_DEVICE"

Torch is used only on CPU for input preparation and numerical validation.
