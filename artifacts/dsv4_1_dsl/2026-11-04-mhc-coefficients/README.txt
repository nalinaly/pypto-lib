DeepSeek V4.1 mHC coefficient integration — 2026-11-04
====================================================

Branch: https://github.com/nalinaly/pypto-lib/tree/dsv4.1-dsl
Model base: 1e9a8a4. Sources: models/deepseek_v4_1_flash/hc.py and hc_dsl.py.

Implemented
-----------
The shared hc_project and hc_sinkhorn interfaces now schedule three DSL
leaves: inverse_rms_row (AIV), project_hf32_tile (AIC), and coefficients_row
(AIV). PyPTO/simpler owns task creation and dependencies. The coefficient
helper returns the same joined readiness contract used by Attention and MoE.
The hand-written hc_linear.cpp and hc_sinkhorn.cpp leaves have no remaining
consumers and are removed. No framework/compiler changes were made.

Inverse RMS consumes the original four FP32 residual streams, before any
collapse/BF16 rounding. Statistics remain FP32 and inverse RMS uses square
root plus precise division. Its reduction grouping differs from the prior
PyPTO vector implementation; bitwise equality with that implementation is
not asserted.

The AIC leaf explicitly selects HF32 with nearest-away rounding for both
operands. Four K5120 splits retain K64 accumulation steps and FP32 partials.
PyPTO schedules M16 tasks; each leaf clips its input/output to runtime-valid
rows and stores 24 coefficient columns at a stride of 32. The scratch row
count remains rounded to a multiple of 16. Full-FP32 projection
(hf32=False) retains its existing PyPTO implementation, using the new shared
DSL inverse RMS and Sinkhorn leaves.

Coefficient generation reduces partials as ((p0+p1)+p2)+p3, multiplies by
inverse RMS and learned scales, adds bias, and computes pre/post Sigmoid
and the 4x4 Sinkhorn residual matrix. Initial row softmax, epsilon placement,
20 normalization iterations, matrix orientation, FP32 storage and delayed
next_pre_mix semantics remain unchanged. The collapse-only hc_head and
prefill's padding-aware post remain in their existing implementations.

Validation
----------
With the paired environment active, run from the repository root using
allocated A5 devices. CPU Torch only prepares data and computes references.

python models/deepseek_v4_1_flash/hc.py --op coefficients --batch 1 -d <device>
python models/deepseek_v4_1_flash/hc.py --op coefficients --batch 17 -d <device>
python models/deepseek_v4_1_flash/hc.py --op coefficients --batch 72 -d <device>
python models/deepseek_v4_1_flash/hc.py --op project --full-fp32 --batch 17 -d <device>
python models/deepseek_v4_1_flash/decode_swa.py -d <device>
python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d <device0,device1>

Nine pre/Sinkhorn/project cases passed at 1,17,72 physical rows. Projection
uses zero-outlier tolerances, exact midpoint probes for positive/negative
values on both operands, and exact preservation checks on HF32 output
padding. RMS cases include zero and near-zero streams. Sinkhorn covers
zero/cancelling partials, Sigmoid saturation and NaNs in unused input
padding. In particular, [2^24,1,1,-2^24] distinguishes sequential from
pairwise split-K reduction.

The full-FP32 projection case passed over its defined output region, and
its exact probes distinguish it from HF32 rounding. Its unused scratch
columns have no preservation contract; the initial new padding assertion
was removed for this existing branch. Numeric tolerances over defined
outputs were retained. Inverse RMS padding is checked in both modes.

The 72-token SWA composition passed all original comparisons (RUN PASS
39.97s). Layer0 EP2 passed all 16 output comparisons on devices 0,1
(RUN PASS 152.29s). See validation-summary.json. Original model
golden numerics and thresholds are retained, including the layer's 1.5%
attention-output outlier budget. The coefficient CPU helper was extracted
for reuse and compared element-for-element with the previous revision in
both precision modes. Run times include build and validation overhead;
this archive makes no performance A/B claim.

Contents and replay
-------------------
fixtures/ freezes the inputs and CPU goldens of all ten standalone cases
as .pt.gz, including probe weights and padding sentinels. data/out files
are CPU references, not saved actual device outputs; the logs report
actual device comparisons. Decompress one fixture directory with its in/
and out/ children, then pass the matching --op, --batch and --golden-data
<directory>. Use --full-fp32 for fp32-17. Replay requires one specific --op.

generated/ contains device_callable.hpp, ABI wrappers/manifests, kernel
configuration, dynamic entry metadata, orchestration, generated PyPTO
in-core sources and runtime text-image bytes. Native ELF files, object
files and installed toolchains are omitted. SWA/layer data was not duplicated;
the deterministic fixture builders remain in the model sources.

validation/ retains passing runs and intermediate authoring diagnostics:
reshape metadata initially used an unrelated scalar instead of tensor.dim;
the standalone harness initially attempted to expose a compile-time flag
as a runtime spec; and the first full-FP32 check asserted preservation of
undefined scratch columns. Those diagnostics are resolved in the current
source and do not require compiler patches.

model-snapshot/ and workflow-snapshots/ preserve source and capture scripts.
artifact-manifest.json inventories paths, sizes, encoding and provenance.
No supplemental file digests or digest-validation steps are used. Text
workspace prefixes are normalized to ${WORKSPACE}; framework-generated
ABI/cache identifiers remain compiler data.

Environment
-----------
Linux aarch64, A5 / dav-3510, Python 3.12.15, Torch 2.12.0+cpu.
SDK directory cann-9.3.0; version.info reports 9.2.0.
PyPTO ae6f6d76 and CANNBotDSL 33c2e5f4 with existing local Python patches;
PTOAS 0.70, simpler 6b6b88f2353c, PTO ISA 217bb98952a8.
Use the existing paired dependency setup and cumulative patches in
../2026-11-03-o-proj/companion-patches for a fresh environment. No new
patches are required by this step. Validation uses the program runtime.
