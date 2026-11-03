DeepSeek V4.1 mHC boundary integration — 2026-11-04
=================================================

Branch: https://github.com/nalinaly/pypto-lib/tree/dsv4.1-dsl
Model base: f1339fb. Sources: models/deepseek_v4_1_flash/hc.py and hc_dsl.py.

Implemented
-----------
The shared hc_head_norm and hc_post interfaces call CANNBotDSL AIV leaves.
PyPTO/simpler schedules one task per physical row with 12 completion slots.
The collapse entry returns a joined TaskId; post retains the caller's
explicit producer dependencies and its FP32 tensor return.

Collapse consumes the delayed incoming pre-mix. Four FP32 products are
summed pairwise, rounded to BF16, and normalized with FP32 RMS statistics,
square root and precise FP32 division. Final output is BF16. RMS reduction
uses a different vector grouping from the previous PyPTO implementation;
bitwise equality with that implementation is not asserted.
Post begins with the BF16 sublayer value widened to FP32 and multiplied by
post_mix, then adds residual streams 0,1,2,3 in order. Mixing matrices use
[input_stream, output_stream]. The stored output remains FP32. Both
operations process all physical rows; callers still own padding clears.

The obsolete PyPTO hc_post_block is removed. Coefficient projection,
Sinkhorn, collapse-only hc_head, and the padding-aware prefill_sp_post
remain in their existing implementations. No compiler/framework changes
were made in this step. Use the paired dependency setup and cumulative
patches in ../2026-11-03-o-proj/companion-patches for a fresh environment.

Validation
----------
With the paired environment active, run from the repository root on
allocated A5 devices. CPU Torch only prepares inputs and goldens.

python models/deepseek_v4_1_flash/hc.py --op boundary --batch 1 -d <device>
python models/deepseek_v4_1_flash/hc.py --op boundary --batch 17 -d <device>
python models/deepseek_v4_1_flash/hc.py --op boundary --batch 72 -d <device>
python models/deepseek_v4_1_flash/decode_swa.py -d <device>
python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d <device0,device1>

All six standalone cases passed. The post comparator retains the original
model-golden check and additionally requires zero-tolerance equality with
an independent sequential FP32 reference. The norm comparator uses
atol=1e-6, rtol=1/128 and zero allowed outliers, including zero and near-zero
rows. Generated entry metadata retains a dynamic leading row dimension.
The 72-token SWA composition passed every original comparison (RUN PASS
40.07s). These timings include compilation and validation, and are not
performance measurements. Layer0 EP2 passed all 16 output comparisons on
allocated devices 0,5 (RUN PASS 151.64s). See validation-summary.json.
The model's original golden functions and thresholds are unchanged.

Initial implementation failure
------------------------------
The first post run failed because cb.tile_slice was passed out_h * D;
its coordinate is a tile index, so the correct expression is out_h. This
was a model-source offset error. Corrected runs passed at 1,17,72 rows.
validation/boundary-17.log.txt and generated/failed-post-offset/ preserve
the diagnostic. The failed generated kernel is not the implementation to
replay; use generated/post-1/, post-17/, or post-72/.

Contents and replay
-------------------
validation/ contains complete stage/composition logs.
fixtures/ contains all six standalone input sets and CPU goldens as .pt.gz.
The data/out files are CPU references, not captured device outputs; logs
record comparisons against actual device output.
generated/ contains device_callable.hpp, entry.cpp, the framework-generated
ABI manifest, dynamic entry metadata, orchestration source, and runtime
text-image bytes. Executable ELF files and installed toolchains are omitted.
model-snapshot/ and workflow-snapshots/ preserve sources and capture scripts.
artifact-manifest.json lists file paths, sizes, encodings and provenance.
Supplemental artifact digests and digest-validation steps are not used.

For replay, decompress one fixture set with its in/ and out/ subdirectories,
then pass --op post or --op head-norm, the matching --batch, and
--golden-data <decompressed-directory> to hc.py. A single --op is required
for replay. Native rebuilds need the paired compiler environment. SWA and
layer inputs were not duplicated; their deterministic builders remain in
the model sources. Absolute workspace prefixes in text are normalized to
${WORKSPACE}; generated ABI/cache identifiers are retained as compiler data.

Environment
-----------
Linux aarch64, A5 / dav-3510, Python 3.12.15; Torch 2.12.0+cpu.
SDK directory cann-9.3.0; installed version.info reports 9.2.0.
PyPTO base ae6f6d76 with existing local foreign-kernel adapter patches;
CANNBotDSL base 33c2e5f4 with existing Python fixes.
PTOAS 0.70; simpler 6b6b88f2353c; PTO ISA 217bb98952a8.
Stage and composed validation uses the program runtime, with no Torch NPU
frontend. See progress.txt and validation-summary.json for task identifiers.
