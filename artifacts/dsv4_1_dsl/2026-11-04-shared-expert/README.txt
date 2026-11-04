DeepSeek V4.1 shared-expert DSL integration — 2026-11-04
======================================================

Collection state
----------------
The new-DSL baseline SWA72 and layer0 EP2 runs passed before the shared
expert was replaced. The shared SwiGLU/quantization and projection checks
passed independently. The first composed shared-FFN run failed, and the
corrected full FFN80 run subsequently passed. Final layer0 EP2 validation
with the replacement also passed: all 16 outputs retain the original model
thresholds. Changes and evidence remain local, without commit or push.

This archive uses the installed CANNBotDSL FP32 interleave fix at 91feb22
and the local PyPTO integration changes. The model work starts from 7431870.
CPU Torch prepares inputs and computes references. Device execution uses
A5 and the program runtime; no performance claim is made by these logs.

Preserved contracts
-------------------
The shared expert consumes ordinary ND/K-N FP8 weights and MX_B_NN scales.
W1/W3 produce FP32 intermediates. SwiGLU rounds both projections to BF16,
applies an upper-only gate clamp and a symmetric up clamp, computes the
activation, and rounds it to BF16 again. Group-32 quantization rounds
max(amax, 1e-4)/448 upward to a power of two, stores E4M3 payload, and packs
E8M0 scales as [M/16, K/64, 16, 2]. W2 returns BF16. The existing model
goldens and thresholds are retained.

Validation present
------------------
validation/baseline-swa72.log.txt: PASS, 72-token SWA composition.
validation/baseline-layer0-ep2.log.txt: PASS, 16 original layer outputs.
validation/vector16.log.txt and vector80.log.txt: random and edge cases
  passed; FP8 payload and E8M0 scales compare byte-for-byte.
validation/projection17.log.txt: W1/W3-style FP32 and W2-style BF16 outputs
  passed using the existing ratio_reldiff(2e-3, 1%) criterion.
validation/shared-full80-fixed.log.txt: PASS, complete shared FFN with
  output [80, 5120] BF16 and the original ratio_reldiff criterion.
validation/layer0-ep2-final.log.txt: PASS, replacement integrated on two A5 devices, 16 outputs.
validation/adapter-tests-final.log.txt: 40 passed.
validation/foreign-planner-jit-regression.log.txt: 74 passed.
validation/nested-specialization-after.log.txt: 3 passed, 7 deselected.

Failure and correction
----------------------
validation/full80-first-failure.resolved.log.txt preserves the original
scheduler S1 running-stalled failure. The corresponding first-attempt
kernel configuration contains only two registrations: one projection and
one vector leaf. The corrected full80 configuration contains three,
including a distinct projection specialization for W2. The PyPTO foreign
call-chain specialization correction and its failing/passing regressions
are recorded by nested-specialization-before.resolved.log.txt and
nested-specialization-after.log.txt. The failure log remains historical
evidence; it is not the current full80 result.

Contents
--------
generated/vector16/ holds the representative AIV wrapper, device source,
SourceBundle ABI manifest, entry metadata, and orchestration.
generated/projection17-gate/ and projection17-down/ hold the FP32 and BF16
projection equivalents. generated/full80/ holds the corrected three-leaf
configuration and generated sources. generated/full80-first-failure/
preserves the first-attempt configuration and orchestration.

fixtures/vector16-edges/ and vector80-edges/ freeze the edge inputs and CPU
goldens. They cover zero rows (scale code 105), the 119-to-120 scale boundary,
clipping, and positive/negative BF16 exact midpoints, across all 2304 columns.
fixtures/full80/ freezes the inputs and CPU reference from the passing
full-FFN run. out/ files are CPU goldens, not saved actual device readbacks.
The logs are the evidence for device comparisons.

Tensors are gzip-compressed .pt files. Decompress one fixture tree while
preserving its in/ and out/ directories. The existing full operator accepts
--golden-data <directory>; the vector files can be loaded as CPU tensors by
the golden harness using shared_swiglu_quant_validation. Inputs are generated
test fixtures, not checkpoint weights or application data.

Text workspace paths become <workspace>; paths outside that workspace but
under the same account become <user-home>. Replace these placeholders before
using archived generated paths, or regenerate from the model sources with
the selected environment. No environment files or credentials are archived.
Abbreviated account names in truncated pytest diagnostics become <user>.
artifact-manifest.json records file provenance, byte sizes and encoding.
No supplemental hashes, checksums or checksum-validation steps are used.

Reproduction entry points
-------------------------
Run from the repository root with the paired environment loaded and devices
allocated by the local task queue:

python models/deepseek_v4_1_flash/expert_shared_dsl.py --rows 16 --case all -d <device>
python models/deepseek_v4_1_flash/expert_shared_dsl.py --rows 80 --case all -d <device>
python models/deepseek_v4_1_flash/expert_shared.py --save-data -d <device>
python models/deepseek_v4_1_flash/decode_swa.py -d <device>
python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d <device0,device1>

The archive capture helper is workflow-snapshots/archive.py.txt. It records
the initial collection; validation-summary.json and the final EP2 log record
the completed validation and must be preserved on recapture.

PyPTO companion changes
-----------------------
The selected PyPTO base is ae6f6d76. companion-patches/pypto-local.patch
contains tracked changes to the JIT call binding, tensor metadata, decorator,
foreign-kernel code, extern tests, and English/Chinese integration docs.
companion-snapshots/ contains complete text snapshots of those files plus
the new cannbotdsl.py, _cannbotdsl_source.py and adapter test modules, which
are untracked locally and therefore absent from git diff HEAD. Restore the
new files from their .py.txt snapshots when reproducing the local adapter.
All patches and snapshots are local evidence, not a published PyPTO change.

The source snapshots preserve kernel discovery, SourceBundle export, native
ABI conversion, wrapper generation and call-chain specialization on the
PyPTO side. The accompanying DSL revision is 91feb22. GitCode credential,
issue-submission and PR-submission utilities or their metadata are excluded.
