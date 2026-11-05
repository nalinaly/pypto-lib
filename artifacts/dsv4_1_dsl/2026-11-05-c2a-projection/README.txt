DeepSeek V4.1 C2A compressor projections in DSL - 2026-11-05
=========================================================

Scope and contracts
-------------------
The C2A proj_cmp task is replaced by local dsl_proj_cmp AIC tasks. KV and gate
retain BF16 operands, M80/N64 tiles, sequential K256 FP32 accumulation and FP32
outputs. NZ weights keep their bytes; whole-storage flattening exposes the
existing [N/16, K/16, 16, 16] faces through the supported PyPTO ND interface.
There is no intermediate BF16 rounding or new quantization rule.

KV and gate tasks run independently. A bounded completion pool retains the
input-ready predecessor and joins both projections before pooling/publication.
Decode and prefill import the same implementation. The former C2A projection
body and tiling constants are removed. C1A, C2A pooling, ring-state ownership,
cache publication and every distributed call retain their code. Neither PyPTO
nor CANNBotDSL is modified.

Complete-layer validation support
---------------------------------
The original complete-layer fixture rejected layer 2 before compilation; it
only supported SWA layers 0 and 1. The fixture now also supports C2A full source
layers, using the existing C2A input builder and CPU golden. C2A comparators are
shared with its standalone entry and applied independently to each EP rank.
Their expressions and thresholds are unchanged, as is the SWA acceptance policy.

The first EP2 run with the original projection passed all 15 numerical/state
outputs but failed the common interface's SWA-only attention_output port: C2A
did not write it, and an entirely-NaN reference was correctly rejected by the
harness. Non-SWA branches now explicitly zero that auxiliary port, and its
comparison is exact. C2A's numerical attention boundary remains attention_hidden.
This is a local output initialization; no communication task was replaced.

Evidence
--------
validation/projection-final.log.txt contains five standalone checks followed by
five direct original-versus-DSL checks on identical A5 inputs:
  physical 17 / active 15, physical 17 / active 0,
  physical 80 / active 72, physical 81 / active 81,
  physical 321 / active 321, exercising completion-pool reuse.
Structured layout cases match CPU results exactly. The random BF16 case uses
FP32 atol=rtol=1e-4 with no allowed outliers. Inactive outputs and the final guard
row remain exact. Both DSL projections match their original-device references
at zero numerical tolerance in all five cases.

validation/baseline-c2a-full.log.txt records the passing original C2A attention
sublayer. baseline-layer2-ep2.log.txt retains the 15-pass/undefined-port baseline
described above. layer2-ep2.log.txt records the final complete layer with the DSL
projection and auxiliary-port fix. The exact task IDs, devices and pass counts
are in validation-summary.json. prefill-compile.log.txt is a TP1 compilation
check of the shared prefill caller; full prefill device accuracy is not claimed.
communication-review.txt records direct AST comparisons and unchanged policies.

generated/ contains wrappers, device source, native ABI and orchestration,
including the original layer orchestration and the projection equivalence entry.
model-snapshot/ contains final and preceding source. fixtures/ holds compressed
CPU inputs and goldens for the random 80-row and structured 321-row cases.
Saved out/ files are CPU references; logs record actual device comparisons.
The deterministic fixture builder reproduces the remaining cases.
workflow/ contains the original projection extracted from the preceding source,
the comparison entry and archive capture script.

diagnostics/ retains the initial unsupported layer entry, the rejected NZ
tensor.view attempt (replaced by supported whole-tensor flattening), and two
validation-wrapper keyword forwarding errors fixed before final validation.
No framework capability or numerical threshold was changed for those failures.

Reproduction
------------
From the model repository root with the paired environment active, allocate
devices through the site's task queue:

python models/deepseek_v4_1_flash/compressor_dsl.py -d <device>
python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 2 -d <device0,device1>

For direct equivalence, copy workflow/baseline_projection.py.txt and
workflow/compare_projection.py.txt to a scratch directory as .py files, put the
model directory and scratch directory on PYTHONPATH, and run compare_projection.py
on one allocated device. The comparison always uses the production DSL scheduler.

The model base is e9371e66, PyPTO is 4a65a9c and CANNBotDSL is 91feb22. Torch is
CPU-only. This is generated-fixture validation, not a checkpoint, whole-network
or performance claim. Text paths are sanitized and trailing whitespace normalized.
Binary fixture bytes are preserved in gzip files. The manifest contains provenance,
sizes and encoding without supplemental checksums.
