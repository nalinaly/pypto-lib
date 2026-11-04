DeepSeek V4.1 local MoE receive gather in DSL - 2026-11-04
=======================================================

Scope and contracts
-------------------
Only the decode dispatch_gather computation is replaced. Each AIV gather_rows
leaf compacts one expert's M16 tile from the current rank's receive windows.
Payload bytes, scale codes, route weights and route indices retain their bits
and source-rank order. E8M0 scales are rearranged from ND transport rows into
the existing MX_A_ZZ storage, without requantization.

The staging, metadata publication/finalization, remote payload stores,
notifications and deferred waits retain their original code. PyPTO supplies
local window views and preserves the push, wait and metadata completion edges.
A bounded completion pool joins the gather tasks before expert compute, even
when every expert is empty. ffn_reduce_scatter and the combine protocol are
unchanged. Neither PyPTO nor CANNBotDSL is modified.

Each task owns an aligned 16-row output tile. Payload, weight and route rows
after the active count retain their contents. Unused scale positions inside
the final active tile become zero; later scale tiles remain untouched. The
temporary ND scale buffer is removed, and scale packing uses per-task UB.

Validation evidence
-------------------
validation/local.log.txt records four A5 cases, each comparing all four output
buffers byte-for-byte, including inactive tails and an extra output guard:
  bounds: expert counts 0, 15, 16, 17, with empty source lanes;
  zero: all experts empty;
  full: all two-source receive lanes at their 80-row capacity;
  multi: four local source lanes with counts 15, 17, 17, 0.
Fixtures include all 256 payload and scale byte codes, signed-zero route
weights, and distinct source rows and route indices. The four-source case
tests local data layout, not four-rank communication.

validation/layer0-ep2.log.txt records the existing TP1/EP2 layer-0 regression
with generated CPU fixtures and the original comparators. The exact task IDs,
allocated devices, and comparison count are in validation-summary.json.
validation/communication-review.txt records direct AST comparisons of the
unchanged dispatch prefix and all other decode_moe functions. Compile-only
logs cover the standalone cases and the composed MoE entry.

Generated code and fixtures
---------------------------
generated/ contains native wrappers, device source, ABI manifests and PyPTO
orchestration for the two/four-source standalone entries and the EP2 layer.
model-snapshot/ contains the final implementation and preceding decode_moe.
fixtures/ contains gzip-compressed CPU input and golden-output .pt files.
The two-source fixture is the last same-shape invocation (full); the four-source
fixture is multi. The deterministic builder reproduces bounds and zero.
Saved out/ files are CPU references; actual device comparisons are in the logs.

diagnostics/ retains the initial model-authoring failures. The first standalone
attempt reshaped a guard-trimmed scale view that the runtime marks noncontiguous.
Removing that unnecessary reshape fixed the device assertion. The integration
initially queried dimensions on DistributedTensor; dimensions now come from
the ordinary output tensors and explicit transport lane widths. A subsequent
scale-tail mismatch came from the default 32-bit predicate in byte clearing;
the final code uses an explicit 8-bit predicate. No framework fix was needed.

Reproduction
------------
From the model repository root, with the paired environment active and A5
devices allocated by the site's task queue:

python models/deepseek_v4_1_flash/moe_dispatch_dsl.py -d <device>
python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d <device0,device1>

The model base is 0b950867, PyPTO is 4a65a9c, and CANNBotDSL is 91feb22.
This is single-layer decode validation with generated weights, not a whole-model
checkpoint, prefill or performance validation. Torch is used only on CPU.
Text paths are sanitized and trailing whitespace normalized. Binary fixture
bytes are preserved inside gzip files. The manifest records provenance,
sizes and encoding without supplemental checksums. Validation logs use .txt
suffixes so the repository's general *.log ignore rule does not exclude them.
