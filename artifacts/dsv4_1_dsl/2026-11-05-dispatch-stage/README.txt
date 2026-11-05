DeepSeek V4.1 local MoE send staging in DSL - 2026-11-05
=====================================================

Scope and contracts
-------------------
Only the decode dispatch_stage body is replaced by stage_local/stage_rows in
moe_dispatch_dsl.py. Each AIV task copies eight physical FP8 activation rows
as INT8 transport bytes and unpacks E8M0 codes from MX_A_ZZ into ND token rows.
There is no quantization or floating-point arithmetic.

Payload staging includes all physical rows, even when num_tokens is zero.
Scale staging writes only the prefix clamped to [0, physical rows]; inactive
scale rows retain their contents. Physical capacity remains M16-aligned. Each
eight-row scale slab owns complete 64-byte cache lines. PyPTO joins the local
tasks and passes the result as the original _stage_tid dependency of
dispatch_push. Input/output allocations and all later dispatch statements are
unchanged, including metadata, remote stores, notifications, waits and gather.
The existing gather functions, routed-expert scatter and combine protocol are
unchanged. Prefill keeps its separate staging implementation. Neither PyPTO
nor CANNBotDSL is modified.

Validation evidence
-------------------
validation/local.log.txt records ten A5 staging cases and four gather
regressions. Staging uses 32 physical rows with active counts -1, 0, 7, 8, 9,
15, 16, 17 and 35, plus the 80-physical/72-active decode shape. Every payload
and scale byte is compared exactly, including inactive scale sentinels and
an extra eight-row output guard. Fixtures include all 256 byte codes.

The unchanged gather checks cover empty experts, M16 tails, full receive
capacity and four-source local compaction. The four-source case tests layout,
not four-rank communication. validation/layer0-ep2.log.txt records the existing
TP1/EP2 layer-0 regression with generated CPU fixtures and original comparators.
Task IDs, allocated devices and comparison counts are in validation-summary.json.
validation/communication-review.txt records direct AST comparisons of all
unchanged code. Compile-only logs cover the combined entry and composed MoE.

Generated code and fixtures
---------------------------
generated/ contains native wrappers, device source, ABI manifests and PyPTO
orchestration for both staging shapes and the EP2 layer. model-snapshot/
contains the final implementation and the preceding versions of both files.
fixtures/ contains gzip-compressed CPU input and golden-output .pt files.
The stage32 snapshot is the last same-shape invocation (active=35, clamped to
32); stage80 has active=72. The deterministic builder reproduces all cases.
Saved out/ files are CPU references, not device readbacks; the logs record
actual device comparisons. The existing gather archive retains its fixtures.

Reproduction
------------
From the model repository root, with the paired environment active and A5
devices allocated by the site's task queue:

python models/deepseek_v4_1_flash/moe_dispatch_dsl.py --op stage -d <device>
python models/deepseek_v4_1_flash/moe_dispatch_dsl.py --op gather -d <device>
python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d <device0,device1>

The default --op all runs all fourteen local cases. The model base is 08f4b64f,
PyPTO is 4a65a9c and CANNBotDSL is 91feb22. This is single-layer decode validation
with generated weights, not checkpoint, prefill or performance validation.
Torch is used only on CPU. Text paths are sanitized and trailing whitespace
normalized. Binary fixture bytes are preserved inside gzip files. The manifest
records provenance, sizes and encoding without supplemental checksums.
