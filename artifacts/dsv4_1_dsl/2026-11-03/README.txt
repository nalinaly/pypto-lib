Window KV integration evidence -- 2026-11-03
==========================================

Scope and source
----------------
The model uses PyPTO/simpler orchestration and a CANNBotDSL AIV leaf for window
KV RMSNorm, adjacent-pair RoPE, group-32 FP8 quantization, and paged publication.
The executable implementation and standalone goldens are owned by
models/deepseek_v4_1_flash/window_kv_dsl.py. The shared qkv_proj_rope.window_publish
entry imports it; the previous C++ writer is removed.

Current source retains the requested sync_start=True on each pl.spmd(1) page
task. The 17 page tasks in the captured fixture remain independent tasks; this
flag does not create a collective barrier across them.

The two-line expert_routed.cpp change is a separate compatibility correction:
the pinned FP4-to-FP8 TCVT supports CompactMode::Normal for that temporary,
with its matching UB allocation size. The independent MoE boundary case passed.

Validation evidence
-------------------
validation/window-boundaries-precise-div-npu.log.txt:
  PASS: 72 capacity / 71 active, 13 pages, two writer/consumer steps;
  zero-active and one-token cases also pass. Payload is compared byte for byte,
  with zero tolerance for scale. These runs precede the sync_start experiment.

validation/swa-rank1-precise-div-npu.log.txt:
  PASS on the frozen rank-1 Attention fixture after precise division was enabled.
  Payload byte differences: zero. Attention outliers: 1101 / 368640 = 0.2987%,
  below the unchanged 1.5% layer-integration budget.

validation/layer0-precise-div-npu.log.txt:
  PASS for the complete generated-weight layer 0, A5 devices 6,7, EP=2,
  72 active tokens per rank. This run precedes sync_start=True. Existing layer
  comparators are retained, including the MoE check conditioned on the validated
  Attention boundary. This is not full-network checkpoint accuracy validation.

swimlanes/sync-start/dsl_sync_start.log.txt:
  PASS for the latest sync_start=True SWA subgraph on A5 device 2, with both
  frozen-input and frozen-golden cache hits. The source change is visible as
  set_require_sync_start(true) in generated/current/l2_decode_swa.cpp.

validation/dsl-export-tests.log.txt and pypto-extern-tests.log.txt:
  24 CANNBotDSL export/direct-call regressions and 29 PyPTO external-kernel
  regressions passed. The former excludes 16 NPU-only tests; actual A5 model
  validation is recorded separately above.

L1 swimlanes
------------
Open these HTML files locally after cloning, or download their swimlanes.zip:
  swimlanes/dsl-vs-cpp/index.html
  swimlanes/sync-start/index.html

Each contains synchronized zoom/pan, PNG/SVG previews, Perfetto JSON, raw L1
records, dependency graph, and callable name map. All captures use A5 device 2
and the same 27 input files: 72 active / 80 capacity, 8 requests, zero history,
17 physical cache pages. Numerical validation passed for every variant.

Single-capture observations, not a repeated-sampling speedup claim:
  Original C++: 605 physical executions, KV 1 SPMD task / 12 AIV blocks,
                observed span 709.107 us, KV first-start/last-end 13.011 us.
  DSL before sync_start: 610 physical executions, KV 17 single-block page tasks,
                observed span 716.427 us, KV interval 38.671 us.
  DSL with sync_start: same task counts, observed span 732.432 us,
                KV interval 48.249 us.

The original C++ writer uses continuous BF16 scales and ordinary division.
The DSL writer uses power-of-two BF16 scales and precise FP32 division.
The first pair therefore includes numerical-contract and scheduling changes.
The sync_start pair changes only that launch flag.

The initial pair's outer shell reported an extra done token after both captures
and validations completed. queue-result.json preserves this fact. Per-variant
RUN PASS records, complete raw/converted event counts and per-core non-overlap
were checked. The sync_start run completed with queue exit 0.

Generated device artifacts
--------------------------
Start in generated/current/:
  device_callable.hpp -- CANNBotDSL-generated device implementation, 898 lines;
                         VF arithmetic begins at line 615, entry at line 759.
  manifest.json       -- symbol, AIV role, POD ABI, argument effects, resources.
  entry.cpp           -- 37-line generated simpler Tensor-to-device-POD wrapper.
  kernel_config.py.txt -- captured registration: func_id=8, external AIV kernel.
  l2_decode_swa.cpp   -- generated PyPTO/simpler orchestration and launch flags.
  window_kv_page.aiv.bin -- the 2072-byte device text image used by the capture.

The device header and text image are identical before/after sync_start; the
flag changes orchestration. The text SHA256 is
982d4b4ae10a720b411757976cf3890a7275e272c7ae619d4090cbe4ac7fec97.
The generated header explicitly selects PRECISION_0ULP_FTZ_TRUE for division.
The export route did not persist a per-stage MLIR dump and does not emit a PTO
file for this external DSL leaf. Historical exports are in generated/jit-cache/.

Frozen numerical data
---------------------
fixtures/swa-rank1/in/*.pt.gz: common frozen inputs.
fixtures/swa-rank1/out/*.pt.gz: power-of-two-codec expected outputs, subsequently
  validated by the precise-division and sync_start captures.
fixtures/swa-cpp/out/*.pt.gz: original C++ codec's own expected outputs.
precision/*.pt.gz: device outputs before/after the precise-division correction.
precision/args_dump/: initial diagnostic manifest and losslessly compressed
  captured argument bytes. Decompress args.bin.xz to args.bin before using the
  runtime dump viewer. The original binary is about 99 MB and includes repeated
  snapshots; XZ preserves every byte with a larger deduplication dictionary.

Raw and compressed hashes are listed in artifact-manifest.json. Extract into
an ignored build_output directory, for example from the repository root:

  python - <<'PY'
  import gzip, lzma, shutil
  from pathlib import Path
  src = Path('artifacts/dsv4_1_dsl/2026-11-03')
  dst = Path('build_output/dsv41-dsl-evidence')
  for f in src.rglob('*'):
      if f.suffix not in ('.gz', '.xz'):
          continue
      out = dst / f.relative_to(src).with_suffix('')
      out.parent.mkdir(parents=True, exist_ok=True)
      opener = gzip.open if f.suffix == '.gz' else lzma.open
      with opener(f, 'rb') as a, out.open('wb') as b:
          shutil.copyfileobj(a, b)
  PY

Toolchain and reproducibility
-----------------------------
The exact versions and patch baselines are in artifact-manifest.json and
toolchain-patches/base-commits.json. The PyPTO runtime/PTOAS/PTO-ISA pins were
checked and match. Apply the companion patches to those baselines when the
corresponding upstream fixes are unavailable. A5 and CANN are required for
device execution; Torch is used only for CPU data/reference calculations.
The companion diffs are stored losslessly as pypto.patch.gz and
cannbotdsl.patch.gz. Decompress them before git apply; this preserves the
significant blank-context markers in unified diffs.

Use the executable model entries and commands documented in
docs/models/deepseek_v4_1_flash/index.md. workflow-snapshots/ preserves the
historical orchestration/debugging scripts as readable .txt snapshots, with
machine-local paths replaced by ${WORKSPACE}, ${USER_ROOT}, or ${USER_HOME}.
They are evidence rather than a separately supported installer or CLI.

issues/rsqrt-3257.txt records the independent rsqrt high-precision selector bug:
https://github.com/hw-native-sys/pypto/issues/3257
It is reproduced by compile inspection and remains separate from the validated
DSL division correction. No PyPTO rsqrt fix is included in this branch.

Python virtual environments, installed compiler packages, font caches and
shared-library dependency builds are not versioned as process data. Original
machine-local artifacts remain available in the development workspace.
