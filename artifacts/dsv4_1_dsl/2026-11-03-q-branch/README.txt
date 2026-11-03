Q RMSNorm, QB projection and RoPE in DSL -- 2026-11-03
====================================================

Implementation
--------------
models/deepseek_v4_1_flash/q_branch_dsl.py supplies three device leaves:
  normalize_quant_rows (AIV): FP32 RMSNorm, BF16 rounding, OCP MXFP8.
  project_q_b (AIC): NZ-weight MXFP8 matrix projection to BF16.
  query_rope_head (AIV): adjacent-pair tail RoPE and final BF16 storage.

qkv_proj_rope.q_proj_rope retains its existing four results: BF16 query
latent, BF16 Query, M80-padded QB payload and paired E8M0 scale backing.
Indexer consumers keep the same interface. Payload and scale padding is
initialized across all allocated groups, including entirely inactive groups.
Inactive query-latent and Query output rows retain their incoming contents.

RMSNorm sums FP32 squares in 64-lane partials, reduces the partials, then
uses sqrt and precise FP32 division to obtain the inverse RMS. The result
is rounded to BF16 before the shared OCP quantizer consumes it. Normalization
and quantization share a VF with VST/VLD ordering at the intermediate store.
The stage golden remains the existing Torch RMSNorm plus quantizer.

QB reuses mxfp8_dsl._mx_project with M80/N128/K128 hardware tiles, existing
NZ weight bytes and packed E8M0 scales. It writes a BF16 GM intermediate.
RoPE consumes that BF16 input, keeps the non-rotary prefix unchanged, rotates
the final 64 lanes as adjacent real/imaginary pairs, and rounds to BF16.
The CPU projection golden uses the existing sequential group-32 reference;
hardware tiling and reference accumulation are distinct descriptions.

PyPTO/simpler owns dependencies among the three task types. This initial
implementation keeps the BF16 intermediate explicit instead of a manual
Cube/Vector event handshake. It does not claim a performance improvement.
The obsolete proj_q_b.cpp and its private q_lora_projection.hpp were removed
after their last consumer was replaced. New kernel implementations use
CANNBotDSL; the C++ changes consist of deleting those obsolete files.
The framework worktrees are unchanged.

This completes the TP1 SWA projection/cache/Attention/output-projection
compute path in DSL. mHC and MoE are separate remaining stages. Other model
paths retain their interfaces, but this capture's integrated hardware checks
are the 72-token SWA path and layer0 EP2, not a full-network checkpoint.

Validation
----------
validation-summary.json records allocation tasks and exact pass lines.
Both standalone stages pass 0/17/81 active-token cases, covering zero and
near-zero rows, inactive NaN inputs, nonuniform norm weights/MX scales,
an M80 boundary and exact inactive-output sentinel preservation.

Normalization compares BF16 values against Torch RMSNorm at rtol=1/128,
atol=1e-6 with no outliers. Its FP8 payload and E8M0 scales compare byte
for byte against quantization of that reference. QB validation separately
checks input quantization bytes, BF16 values before RoPE, and final Query
using the existing projection tolerance. No acceptance threshold was relaxed.

The complete 72-token SWA subgraph and layer0 EP2 pass every original
comparator. Layer attention_output retains the branch's 1.5% per-rank
outlier budget. The composed model goldens are unchanged.

Artifacts
---------
fixtures/ preserves independent first 17-active captures and final 81-active
captures of both stages, including generated QB weights. .pt.gz files are
lossless gzip archives. data/out contains CPU goldens; actual device-value
comparisons are recorded in the logs. Each stage reused one work directory
in the all-case run, so earlier empty/17 files there were overwritten by
the final 81-active case. Reconstruct other cases through the same entry.

generated/ contains device_callable.hpp, ABI manifests, PyPTO entry.cpp
wrappers, kernel configs, orchestration source and available runtime
incore_*.bin device text images. The composed SWA manifest identifies the
three new task types. ELF files, shared libraries and installed toolchains
are excluded. Text paths use ${WORKSPACE}; artifact-manifest.json records
source paths, stored file sizes and the model-source file list.

Full SWA and layer runs did not request save_data; their deterministic
generated-weight specifications remain in the model. All device execution
was allocated through task-submit. Timings are single correctness runs,
not a repeated-sampling performance comparison.

The paired toolchain requirements remain those of the output-projection
archive. Apply ../2026-11-03-o-proj/companion-patches to its recorded bases;
do not stack different cumulative patch captures. Environment: Python
3.12.15, CANN version.info 9.2.0 under cann-9.3.0, PTOAS 0.70, simpler
6b6b88f2353cca12cfafe809750b50b59931b0d1, and PTO ISA
217bb98952a8f9004b95fe938da97d8d5f68f702. Torch is used on CPU only.

From the model repository root with the paired environment active:
  python models/deepseek_v4_1_flash/qkv_proj_rope.py --stage all --case all -d "$TASK_DEVICE"
  python models/deepseek_v4_1_flash/decode_swa.py -d "$TASK_DEVICE"
Allocate two devices and pass their comma-separated IDs for the layer:
  python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d "$TASK_DEVICE"
