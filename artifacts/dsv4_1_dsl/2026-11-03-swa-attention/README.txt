SWA Attention device-leaf integration -- 2026-11-03
=================================================

Source and scope
----------------
models/deepseek_v4_1_flash/swa_dsl.py implements one complete 128-entry
window for one Query, 64 Query heads and head dimension 512, on A5.
The shared split-cache SWA entry selects it for TP1. Compressed-KV,
packed544 and other TP implementations remain their existing consumers.

PyPTO/simpler dispatches one MIX task per Query, with one AIC and two AIV
participants. A 12-entry completion pool bounds task chains. Each task uses
sync_start=True; dependencies join all Query consumers. DSL RingBuffers own
only local compute/data-transfer synchronization within that MIX cohort.
There is no cross-task S2 split, reduction or global barrier.

The leaf preserves split FP8 payload/BF16 group-32 scales and physical cache
indices. It rounds dequantized KV to BF16, computes FP32 QK, applies the
index mask, casts unnormalized probabilities to BF16 for PV, then includes
the attention sink in final FP32 normalization and stores BF16 output.
Stored scale values are consumed directly; the leaf does not choose a new
quantizer. The implementation retains the existing validation thresholds.

The reference for the mixed gather/QK/Softmax/PV structure is the original-KV
branch of cannbot-dsl samples/mixed_quant_sparse_flash_mla at da23f1fe8c93.
The model leaf uses the existing split-cache ABI and current model's rounding
boundaries. It does not call the sample's Torch/NPU host launcher or its
AICPU metadata scheduler. No C++ source or toolchain implementation changed.

Validation
----------
validation/core-second-npu.log.txt: first actual device comparison PASS,
3 active Queries plus an inactive sentinel row, A5 device 6.

validation/core-cases-npu.log.txt: all four existing-harness cases PASS,
0/3/33/129 active Queries, including all-masked rows, partial windows,
cross-page and wrap indices, irregular/repeated indices, and inactive rows.
The FP32-reference comparator retains relative-L2 < 0.006, zero elements
outside its 0.003 + 0.03*abs(reference) band, exact masked zeros, and exact
inactive sentinels.

validation/swa-npu.log.txt: complete 72-token SWA subgraph PASS, including
cache payload/scales, attention output, HC hidden output and next pre-mix.

validation/layer0-npu.log.txt: complete EP2 layer 0 PASS, 72 active tokens
per rank, with every existing layer comparator passing (RUN PASS 174.79s).
validation-summary.json records the allocation tasks and execution scope.
These are generated-weight checks, not full-network checkpoint accuracy or
a repeated-sampling performance comparison.

Generated artifacts
-------------------
generated/device_callable.hpp, entry.cpp and manifest.json are the passing
device export. kernel_config.py.txt and l2_sparse_attention_swa.cpp show the
two registered device roles and the PyPTO/simpler MIX submission.
incore_*.bin are the actual runtime's raw device text images, not ELF files.
generated/failed-native-attempt/ preserves the earlier export with two CCE
errors; it is diagnostic history and is not the current executable.

Frozen core data
----------------
fixtures/gather-3-active/ preserves the first passing standalone capture.
fixtures/runs-129-active/ preserves the final case of the four-case run.
The multi-case harness reused one compiled work directory and overwrote its
data on each case, so separate captured 0- and 33-active files are not claimed.
Their deterministic construction remains in sparse_attention.make_swa_case.
data/out files are CPU golden outputs, not captured device output tensors.

Each .pt.gz is a lossless gzip of the harness's original .pt file. Decompress
to an ignored build_output directory before loading with Torch. Source paths
and stored file sizes are listed in artifact-manifest.json. Full SWA and layer runs
did not request save_data; their generated-weight specs remain reproducible
through the public validation entries and the pinned environment.

Compiler diagnostics
--------------------
The first export needed separate PyPTO slice/reshape statements for metadata
inference. Native compilation then caught own-code FP8 unpack typing and an
off-chip tensor captured inside a VF; both are corrected in the model leaf.
The first queued attempt failed during native compilation and did not pass
numerical execution. Both current native AIC/AIV builds pass.

workflow-snapshots/repro-vf-mask-scope.py.txt is a standalone compile-only DSL
reproducer. Saving it as a .py and passing bad reproduces a compiler crash
when a VF mask name is rebound inside a scalar loop; good uses a distinct
inner mask name and exports successfully. Current model code uses distinct
VF mask names. This compiler limitation is recorded but not repaired here.

The paired PyPTO/CANNBotDSL toolchain and companion patches are unchanged
from ../2026-11-03/toolchain-patches/base-commits.json. Historical script/log
snapshots replace machine-local workspace paths with ${WORKSPACE} and
${USER_ROOT}. Installed packages, core dumps and native debug ELFs are not
versioned; the runtime text images and generated source are retained.
