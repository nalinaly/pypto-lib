DSL output projection and PyPTO adapter ownership -- 2026-11-03
============================================================

Source and contract
-------------------
models/deepseek_v4_1_flash/o_proj_dsl.py implements the TP1 A5 inverse RoPE
and OCP MXFP8 quantizer, grouped WoA, BF16 requantizer and WoB. PyPTO/simpler
owns task scheduling. DSL owns the AIV vector math and AIC matrix compute.
Other TP configurations remain consumers of the existing kernels until
their local head/group shapes have been migrated and validated.

The weight bytes remain NZ [N/32,K/16,16,32]. Activation scales retain
[M/16,K/64,16,2] order; weight scales retain [N/16,K/64,16,2] order. E8M0
bytes are preserved. Inverse RoPE rounds to BF16 before quantization, WoA
rounds to BF16 before requantization, and WoB stores BF16. The OCP scale
selection and FP8 nearest-even conversion are unchanged.

One inverse-RoPE task handles one head and 16 token rows. The projection
tasks use M80/N128/K128 tiles. Requantization uses 16x256 tiles. Explicit
runtime active/valid-row arguments distinguish valid data from a view's
static capacity. Final padded views carry valid_shape, and Fixpipe writes
only the requested rows. Inactive rows are zeroed; inactive NaN input
sentinels must not leak into output. Existing validation gates are retained.

The samples/attn_epilogue implementation in cannbot-dsl at da23f1fe8c93 is
the reference for the DSL MX matmul structure. This model implementation
consumes the model's existing NZ weights and packed scales directly instead
of introducing sample-specific ND weights or a separate host launcher.

Adapter migration
-----------------
The user selected PyPTO ownership. python/pypto/jit/cannbotdsl.py now owns
TensorSpec/dtype adaptation, export_device invocation, ABI wrapper creation,
content-addressed source publication, and ExternalKernelSpec configuration.
E8M0 and INDEX mapping live there. PyPTO also infers shaped tensor.view
metadata so an explicit ND view of MX scale backing can reach a DSL leaf.
The model does not numerically convert scales to another representation.

CANNBotDSL keeps its generic device export. Its interop/pypto.py is a lazy
compatibility forwarder. Importing either package alone does not require
the other. Runtime use of the compatibility wrapper needs the paired PyPTO
adapter version. No C++ implementation was changed in this integration.

The cumulative patches in companion-patches/ are relative to:
  PyPTO: ae6f6d76d32c2506fa1764a254a5f46a251431e9
  CANNBotDSL: 33c2e5f4af7541f64722284668a856a2e8fe9cec
They include the prior local BF16/FP8/INDEX bridge changes and CANNBotDSL
FP32 interleaved-store fix. Apply them to clean checkouts of those bases;
do not stack them on earlier archived patches. The newly added PyPTO
adapter and its tests are included, not just changes to tracked files.

Validation and diagnostics
--------------------------
validation-summary.json identifies the allocation tasks, devices and exact
scope. All NPU execution was allocated through task-submit. The validation
logs preserve the original comparator decisions; failed exploratory runs
are included as diagnostic history and are not claimed as passing.

The rope stage compares all FP8 payload and E8M0 scale bytes exactly,
including zero padding after 17 active rows. The MX stage uses nonuniform
scale codes and integer FP8 payloads to check packed addressing and a
17-row final store at zero tolerance. The standalone o_proj checks cover
0/12/17 active tokens and FP32 HC post with 72/17/0/81 active tokens.
The 81-token case crosses the M80 boundary. The composed SWA and layer0
EP2 entries use 72 active tokens and the existing generated-weight goldens.
These checks do not claim whole-network checkpoint accuracy or measured
performance improvement.

Earlier RoPE attempts inferred live rows from static source capacity and
separated the RoPE producer from quantization across VF regions. The final
leaf passes runtime counts explicitly and performs the RoPE/store/reload/
quantization within one VF with VST/VLD ordering. The stage and composed
checks pass without changing quantizer policy or loosening thresholds.

Stored artifacts and reproduction
---------------------------------
generated/ contains device_callable.hpp, ABI manifests, PyPTO entry.cpp
wrappers, orchestration source and kernel configs for the two stages and
the complete projection. incore_*.bin are actual runtime device text images,
not ELF files. Generated code is evidence, not a hand-maintained kernel.

fixtures/ contains losslessly compressed harness data for rope-17, mx-17
and the final 17-active standalone projection, including generated weights.
The multi-case harness reused its work directory, so earlier active counts
were overwritten and are represented by logs and deterministic constructors.
data/out files are CPU goldens; the logs record comparisons with actual
device outputs. HC-post/SWA/layer inputs are reproducible from model specs;
their duplicate weights are not stored in this archive.

Decompress .pt.gz files into an ignored build_output directory before loading
with Torch. artifact-manifest.json records original and stored hashes.
Machine-specific workspace paths in text are replaced with ${WORKSPACE}.
The full installed environment, ELF files and shared libraries are excluded.

With the paired environment active, run from the model repository root,
using a device supplied by task-submit in TASK_DEVICE:
  python models/deepseek_v4_1_flash/o_proj_dsl.py --stage rope --active 17 -d "$TASK_DEVICE"
  python models/deepseek_v4_1_flash/o_proj_dsl.py --stage mx --active 17 -d "$TASK_DEVICE"
  python models/deepseek_v4_1_flash/o_proj.py --case all --save-data -d "$TASK_DEVICE"
  python models/deepseek_v4_1_flash/decode_swa.py -d "$TASK_DEVICE"
For layer0, allocate two devices and pass their comma-separated IDs:
  python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d "$TASK_DEVICE"

Toolchain: Python 3.12.15, CANN version.info 9.2.0 (installed under the
cann-9.3.0 directory), PTOAS 0.70, simpler 6b6b88f2353cca12cfafe809750b50b59931b0d1,
PTO ISA 217bb98952a8f9004b95fe938da97d8d5f68f702. Torch is used on CPU only
for input preparation and golden validation.
