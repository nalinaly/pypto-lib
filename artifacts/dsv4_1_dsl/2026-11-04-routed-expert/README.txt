DeepSeek V4.1 local routed-expert DSL integration — 2026-11-04
===========================================================

Result and task boundary
------------------------
The former fused_expert_ffn C++ compute task is replaced by three local DSL
phases: dsl_routed_up (MIX W1/W3), dsl_routed_act_quant (AIV), and
dsl_routed_down (MIX W2). Each phase has 32 workers and an explicit TaskId
join. MIX leaves reserve AIC/AIV participants with sync_start=True.

The first phase retains input_ready. The completion handle returned to the
caller joins every W2 worker. The ffn_reduce_scatter consumer, remote stores,
notifications and deferred waits are unchanged. The existing two-FP32-partial
sum and final BF16 rounding are unchanged. The previous model C++ kernel is
removed; its text is retained here solely as a historical implementation.

This is a staged implementation with GM intermediates, not a single fused
DSL kernel. In the current EP2 configuration the declared gate/up/payload/
scale capacity is about 76.2 MiB per rank. On-chip fusion and performance
tuning are deferred. No performance A/B is claimed.

Preserved contracts
-------------------
- Original MXFP4 panel bytes: [expert, N/256, K/256, 256, N_tile/2], with
  even N in the low nibble. This differs from the K-adjacent input layout
  of the official A8W4 sample refreshed to 48fa202.
- Every E2M1 code, including signed zero, widens exactly to E4M3. There is
  no temporary 1/64 scaling. AIVs publish K64/N256 NZ halves to AIC; AIC
  accumulates M16/N256/K128 MX products in FP32.
- Original packed MX_A_ZZ/MX_B_NN scale bytes. PyPTO passes ND storage views
  to the provider; the local kernel reconstructs the native scale layouts.
- Upper-only gate clipping and symmetric up clipping, FP32 SwiGLU, then
  route-weight multiplication before quantization. No extra BF16 rounding.
- Group-32 upward E8M0 scales and nearest-even E4M3 payload, using the same
  helper as the shared expert and router input quantizer.
- Two W2 partial planes over hidden intervals [0,1024) and [1024,2304),
  compact unpadded expert row prefixes, and the original plane stride.
- Counts are immutable after input_ready and retain their original bounds.
  Private intermediate expert slabs are padded to 16 rows; inactive rows
  are zero-filled during activation without reading poisoned gate/up/weights.

Validation
----------
baseline-bounds.log.txt: original C++ implementation, counts 0/1/159/160,
  PASS against the existing local expert golden.
probes.log.txt: all 16 FP4 encodings and every NZ output position compare
  byte-exactly; mixed A8W4 projection passes the original numerical criterion.
activation.log.txt: weighted FP32 activation with zero weights, clipping,
  BF16-midpoint-valued FP32 inputs and NaN inactive guards. Payload and E8M0
  scale bytes both compare exactly.
bounds.log.txt: complete DSL local FFN, receive-capacity boundary case, PASS.
local-all.log.txt: four consecutive calls through the same implementation:
  bounds; 15/16/17/63 rows; 64/65/15/16 rows; all-zero counts. All PASS.
  Active outputs retain ratio_reldiff(diff_thd=2e-3, pct_thd=0.01); inactive
  output sentinel rows remain byte-for-byte unchanged.
layer0-ep2.log.txt: TP1/EP2 layer 0, 72 active tokens per rank, devices 4 and 7,
  all 16 outputs PASS with original acceptance criteria. Reported x_next
  rel_l2=0.1408408 and max_abs=3.2990062 match the preceding gate regression
  at the printed precision.
static-checks.log.txt: Ruff, headers, English-only and diff review.
communication-review.txt.txt: direct AST comparison of the scatter consumer,
  final reduction and goldens, plus unchanged decode_moe/decode_layer sources.

Files
-----
model-snapshot/ contains final model source text and the previous Python/C++
implementation. generated/ contains provider wrappers, generated device code,
native ABI manifests and orchestration for the isolated probes, full local
compute and two-card layer. fixtures/ contains gzip-compressed CPU .pt inputs
and expected outputs for the isolated probes. out/ tensors are CPU goldens,
not actual device readbacks; the logs record the device comparisons.

All execution used A5 and the task queue. Torch was used only on CPU for data
preparation and validation. The installed DSL remains 91feb22; PyPTO is
4a65a9c with its existing pinned runtime/toolchain. This task changes neither
framework. Source and evidence remain local on dsv4.1-dsl, without commit or
push. No full checkpoint, prefill or other TP/EP configuration is claimed.

Reproduction
------------
Load the paired environment, allocate devices through the site's task queue,
and run from the model repository root:

python models/deepseek_v4_1_flash/expert_routed_dsl.py -d <device>
python models/deepseek_v4_1_flash/expert_routed.py --case all -d <device>
python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d <device0,device1>

Text workspace paths are replaced by <workspace> or <user-home>. Regenerate
sources or replace placeholders before using archived paths. Credentials and
environment files are excluded. artifact-manifest.json records provenance,
sizes and encoding. No supplemental hash/checksum checks are performed.
