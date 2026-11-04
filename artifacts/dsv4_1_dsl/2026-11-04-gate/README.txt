DeepSeek V4.1 local MoE router DSL integration — 2026-11-04
========================================================

Result
------
Four local computation tasks now call CANNBotDSL leaves through PyPTO:
rms_norm -> dsl_gate_norm, x_norm_mx_quant -> dsl_gate_quant,
gate -> dsl_gate, and route_sort -> dsl_route_sort.
The gate's pre-route initialization remains in PyPTO.

Communication-bearing tasks are retained, including dispatch_meta_publish,
dispatch_meta_wait, dispatch_push, dispatch_publish_epoch, dispatch_wait,
ffn_reduce_scatter, combine_notify, and combine_wait. The source files
decode_moe.py, expert_routed.py and decode_layer.py are unchanged.

Changes and evidence remain local on dsv4.1-dsl, without a commit or push.
The execution environment uses CANNBotDSL 91feb22, PyPTO 4a65a9c, the pinned
simpler/PTO ISA and PTOAS 0.70. CANN compiler metadata is 9.2.0; the existing
SDK installation directory is named cann-9.3.0. All device runs used A5 and
the task queue; Torch was used only on CPU for inputs and references.

Official samples refresh
------------------------
The separate cann/cannbot-dsl shallow checkout was updated from da23f1f to
48fa202. Samples are now grouped under attention/, matmul/, vector/ and mc2/.
The previous checkout is retained by its archive branch. This source refresh
did not replace the installed mabing1118/cannbot-dsl-dev compiler.

Relevant samples were reviewed:
- vector/dynamic_mx_quant_sr: scale_alg=1 uses upward power-of-two scales,
  but stochastic rounding and all-zero scale code 0 differ from this model.
  The model retains nearest-even FP8 conversion and zero-group scale 105.
- vector/block_attn_res_update_rms_norm: fuses residual/online-softmax work
  and targets D=7168. It is not the D=5120 MoE RMSNorm contract.
- vector/engram_gate: signed-sqrt/sigmoid residual gating, not MoE routing.
- attention/mega_full_index: contains indexer-specific TopK machinery.
- mc2/moonep: cross-rank weight prefetch; excluded from this migration.

Numerical contracts
-------------------
Normalization accumulates FP32 sums over 512-column chunks, applies precise
sqrt/div, and rounds to BF16 before widening to FP32 for the router.
Inactive rows are zero. Gate and shared-expert quantization reuse one helper:
group32 scale = ceil_pow2(max(amax, 1e-4) / 448), E4M3 nearest-even payload,
E8M0 scale storage [M/16, K/64, 16, 2]. The attention OCP helper is separate.

Projection uses full FP32 M16/N16/K512 accumulation and sqrt-softplus scores,
with the existing negative-logit floor. A MIX leaf performs the projection
and score calculation with one AIC and two AIV participants. PyPTO reserves
the participants with sync_start=True.

TopK uses repeated vector maximum reductions and chooses the lower expert
index for exact ties. Correction bias affects selection only; weights use
the selected unbiased scores, normalized and multiplied by the route scale.
Runtime active-token counts control normalization and routing tails.

Validation
----------
validation/baseline-gate71.log.txt: original gate, 71 active tokens, PASS.
validation/input-and-shared.log.txt: normalization/quantization substitution
  at 71 and 0 active tokens, PASS; shared-expert quantization edges, PASS.
validation/router.log.txt: FP8 and E8M0 bytes compare exactly; stable TopK
  ids compare exactly, with ties across register boundaries and nontrivial
  unbiased weights; 7 and 0 active routing rows, PASS. Complete gate at
  71 and 0 active tokens also passes the original comparators.
validation/gate80.log.txt: complete gate with all 80 rows active, PASS.
validation/layer0-ep2.log.txt: TP1/EP2 layer 0, 72 active tokens per rank,
  devices 6 and 7, all 16 outputs PASS. Existing precision thresholds remain
  unchanged. The reported x_next rel_l2=0.1408408 and max_abs=3.2990062 match
  the preceding PyPTO-PR regression at the printed precision.
validation/static-checks.log.txt: Ruff, headers, English-only and diff checks.

Contents and reproduction
-------------------------
generated/gate80/ contains the orchestration, provider wrappers, generated
device source and native ABI manifests. generated/quant16/ and routes/ hold
isolated leaf equivalents. generated/layer0-ep2/ preserves the composed
configuration and orchestration. Model sources are copied as text under
model-snapshot/, including the preceding gate for comparison.

fixtures/ stores gzip-compressed CPU .pt inputs and goldens from the passing
validation entries. The routes fixture is the last saved runtime-scalar
case; consult its num_tokens.pt.gz. out/ contains CPU expected tensors,
not captured device readbacks. Logs record the actual comparisons. The
fixture builders retain deterministic CPU seeds for reproducing other cases.

Run from the model repository root with the paired environment loaded and
A5 devices allocated by the site's task queue:

python models/deepseek_v4_1_flash/gate_dsl.py -d <device>
python models/deepseek_v4_1_flash/gate.py --num-tokens 71 -d <device>
python models/deepseek_v4_1_flash/gate.py --num-tokens 0 -d <device>
python models/deepseek_v4_1_flash/gate.py --num-tokens 80 --save-data -d <device>
python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d <device0,device1>

Text workspace paths are replaced by <workspace> or <user-home>; regenerate
sources or replace placeholders before replaying archived paths. No credentials
or environment files are archived. artifact-manifest.json records provenance,
sizes and encoding, with no supplemental checksum or hash validation.
No performance study, full checkpoint, or prefill validation is claimed.
