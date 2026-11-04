DeepSeek V4.1 local MoE combine DSL integration — 2026-11-04
=========================================================

Scope
-----
Only the decode combine_reduce computation is replaced, by the AIV leaf
combine_rows and its PyPTO scheduler in moe_combine_dsl.py. The existing
combine_notify and combine_wait scopes remain unchanged. All other functions
in decode_moe.py are unchanged; expert_routed.py and decode_layer.py are unchanged.

The DSL receives a view of the local rank's routed-output window. It only
reads local memory. PyPTO/simpler owns window binding, notification, deferred
completion, slicing and scheduling. The completion pool starts from the
original combine_wait TaskId, chains each slot's subsequent tasks and joins
all final slot handles before mHC post-processing.

Numerical contract
------------------
Two physical rows per task. Widen the shared expert output to FP32, add
routed lanes 0 through 5 sequentially, then round once to BF16 nearest-even.
Inactive rows are written as zero without loading their poisoned inputs.
An explicit physical-row count protects the final row of an odd-size tensor.

Validation
----------
validation/local.log: all four cases PASS at atol=rtol=max_error_ratio=0:
  physical 5 / active 3, physical 5 / active 0,
  physical 80 / active 71, physical 80 / active 80.
The fixtures exercise FP32 cancellation order, BF16 rounding midpoints,
distinct route values and NaN inactive inputs. One output guard row must
retain the original BF16 sentinel 37.0 in every case.

validation/layer0-ep2.log: TP1/EP2 layer 0, 72 active tokens per rank, devices
1 and 2; all 16 outputs PASS at the original model thresholds. Reported
x_next rel_l2=0.1408408 and max_abs=3.2990062 match the preceding routed-FFN
regression at the printed precision.

validation/communication-review.txt records direct AST comparisons of the
unchanged communication scopes and surrounding model functions. The local
and MoE code-generation logs are also retained.

Contents and reproduction
-------------------------
generated/ contains the wrapper, device source, native ABI manifest and
orchestration for local and EP2 use. The EP2 orchestration shows a normal
Tensor view of ext_routed_output passed as a read-only local argument.
model-snapshot/ contains the implementation and preceding decode_moe source.

fixtures/ contains gzip-compressed CPU .pt inputs and expected outputs.
Each shape's fixture is the last saved scalar-count case: local5 has zero
active rows; local80 has all 80 rows active. The deterministic fixture
builder reproduces the other cases. out/ contains CPU goldens, not saved
device readbacks; the validation logs record actual device comparisons.

From the model repository root, with the paired environment active and A5
devices allocated by the site's task queue:

python models/deepseek_v4_1_flash/moe_combine_dsl.py -d <device>
python models/deepseek_v4_1_flash/decode_layer.py --ep 2 --layer-id 0 -d <device0,device1>

The model base is 50fb049, PyPTO is 4a65a9c and the DSL is 91feb22. Neither
framework is changed. Source and evidence belong to dsv4.1-dsl. No checkpoint,
prefill or performance claim is made. Torch is used only on CPU.

Text paths are sanitized to <workspace>/<user-home>, and trailing whitespace
is normalized in archived text. Binary fixture bytes are preserved inside
gzip files. Credentials and environment files are excluded. The manifest
records provenance, sizes and encoding without supplemental checksums.
