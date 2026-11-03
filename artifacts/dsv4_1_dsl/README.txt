DeepSeek V4.1 DSL development records
===================================

Development branch: nalinaly/pypto-lib:dsv4.1-dsl
https://github.com/nalinaly/pypto-lib/tree/dsv4.1-dsl

Keep subsequent DeepSeek V4.1 DSL implementations and their supporting process
data on this branch. The model implementation lives in
models/deepseek_v4_1_flash/. Store evidence in a dated directory here, preserving
previous captures rather than replacing their data with later runs.

2026-11-03/ contains the first window-KV integration, numerical investigation,
passing validation logs, L1 traces, generated device source, and frozen tensors.
Start with that directory's README.txt and artifact-manifest.json.

2026-11-03-ascendc-kv-scale/ records upstream pypto-lib issue #1666 about
window KV scale selection versus recipes AscendC, including source revisions
and a standalone CPU numerical example. It does not change the model kernels.
