AscendC window KV scale contract question -- 2026-11-03
======================================================

Upstream issue: https://github.com/hw-native-sys/pypto-lib/issues/1666

This record compares upstream PyPTO-lib main with the pinned recipes AscendC
window KV quantizer. It asks whether the different scale choices are intended
and whether their model accuracy impact has been validated. It does not report
a proven end-to-end accuracy regression or change the local DSL implementation.

issue.txt contains the published Chinese title and complete body, including
fixed-revision source links and the standalone CPU reproduction.
repro.py.txt is the same runnable example; save it as repro.py and run with
a Torch installation that supports CPU FP8 conversion.
repro-output.txt contains the observed output.
issue-metadata.json records the public issue and compared revisions.
artifact-manifest.json records checksums for this archive.

The upstream main writer is kernels/norm_rope_publish.cpp in this snapshot.
The earlier name window_norm_rope_publish.cpp no longer exists on that main.
The existing dsv4.1-dsl branch and its earlier device validation are unchanged.

The numerical example isolates scale selection on one BF16 group. Both FP8
payload and reconstructed values differ. Neither the scale ratio nor this
single group establishes model-level error or ranks the two policies globally.
