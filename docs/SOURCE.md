# Source provenance

`EXPORTED_FROM` identifies the archive source:
`eee64280b9f50cf02c2ad427a0294cdb5b431bc9`.
The a17 package was built from the matching source-table lineage
`7e98e828176a99e53a2975e9814d9b660c1d1ca2`. Both are correct for the exported
runtime: all 314 warm, SYCL and shared helper entries in the measured package's
native source table match both revisions before export scrubbing. Later
experiment bookkeeping does not make this a newer engine build.

The export applies privacy-only path/comment rewrites. Public build scripts,
reference presets and documentation are maintained on the public history.
The a17 execution setup uses batched SwiGLU, small fusion, Engram priority,
512 MiB less fixed reserve on each GPU and a rebalanced placement. Its
26.27 tok/s (25.93–26.42) and strict 6/10 result used the same compiled package
with six pinned ROCm 7.2.1 userspace ELFs, ten answers and an 11,900-token cap.

`--preset a17` supplies the measured modes, capacities, profile and placement;
`--preset a16` retains the earlier configuration. The public benchmark keeps
its three-answer, 8,192-token, cumulative-counter protocol. Its results must
be measured separately. Public scripts and runtime staging have CPU checks;
the complete public build/pack/benchmark workflow has never run on GPUs.
