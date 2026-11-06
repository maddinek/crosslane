# Contributing

Thanks for looking. This is research code that grew out of one machine, so the bar for a useful
contribution is low: a measurement on different hardware is as welcome as a kernel.

## Ground rules

- **Measure, then claim.** A speed claim needs the same model file, quant and prompts, at least 3
  repetitions with the spread, and the raw log attached. Use `tools/bench.sh` (one command, see below).
- **Defaults stay exact.** Anything that changes outputs is opt-in (a freeze/config flag) and comes
  with a numerical check: per-matrix and whole-expert error against the CPU reference, and token
  agreement on the canonical prompt. A kernel that sums in a different order than the reference has to
  pass the gate in [docs/RESULTS.md](docs/RESULTS.md#the-mmvq-kernel-and-the-quality-gate) before its
  speed counts; none has yet. Kernels may not change routing or the math semantics.
- **Negative results count.** If something was slower or broke, add a row to `docs/DEAD_ENDS.md`
  with the reason and the raw numbers. That saves the next person a night.
- **No weights in the repo.** Link the public GGUF and the pack-building script.

## One-command benchmark

```bash
tools/bench.sh --pack /absolute/models/pack
```

It runs the canonical coding prompt (one warm-up + 3 measured answers, T=1, top_k 64, 8,192-token cap),
prints decode tok/s per answer with median/min/max, and writes the raw logs to a new `bench-out/`.
It uses a default placement, so expect less than the measured-placement numbers in the README. Please paste the summary and attach the log in your
issue or PR.

## Where to help

Ranked by expected gain on the reference machine (≈63 ms/token today):

1. **B65 expert kernel** (biggest). A whole expert takes ~0.32 ms in-run (~5% of the card's 608 GB/s).
   An MMVQ-style kernel reaches ~0.19 ms in a real run but failed the quality gate, so it isn't adopted.
   An exact-order version on the same weight layout is being tested. What's wanted is a fast kernel that
   keeps the order, or one that passes the gate.
2. **R9700 small dense kernels** (~11 ms/token). Norm/RoPE "finish", HC mixing/Sinkhorn, routing,
   ring copies and batch-1 sparse attention. Fusing them is the lever; HIP graph replay was measured
   slower on ROCm at batch 1 (see DEAD_ENDS).
3. **Engram disk reads** (~5 ms/token). Two layers read 48 rows per token from 103 GB of tables, one
   after another. Same-token read-ahead is being tested.
4. **CPU expert lane** (~0.29 ms per expert, ~30 GB/s). Parallel experts across all cores, AVX-512
   VNNI IQ2/IQ1 kernels, CCD-aware pinning, prefetch from the RAM store.
5. **Speculative decoding.** It's modelled at only 1.2–1.3× today, because a 2-token verify nearly
   doubles the experts touched. It becomes interesting once the expert lanes get cheaper.
6. **Other hardware.** Two AMD cards, AMD + NVIDIA, Intel + NVIDIA, other MoE models (GLM, MiniMax,
   Qwen MoE). The placement and three-lane runtime are model-agnostic; the kernels cover IQ2_XXS,
   IQ2_XS and IQ1_M today.
7. **Quality at long outputs.** Many answers run into the token cap while still reasoning. Better
   stopping or a thinking budget would cut time-to-working-code more than raw tok/s.

## Commits

Small commits, one change each, with what was measured in the message. Please don't reformat
upstream Strata files; keep our changes easy to rebase onto new upstream pins.
