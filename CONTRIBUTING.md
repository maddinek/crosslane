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

Ranked by expected gain on the reference machine (39.1 ms/token today, 33.3 ms is the goal):

1. **R9700 matrix-vector kernels** (about 11.7 of 22 ms of dense). Big projections are already near the measured stream limit. HIP graph replay, SoA and reordered Q5 sums were measured and didn't help, and fusing small kernels saves under 1 ms. A real new idea here is the only exact lever big enough to matter.
2. **Speculative decoding with DeepSeek's DSpark drafter.** Projections batch well and byte-exactly. Unmeasured: acceptance on this edited model, full-row dense and expert cost, and the whole cycle. A verify path needs at least 1.5× to be worth building.
3. **Batched prefill.** An 8,192-token prompt takes about 6 minutes today because it runs token by token. Estimated 38–72 tok/s on exact paths.
4. **Reaching working code sooner.** Many answers hit the token cap while still reasoning. A lower reasoning effort or better planning could matter more than raw tok/s, but it has to keep quality.
5. **Placement that knows about batching and co-selection.** Batch cost is non-linear, and the optimizer uses averages. Estimated under 1 ms, and it overlaps with more VRAM.
6. **Join and shared-FFN overlap.** Early start proved this was worth about 3.5 ms; ordered timestamps would show what's left.
7. **Other hardware.** Two AMD cards, AMD + NVIDIA, Intel + NVIDIA. The placement and three-lane ideas transfer; the engine is DeepSeek-specific and the kernels cover IQ2_XXS, IQ2_XS and IQ1_M.

## Commits

Small commits, one change each, with what was measured in the message. Please don't reformat
upstream Strata files; keep our changes easy to rebase onto new upstream pins.
