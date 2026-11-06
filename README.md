# crosslane

Run a big Mixture-of-Experts model across **mixed-vendor GPUs plus system RAM**, with every device
computing its share of each layer's experts **at the same time**.

crosslane is a research fork of [Strata](https://github.com/Niko1221/Strata) (pinned at
`ffa80d73`, v0.2.10) for DeepSeek V4.1 Flash (GSQ-RCO IQ2_XXS pack, 40 layers × 384 routed
experts, 6 per token). The test machine has one AMD Radeon AI PRO R9700 (HIP), one Intel Arc Pro B65
(SYCL/ESIMD), a Ryzen 9 9950X3D and 128 GB DDR5. Neither GPU can hold the model alone.

> Status: research code. It works on the machine above and has not been tested anywhere else.
> Expect to read code.

## What it does

- **Three expert lanes per layer.** The R9700 holds a resident bank of the hottest experts (~21.6 GB),
  the B65 a warm bank (~25 GB) and a 110 GB RAM store with 32 readers and full preload holds the rest.
  For each token and layer, all three lanes run their selected experts concurrently and join with a
  fixed-order combine.
- **Dense layers on the faster card.** All attention/shared-expert work runs on the R9700 (cut 0); the
  B65 serves experts only, so no activations cross between the cards per layer.
- **Whole-expert packed dp4a kernels for the B65.** ESIMD kernels for IQ2_XXS, IQ2_XS and IQ1_M (gate,
  up and down), plus an exact-order subgroup attention kernel.
- **Critical-path placement.** A rebalancer that uses measured per-lane costs from a real run moves
  experts between lanes so the three finish each layer together.
- **Measured, not guessed.** Per-layer lane service times, store counters and per-kernel launch spans are
  logged in every run.

## Results so far

Realistic coding task (one 325-token prompt asking for a complete Python module with tests), sampled
decoding (T=1, top_k 64), 8,192-token cap, warm engine, 3 measured answers after one warm-up. Same
model file throughout. "Strict" is a frozen scorer that runs the generated module's own tests plus
independent checks.

| step | change | decode tok/s (median of 3) | strict |
|---|---|---:|---|
| stock Strata, R9700 + RAM only | reference | 9.89 | — |
| a3 | first fork run: both GPUs + RAM store | 10.28 | 1/3 |
| a5 | subgroup B65 attention | 12.22 | 1/3 |
| a6r | IQ2_XS B65 kernels + rebalance | 12.96 | 2/3 |
| a8c | all dense on the R9700, packed-strip B65 kernel | 14.41 | 1/3 |
| a8r2 | two rebalance rounds | **15.85** | 0/3 |

The quality score is noisy at n=3, and the faster runs often hit the 8,192-token cap while still
"thinking". See [docs/RESULTS.md](docs/RESULTS.md) for every run, its spread and its raw-data hashes.

Where the time goes now (≈63 ms/token): R9700 dense ~23 ms (the matrix-vector products are already
near its memory-bandwidth floor; ~11 ms is small kernels), the joined expert lanes ~25 ms, and the
rest in sampling, routing and host work. The next step is an MMVQ-style B65 expert kernel: on the card
it measures 128–140 µs per expert vs ~400 µs for the exact path (NRMSE 5e-6). It isn't in a full run
yet.

## How this was built

This was built by **Martin (the owner of the machine)** with two AI coding agents working as a pair:
**Claude** (Anthropic) and **Codex** (OpenAI). Every design decision and every code change was reviewed
by the other agent before it ran on the hardware. The agents wrote most of the code; Martin set the
goals ("both GPUs fully involved", "no stock runs", "30 tok/s on this hardware"), made the calls on
trade-offs and ran the machines. Every experiment, including the failures, was logged with its raw
data. A selection of the dead ends is in [docs/DEAD_ENDS.md](docs/DEAD_ENDS.md), because they are as
useful as the wins.

## Building

See [docs/BUILD.md](docs/BUILD.md). You need ROCm (HIP) for the AMD card, oneAPI 2026.1 (SYCL/ESIMD) for
the Intel card and the upstream Strata source at the pinned commit. Model weights are not included;
the pack is built from the public GGUF with the scripts in `tools/`.

## Where to help

See [CONTRIBUTING.md](CONTRIBUTING.md#where-to-help). The biggest open levers are the B65 expert kernel,
fusing the small dense kernels on the R9700 and a faster CPU expert lane. Other GPU pairs (two AMD
cards, AMD + NVIDIA, Intel + NVIDIA) are untested and very welcome.

## Licence

MIT. Upstream Strata and Project Maya (MIT) and ggml/llama.cpp (MIT) notices are kept in
[third_party/](third_party/). Intel dpct helper headers keep their Apache-2.0 WITH LLVM-exception headers.
