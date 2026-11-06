# crosslane

Run a big Mixture-of-Experts model across **mixed-vendor GPUs plus system RAM**, with every device
computing its share of each layer's experts **at the same time**.

crosslane is a research fork of [Strata](https://github.com/Niko1221/Strata) (pinned at
`ffa80d73`, v0.2.10) for DeepSeek V4.1 Flash (GSQ-RCO IQ2_XXS pack, 40 layers × 384 routed
experts, 6 per token). The test machine has one AMD Radeon AI PRO R9700 (HIP), one Intel Arc Pro B65
(SYCL/ESIMD), a Ryzen 9 9950X3D and 128 GB DDR5. Neither GPU can hold the model alone.

> Status: research code, tested on the machine above only. The standalone build, pack and benchmark
> scripts in this repo have passed CPU tests but have **not yet been run end to end on GPUs**; the
> results below come from the same engine sources, run under our private run guards. Expect to read code.

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
| a9r | MMVQ-style B65 kernel + rebalance: **failed our quality gate, not adopted** | 18.19 | 0/3 |

15.85 tok/s is the best result on kernels we accept. The quality score is noisy at n=3, and the faster runs often hit the 8,192-token cap while still
"thinking". See [docs/RESULTS.md](docs/RESULTS.md) for every run, its spread and its raw-data hashes.

Where the time goes now (≈63 ms/token): R9700 dense ~28 ms (~23 ms of kernels, where the matrix-vector
products are near the memory-bandwidth floor and ~11 ms is small kernels, plus a ~5 ms gap that is mostly
disk reads of the two Engram tables), the joined expert lanes ~25 ms, and the rest in sampling, routing
and host work. The default uniform placement in the quick benchmark won't reproduce these numbers: they
use measured placement (`engine/strata/warm/make_costs.py` + `rebalance.py`).

The MMVQ-style B65 expert kernel (opt-in, `--sycl-experts mmvq-soa-v1`) cuts the B65 from ~0.32 to ~0.19 ms
per expert in a real run, but it reorders sums and **failed our gate**: teacher-forced against the exact
kernels on the real prompt, top-1 agreement was 95.35% (gate ≥ 99%), mean KL 0.0137 (≤ 0.001) and max KL
1.486 (≤ 0.01), even though the per-expert error on real activations is small (NRMSE ≤ 4.3e-4). A control
between exact placements is still to run, and an exact-order version is next. See
[docs/RESULTS.md](docs/RESULTS.md#the-mmvq-kernel-and-the-quality-gate).

## How this was built

This was built by **Marty (the owner of the machine)** with two AI coding agents working as a pair:
**Claude** (Anthropic) and **Codex** (OpenAI). Every design decision and every code change was reviewed
by the other agent before it ran on the hardware. The agents wrote most of the code; Marty set the
goals ("both GPUs fully involved", "no stock runs", "30 tok/s on this hardware"), made the calls on
trade-offs and ran the machines. Every experiment, including the failures, was logged with its raw
data. A selection of the dead ends is in [docs/DEAD_ENDS.md](docs/DEAD_ENDS.md), because they are as
useful as the wins.

## Building

See [docs/BUILD.md](docs/BUILD.md): `tools/build.sh`, `tools/pack.py`, `tools/bench.sh`. You need ROCm
(HIP) for the AMD card, oneAPI 2026.1 (SYCL/ESIMD) for the Intel card and the upstream Strata source at
the pinned commit. Model weights are not included; the pack is built from the public GGUF.

## Where to help

See [CONTRIBUTING.md](CONTRIBUTING.md#where-to-help). The biggest open levers are the B65 expert kernel,
fusing the small dense kernels on the R9700 and a faster CPU expert lane. Other GPU pairs (two AMD
cards, AMD + NVIDIA, Intel + NVIDIA) are untested and very welcome.

## Licence

MIT. Upstream Strata and Project Maya (MIT) and ggml/llama.cpp (MIT) notices are kept in
[licenses/](licenses/) and `engine/strata/sycl/moe/licenses/`. Intel dpct helper headers keep their Apache-2.0 WITH LLVM-exception headers.
