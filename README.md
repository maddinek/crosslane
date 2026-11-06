# crosslane

**A 763-billion-parameter DeepSeek on one desktop, at 25.6 tokens per second, with an AMD GPU, an Intel GPU and system RAM computing each layer's experts concurrently.**

[![Decode speed rose from 9.89 to 25.57 tokens per second over eleven steps, against a goal of 30](docs/progress.svg)](docs/RESULTS.md)

| | |
|---|---|
| **Speed** | **25.57 tok/s** median (25.47 / 25.57 / 25.65), generating real code |
| **Against stock** | **2.59×** our recorded stock [Strata](https://github.com/Niko1221/Strata) run on the same machine and model file (9.89 tok/s, one GPU, 16 readers, a 51 GB store). A whole-system comparison, not a kernel A/B |
| **How exact** | No speculation, no skipped experts, routing algorithm unchanged. Tested kernel and schedule modes are byte-identical at a fixed placement over 581 teacher-forced positions; placement changes alter results slightly (see below) |
| **Fastest working code** | **295 s** at 25.59 tok/s with an 11,900-token cap: one answer, passing its own 13 tests and independent checks. Includes prefill, excludes engine load |
| **Hardware** | Radeon AI PRO R9700 32 GB, Arc Pro B65 32 GB, Ryzen 9 9950X3D, 128 GB DDR5 |
| **Goal** | 30 tok/s. **Not reached yet**; see [what's left](#whats-left) |

crosslane is a research fork of Strata (pinned at `ffa80d73`, v0.2.10). Neither GPU can hold this model alone, so the active lanes compute that layer's routed experts concurrently, and dense work stays on the R9700.

> **Status: research code, tested on one machine.** The results come from these engine sources run under our private safety guards. The standalone build, pack and benchmark scripts have passed CPU tests but **haven't yet run end to end on GPUs**, and the benchmark doesn't enable every setting used for the best result yet. Expect to read code.

## How it works

The model is DeepSeek V4.1 Flash 1017: 40 layers, 384 routed experts per layer, 6 picked per token, so 240 expert computations for every token. We split them by what each device does best:

```mermaid
flowchart LR
    T[Token] --> D["R9700: all dense work<br/>attention, shared FFN, routing"]
    D --> R{"Route: 6 experts<br/>for this layer"}
    R --> A["R9700 bank<br/>2,532 experts, 21.6 GB<br/>154.7 picks per token"]
    R --> B["B65 bank<br/>2,915 experts, 25.0 GB<br/>55.6 picks per token"]
    R --> C["RAM store, 32 readers<br/>9,913 experts, 102 GB<br/>29.7 picks per token"]
    A --> J["Join and combine<br/>in fixed order"]
    B --> J
    C --> J
    J --> N[Next layer]
```

Every device is doing real work, in a different role:

| | R9700 | Arc Pro B65 | CPU + RAM |
|---|---:|---:|---:|
| Dense work (attention, shared FFN, routing) | all of it, 22 ms | none | none |
| Expert bank | 2,532 experts, 21.6 GB | **2,915 experts, 25.0 GB, the largest GPU bank** | 9,913 experts, 102 GB |
| Mean expert picks per token, of 240 | 154.7 | 55.6 | 29.7 |
| Time per pick | 0.05 ms | 0.17 ms | 0.29 ms |
| Expert work per token | 8.2 ms | **9.6 ms, the longest lane** | 8.7 ms |

The R9700 is faster per pick, so the rebalancer gives it the most-used experts, and the B65 and CPU lanes get fewer picks. That brings the lanes' total service close, though individual layers still differ. The B65 runs no dense work: its 9.6 ms of expert service is out of a 39.1 ms token (service time, not a GPU-utilisation measurement). It did in earlier versions, but dense layer 0 took 11.5 ms there against 0.48 ms on the R9700, and moving all dense over, together with a packed-strip B65 kernel, took us from 12.96 to 14.41 tok/s. That's the part of "both GPUs equally involved" we haven't reached.

What made the difference, in the order we found it:

- **Three expert lanes per layer.** Each lane runs its picks at once, then they join in a fixed order, so the math doesn't depend on who finished first.
- **Measured placement.** A rebalancer uses real per-lane costs from a real run and moves experts to bring the lanes' total time together. Rebalancing took 20.69 to 25.57 tok/s in the last step; placement also changes the answers, so that's an unpaired comparison of medians.
- **New kernels for both vendors.** An exact-order structure-of-arrays dot-product kernel for IQ2_XXS, IQ2_XS and IQ1_M on the B65, per-layer expert batching on the R9700 (six experts in 227 µs instead of 649 µs).
- **Dense on the faster card only**, with a queued schedule, so there's no dense-stage hand-off between GPUs. Expert inputs and outputs still transfer.
- **Parallel Engram reads.** DeepSeek's memory lookup hits 24 rows per layer on disk. Parallel reads, read-ahead and a 128 MiB row cache cut a noisy 5–17 ms per token; the whole Engram-containing regions now total 1.74 ms, including other work in those layers.
- **Starting expert work early.** CPU and B65 start before the shared FFN drains, instead of after it. Together with a faster AVX2 combine, that cut 5.8 ms per token on token-identical answers. We only measured the two together.

## Results

Realistic task: one 325-token prompt asking for a complete Python module with tests. Sampled (T=1, top-k 64), warm engine, one warm-up then 3 measured answers, 8,192-token cap, same model file throughout. Full history and spreads are in [docs/RESULTS.md](docs/RESULTS.md). The answers from the best setup are in [docs/evidence/](docs/evidence/). The guarded-run logs behind the numbers stay private for now, because they're full of host details.

| Step | Change | tok/s | Strict |
|---|---|---:|---:|
| Stock | Strata, R9700 + RAM only | 9.89 | — |
| a3 | First fork run, both GPUs + RAM store | 10.28 | 1/3 |
| a5 | Subgroup B65 attention | 12.22 | 1/3 |
| a6r | IQ2_XS B65 kernels + rebalance | 12.96 | 2/3 |
| a8c | All dense on the R9700, packed B65 kernel | 14.41 | 1/3 |
| a8r2 | Two rebalance rounds | 15.85 | 0/3 |
| a11 | Queued R9700 schedule + rebalance | 17.04 | 0/3 |
| a13r | Exact-order B65 SoA kernel + rebalance | 18.04 | 0/3 |
| a14r | R9700 expert batching + rebalance | 21.27 | 2/3 |
| a15r | Parallel Engram reads + rebalance | 22.57 | 0/3 |
| **a16r** | **Faster combine, early expert start + rebalance** | **25.57** | **0/3** |

### Where a token goes now

39.1 ms per token in the best run:

| Part | ms | |
|---|---:|---|
| Dense, excluding Engram layers | 22.15 | Matrix-vector products are over half; large ones already run at 380–618 GB/s of weights against a measured 612 GB/s stream |
| Engram-containing regions | 1.74 | Includes those layers' other work |
| Expert join | 11.91 | CPU 8.7, B65 9.6, R9700 8.2 ms of overlapping service |
| Residual | 3.32 | Wall minus the regions above, not an exclusive breakdown. A separate timed diagnostic found 0.95 ms of small host and copy steps outside dense, join, shared-FFN wait and head |

### The strict score, honestly

**All three answers in the best run hit the 8,192-token cap before finishing, so it scored 0/3.** The model reasons for a long time first, and these answers were cut off, not looping. Across our history 43 of 72 comparable answers hit that cap (a descriptive count with repeated streams, not independent trials), so three in a row isn't surprising. That neither shows nor rules out a quality loss.

With room up to 11,900 tokens at the same setup, the one measured answer finished on its own at 25.59 tok/s and passed strict with working code in 295 s. One answer proves little. A 10-answer test at that cap is next and will add real evidence.

Moving experts between devices also changes the numbers slightly, even with exact kernels (top-1 96.6%, mean KL 0.014), most likely because CPU and GPU lanes quantize activations differently. Routes change too (6,295 of 23,240 layer selections differ). So "exact" here means byte-identical logits at a fixed placement, checked over 581 teacher-forced positions for the tested kernel and schedule modes. Different placements are different runs of a very close model, not the same bytes.

### Also measured

- **Long prompts:** an 8,192-token prefill took 369 s at 22.2 tok/s on the a15r setup, with no slowdown by position. It's one diagnostic stream, token by token, and the weak spot.
- **Long context:** a 12,001-position run reached 28.5 GB on the R9700 (whole device) and 25.3 GB on the B65, flat with position. Sampling can miss spikes.
- **Memory:** 21.6 GB and 25.0 GB of expert banks, plus 102 GB preloaded in RAM. Experts never come from disk during decode.

## What's left

30 tok/s needs 33.3 ms per token, 5.8 ms less than today.

- **Exact kernels:** our estimate is a plateau around 26–28 tok/s. The remaining levers are small and overlap, and no credible exact route to 30 has shown up yet. Being wrong would be welcome.
- **DeepSeek's DSpark drafter:** a 1.5× gain would justify building a verify path. So far: checking several rows at once stays byte-identical, and three rows cost 0.34–0.52 of three separate runs on the big matrices. Acceptance on this model and the whole cycle are unmeasured.
- **Re-encoding experts to Q2_K: rejected.** Measured kernels plus a capacity model project 3.18 ms slower per token, with 1,434 experts pushed to the CPU. There was no full-model Q2_K run.
- **Quality:** the 10-answer test, and whether a lower reasoning effort reaches working code sooner.
- **Prefill:** batching it, estimated at 38–72 tok/s on exact paths.

The dead ends are as useful as the wins, so they're in [docs/DEAD_ENDS.md](docs/DEAD_ENDS.md).

## The model

[`AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-GGUF`](https://huggingface.co/AtomicChat/DeepSeek-V4.1-Flash-1017-Uncensored-GSQ-RCO-GGUF), rung IQ2_XXS: 138 GB of backbone plus a 110 GB Engram table. It's [ISTA-DASLab's GSQ-RCO](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) quantization of DeepSeek's weights, where GSQ learns the low-bit rounding and RCO picks each tensor's type, with AtomicChat's one-direction refusal edit on 82 dense tensors. crosslane changes how the file runs, not what's in it. Weights aren't included here.

The kernels cover IQ2_XXS, IQ2_XS and IQ1_M. The engine is DeepSeek-specific today; the placement and three-lane ideas aren't.

## How this was built

Built by **Marty** with two AI coding agents working as a pair: **Claude** (Anthropic) and **Codex** (OpenAI). Each one reviewed the other's designs and code before anything ran on the hardware, and each caught real mistakes in the other's work. The agents wrote most of the code. Marty set the goals ("both GPUs fully involved", "no stock runs", "30 tok/s on this hardware"), made the calls on trade-offs and refused the easy answers.

The rules that kept it honest: raw data for every run, at least 3 repetitions with spread reported, the same model file, stop rules written before a run, and failures logged the same day.

## Try it

See [docs/BUILD.md](docs/BUILD.md): `tools/build.sh`, `tools/pack.py`, `tools/bench.sh`. You need ROCm (HIP) for the AMD card, oneAPI 2026.1 (SYCL/ESIMD) for the Intel card and the upstream Strata source at the pinned commit. The quick benchmark uses a default placement and won't reproduce the numbers above; they use measured placement (`engine/strata/warm/make_costs.py` + `rebalance.py`).

## Help wanted

[CONTRIBUTING.md](CONTRIBUTING.md#where-to-help) ranks what would help most: R9700 matrix-vector kernels, co-selection-aware placement, batched prefill, and other GPU pairs. Two AMD cards, AMD + NVIDIA and Intel + NVIDIA are all untested and very welcome.

## Licence

MIT. Upstream Strata and Project Maya (MIT) and ggml/llama.cpp (MIT) notices are kept in
[licenses/](licenses/) and `engine/strata/sycl/moe/licenses/`. Intel dpct helper headers keep their Apache-2.0 WITH LLVM-exception headers.
