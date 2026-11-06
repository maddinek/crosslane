# Results

Hardware: AMD Radeon AI PRO R9700 (32 GB, HIP/ROCm), Intel Arc Pro B65 (32 GB, SYCL/oneAPI 2026.1,
driver 1.17.39758), Ryzen 9 9950X3D, 128 GB DDR5, both GPUs on PCIe 5.0 ×8.
Model: DeepSeek V4.1 Flash GSQ-RCO IQ2_XXS pack (40 layers, 384 routed experts, 6 per token,
~132 GB of expert payload). Upstream Strata `ffa80d73` (v0.2.10).

Protocol for fork rows, with stock noted below: one persistent warm engine; canonical 325-token coding prompt (complete
Python module + tests); sampled decoding T=1, top_p 1, top_k 64, min_p 0; seeds 1000 (warm-up,
excluded), then 1001–1003; 8,192-token cap; context 12,288; frozen scorer. tok/s = generated
outputs / decode wall time. Placement in the latest runs: R9700 resident bank ~21.6 GB, B65 warm bank ~25 GB
(under a 28.5 GB ceiling incl. runtime overhead), RAM store 110 GB budget with 32 readers and full
preload (~101.9 GB used). Experts never come from disk during decode (100% store hits); the two Engram
tables (110.6 GB, 103 GiB) are still read from disk, 48 rows per token.

The stock row used one GPU, 16 readers and about 89% store hits with about 53 MB per token of disk misses, so 2.59× is an observed whole-system comparison, not a matched kernel A/B. The speed-run logs stay private for now; sample answers are in [evidence/](evidence/).

| run | change | r0 | r1 | r2 | median | strict |
|---|---|---:|---:|---:|---:|---|
| stock | Strata v0.2.10, R9700 + RAM store, no B65 | — | — | — | 9.89 | — |
| a3-cut1 | first fork run: three concurrent lanes, B65 holds layer 0 dense | 9.66 | 10.54 | 10.28 | 10.28 | 1/3 |
| a5 | exact-order subgroup32 B65 attention (bit-identical tokens to a3) | 12.22 | 11.75 | 12.23 | 12.22 | 1/3 |
| a6 | IQ2_XS packed B65 kernels (B65 covers all 40 layers) | 12.06 | 12.04 | 12.76 | 12.06 | 0/3 |
| a6r | + critical-path rebalance from measured lane costs | 13.51 | 12.81 | 12.96 | 12.96 | 2/3 |
| a8c | all dense on the R9700 (cut 0), packed-strip B65 kernel | 14.41 | 13.72 | 15.39 | 14.41 | 1/3 |
| a8r | + rebalance on a8c costs | 15.63 | 15.71 | 15.01 | 15.63 | 0/3 |
| a8r2 | + second rebalance round | 16.02 | 15.85 | 15.38 | 15.85 | 0/3 |
| a11 | queued R9700 dense schedule + a third rebalance round | 17.04 | 14.97 | 17.23 | 17.04 | 0/3 |
| a13 | exact-order B65 SoA kernel, CPU layer batching, default placement | 16.44 | 16.73 | 15.63 | 16.44 | 1/3 |
| a13r | + rebalance | 18.04 | 18.48 | 17.80 | 18.04 | 0/3 |
| a14 | + R9700 per-layer expert batching, default placement | 17.25 | 14.60 | 17.58 | 17.25 | 1/3 |
| a14r | + rebalance | 21.29 | 21.27 | 19.62 | 21.27 | 2/3 |
| a15 | + parallel Engram reads (QD8, read-ahead; called Engram a3 privately), default placement | 18.22 | 18.84 | 18.31 | 18.31 | 1/3 |
| a15r | + rebalance | 22.40 | 22.57 | 22.61 | 22.57 | 0/3 |
| a16 | + AVX2 combine, early expert start, default placement (answers token-identical to a15) | 20.63 | 20.69 | 20.73 | 20.69 | 1/3 |
| a16r | + rebalance | 25.47 | 25.57 | 25.65 | **25.57** | 0/3 |
| a9r | MMVQ-style B65 kernel + rebalance: **failed the quality gate below, not adopted** | 18.19 | 17.05 | 18.53 | 18.19 | 0/3 |

Strict = the generated module's own tests plus independent functional checks pass. With 3 sampled
answers per run this is noisy, and many of the faster runs hit the 8,192 cap while still reasoning;
the fastest measured passing code at this cap is 333 s (a16 r2). All three a16r answers were cut off at
the cap (text normal, not looping); across our history 43 of 72 comparable answers hit it (a descriptive count
with repeated streams, not independent trials), which neither shows nor rules out a regression.

With a cap of 11,900 at the a16r setup (one warm-up, one measured answer): 6,288 tokens at 25.64 tok/s, then
7,179 tokens at 25.59 tok/s, finished by itself, strict pass, 13 own tests, working code in **295 s**. That's one
answer. A 10-answer test at this cap is the real check and hasn't run yet.

## Where a token goes in earlier runs (a8c kernel timers and later a9 medians, separate diagnostics)

R9700 dense, ~28 ms/token, of which 23.2 ms are kernel spans: matrix-vector products 11.9 (≈ the 10.6–13.2 ms memory-bandwidth floor),
sparse attention 3.0, norm/RoPE finish 1.8, HC mix/split + pre-norm 2.5, router + route + top-k 1.9,
output head 0.7, ring/ext copies, SwiGLU and hc_post ~1.4. The ~5 ms gap between spans is about 90% Engram
disk reads (layers 1 and 14, 48 serial reads per token).

Per expert selection in a real run (from per-layer lane service deltas): CPU ~0.29 ms, R9700
~0.12–0.13 ms, B65 ~0.32 ms. Joined expert time per token (a8r2): ~25 ms.

## Where a token goes (a16r)

Decode-only per fed token, pooled over three answers: wall 39.124 ms; dense excluding Engram-containing regions
22.149 ms; those regions 1.741 ms; expert join 11.914 ms; wall minus dense minus join 3.321 ms. Overlapping service: CPU 8.692,
B65 9.583, R9700 8.153 ms. Selected experts per token: R9700 154.7, B65 55.6, CPU 29.7 (default placement 113.3 / 52.0 / 74.6).
Separate diagnostic, one measured answer with kernel timers and an exclusive host partition (25.57 tok/s, no timer-off pair, not a
speed claim; its boundaries differ from the numbers above): dense 19.752, join 11.823, shared-FFN wait 5.603, head 0.995, remaining small steps including copies and device waits 0.946 ms.

Kernels over seven decode samples, launch spans including queue gaps: matrix-vector 8.020 + 3.638, attention 2.627, finish 1.641,
HC mix 1.337, HC pre-norm 1.007, router 0.842, head 0.736, route 0.629, ring copies 0.681, hc_post 0.334, SwiGLU 0.296, top-k 0.270 ms.
Traced time grows only 0.18 ms from position 1,024 to 7,168. Large Q5_K projections move 380–618 GB/s of weight
payload against a measured 612 GB/s stream, so dense is near the memory limit, though the timers can't separate
arithmetic from queue gaps.

Estimates for what's left on exact kernels: 26–28 tok/s, no credible exact route to 30 yet.

## Long prompts and long context

- Prefill, one teacher-forced 8,192-token stream on a15r, a slope diagnostic (not independent requests): 1,024 tokens 48.7 s, 4,096 182.4 s, 8,192 369.4 s (22.2 tok/s), with no slowdown by position.
- 12,001 positions on a16r, unchanged reserves: R9700 28.47 GB whole device (minimum free 5.73 GB), B65 25.31 GB resident (28.5 GB ceiling), no growth with position; forced tokens ran at 23.5–25.3 tok/s. Sampling can miss spikes.

## DSpark and Q2_K

- DSpark projection micro on real weights, R9700: batched rows byte-identical to serial on all 15 tensor signatures, T=1..6. Three rows cost 0.34–0.52 of three serial calls on Q5_K and Q8_0 tensors (head 1.02× one row, attention output 1.52×). Synthetic inputs, projections only. Acceptance on this edited model and the whole cycle are unmeasured.
- Q2_K re-encoding of experts: stopped by a rule set before the run. Warm µs per selection at N=1/2/3/6: R9700 IQ 80.6/62.7/50.8/40.6, Q2_K 80.1/52.0/43.6/33.9; B65 IQ 187.7/152.3/139.4/124.8, Q2_K 212.6/173.1/157.1/142.2. Projected on the a14r base, not measured end to end: net −3.18 ms per token, because bigger experts push 1,434 back to the CPU.

## Where a token goes (a14r)

Per fed token: dense excluding Engram 22.0–22.2 ms, Engram layers 4.7–8.9 ms, expert join 14.5 ms (service: CPU
8.3, R9700 8.5, B65 9.4 ms), and about 6 ms in everything else. Answers of identical tokens have shown Engram
at anywhere from about 5 to 17 ms, which accounts for most of the spread between answers (the 14.60 in a14).

## B65 kernel microbenchmarks (native, warm)

| B65 expert path | per expert |
|---|---:|
| scalar reference | 1.89 ms (IQ2_XS) |
| packed whole-expert dp4a (exact order) | 0.28–0.38 ms |
| packed strip (exact order) | 0.27–0.33 ms |
| MMVQ-style SoA, 4–8 experts per launch (reordered sums, failed the gate below) | 0.128–0.140 ms |
| exact-order SoA (`mmvq-soa-v2`), 4–8 experts per launch | 0.114–0.134 ms |

In a real run the exact-order kernel costs about 0.18 ms per selection, because the B65 averages about 1.3 selections per layer. The R9700's per-layer batching: 95 µs at 1 expert, 227 µs for 6 (38 µs each), against 649 µs serial.

## The MMVQ kernel and the quality gate

A kernel that changes summation order has to pass all five points, in this order: exact integer dot
products; projection error within the bound; whole-expert NRMSE ≤ 0.005 and max ≤ 2% of peak;
teacher-forced top-1 ≥ 99%, mean KL ≤ 0.001 and max KL ≤ 0.01 on the real prompt against the exact
kernels; and no fewer scorer passes than the exact kernels.

`mmvq-soa-v1` passed the per-expert points (projections ≤ 2.6e-7, whole expert ≤ 4.27e-4 on real
activations; 5.3e-6 was on synthetic inputs and understated it) and failed the teacher-forced one:

| comparison with exact kernels | top-1 | mean KL | max KL |
|---|---:|---:|---:|
| all 581 positions | 95.35% | 0.0137 | 1.486 |
| 256 continuation positions | 95.31% | 0.0047 | 0.071 |

It is not adopted, and its 18.19 tok/s doesn't count as a result.

A later control complicates the picture: exact kernels at a different placement differ by about as much
(top-1 96.56%, mean KL 0.0136, max KL 1.644), while repeating a run is byte-identical. So this gate is
stricter than the difference between our own placements. The exact-order kernel passes it outright: byte-identical at
581 positions with no route changes, and just as fast in a run.
