# Results

Hardware: AMD Radeon AI PRO R9700 (32 GB, HIP/ROCm), Intel Arc Pro B65 (32 GB, SYCL/oneAPI 2026.1,
driver 1.17.39758), Ryzen 9 9950X3D, 128 GB DDR5, both GPUs on PCIe 5.0 ×8.
Model: DeepSeek V4.1 Flash GSQ-RCO IQ2_XXS pack (40 layers, 384 routed experts, 6 per token,
~132 GB of expert payload). Upstream Strata `ffa80d73` (v0.2.10).

Protocol for every row: one persistent warm engine; canonical 325-token coding prompt (complete
Python module + tests); sampled decoding T=1, top_p 1, top_k 64, min_p 0; seeds 1000 (warm-up,
excluded), then 1001–1003; 8,192-token cap; context 12,288; frozen scorer. tok/s = generated
outputs / decode wall time. Placement per run: R9700 resident bank ~21.6 GB, B65 warm bank ~25 GB
(under a 28.5 GB ceiling incl. runtime overhead), RAM store 110 GB budget with 32 readers and full
preload (~101.9 GB used). Experts never come from disk during decode (100% store hits); the two Engram
tables (103 GB) are still read from disk, 48 rows per token.

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

| a11 | queued R9700 dense schedule + a third rebalance round | 17.04 | 14.97 | 17.23 | **17.04** | 0/3 |
| a9r | MMVQ-style B65 kernel + rebalance: **failed the quality gate below, not adopted** | 18.19 | 17.05 | 18.53 | 18.19 | 0/3 |

Strict = the generated module's own tests plus independent functional checks pass. With 3 sampled
answers per run this is noisy, and many of the faster runs hit the 8,192 cap while still reasoning;
the best time to passing code so far is 448 s (a8c r2).

## Where a token goes (a8c, per-kernel timers)

R9700 dense, ~28 ms/token, of which 23.2 ms are kernel spans: matrix-vector products 11.9 (≈ the 10.6–13.2 ms memory-bandwidth floor),
sparse attention 3.0, norm/RoPE finish 1.8, HC mix/split + pre-norm 2.5, router + route + top-k 1.9,
output head 0.7, ring/ext copies, SwiGLU and hc_post ~1.4. The ~5 ms gap between spans is about 90% Engram
disk reads (layers 1 and 14, 48 serial reads per token).

Per expert selection in a real run (from per-layer lane service deltas): CPU ~0.29 ms, R9700
~0.12–0.13 ms, B65 ~0.32 ms. Joined expert time per token (a8r2): ~25 ms.

## B65 kernel microbenchmarks (native, warm)

| B65 expert path | per expert |
|---|---:|
| scalar reference | 1.89 ms (IQ2_XS) |
| packed whole-expert dp4a (exact order) | 0.28–0.38 ms |
| packed strip (exact order) | 0.27–0.33 ms |
| MMVQ-style SoA, 4–8 experts per launch (reordered sums, failed the gate below) | 0.128–0.140 ms |

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

It is not adopted, and 18.19 tok/s doesn't count as a result. One suspect is that it rounds in a different
place and pairs accumulators across lanes. A control between exact placements has yet to show how much
of this difference is also there between the exact lanes themselves. Passing the control wouldn't waive the gate.
