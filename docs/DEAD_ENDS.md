# Dead ends worth knowing

Things that looked promising and were measured not to help, or broke, on the reference machine.

| What | What happened | Lesson |
|---|---|---|
| Filling both GPUs without rebalancing | Putting all supported experts on the B65 (fwarm-a6) was flat (12.06 vs 12.22 tok/s): the CPU became the slowest lane. | Placement must use measured per-lane costs; a rebalance gave +7.5%. |
| A second rebalance round | +1.4% only (15.63 → 15.85). | Placement is converged; kernels are the lever. |
| Balancing the dense split between the two GPUs | Single-stream decode runs layers one after another, so splitting dense layers across GPUs only adds a hand-off. The B65 dense path (exact-order port) took 11.5 ms for one layer vs ~0.5 ms on the R9700. | Put dense on the faster card; split only to free VRAM, and only if per-layer times are close. |
| HIP static graph replay for batch-1 GEMVs (ROCm) | Bit-identical, but 3–27% slower than direct launches (median +14% wall). | Fuse kernels instead of graphing them. |
| Strata's built-in draft model (DSpark), stock | 0.88–0.97× despite 79–85% acceptance on code. | A 2-token verify nearly doubles the experts touched per layer; speculation needs cheap expert lanes first. |
| Treating summation order as the main B65 cost | The first scalar port in the reference's order took 1.89 ms per expert. Packed kernels in the same exact order reach ~0.3 ms. | Packing and batching gave most of the gain without changing a bit. Cross-vendor bit identity isn't a goal in itself, but a reordering has to pass the quality gate. |
| MMVQ-style B65 kernel with reordered sums | 18.19 tok/s after a rebalance (exact kernels at the time: 15.85), but teacher-forced top-1 was 95.35% against a 99% gate, with max KL 1.486. Per-expert NRMSE stayed ≤ 4.3e-4 on real activations (5.3e-6 on synthetic inputs). | Not adopted. Small per-expert error doesn't predict the model-level effect, and synthetic inputs flatter it. Test on the real prompt early. |
| CPU shim for SYCL code | A shim that accepted `wait_and_throw()` on a `const sycl::event` hid a real oneAPI compile error. | Run the real compile-only preflight early. |
| Comparing device `exp` with host libm bitwise | 2-ulp differences in SwiGLU flagged a correct kernel. | Compare transcendental stages against the same device's scalar path. |
| Python 3.11 vs 3.12+ `sum()` of floats | Timing replays differed in the last bit across machines. | Use `math.fsum` for anything that must reproduce. |
