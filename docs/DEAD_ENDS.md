# Dead ends worth knowing

Things that looked promising and were measured not to help, or broke, on the reference machine.

| What | What happened | Lesson |
|---|---|---|
| Filling both GPUs without rebalancing | Putting all supported experts on the B65 (fwarm-a6) was flat (12.06 vs 12.22 tok/s): the CPU became the slowest lane. | Placement must use measured per-lane costs; a rebalance gave +7.5%. |
| A second rebalance round | +1.4% only (15.63 → 15.85). | Placement is converged; kernels are the lever. |
| Balancing the dense split between the two GPUs | Single-stream decode runs layers one after another, so splitting dense layers across GPUs only adds a hand-off. The B65 dense path (exact-order port) took 11.5 ms for one layer vs ~0.5 ms on the R9700. | Put dense on the faster card; split only to free VRAM, and only if per-layer times are close. |
| HIP static graph replay for batch-1 GEMVs (ROCm) | Bit-identical, but 3–27% slower than direct launches (median +14% wall). | Fuse kernels instead of graphing them. |
| Strata's built-in draft model (DSpark), stock | 0.88–0.97× despite 79–85% acceptance on code. | A 2-token verify nearly doubles the experts touched per layer; speculation needs cheap expert lanes first. |
| Bit-identical cross-vendor kernels | Keeping the B65 kernels in the CPU reference's summation order cost ~13× per expert vs llama.cpp-style kernels on the same silicon class. | Cross-vendor bit identity isn't needed; gate on numerical error and output quality instead. |
| CPU shim for SYCL code | A shim that accepted `wait_and_throw()` on a `const sycl::event` hid a real oneAPI compile error. | Run the real compile-only preflight early. |
| Comparing device `exp` with host libm bitwise | 2-ulp differences in SwiGLU flagged a correct kernel. | Compare transcendental stages against the same device's scalar path. |
| Python 3.11 vs 3.12+ `sum()` of floats | Timing replays differed in the last bit across machines. | Use `math.fsum` for anything that must reproduce. |
