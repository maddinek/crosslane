# SYCL model stages

SYCL implementations of DeepSeek V4.1 dense primitives, routing, Engram and
quantized experts. `stage/wiring` composes these with the HIP backend in one
process. Warm serving adds concurrent AMD, Intel and CPU expert lanes.

Use the standalone build instructions in `docs/BUILD.md`. CPU cooperative
fixtures test arithmetic and ownership; native checks qualify device code.
