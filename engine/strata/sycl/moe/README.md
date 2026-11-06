# Quantized expert primitives

`row2.cpp` implements SYCL expert projections. `cpu_lane.cpp` binds the pinned
upstream CPU rows. `core.hpp`, `format.hpp` and packed-dot helpers preserve
quantized payloads. CPU cooperative fixtures exercise actual projection code.

The default exact paths retain their reduction order. The opt-in warm MMVQ SoA
path uses a different floating-point reduction and bounded numeric error;
cross-vendor bit identity is not promised. See `docs/BUILD.md` for native checks.

Upstream Maya and ggml notices and MIT licences are in `licenses/`.
