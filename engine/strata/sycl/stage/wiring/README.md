# Model backend wiring

`CMakeLists.txt` builds the host runners or SYCL backend with separate
toolchains. `hip_top_level.cmake` adds the HIP backend and shared CPU lane to the
pinned upstream configure. `tools/build.sh` drives all three builds.

`pack_source.cpp` reads the upstream text pack and file-backed GGUF tensors.
`host_plan.hpp` accounts for model geometry, runtime reserves and resident
experts. The two backends export a C ABI and share the same CPU lane library,
which owns the warm RAM store and concurrent expert workers.

`parity/` contains native ABI, retry, embedding and numeric controls. Warm
expert checks live in `engine/strata/warm`. CPU tests simulate the cooperative
API; they cannot establish device performance or compiler correctness.
