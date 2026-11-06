# Build and run

This is research code tested on one Linux machine: Ryzen 9950X3D, 128 GB RAM,
Radeon AI PRO R9700 32 GB and Intel Arc Pro B65 32 GB. The standalone scripts
have CPU fixture tests; the complete public build/pack/benchmark workflow has
not yet been run on GPUs. There is no portability or speed guarantee for other
hardware. This entrypoint targets DeepSeek V4.1 Flash text generation only.

## Prerequisites

Install your own compatible ROCm/HIP and oneAPI DPC++/Level Zero toolchains,
Intel compute runtime, CMake >=3.24, Ninja, Git, binutils (`readelf`, `ldd`),
Python >=3.11 and a C++20 host compiler. The reference used oneAPI 2026.1.1.
Build and run on Linux; the CPU cooperative tests do not qualify GPU code.
The Python tokenizer also needs `regex`; tests need `pytest`:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install regex pytest
export CROSSLANE_PYTHON="$PWD/.venv/bin/python"
```

Both GPUs need to be visible to the same process. HIP and SYCL use separate
translation units and libraries, with a single shared CPU expert library.
Nothing starts a server or opens a network listener.

## Build all three toolchains

From this checkout:

```sh
export CROSSLANE_SYCL_ENV=/opt/intel/oneapi/setvars.sh # if icpx is not already on PATH
./tools/build.sh
```

This fetches Strata at `ffa80d73133ea5c548aefe57889a16e8085a938f` and llama.cpp
at `3cf03257f219afbe7334045ff7c6a06ac68c627d`, applies the retained HIP alignment
patch, verifies the source contract and builds HIP, SYCL and host in order with
`AIOPT_FORK_WARM=1` / `STRATA_FORK_WARM=ON`. Every CMake build uses `-j2` by
default. It composes `build/package` with the shared CPU library, both backend
DSOs, host runners, native expert checks, an inspected Intel runtime closure,
compile commands, build logs and SHA256 inventories. Sources and weights are
never fetched from this fork's private development environment.

Configuration (environment variables):

| Variable | Default / purpose |
|---|---|
| `CROSSLANE_BUILD_DIR` | absolute `$PWD/build`; must be fresh |
| `CROSSLANE_JOBS` | `2` concurrent compiler jobs |
| `CROSSLANE_HIP_ARCH` | `gfx1201` (reference card; change for yours) |
| `CROSSLANE_HIP_CXX`, `CROSSLANE_HOST_CXX` | `c++` |
| `CROSSLANE_HIP_COMPILER`, `CROSSLANE_HIP_FLAGS` | optional CMake HIP compiler and flags |
| `CROSSLANE_SYCL_CXX` | `icpx` |
| `CROSSLANE_SYCL_ENV` | optional setup script sourced in the SYCL phase |
| `CROSSLANE_HIP_RUNNER`, `CROSSLANE_SYCL_RUNNER`, `CROSSLANE_HOST_RUNNER` | optional command prefixes, e.g. `distrobox enter my-rocm --`; empty runs directly |
| `CROSSLANE_RUNTIME_CONFIG` | optional JSON with `library_roots` globs and `exclude_prefixes` for runtime collection |
| `CROSSLANE_STRATA_URL`, `CROSSLANE_GGML_URL` | optional local mirrors; pinned revisions still checked |

Runner prefixes are whitespace-separated argv, without shell evaluation. A
runner must expose the same absolute source/build paths and forward the
`CROSSLANE_*` environment. No container name, image or distrobox is required.
The runtime collector defaults to installed compiler libraries and common
Linux Intel runtime locations. A custom runtime installation may need the
JSON override. Failed builds retain logs; choose a fresh build directory for
another complete attempt. No GPU checks execute during compilation.

## Pack the public model

Obtain the public **DeepSeek V4.1 Flash GSQ-RCO IQ2_XXS** GGUF shard set you want
to test, including its Engram tables. Keep the model's own licence and source
checksums. Model downloads are separate from this script. Do not mix arbitrary
quants and assume the published speed applies. The Intel warm lane supports
IQ2_XXS, IQ2_XS and IQ1_M expert matrices; dense formats follow the retained port.

```sh
.venv/bin/python tools/pack.py --model /absolute/models/gguf-shards --out /absolute/models/pack
# If that quant omits the Engram tables, add: --engram /absolute/models/engram-shards
# If sparse metadata is missing, add: --config-json /absolute/models/config.json
```

This calls the pinned upstream `tools/dsv41/pack_gguf.py`, which writes
`ds4.txt`, `tensors.txt`, `engram.bin` and `tokenizer/`. Weight bytes are read
in place from the public shards. Keep the shards at their recorded locations.
`--upstream` selects a custom build directory's pinned Strata checkout.

## Benchmark

```sh
./tools/bench.sh --pack /absolute/models/pack
# Alternative expert kernel (different reduction order, opt-in):
./tools/bench.sh --pack /absolute/models/pack --sycl-experts mmvq-soa-v1 --out bench-out-mmvq
```

One persistent native engine runs the canonical coding prompt from
`prompts/code.json`, one warm-up and three measured answers, seeds 1000–1003,
temperature 1, top-k 64, top-p 1, min-p 0, 8192 output-token cap and 12288 context.
The tokenizer must reproduce all 325 recorded prompt IDs using the pack's own
pre-tokenizer and special-token types; a mismatch refuses the benchmark.
The command prints median/min/max emitted tokens per native decode second,
excluding the warm-up. `bench-out/` must be new and contains raw stdout/stderr,
config, argv/environment, metadata/package/profile hashes, chosen plan,
responses, generated code text and `summary.json`. Completed decode steps per
second are reported separately. A length-capped answer counts for speed; it
can still be unusable. This entrypoint does not score generated code.

Defaults reserve 28.5 GB on Intel, 30.45 GB on AMD, 3 GiB Intel runtime overhead,
~1.53 GB AMD fixed overhead, and a 110 GB RAM store with an 8 GiB available-memory
floor. Use `--help` for explicit byte budgets and CPU/store thread counts.
The reference code requires at least a 20 GB HIP expert bank; smaller cards are
not supported by this path. The Intel driver can spill to host memory rather
than rejecting allocation: measure your runtime overhead and keep headroom.
The full preload refuses when the non-GPU experts do not fit the store.

Without `--profile`, layer/expert counts are uniform and placement differs from
the published runs. Supply an actual uint32 little-endian count profile to rank
experts, and optionally `--placement` for an already computed compatible warm
placement. New measured selection profiles are retained as `decode-profile*`.
The pure offline optimiser remains in `engine/strata/warm/rebalance.py`:
`optimise(plan, profile, costs)` returns the assignment and a report, and
`assignment_text(assignment, plan)` serializes a compatible placement. Its
inputs are the host runner's complete `--plan-only` plan, an actual measured
decode profile, and measured service costs (`make_costs.py` documents the
native counter/timing inputs). The benchmark's small `chosen-plan.json` is a
calibration report, not that complete planner input. Reconstruct any proposed
placement with the native planner before use. Profiles and costs must agree;
CPU simulations cannot supply native timing. There is no speed claim for
uniform placement. HIP graphs stay off: the native
reference measurement found them slower. MMVQ is available for experiments;
CPU numeric checks alone do not certify end-to-end output quality.

Optional native checks (run with the same runtime search path as the benchmark):

```sh
export LD_LIBRARY_PATH="$PWD/build/package:$PWD/build/package/runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
./build/package/strata-expert-bitcheck --pack /absolute/models/pack
./build/package/strata-strip-bitcheck --pack /absolute/models/pack
./build/package/strata-mmvq-accuracy --pack /absolute/models/pack
```

## Host policy

The standalone tree has no host profile. `CROSSLANE_B65_GUARD=/path/to/wrapper`
adds an optional wrapper command that accepts the benchmark argv. Empty means
no wrapper. `CROSSLANE_LOCK` changes the default
`~/.cache/crosslane/gpu.lock`. The wrapper runs outside the GPU lock, including
its entry/exit actions; the benchmark inherits fd8 for the whole run.
`CROSSLANE_PYTHON` selects the Python executable.

An optional `CROSSLANE_HOST_PROFILE=/absolute/site.json` can define `lock`,
`b65_guard`, `b65_lock`, `require_b65_guard`, `production_checks`, `pause_file`
and `idle_inputs`. Enabled production checks require the configured pause file.
A profile requiring a B65 guard rejects an empty wrapper override. The reference
site's strict profile is maintained privately and is excluded from this export.
These standalone scripts do not replace a site's full telemetry or allocation
monitoring. Private reference windows retain their existing memory/device gates.

## CPU tests

```sh
CXX=c++ .venv/bin/python tools/test.py -j2
```

Two test files run at a time. Tests execute actual packed-strip/MMVQ kernels in
cooperative CPU shims, exercise the attention reduction tree, reject deliberate
kernel mutations, and check standalone build/pack/benchmark boundaries with
mock toolchains. No model, GPU, external git history or network is needed.
Use `-j1` on smaller hosts. Device correctness and performance still require
native checks on your own hardware.
