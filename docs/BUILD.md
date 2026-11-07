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
# Frozen a16r placement and execution modes (reference model/hardware only):
./tools/bench.sh --pack /absolute/models/pack --preset a16 --out bench-a16
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

The default `--preset baseline` preserves the original benchmark: uniform
profile (unless supplied), native default placement, packed-strip B65 experts,
baseline HIP schedule, serial HIP experts, no early issue or Engram read-ahead.
It does not reproduce the best measured result.

`--preset a16` selects the execution settings and measured placement from the
three-answer **a16r** run: `mmvq-soa-v2` on B65, `layer-batch-v1` HIP experts,
CPU layer batching, `queued-v1`, `early-v1`, AVX2/FMA SIMD combine, and
`read-ahead-v1` Engram with QD8 and a 128 MiB packed row cache. HIP graphs,
fusion, GEMV alternatives, CPU affinity/prefetch/spin and diagnostic host
timers remain off. It uses the same default memory/thread budgets listed
below. Explicit budget/thread overrides describe a different setup.

The preset ships the frozen 40×384 selection counts and placement under
`presets/a16/` as UTF-8 text. It reconstructs the uint32 little-endian profile
in the output directory and checks both original hashes:

| Input | SHA256 |
|---|---|
| Reconstructed profile | `4fd7692bfcec8d54938c5beb6fd46ff4d7f90932cb6ef955343804426b6e73d0` |
| Placement | `11c00818ad1ab61500f844e3dfec669e5e2ad123def4553aeac400e31b7a19ca` |

Use the exact GGUF files identified in `presets/a16/metadata.json`. The wrapper
checks dimensions, text hashes and the canonical prompt IDs; it does **not**
hash all model weights on each launch. Verify the model checksums separately.
A different model with the same dimensions is not the reference model.
The a16 preset rejects custom `--profile`, `--placement` or another B65 kernel;
use baseline for those inputs. Non-reference hardware needs new measured
placement and memory qualification, not these frozen counts.

For a16, inherited `DS4_*` and `AIOPT_ENGRAM_*` switches are replaced by the
recorded settings, including `DS4_SYNC=1`, `DS4_ENGRAM_CACHE_MB=0`, two Engram
threads and full store preload. The library search order is package/runtime
then package. Logical selectors match the original launch:
`HIP_VISIBLE_DEVICES=0`, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`.
Physical GPU selection (for example `ROCR_VISIBLE_DEVICES`) is site-specific
and remains inherited; no reference device UUID is shipped. `command.json`
records the selected preset, placement hash and effective execution environment.

The CPU entrypoint test compares the generated configuration, actual native
argv/environment and request bytes against an independently extracted a16r
freeze fixture in `tests/fixtures/a16r-runtime.json`, with paths relocated.
It checks real profile/placement hashes and deliberately contaminated ambient
settings. The native process alone is mocked. This closes the launch-input gap;
the standalone toolchain build, tokenizer, native bank reconstruction, GPU
identity, memory safety, quality scoring and speed still need an end-to-end
run on the reference hardware. The separate 295-second passing answer used
an 11,900-token cap and one measured draw; this command reproduces the
8,192-cap, three-answer speed-run protocol, not that quality experiment.

Defaults reserve 28.5 GB on Intel, 30.45 GB on AMD, 3 GiB Intel runtime overhead,
~1.53 GB AMD fixed overhead, and a 110 GB RAM store with an 8 GiB available-memory
floor. Use `--help` for explicit byte budgets and CPU/store thread counts.
The reference code requires at least a 20 GB HIP expert bank; smaller cards are
not supported by this path. The Intel driver can spill to host memory rather
than rejecting allocation: measure your runtime overhead and keep headroom.
The full preload refuses when the non-GPU experts do not fit the store.

With baseline and no `--profile`, layer/expert counts are uniform and placement differs from
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

## ROCm 7.2.1 without installing it

Fedora's `rocm-hip-7.1.1-3.fc44` retained memory per generated token on this workload. We ran AMD's 7.2.1 userspace from a private folder: no `dnf`, no scriptlets, no driver or kernel change, and the same compiled HIP binaries. It worked on our kernel and card; yours may differ, so check your output against a run you trust.

From `https://repo.radeon.com/rocm/rhel9/7.2.1/main/` take four RPMs: `hip-runtime-amd-7.2.53211.70201-81.el8.x86_64.rpm`, `hsa-rocr-1.18.0.70201-81.el8.x86_64.rpm`, `comgr-3.0.0.70201-81.el8.x86_64.rpm` and `rocprofiler-register-0.6.0.70201-81.el8.x86_64.rpm`. Unpack them without installing, keep `libamdhip64`, `libhsa-runtime64`, `libamd_comgr`, `librocprofiler-register` and the two `libhiprtc` libraries with their SONAME links in one folder, and put that folder first:

```bash
LD_LIBRARY_PATH=/absolute/rocm-7.2.1:/absolute/package/runtime:/absolute/package ...
```

Check `/proc/<pid>/maps` to confirm the engine really loaded those files. Plain `hipMemcpy` still grew about 1 KB per copy on 7.2.1 in a small probe, but the decode path doesn't use it.

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
