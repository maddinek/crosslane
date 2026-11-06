#!/usr/bin/env bash
# One toolchain per configure. Runner prefixes are optional whitespace-separated argv.
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
python=${CROSSLANE_PYTHON:-python3}
build=${CROSSLANE_BUILD_DIR:-$root/build}
jobs=${CROSSLANE_JOBS:-2}
[[ $jobs =~ ^[1-9][0-9]*$ ]] || { echo 'CROSSLANE_JOBS must be positive' >&2; exit 2; }
export AIOPT_FORK_WARM=1
wiring=$root/engine/strata/sycl/stage/wiring
source_root=$build/upstream
src=$source_root/strata
if [[ ${1:-} == --help ]]; then
  echo 'tools/build.sh: fetch pinned sources, build HIP/SYCL/host, compose build/package.'
  echo 'See docs/BUILD.md for prerequisites, paths, optional runners and toolchain overrides.'
  exit
fi
if [[ ${1:-} == _phase ]]; then
  backend=${2:?backend}
  mkdir -p "$build/$backend"
  if [[ $backend == hip ]]; then
    hip_options=()
    if [[ -n ${CROSSLANE_HIP_COMPILER:-} ]]; then hip_options+=("-DCMAKE_HIP_COMPILER=$CROSSLANE_HIP_COMPILER"); fi
    if [[ -n ${CROSSLANE_HIP_FLAGS:-} ]]; then hip_options+=("-DCMAKE_HIP_FLAGS=$CROSSLANE_HIP_FLAGS"); fi
    cmake -S "$src" -B "$build/hip" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER="${CROSSLANE_HIP_CXX:-c++}" \
      -DSTRATA_ENABLE_HIP=ON -DCMAKE_HIP_ARCHITECTURES="${CROSSLANE_HIP_ARCH:-gfx1201}" \
      -DSTRATA_BUILD_TESTS=OFF -DSTRATA_BUILD_DSV41=ON -DSTRATA_NATIVE_EXPERTS=ON \
      -DSTRATA_BUILD_ID="$(cat "$root/engine/strata/upstream.rev")" \
      -DSTRATA_GGML_DIR="$source_root/llama.cpp" \
      -DCMAKE_PROJECT_TOP_LEVEL_INCLUDES="$wiring/hip_top_level.cmake" \
      -DSTRATA_WIRING_CPU_LANE=shared -DSTRATA_FORK_WARM=ON \
      "${hip_options[@]}"
    cmake --build "$build/hip" -j "$jobs" --target strata-stage-hip-model strata-row2-cpu strata-row2-cpu-fixture strata-row2-pool-fixture strata-hip-dense-check
    "$build/hip/strata-row2-pool-fixture"
  elif [[ $backend == sycl ]]; then
    if [[ -n ${CROSSLANE_SYCL_ENV:-} ]]; then
      set +eu
      source "$CROSSLANE_SYCL_ENV"
      rc=$?
      set -eu
      (( rc == 0 )) || exit "$rc"
    fi
    compiler=${CROSSLANE_SYCL_CXX:-icpx}
    compiler_runtime=$(dirname "$("$compiler" -print-file-name=libsvml.so)")
    export LD_LIBRARY_PATH="$build/hip:$compiler_runtime${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    cmake -S "$wiring" -B "$build/sycl" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER="$compiler" \
      -DSTRATA_WIRING_BACKEND=sycl -DSTRATA_WIRING_ROW2=moe -DSTRATA_WIRING_CPU_LANE=shared \
      -DSTRATA_ROW2_CPU_LIBRARY="$build/hip/libstrata-row2-cpu.so" -DSTRATA_FORK_WARM=ON
    cmake --build "$build/sycl" -j "$jobs"
    if [[ -n ${CROSSLANE_RUNTIME_CONFIG:-} ]]; then
      runtime_config=$CROSSLANE_RUNTIME_CONFIG
    else
      runtime_config=$build/sycl/runtime-config.json
      "$python" "$root/tools/build_support.py" runtime-config "$compiler_runtime" "$runtime_config"
    fi
    "$python" "$root/tools/collect_runtime.py" --config "$runtime_config" \
      --backend "$build/sycl/libstrata-stage-sycl-model.so" --output "$build/sycl/runtime"
  elif [[ $backend == host ]]; then
    cmake -S "$wiring" -B "$build/host" -G Ninja -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DCMAKE_CXX_COMPILER="${CROSSLANE_HOST_CXX:-c++}" \
      -DSTRATA_WIRING_BACKEND=host -DSTRATA_FORK_WARM=ON
    cmake --build "$build/host" -j "$jobs"
  else exit 2; fi
  exit
fi
[[ $# == 0 ]] || { echo 'unexpected arguments; see --help' >&2; exit 2; }
[[ ! -e $build ]] || { echo "fresh build directory required: $build" >&2; exit 1; }
mkdir -p "$source_root"
git clone --no-checkout "${CROSSLANE_STRATA_URL:-https://github.com/Niko1221/Strata.git}" "$src"
git -C "$src" checkout --detach "$(cat "$root/engine/strata/upstream.rev")"
git clone --no-checkout "${CROSSLANE_GGML_URL:-https://github.com/ggml-org/llama.cpp.git}" "$source_root/llama.cpp"
ggml_pin=$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["ggml_pin"])' "$root/engine/strata/source-contract.json")
git -C "$source_root/llama.cpp" checkout --detach "$ggml_pin"
git -C "$src" apply --check "$root/engine/strata/patches/0001-hip-shared-align-attribute.patch"
git -C "$src" apply "$root/engine/strata/patches/0001-hip-shared-align-attribute.patch"
"$python" "$root/tools/build_support.py" verify "$root" "$src"
for backend in hip sycl host; do
  runner_var=CROSSLANE_${backend^^}_RUNNER
  read -r -a runner <<< "${!runner_var:-}"
  "${runner[@]}" bash "$root/tools/build.sh" _phase "$backend" \
    > "$build/$backend.stdout" 2> "$build/$backend.stderr"
done
"$python" "$root/tools/build_support.py" compose "$root" "$build"
echo "Package: $build/package"
