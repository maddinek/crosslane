#!/usr/bin/env bash
set -euo pipefail
# --preset a16 retains the earlier setup; --preset a17 uses the VRAM placement
# and the verified ROCm 7.2.1 folder staged by tools/stage_rocm.py.
root=$(cd "$(dirname "$0")/.." && pwd)
if [[ ${1:-} == --help ]]; then
  exec "${CROSSLANE_PYTHON:-python3}" "$root/tools/bench.py" --help
fi
exec "${CROSSLANE_PYTHON:-python3}" "$root/engine/strata/parity-g01/host_profile.py" \
  "${CROSSLANE_PYTHON:-python3}" "$root/tools/bench.py" "$@"
