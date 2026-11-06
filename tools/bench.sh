#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
if [[ ${1:-} == --help ]]; then
  exec "${CROSSLANE_PYTHON:-python3}" "$root/tools/bench.py" --help
fi
exec "${CROSSLANE_PYTHON:-python3}" "$root/engine/strata/parity-g01/host_profile.py" \
  "${CROSSLANE_PYTHON:-python3}" "$root/tools/bench.py" "$@"
