#!/usr/bin/env python3
"""Build a metadata-only Strata pack beside public GGUF shards; never copy weights."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--upstream', type=Path, default=ROOT/'build/upstream/strata')
    p.add_argument('--model', type=Path, required=True, help='directory containing public GGUF shards')
    p.add_argument('--out', type=Path, required=True, help='new pack directory')
    p.add_argument('--engram', type=Path, help='separate Engram GGUF directory/file if omitted from main quant')
    p.add_argument('--config-json', type=Path, help='original model config, if sparse geometry is missing')
    a = p.parse_args()
    if a.out.exists():
        p.error('pack output must be new')
    pin = (ROOT/'engine/strata/upstream.rev').read_text().strip()
    actual = subprocess.check_output(['git', '-C', str(a.upstream), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != pin:
        p.error('upstream must match engine/strata/upstream.rev')
    command = [sys.executable, str(a.upstream/'tools/dsv41/pack_gguf.py'),
               '--model', str(a.model.resolve()), '--out', str(a.out.resolve())]
    for name in ('engram', 'config_json'):
        if getattr(a, name) is not None:
            command += ['--'+name.replace('_', '-'), str(getattr(a, name).resolve())]
    subprocess.run(command, check=True)
    print('Pack:', a.out.resolve())


if __name__ == '__main__': main()
