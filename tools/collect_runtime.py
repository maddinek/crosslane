#!/usr/bin/env python3
"""Copy an inspected runtime dependency closure to an isolated directory, with hashes.

Run in the configured SYCL toolchain environment. No package installation
or writes to the source toolchain; dlopen-only roots are explicit in config.
"""
import argparse
import glob
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--config', type=Path, required=True)
p.add_argument('--backend', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
cfg = json.loads(a.config.read_text())
a.output.mkdir(parents=True, exist_ok=False)
roots = [Path(x) for pattern in cfg['library_roots'] for x in glob.glob(pattern)]
if not roots:
    raise SystemExit('no configured runtime libraries found')
pending = [a.backend, *roots]
seen, libraries, skipped = set(), {}, []
while pending:
    path = pending.pop()
    real = path.resolve(strict=True)
    with real.open("rb") as source:
        is_elf = source.read(4) == b"\x7fELF"
    if not is_elf:
        if path == a.backend:
            raise RuntimeError(f"backend is not ELF: {path}")
        skipped.append(str(path))
        continue
    if not path.name.startswith(tuple(cfg['exclude_prefixes'])):
        if path.name in libraries and libraries[path.name] != real:
            raise RuntimeError(f'ambiguous library name: {path.name}')
        libraries[path.name] = real
    if real in seen:
        continue
    seen.add(real)
    result = subprocess.run(['ldd', str(real)], text=True, capture_output=True)
    if result.returncode or 'not found' in result.stdout:
        raise RuntimeError(f'ldd failed for {real}: {result.stdout} {result.stderr}')
    for name in re.findall(r'=> (/\S+) \(', result.stdout):
        dependency = Path(name)
        if not dependency.name.startswith(tuple(cfg['exclude_prefixes'])):
            pending.append(dependency)
manifest = []
for name, real in sorted(libraries.items()):
    target = a.output / name
    shutil.copyfile(real, target)
    with target.open('rb') as source:
        digest = hashlib.file_digest(source, 'sha256').hexdigest()
    manifest.append({'name': name, 'source': str(real), 'bytes': target.stat().st_size, 'sha256': digest})
(a.output / 'skipped-non-elf.json').write_text(json.dumps(skipped, indent=2) + '\n')
(a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(json.dumps({'libraries': len(manifest), 'bytes': sum(x['bytes'] for x in manifest)}))
