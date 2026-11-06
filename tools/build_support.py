"""Pinned source checks and standalone package composition (stdlib only)."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify(root, src):
    contract = json.loads((root/'engine/strata/source-contract.json').read_text())
    pin = (root/'engine/strata/upstream.rev').read_text().strip()
    actual = subprocess.check_output(['git', '-C', str(src), 'rev-parse', 'HEAD'], text=True).strip()
    ggml = subprocess.check_output(['git', '-C', str(src.parent/'llama.cpp'), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != pin or contract['upstream_pin'] != pin or ggml != contract['ggml_pin']:
        raise ValueError('source revision differs from pinned contract')
    for name, digest in contract['required_source_sha256'].items():
        if sha(src/name) != digest:
            raise ValueError('source differs from contract: '+name)
    for name, digest in contract['patch_sha256'].items():
        if sha(root/'engine/strata/patches'/name) != digest:
            raise ValueError('patch differs from contract: '+name)


def runtime_config(compiler, out):
    dirs = [compiler, '/usr/lib64', '/usr/lib/x86_64-linux-gnu', '/usr/lib']
    names = ['libze_loader.so*', 'libze_intel_gpu.so*', 'libigc.so*', 'libigdfcl.so*', 'libigdgmm.so*']
    roots = [str(Path(compiler)/'*.so*')]+[str(Path(d)/n) for d in dirs[1:] for n in names]
    Path(out).write_text(json.dumps({'library_roots':roots, 'exclude_prefixes':[
        'libstrata-stage', 'ld-linux', 'libc.so', 'libm.so', 'libpthread.so', 'libdl.so', 'librt.so', 'libresolv.so']}, indent=2)+'\n')


def compose(root, build):
    out = build/'package'
    out.mkdir()
    names = {'hip':['libstrata-stage-hip-model.so', 'libstrata-row2-cpu.so', 'strata-hip-dense-check'],
             'sycl':['libstrata-stage-sycl-model.so', 'strata-expert-bitcheck', 'strata-strip-bitcheck', 'strata-mmvq-accuracy'],
             'host':['strata-warm-run', 'strata-split-run']}
    for backend, artifacts in names.items():
        for name in artifacts:
            candidates = list((build/backend).rglob(name))
            if len(candidates) != 1:
                raise ValueError('missing/ambiguous build artifact: '+name)
            shutil.copy2(candidates[0], out/name)
    shutil.copytree(build/'sycl/runtime', out/'runtime')
    if sha(out/'runtime/libstrata-row2-cpu.so') != sha(out/'libstrata-row2-cpu.so'):
        raise ValueError('runtime must use the exact shared CPU lane')
    for backend in names:
        proof = out/'build-proof'/backend
        proof.mkdir(parents=True)
        for name in ('CMakeCache.txt', 'compile_commands.json'):
            shutil.copy2(build/backend/name, proof/name)
        for kind in ('stdout', 'stderr'):
            shutil.copy2(build/(backend+'.'+kind), proof/('build.'+kind))
    sources = {str(p.relative_to(root)):sha(p) for folder in ('engine', 'tools', 'prompts')
               for p in sorted((root/folder).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    artifacts = {str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}
    (out/'manifest.json').write_text(json.dumps({'schema':1, 'upstream_pin':(root/'engine/strata/upstream.rev').read_text().strip(),
        'source_sha256':sources, 'artifact_sha256':artifacts, 'native_test_scope':'build and CPU pool control; GPU correctness and timing require native runs'}, indent=2)+'\n')


if __name__ == '__main__':
    action, *args = sys.argv[1:]
    if action == 'runtime-config': runtime_config(*args)
    elif action == 'verify': verify(*map(Path, args))
    elif action == 'compose': compose(*map(Path, args))
    else: raise SystemExit('unknown action')
