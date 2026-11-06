#!/usr/bin/env python3
"""One persistent native engine: canonical coding prompt, warm-up, three answers."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import statistics
import struct
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'engine/strata/warm'), str(ROOT/'engine/strata/parity-g01')]
from warm_config import config
import host_profile


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def tokenizer(upstream, pack):
    spec = importlib.util.spec_from_file_location('crosslane_tokenizer', upstream/'tools/strata_tokenizer.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    folder = pack/'tokenizer'
    cfg = json.loads((folder/'tokenizer.json').read_text())
    vocab = json.loads((folder/'vocab.json').read_text())
    # Preserve the model pre-tokenizer and special-token types exactly.
    tokens = vocab if isinstance(vocab, list) else [t for t, i in sorted(vocab.items(), key=lambda x:x[1])]
    merges = (folder/'merges.txt').read_text().splitlines()
    return mod.Tokenizer(tokens, [m for m in merges if m and not m.startswith('#')],
                         token_types=json.loads((folder/'token_type.json').read_text()), pre=cfg['pre'], special_ids=cfg.get('special_ids'))


def parse_stream(path):
    rows = []
    active = None
    counter = 0
    ready = False
    for line in Path(path).read_text().splitlines():
        s = line.split()
        if s == ['READY'] and not ready and not rows:
            ready = True
        elif len(s) == 4 and s[0] == 'BEGIN' and ready and active is None and len(rows) < 4:
            if list(map(int, s[1:])) != [len(rows), 1000+len(rows), counter]:
                raise ValueError('BEGIN order/seed/counter differs')
            active = []
        elif len(s) == 2 and s[0] == 'TOKEN' and active is not None:
            active.append(int(s[1]))
        elif len(s) == 7 and s[0] == 'END' and active is not None:
            n, prefill, decode = map(int, s[1:4])
            begin, end = map(int, s[5:7])
            if n != len(active) or not 1 <= n <= 8192 or prefill <= 0 or decode <= 0 or s[4] not in ('eos', 'length') or begin != counter or end-begin != n:
                raise ValueError('invalid native END record')
            rate = n/(decode/1e9)
            if not math.isfinite(rate): raise ValueError('nonfinite rate')
            rows.append(dict(index=len(rows), phase='warmup' if not rows else 'measured',
                tokens=active, finish_reason=s[4], tok_s=rate, prefill_s=prefill/1e9, decode_s=decode/1e9,
                decode_steps_tok_s=(n-1)/(decode/1e9)))
            active = None
            counter = end
        else:
            raise ValueError('unexpected native stream record: '+line[:120])
    if not ready or len(rows) != 4 or active is not None:
        raise ValueError('requires exactly one warm-up and three complete answers')
    rates = [r['tok_s'] for r in rows[1:]]
    return rows, dict(median=statistics.median(rates), min=min(rates), max=max(rates), n=3)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pack', type=Path, required=True)
    p.add_argument('--package', type=Path, default=ROOT/'build/package')
    p.add_argument('--upstream', type=Path, default=ROOT/'build/upstream/strata')
    p.add_argument('--out', type=Path, default=ROOT/'bench-out')
    p.add_argument('--profile', type=Path, help='uint32 LE layer/expert counts; default uniform, not the published placement')
    p.add_argument('--placement', type=Path, help='optional precomputed compatible warm placement file')
    p.add_argument('--sycl-experts', choices=['serial-v1', 'grouped-v1', 'packed-strip-v1', 'mmvq-soa-v1'], default='packed-strip-v1')
    for key, default in [('sycl-cap',28500000000), ('hip-cap',30450647040), ('sycl-fixed',3*2**30),
                         ('hip-fixed',1529365504), ('store-bytes',110000000000), ('cpu-threads',16), ('store-threads',32)]:
        p.add_argument('--'+key, type=int, default=default)
    a = p.parse_args()
    for name in ('pack', 'package', 'upstream', 'out'):
        setattr(a, name, getattr(a, name).resolve())
    if any(getattr(a, k) <= 0 for k in ('sycl_cap','hip_cap','sycl_fixed','hip_fixed','store_bytes','cpu_threads','store_threads')):
        p.error('memory and thread budgets must be positive')
    host_profile.require_pause()
    prompt = json.loads((ROOT/'prompts/code.json').read_text())
    tok = tokenizer(a.upstream, a.pack)
    ids = tok.encode(prompt['rendered'], parse_special=True)
    if ids != prompt['ids'] or len(ids) != 325:
        raise ValueError('pack tokenizer does not match the canonical prompt; refusing benchmark')
    a.out.mkdir(parents=True)
    if a.profile:
        profile = a.profile.resolve()
        (a.out/'profile.bin').write_bytes(profile.read_bytes())
    else:
        meta = dict(line.split(' ',1) for line in (a.pack/'ds4.txt').read_text().splitlines() if line)
        (a.out/'profile.bin').write_bytes(struct.pack('<I',1)* (int(meta['num_hidden_layers'])*int(meta['n_routed_experts'])))
    profile = a.out/'profile.bin'
    cfg = config(profile, True, dense_owner='hip-v1', sycl_attention='subgroup32-v1', sycl_experts=a.sycl_experts)
    cfg = cfg.replace('cpu_threads=16\n',f'cpu_threads={a.cpu_threads}\n').replace('warm_sycl_fixed='+str(3*2**30), 'warm_sycl_fixed='+str(a.sycl_fixed)).replace('warm_hip_fixed=1529365504',f'warm_hip_fixed={a.hip_fixed}')
    if a.placement:
        (a.out/'placement.txt').write_bytes(a.placement.read_bytes())
        cfg += 'warm_placement='+str(a.out/'placement.txt')+'\n'
    (a.out/'warm.conf').write_text(cfg)
    (a.out/'prompt.json').write_text(json.dumps(prompt,indent=2,ensure_ascii=False)+'\n')
    (a.out/'requests.txt').write_text(''.join(f'{1000+i} 8192 325 '+' '.join(map(str,ids))+'\n' for i in range(4)))
    env = {**os.environ, 'DS4_FORK_WARM':'1', 'DS4_STORE_BYTES':str(a.store_bytes),
           'DS4_STORE_THREADS':str(a.store_threads), 'DS4_STORE_FLOOR_BYTES':str(8*2**30),
           'DS4_STORE_PRELOAD':'1', 'DS4_STORE_DIAGNOSTIC':'1', 'DS4_WARM_PROFILE_OUTPUT':str(a.out/'decode-profile'),
           'LD_LIBRARY_PATH':str(a.package)+':'+str(a.package/'runtime')+':'+os.environ.get('LD_LIBRARY_PATH','')}
    command = [str(a.package/'strata-warm-run'), '--pack',str(a.pack),'--ctx','12288','--cut','0','--eligible-cuts','0',
        '--stage-config',str(a.out/'warm.conf'),'--profile',str(profile),'--requests',str(a.out/'requests.txt'),
        '--chosen-plan',str(a.out/'chosen-plan.json'),'--sycl',str(a.package/'libstrata-stage-sycl-model.so'),
        '--hip',str(a.package/'libstrata-stage-hip-model.so'),'--cpu-library',str(a.package/'libstrata-row2-cpu.so')]
    for key in ('sycl_cap','hip_cap','sycl_fixed','hip_fixed'):command += ['--'+key.replace('_','-'),str(getattr(a,key))]
    (a.out/'command.json').write_text(json.dumps({'argv':command, 'environment':{k:v for k,v in env.items() if k.startswith('DS4_') or k=='LD_LIBRARY_PATH'},
        'profile_kind':'provided' if a.profile else 'uniform', 'package_manifest_sha256':sha(a.package/'manifest.json'),
        'pack_metadata_sha256':{n:sha(a.pack/n) for n in ('ds4.txt','tensors.txt','engram.bin')},
        'profile_sha256':sha(profile)}, indent=2)+'\n')
    with (a.out/'engine.stdout').open('x') as stdout, (a.out/'engine.stderr').open('x') as stderr:
        proc = subprocess.Popen(command, env=env, stdout=stdout, stderr=stderr)
        def stop(signum, frame):
            if proc.poll() is None:proc.terminate()
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
        rc = proc.wait()
    (a.out/'exit.json').write_text(json.dumps({'returncode':rc})+'\n')
    if rc: raise SystemExit(f'native engine failed ({rc}); raw logs in {a.out}')
    rows, stats = parse_stream(a.out/'engine.stdout')
    for row in rows:
        row['text'] = tok.decode(row['tokens'])
        (a.out/f'answer-{row["index"]:02d}.txt').write_text(row['text'])
    (a.out/'summary.json').write_text(json.dumps({'requests':rows,'tok_s':stats,
        'scope':'emitted tokens / native decode seconds; warm-up excluded; speed does not score code correctness'},indent=2)+'\n')
    print(f'tok/s median {stats["median"]:.3f}, min {stats["min"]:.3f}, max {stats["max"]:.3f} (n=3, warm-up excluded)')
    print('Raw logs and answers:',a.out)


if __name__ == '__main__': main()
