#!/usr/bin/env python3
"""Prepare measured service coefficients from actual warm fork/profile/native raw.

Cumulative layer counters are differenced at adjacent request boundaries.
Warmup and prefill are excluded. GPU busy counters are never treated as service.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from timing_summary import summarise


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(stderr,profile,plan,timing_stdout,run_revision,native_revision):
    if any(not re.fullmatch('[0-9a-f]{40}',r) for r in (run_revision,native_revision)):raise ValueError('full evidence/source revisions required')
    timing=summarise(timing_stdout)
    native={c['signature'].rstrip('/'):c['metrics']['host_ns']['median'] for c in timing['cells'] if c['packed'] and c['cache']=='warm'}
    raw=Path(profile).read_bytes();nl=plan['layers'];ne=len(plan['warm_inventory'])//nl
    if len(raw)!=4*nl*ne:raise ValueError('target-only decode profile extent')
    counts=[int.from_bytes(raw[i:i+4],'little') for i in range(0,len(raw),4)]
    boundaries={};layers={};event=None
    for line in Path(stderr).read_text().splitlines():
        if line.startswith('store_diag_v1 '):
            row=json.loads(line.removeprefix('store_diag_v1 '));event=row['event'];key=(row['request'],event)
            if key in boundaries:raise ValueError('duplicate request boundary')
            boundaries[key]=row
        elif line.startswith('AIOPT_WARM_LAYER_TOTALS '):
            row={k:int(v) for k,v in (word.split('=',1) for word in line.split()[1:])}
            key=(row['request'],event,row['layer'])
            if key in layers:raise ValueError('duplicate layer/boundary')
            layers[key]=row
    tokens=sum(boundaries[i,'decode_end']['interval_delta']['model_fed_tokens'] for i in (2,3,4))
    if tokens<=0 or sum(counts)%(nl*tokens):raise ValueError('invalid fed-token/profile count')
    picks=sum(counts)//(nl*tokens)
    if picks<=0 or any(sum(counts[l*ne:(l+1)*ne])!=picks*tokens for l in range(nl)):raise ValueError('profile per-layer selection count differs')
    costs=[];details=[]
    for l in range(nl):
        delta={name:sum(layers[i,'request_end',l][name]-layers[i,'prefill_end',l][name] for i in (2,3,4)) for name in ('cpu_service_ns','hip_service_ns','sycl_service_ns','hip_calls','sycl_calls')}
        if any(v<0 for v in delta.values()):raise ValueError('negative cumulative-counter delta')
        calls={'hip':delta['hip_calls'],'sycl':delta['sycl_calls'],'cpu':tokens*picks-delta['hip_calls']-delta['sycl_calls']}
        signatures={'/'.join(row[4:]) for row in plan['warm_inventory'] if row[0]==l}
        if len(signatures)!=1:raise ValueError('nonuniform within-layer signature needs per-expert measured costs')
        signature=signatures.pop();layer={}
        for name in ('cpu','hip','sycl'):
            if calls[name]>0:value=delta[name+'_service_ns']/calls[name];source='warm-fork-layer-service-delta/per-selection'
            elif name=='sycl' and signature in native:value=native[signature];source='native-packed-chain-host-with-input-upload-and-output-readback'
            else:raise ValueError(f'no measured {name} cost for layer {l}; supply a measured cost file')
            layer[name]={'per_expert_ns':value,'issue_ns':0,'measurement_source':source}
        costs.append(layer);details.append({'layer':l,'signature':signature,'delta':delta,'calls':calls})
    return {'schema':'strata-warm-costs-v1','scope':'measured-decode-excluding-warmup','decode_tokens':tokens,'profile_sha256':sha(profile),'layers':costs,
            'evidence':{'run_revision':run_revision,'native_source_revision':native_revision,'stderr_sha256':sha(stderr),'native_stdout_sha256':sha(timing_stdout),
                        'baseline_plan_sha256':hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),'layer_deltas':details},
            'assumptions':['Per-selection service coefficients absorb measured issue/loop overhead; issue_ns=0 avoids double counting.',
                           'B65 previously empty signatures use newly measured synchronous micro host service, including H2D/D2H; full-fork contention remains unmeasured.',
                           'Marginal profile objective is a labelled surrogate; native performance must confirm it.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stderr','profile','plan','timing-stdout','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--run-revision',required=True);p.add_argument('--native-source-revision',required=True);a=p.parse_args()
    result=build(a.stderr,a.profile,json.loads(a.plan.read_text()),a.timing_stdout,a.run_revision,a.native_source_revision)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
