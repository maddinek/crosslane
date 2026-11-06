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
from warm_config import cost_modes


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(stderr,profile,plan,timing_stdout,run_revision,native_revision,*,native_fallback=True,warm_freeze=None,warm_conf=None):
    if any(not re.fullmatch('[0-9a-f]{40}',r) for r in (run_revision,native_revision)):raise ValueError('full evidence/source revisions required')
    timing=summarise(timing_stdout) if native_fallback else {'cells':[]}
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
    execution={}
    if (warm_freeze is None)!=(warm_conf is None):raise ValueError('both warm freeze and config required')
    if warm_freeze is not None:
        frozen=json.loads(Path(warm_freeze).read_text())
        hashes=[h for p,h in frozen['warm_inputs'].items() if Path(p).name=='warm.conf']
        if hashes!=[sha(warm_conf)] or frozen['plans'][str(plan['cut'])]!=plan:raise ValueError('cost freeze/config/plan binding differs')
        cfg=dict(line.split('=',1) for line in Path(warm_conf).read_text().splitlines() if line and not line.startswith('#'))
        execution=cost_modes(cfg,frozen.get('cpu_layer_batch',False),frozen.get('cpu_affinity',''),
                             **{k:frozen[k] for k in ('cpu_prefetch_rows','cpu_spin_us','cpu_store_thp','engram_row_cache','engram_io','other_options') if k in frozen})
        if native_fallback and execution['warm_sycl_experts']!='packed-strip-v1':raise ValueError('packed native fallback cannot price a different SYCL mode')
    measured=sorted(i for i,event in boundaries if event=='decode_end' and i>1)
    if not measured or measured!=list(range(2,max(measured)+1)):raise ValueError('measured request census is incomplete')
    if warm_freeze is not None and len(measured)!=frozen.get('quality_protocol',{}).get('measured_answers',3):raise ValueError('cost request census differs from freeze')
    tokens=sum(boundaries[i,'decode_end']['interval_delta']['model_fed_tokens'] for i in measured)
    if tokens<=0 or sum(counts)%(nl*tokens):raise ValueError('invalid fed-token/profile count')
    picks=sum(counts)//(nl*tokens)
    if picks<=0 or any(sum(counts[l*ne:(l+1)*ne])!=picks*tokens for l in range(nl)):raise ValueError('profile per-layer selection count differs')
    # Event calls count batches on grouped lanes. Decode profile entries count
    # selections irrespective of batching and are bound to this exact placement.
    ownership={}
    for name,rows in (('hip',plan['hip_resident']['experts']),('sycl',plan['warm_sycl']['members'])):
        for row in rows:
            key=tuple(row[:2])
            if (key in ownership or len(key)!=2 or any(type(v) is not int for v in key) or
                not 0<=key[0]<nl or not 0<=key[1]<ne):raise ValueError('invalid/overlapping cost placement')
            ownership[key]=name
    costs=[];details=[]
    for l in range(nl):
        delta={name:sum(layers[i,'request_end',l][name]-layers[i,'prefill_end',l][name] for i in measured) for name in ('cpu_service_ns','hip_service_ns','sycl_service_ns','hip_calls','sycl_calls')}
        if any(v<0 for v in delta.values()):raise ValueError('negative cumulative-counter delta')
        calls={name:sum(counts[l*ne+e] for e in range(ne) if ownership.get((l,e),'cpu')==name) for name in ('cpu','hip','sycl')}
        if any(delta[name+'_calls']>calls[name] for name in ('hip','sycl')):raise ValueError('GPU event calls exceed selections; profile/plan/raw mismatch')
        signatures={'/'.join(row[4:]) for row in plan['warm_inventory'] if row[0]==l}
        if len(signatures)!=1:raise ValueError('nonuniform within-layer signature needs per-expert measured costs')
        signature=signatures.pop();layer={}
        for name in ('cpu','hip','sycl'):
            if calls[name]>0:value=delta[name+'_service_ns']/calls[name];source='warm-fork-layer-service-delta/per-selection'
            elif name=='sycl' and signature in native:value=native[signature];source='native-packed-chain-host-with-input-upload-and-output-readback'
            else:raise ValueError(f'no measured {name} cost for layer {l}; supply a measured cost file')
            layer[name]={'per_expert_ns':value,'issue_ns':0,'measurement_source':source}
        costs.append(layer);details.append({'layer':l,'signature':signature,'delta':delta,'calls':calls,
            'selection_source':'decode-profile-and-frozen-ownership','event_calls':{name:delta[name+'_calls'] for name in ('hip','sycl')}})
    return {'schema':'strata-warm-costs-v1','scope':'measured-decode-excluding-warmup','decode_tokens':tokens,'profile_sha256':sha(profile),'layers':costs,
            'evidence':{'run_revision':run_revision,'native_source_revision':native_revision,'stderr_sha256':sha(stderr),'native_stdout_sha256':sha(timing_stdout),
                        'measured_requests':measured,'native_packed_fallback_enabled':native_fallback,
                        'execution_modes':execution,
                        'warm_freeze_sha256':sha(warm_freeze) if warm_freeze is not None else None,
                        'baseline_plan_sha256':hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),'layer_deltas':details},
            'assumptions':['Per-selection service coefficients absorb measured issue/loop overhead; issue_ns=0 avoids double counting.',
                           'Selection counts come from the exact decode profile and placement, never GPU event counts (which count batches).',
                           'Batched coefficients are measured averages at the observed batch-size mix; rebalance is a surrogate and requires fresh native confirmation/recalibration.',
                           'B65 previously empty signatures use newly measured synchronous micro host service, including H2D/D2H; full-fork contention remains unmeasured.',
                           'Marginal profile objective is a labelled surrogate; native performance must confirm it.']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stderr','profile','plan','timing-stdout','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--no-native-fallback',action='store_true',help='require full-fork measurements in every lane; mandatory for v2/batched combination costs')
    p.add_argument('--warm-freeze',type=Path);p.add_argument('--warm-conf',type=Path)
    p.add_argument('--run-revision',required=True);p.add_argument('--native-source-revision',required=True);a=p.parse_args()
    result=build(a.stderr,a.profile,json.loads(a.plan.read_text()),a.timing_stdout,a.run_revision,a.native_source_revision,native_fallback=not a.no_native_fallback,warm_freeze=a.warm_freeze,warm_conf=a.warm_conf)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
