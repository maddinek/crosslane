#!/usr/bin/env python3
"""Fail-closed native MMVQ timing reducer. No end-to-end performance inference."""
import argparse
import json
import statistics
from pathlib import Path
from grouped_summary import SIGNATURES, native_signatures


def summarise(path):
    text=Path(path).read_text()
    signatures=native_signatures(text)
    grids=[l for l in text.splitlines() if l.startswith('MMVQ_GRID ')]
    if grids not in ([],['MMVQ_GRID path-a15']):raise ValueError('unknown/duplicate timing grid')
    sizes=(1,2,3,4,5,6,8) if grids else (1,2,4,8)
    if ('HOST_SIMULATION_ONLY' in text or text.count('NATIVE device=')!=1 or
            text.count(f'PASS: MMVQ {len(signatures)} batches;')!=1 or 'FAIL:' in text):
        raise ValueError('complete native real-pack accuracy and identity required')
    marker='MMVQ_TIMING '
    rows=[json.loads(line[len(marker):]) for line in text.splitlines() if line.startswith(marker)]
    keys={'signature','layer','n','cache','mode','rep','host_ns','span_ns','gate_up_ns','down_ns'}
    cells={};layers={};metrics=('host_ns','span_ns','gate_up_ns','down_ns')
    for r in rows:
        if set(r)!=keys:raise ValueError('timing schema')
        s=r['signature']
        if (s not in signatures or type(r['layer']) is not int or r['layer']<0 or
                type(r['n']) is not int or r['n'] not in sizes or
                r['cache'] not in ('warm','scrubbed') or r['mode'] not in ('exact-grouped','packed-strip-v1','mmvq-soa','mmvq-soa-v2') or
                type(r['rep']) is not int or r['rep'] not in range(5)):
            raise ValueError('timing identity')
        if s in layers and layers[s]!=r['layer']:raise ValueError('changed signature layer')
        if signatures[s] is not None and signatures[s]!=r['layer']:raise ValueError('manifest layer changed')
        layers[s]=r['layer']
        if any(type(r[m]) is not int or r[m]<=0 for m in metrics):raise ValueError('positive integer timing required')
        if r['gate_up_ns']+r['down_ns']>r['span_ns']:raise ValueError('projection span exceeds chain')
        key=(s,r['n'],r['cache'],r['mode']);cell=cells.setdefault(key,{})
        if r['rep'] in cell:raise ValueError('duplicate repetition')
        cell[r['rep']]=r
    controls={r['mode'] for r in rows if r['mode'] in ('exact-grouped','packed-strip-v1')}
    if len(controls)!=1:raise ValueError('mixed control timing modes')
    control=controls.pop()
    candidates={r['mode'] for r in rows if r['mode']!=control}
    if len(candidates)!=1:raise ValueError('mixed candidate timing modes')
    candidate=candidates.pop()
    if grids and (control,candidate)!=('packed-strip-v1','mmvq-soa-v2'):raise ValueError('a15 requires packed-strip/v2 pairing')
    expected={(s,n,c,m) for s in signatures for n in sizes for c in ('warm','scrubbed') for m in (control,candidate)}
    if set(cells)!=expected or any(set(v)!=set(range(5)) for v in cells.values()):raise ValueError('incomplete real-pack timing inventory')
    service_rows=[json.loads(l[len('MMVQ_SERVICE '):]) for l in text.splitlines() if l.startswith('MMVQ_SERVICE ')]
    service={}
    for r in service_rows:
        if set(r)!={'layer','n','cache','mode','rep','service_ns'}:raise ValueError('service schema')
        k=(r['layer'],r['n'],r['cache'],r['mode'],r['rep'])
        if k in service or type(r['service_ns']) is not int or r['service_ns']<=0:raise ValueError('service duplicate/value')
        service[k]=r['service_ns']
    expected_service={(r['layer'],r['n'],r['cache'],r['mode'],r['rep']) for r in rows} if grids else set()
    if set(service)!=expected_service:raise ValueError('incomplete host service census')
    if grids:
        for r in rows:
            r['service_ns']=service[r['layer'],r['n'],r['cache'],r['mode'],r['rep']]
            if r['service_ns']<r['host_ns']:raise ValueError('service less than submit/wait wall')
        metrics=(*metrics,'service_ns')
    stats=[];pairs=[]
    for key,cell in sorted(cells.items()):
        summary={}
        for m in metrics:
            values=[cell[i][m] for i in range(5)]
            summary[m]=dict(samples=values,median=statistics.median(values),min=min(values),max=max(values),sample_sd=statistics.stdev(values))
        stats.append(dict(zip(('signature','n','cache','mode'),key),layer=layers[key[0]],metrics=summary))
    for s,n,c in sorted({k[:3] for k in cells}):
        old=statistics.median(cells[s,n,c,control][i]['host_ns'] for i in range(5))
        new=statistics.median(cells[s,n,c,candidate][i]['host_ns'] for i in range(5))
        pairs.append(dict(signature=s,n=n,cache=c,host_speedup=old/new,host_ns_per_selection=new/n,saved_ns_per_selection=(old-new)/n))
    return dict(batch_sizes=list(sizes),control_mode=control,candidate_mode=candidate,scope='native same-pack grouped expert micro; numerical screens only, full-model quality and speed pending',rows=len(rows),cells=stats,comparisons=pairs)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('raw');p.add_argument('output');a=p.parse_args()
    result=summarise(a.raw)
    with Path(a.output).open('x') as f:json.dump(result,f,indent=2);f.write('\n')
