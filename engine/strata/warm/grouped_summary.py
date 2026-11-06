#!/usr/bin/env python3
"""Strict native grouped-expert micro reducer; never promotes to decode throughput."""
import argparse
import json
import math
import re
import statistics
from pathlib import Path

SIGNATURES={
    'IQ2_XXS/IQ2_XXS/IQ2_XXS/', 'IQ1_M/IQ1_M/IQ1_M/',
    'IQ2_XS/IQ2_XS/IQ1_M/', 'IQ2_XXS/IQ2_XXS/IQ2_XS/',
    'IQ2_XS/IQ2_XS/IQ2_XXS/', 'IQ2_XS/IQ2_XS/IQ2_XS/',
}
SUPPORTED_SIGNATURES={f'{gu}/{gu}/{dn}/' for gu in ('IQ1_M','IQ2_XS','IQ2_XXS') for dn in ('IQ1_M','IQ2_XS','IQ2_XXS')}


def native_signatures(text):
    """New native manifests bind the complete available pack subset.

    Old archives without a manifest must still cover all six signatures.
    Never infer coverage from timing rows: deleting a complete signature must fail.
    """
    counts=re.findall(r'^REAL_SIGNATURES ([0-9]+)$',text,re.M)
    manifest=re.findall(r'^REAL_SIGNATURE layer=([0-9]+) signature=(\S+)$',text,re.M)
    if not counts and not manifest:return {s:None for s in SIGNATURES}
    if len(counts)!=1 or not 1<=int(counts[0])<=len(SUPPORTED_SIGNATURES) or len(manifest)!=int(counts[0]):
        raise ValueError('incomplete real signature manifest')
    result={s:int(layer) for layer,s in manifest}
    if len(result)!=len(manifest) or not set(result)<=SUPPORTED_SIGNATURES or len(set(result.values()))!=len(result):
        raise ValueError('invalid real signature manifest')
    return result


def summarise(path,strip=False):
    modes=("serial","grouped","packed-strip") if strip else ("serial","grouped")
    marker="STRIP_TIMING " if strip else "GROUP_TIMING "
    text=Path(path).read_text()
    signatures=native_signatures(text)
    if ('HOST_SIMULATION_ONLY' in text or text.count('NATIVE device=')!=1 or
            text.count(f'PASS: {4*len(signatures)} grouped batch checks;')!=1 or 'FAIL:' in text):
        raise ValueError('native identity and all real-pack bitchecks required')
    rows=[json.loads(line.removeprefix(marker)) for line in text.splitlines() if line.startswith(marker)]
    cells={}
    expected_keys={'signature','layer','n','cache','mode','rep','host_ns','span_ns','matrix_ns'}
    if strip:expected_keys|={'gate_up_ns','down_ns'}
    metrics=('host_ns','span_ns','matrix_ns','gate_up_ns','down_ns') if strip else ('host_ns','span_ns','matrix_ns')
    layers={}
    for row in rows:
        if set(row)!=expected_keys:raise ValueError('timing schema')
        sig=row['signature'];key=(sig,row['n'],row['cache'],row['mode'])
        if (sig not in signatures or type(row['n']) is not int or row['n'] not in (1,2,4,8) or
                row['cache'] not in ('warm','scrubbed') or row['mode'] not in modes or
                type(row['rep']) is not int or row['rep'] not in range(5) or
                type(row['layer']) is not int or row['layer']<0):raise ValueError('timing identity')
        if sig in layers and layers[sig]!=row['layer']:raise ValueError('signature layer changed')
        if signatures[sig] is not None and signatures[sig]!=row['layer']:raise ValueError('manifest layer changed')
        layers[sig]=row['layer']
        for field in metrics:
            if type(row[field]) is not int or row[field]<=0:raise ValueError('positive native timings required')
        if row['matrix_ns']>row['span_ns']:raise ValueError('matrix busy exceeds ordered span')
        if strip and row['gate_up_ns']+row['down_ns']!=row['matrix_ns']:raise ValueError('projection timing sum')
        cell=cells.setdefault(key,{})
        if row['rep'] in cell:raise ValueError('duplicate repetition')
        cell[row['rep']]=row
    expected={(s,n,c,m) for s in signatures for n in (1,2,4,8) for c in ('warm','scrubbed') for m in modes}
    if set(cells)!=expected or any(set(v)!=set(range(5)) for v in cells.values()):raise ValueError('incomplete '+str(len(expected)*5)+'-row timing inventory')
    result=[]
    for key,cell in sorted(cells.items()):
        stats={}
        for metric in metrics:
            samples=[cell[i][metric] for i in range(5)]
            stats[metric]={'samples':samples,'median':statistics.median(samples),'min':min(samples),'max':max(samples),'sample_sd':statistics.stdev(samples)}
        result.append(dict(zip(('signature','n','cache','mode'),key),layer=layers[key[0]],metrics=stats))
    pairs=[]
    for s,n,c in sorted({key[:3] for key in cells}):
        for control,candidate in (("serial","grouped"),("grouped","packed-strip"),("serial","packed-strip")) if strip else (("serial","grouped"),):
            old=statistics.median(cells[s,n,c,control][i]['host_ns'] for i in range(5))
            new=statistics.median(cells[s,n,c,candidate][i]['host_ns'] for i in range(5))
            busy_old=statistics.median(cells[s,n,c,control][i]['matrix_ns'] for i in range(5))
            busy_new=statistics.median(cells[s,n,c,candidate][i]['matrix_ns'] for i in range(5))
            pairs.append({'signature':s,'n':n,'cache':c,'control':control,'candidate':candidate,
                          'host_speedup':old/new,'matrix_speedup':busy_old/busy_new,
                          'saved_ns_per_batch':old-new,'saved_ns_per_selection':(old-new)/n,
                          'b65_service_ms_saved_at_48_70':[48*(old-new)/n/1e6,70*(old-new)/n/1e6]})
    return {'scope':'native same-pack micro; no end-to-end speed claim','rows':len(rows),'cells':result,'comparisons':pairs}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--strip',action='store_true');p.add_argument('raw');p.add_argument('output');a=p.parse_args()
    with Path(a.output).open('x') as f:json.dump(summarise(a.raw,strip=a.strip),f,indent=2);f.write('\n')
