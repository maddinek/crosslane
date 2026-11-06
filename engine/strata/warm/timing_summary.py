#!/usr/bin/env python3
"""Reduce native whole-expert timing stdout, retaining five-rep spread per cell."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarise(path):
    text=Path(path).read_text()
    if 'HOST_SIMULATION_ONLY' in text or not any(s.startswith('NATIVE device=') for s in text.splitlines()):
        raise ValueError('native Intel GPU identity required; CPU simulation cannot qualify')
    if 'PASS:' not in text or 'FAIL:' in text or 'MISMATCH' in text:raise ValueError('bitcheck did not pass')
    rows=[json.loads(s.removeprefix('TIMING ')) for s in text.splitlines() if s.startswith('TIMING ')]
    cells={};signatures=set()
    for row in rows:
        sig=row['signature'];signatures.add(sig);key=(sig,row['cache'],row['packed']);cells.setdefault(key,[]).append(row)
        if row['cache'] not in ('warm','scrubbed') or type(row['packed']) is not bool or row['expert']!=0:
            raise ValueError('unadmitted timing cell')
        for metric in ('chain_event_ns','host_ns','gate_up_event_ns','down_event_ns'):
            if type(row[metric]) is not int or row[metric]<=0:raise ValueError('invalid timing duration')
        if row['gate_up_event_ns']+row['down_event_ns']>row['chain_event_ns']:raise ValueError('event span contradicts project durations')
    expected={'IQ2_XS/IQ2_XS/IQ1_M/','IQ2_XXS/IQ2_XXS/IQ2_XS/','IQ2_XS/IQ2_XS/IQ2_XXS/','IQ2_XS/IQ2_XS/IQ2_XS/'}
    if signatures!=expected or len(rows)!=80 or len(cells)!=16:raise ValueError('four-signature/two-cache/two-mode/five-rep inventory differs')
    result=[]
    for (sig,cache,packed),samples in sorted(cells.items()):
        if sorted(s['rep'] for s in samples)!=list(range(5)) or len({s['layer'] for s in samples})!=1:raise ValueError('duplicate/missing/cross-layer repetition')
        metrics={}
        for metric in ('chain_event_ns','host_ns','gate_up_event_ns','down_event_ns'):
            values=[s[metric] for s in sorted(samples,key=lambda s:s['rep'])]
            metrics[metric]={'samples':values,'median':statistics.median(values),'min':min(values),'max':max(values),'sample_sd':statistics.stdev(values)}
        result.append({'signature':sig,'cache':cache,'packed':packed,'layer':samples[0]['layer'],'metrics':metrics})
    return {'schema':'strata-warm-expert-timing-v1','source_stdout_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'cells':result,'scope':'whole chain event span includes issue gaps; project event durations measure GPU busy; host is synchronous service wall time. No end-to-end inference claim.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stdout',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    with a.output.open('x') as f:json.dump(summarise(a.stdout),f,indent=2);f.write('\n')
