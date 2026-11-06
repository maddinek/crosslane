#!/usr/bin/env python3
"""Read allowed physical cores/L3 domains; emit an explicit row-worker CPU map."""
import argparse
import json
import os
from pathlib import Path


def discover():
    cores={}
    for cpu in sorted(os.sched_getaffinity(0)):
        root=Path('/sys/devices/system/cpu')/f'cpu{cpu}'
        core=tuple(int((root/'topology'/name).read_text()) for name in ('physical_package_id','core_id'))
        caches=[]
        for index in (root/'cache').glob('index*'):
            if (index/'level').read_text().strip()!='3':continue
            size=(index/'size').read_text().strip();scale=1024 if size[-1]=='K' else 1024**2 if size[-1]=='M' else 1
            caches.append(((index/'shared_cpu_list').read_text().strip(),int(size[:-1] if scale!=1 else size)*scale))
        if len(caches)!=1:raise ValueError('one L3 domain per CPU required for this map')
        if core not in cores:cores[core]={'package':core[0],'core':core[1],'logical_cpus':[],'l3_cpus':caches[0][0],'l3_bytes':caches[0][1],
                                        'numa_nodes':sorted(p.name for p in root.glob('node[0-9]*'))}
        cores[core]['logical_cpus'].append(cpu)
    return list(cores.values())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('all','largest','smallest'),default='all')
    p.add_argument('--threads',required=True,type=int);p.add_argument('--record',type=Path);a=p.parse_args()
    cores=discover();domains={}
    for core in cores:domains.setdefault((core['l3_bytes'],core['l3_cpus']),[]).append(core['logical_cpus'][0])
    order=sorted(domains,reverse=True)
    if a.mode=='largest':order=order[:1]
    elif a.mode=='smallest':order=order[-1:]
    cpus=[domains[key][i] for i in range(max(map(len,domains.values()))) for key in order if i<len(domains[key])]
    if not 1<=a.threads<=len(cpus):raise ValueError('requested workers exceed allowed physical cores in selected domains')
    selected=cpus[:a.threads]
    if a.record:
        with a.record.open('x') as f:json.dump({'cores':cores,'mode':a.mode,'selected':selected},f,indent=2);f.write('\n')
    print(','.join(map(str,selected)))


if __name__=='__main__':main()
