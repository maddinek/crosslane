#!/usr/bin/env python3
"""Hash-check AMD RPMs and extract only pinned ELF libraries to a FRESH prefix.

Stdlib only, no install/scripts, no runtime execution, no existing file changes.
"""
import argparse
import hashlib
import json
import lzma
from pathlib import Path
import struct

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        while b:=stream.read(1<<20):h.update(b)
    return h.hexdigest()

def regular_files(rpm):
    b=rpm.read_bytes()
    if b[:4]!=b'\xed\xab\xee\xdb':raise ValueError('not RPM')
    def endhdr(i):
        if b[i:i+3]!=b'\x8e\xad\xe8':raise ValueError('bad RPM header')
        n,size=struct.unpack_from('>II',b,i+8)
        return i+16+n*16+size
    offset=endhdr(96);offset=endhdr((offset+7)&~7)
    b=lzma.decompress(b[offset:]);offset=0
    while b[offset:offset+6] in (b'070701',b'070702'):
        h=b[offset:offset+110];size=int(h[54:62],16);namesize=int(h[94:102],16)
        name=b[offset+110:offset+110+namesize-1].decode();offset=(offset+110+namesize+3)&~3
        data=b[offset:offset+size];offset=(offset+size+3)&~3
        if name=='TRAILER!!!':return
        if int(h[14:22],16)&0o170000==0o100000 and size:yield name.removeprefix('./'),data
    raise ValueError('incomplete RPM payload')

def stage(downloads,destination):
    here=Path(__file__).resolve().parent/'rocm-7.2.1'
    packages=json.loads((here/'packages.json').read_text())['packages']
    expected={x['path']:{**x,'name':Path(x['path']).name} for x in json.loads((here/'elf-manifest.json').read_text())}
    for row in packages:
        p=downloads/row['url'].rsplit('/',1)[1]
        if p.stat().st_size!=row['bytes'] or digest(p)!=row['sha256']:raise ValueError('RPM differs: '+p.name)
    destination.mkdir(parents=True,exist_ok=False)
    found=set()
    for row in packages:
        for name,data in regular_files(downloads/row['url'].rsplit('/',1)[1]):
            if name not in expected:continue
            r=expected[name]
            if name in found or len(data)!=r['bytes'] or hashlib.sha256(data).hexdigest()!=r['sha256']:raise ValueError('ELF payload differs: '+name)
            if data[:4]!=b'\x7fELF':raise ValueError('not ELF')
            p=destination/r['name'];p.write_bytes(data);p.chmod(0o644);found.add(name)
            soname=next(t['value'] for t in r['dynamic'] if t['tag']=='DT_SONAME')
            if soname!=p.name:(destination/soname).symlink_to(p.name)
    if found!=set(expected):raise ValueError('missing ELF payload')
    (destination/'STAGED.json').write_text(json.dumps({'packages':packages,'files':[{'path':r['name'],'sha256':r['sha256']} for r in expected.values()]},indent=2)+'\n')
    print(json.dumps({'destination':str(destination),'libraries':len(found),'status':'EXTRACTED_ONLY'},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--downloads',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();stage(a.downloads,a.destination)
