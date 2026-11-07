"""Exact arithmetic, bounded loads, real grouped stages and frozen opt-in."""
from pathlib import Path
import subprocess
import importlib.util
import json
import pytest
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
import os
CXX=os.environ.get('CXX','c++')


def build(tmp_path,source,defines=(),link=False):
    binary=tmp_path/'proof'
    args=[CXX,'-std=c++20','-O2','-ffp-contract=off','-Wno-attributes','-DSTRATA_FORK_WARM=1','-DSTRATA_IQ2_HOST_TEST',
          *['-D'+d for d in defines],'-I'+str(ROOT/'engine/strata/fixtures/iq2-port/host_api'),'-I'+str(ROOT),'-I'+str(ROOT/'engine/strata/sycl'),'-iquote',str(HERE),str(source)]
    if link:args += [str(ROOT/'engine/strata/sycl/moe/row2.cpp'),str(ROOT/'engine/strata/sycl/stage/wiring/pack_source.cpp'),str(ROOT/'engine/strata/sycl/stage/wiring/cpu_lane_stub.cpp')]
    r=subprocess.run(args+['-o',str(binary)],capture_output=True,text=True,timeout=180)
    assert r.returncode==0,r.stdout+r.stderr
    return binary


def test_exhaustive_half_words_tiles_and_subdots(tmp_path):
    binary=build(tmp_path,HERE/'strip_proof.cpp')
    r=subprocess.run([str(binary)],capture_output=True,text=True,timeout=180)
    assert r.returncode==0,r.stdout+r.stderr
    assert '65536 half encodings' in r.stdout
    print(r.stdout.strip())


def test_actual_fast_grouped_chains_and_tail_barriers(tmp_path):
    binary=build(tmp_path,HERE/'expert_check.cpp',['STRATA_EXPERT_GROUPED=1','STRATA_EXPERT_STRIP=1'],True)
    r=subprocess.run([str(binary)],capture_output=True,text=True,timeout=300)
    assert r.returncode==0,r.stdout+r.stderr
    assert 'ACTUAL_K_CHECKS 96' in r.stdout and '66 grouped batch checks' in r.stdout
    print(r.stdout.strip())


def test_changed_sign_word_fails_proof(tmp_path):
    header=(HERE/'packed_strip.hpp').read_text()
    assert 'U word=(mag^mask)+(mask&0x01010101u);' in header
    (tmp_path/'packed_strip.hpp').write_text(header.replace('U word=(mag^mask)+(mask&0x01010101u);','U word=mag;'))
    source=tmp_path/'strip_proof.cpp';source.write_text((HERE/'strip_proof.cpp').read_text())
    binary=build(tmp_path,source)
    r=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
    assert r.returncode==1 and 'packed magnitude/sign word' in r.stderr
    print('EXPECTED:',r.stderr.strip())


def test_packed_strip_frozen_with_all_a7_flags():
    import runpy
    ns=runpy.run_path(str(HERE/'warm_config.py'))
    cfg=ns['config']
    assert all(k not in cfg('/p',True) for k in ('warm_dense_owner','warm_hip_schedule','warm_kernel_timing','warm_sycl_experts'))
    value=cfg('/p',True,dense_owner='hip-v1',hip_schedule='queued-v1',kernel_timing=True,sycl_experts='packed-strip-v1')
    for key in ('warm_dense_owner=hip-v1','warm_hip_schedule=queued-v1','warm_kernel_timing=1','warm_sycl_experts=packed-strip-v1'):assert key+'\n' in value
    with pytest.raises(ValueError):cfg('/p',False,sycl_experts='packed-strip-v1')


def test_native_strip_reducer_rejects_partial_simulated_and_unpaired(tmp_path):
    spec=importlib.util.spec_from_file_location('strip_summary',HERE/'grouped_summary.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    rows=[dict(signature=s,layer=l,n=n,cache=c,mode=mode,rep=r,host_ns=300+r,span_ns=310+r,matrix_ns=250,gate_up_ns=150,down_ns=100)
          for l,s in enumerate(sorted(m.SIGNATURES)) for n in (1,2,4,8) for c in ('warm','scrubbed') for mode in ('serial','grouped','packed-strip') for r in range(5)]
    p=tmp_path/'timing'
    def write(rows,prefix='NATIVE device=TEST driver=TEST\n',end='PASS: 24 grouped batch checks; fixture\n'):
        p.write_text(prefix+''.join('STRIP_TIMING '+json.dumps(r)+'\n' for r in rows)+end)
    write(rows);report=m.summarise(p,strip=True);assert report['rows']==720 and len(report['comparisons'])==144
    for bad in (rows[:-1],rows+[rows[0]],[{**r,'down_ns':99} for r in rows],[{**r,'mode':'grouped'} for r in rows]):
        write(bad)
        with pytest.raises(ValueError):m.summarise(p,strip=True)
    write(rows,prefix='HOST_SIMULATION_ONLY\n')
    with pytest.raises(ValueError):m.summarise(p,strip=True)
    write(rows,end='')
    with pytest.raises(ValueError):m.summarise(p,strip=True)


def test_native_timing_branch_requires_real_pack(tmp_path):
    binary=build(tmp_path,HERE/'expert_check.cpp',['STRATA_EXPERT_GROUPED=1','STRATA_EXPERT_STRIP=1','STRATA_GROUPED_TIMING=1','STRATA_EXPERT_NATIVE_SIMULATION=1'],True)
    r=subprocess.run([str(binary)],capture_output=True,text=True,timeout=30)
    assert r.returncode==1 and 'native timing requires real pack' in r.stderr
    print('EXPECTED:',r.stderr.strip())
