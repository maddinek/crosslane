"""Compile and execute actual MMVQ SoA kernels in the CPU cooperative shim."""
from pathlib import Path
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
import os
CXX=os.environ.get('CXX','c++')


def build(binary,source,extra=(),definitions=()):
    cmd=[CXX,'-std=c++20','-O2','-ffp-contract=off','-pthread','-DSTRATA_FORK_WARM=1','-DSTRATA_IQ2_HOST_TEST','-Wno-attributes',
         '-I'+str(ROOT/'engine/strata/fixtures/iq2-port/host_api'),'-I'+str(ROOT),'-I'+str(ROOT/'engine/strata/sycl'),*definitions,str(source),*map(str,extra),'-o',str(binary)]
    result=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
    assert result.returncode==0,result.stdout+result.stderr


def run(binary,*args):
    return subprocess.run([str(binary),*map(str,args)],capture_output=True,text=True,timeout=300)


def test_lossless_layout_and_exhaustive_integer_decoder(tmp_path):
    binary=tmp_path/'proof';build(binary,HERE/'mmvq_proof.cpp')
    result=run(binary);assert result.returncode==0,result.stdout+result.stderr
    assert 'PASS:' in result.stdout;print(result.stdout)


def primitives():
    return [ROOT/'engine/strata/sycl/moe/row2.cpp',ROOT/'engine/strata/sycl/stage/wiring/pack_source.cpp',ROOT/'engine/strata/sycl/stage/wiring/cpu_lane_stub.cpp']


def test_actual_k_grouped_chain_with_proposed_numeric_gates(tmp_path):
    binary=tmp_path/'accuracy';build(binary,HERE/'expert_check.cpp',primitives(),['-DSTRATA_EXPERT_MMVQ=1'])
    result=run(binary);assert result.returncode==0,result.stdout+result.stderr
    assert 'HOST_SIMULATION_ONLY' in result.stdout and 'MMVQ_ACTUAL_K 18' in result.stdout and 'PASS: MMVQ 37 batches' in result.stdout
    print(result.stdout)


def test_native_timing_branch_compiles_and_requires_pack(tmp_path):
    binary=tmp_path/'timing';build(binary,HERE/'expert_check.cpp',primitives(),['-DSTRATA_EXPERT_MMVQ=1','-DSTRATA_MMVQ_TIMING=1','-DSTRATA_EXPERT_NATIVE_SIMULATION=1'])
    result=run(binary);assert result.returncode==1 and 'native MMVQ timing requires real pack' in result.stderr,result.stdout+result.stderr


@pytest.mark.parametrize("candidate",["mmvq-soa-v1","mmvq-soa-v2"])
def test_opt_in_config_and_real_bank(tmp_path,candidate):
    source=(HERE/'grouped_bank_test.cpp').read_text()
    source=source.replace('#include "sycl_bank.hpp"','#include "sycl_bank.hpp"\n#include "mmvq_metrics.hpp"')
    source=source.replace('cfg.warm_sycl_experts=mode==2?', 'cfg.warm_sycl_experts=mode==3?"mmvq-soa-v1":mode==2?')
    source=source.replace('else assert(out.size()==baseline[case_number].size()&&!std::memcmp(out.data(),baseline[case_number].data(),out.size()*4));', 'else if(mode==3)assert(mmvq::metrics(out,baseline[case_number]).expert_ok());else assert(out.size()==baseline[case_number].size()&&!std::memcmp(out.data(),baseline[case_number].data(),out.size()*4));')
    source=source.replace('exercise(0);exercise(1);exercise(2);','exercise(0);exercise(3);')
    # Direct serial entry must never interpret SoA as raw AoS.
    source=source.replace('size_t case_number=0;', 'size_t case_number=0; if(mode==3){auto direct=bank->device(0,5,x);auto* redirect=dynamic_cast<WarmRedirect*>(bank.get());auto batch=redirect->warm_compute_many(0,std::array<int,1>{5},x);assert(direct==batch.at(0));}')
    source=source.replace('mmvq-soa-v1',candidate)
    if candidate in ('mmvq-soa-v2','mmvq-soa-v3'):
        source=source.replace('else if(mode==3)assert(mmvq::metrics(out,baseline[case_number]).expert_ok());','')
    p=tmp_path/'bank.cpp';p.write_text(source)
    inc=ROOT/'engine/strata/fixtures/warm-store'
    binary=tmp_path/'bank';build(binary,p,[HERE/'sycl_bank.cpp',HERE/'runtime.cpp',ROOT/'engine/strata/sycl/moe/row2.cpp'],['-I'+str(inc),'-I'+str(HERE)])
    result=run(binary,tmp_path/'weights',tmp_path/'profile');assert result.returncode==0,result.stdout+result.stderr
    assert 'AIOPT_WARM_MMVQ layout=soa-v1 same_payload=1' in result.stderr;print(result.stdout)


def test_default_config_and_freeze_opt_in(tmp_path):
    import ast
    module=ast.parse((HERE/'warm_config.py').read_text());nodes=[n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='config']
    ns={'SYCL_FIXED_BYTES':3*2**30};exec(compile(ast.Module(body=nodes,type_ignores=[]),'driver.py','exec'),ns)
    default=ns['config']('/profile',True)
    assert 'warm_sycl_experts' not in default
    assert ns['config']('/profile',True,sycl_experts='mmvq-soa-v1')==default+'warm_sycl_experts=mmvq-soa-v1\n'
    with pytest.raises(ValueError):ns['config']('/profile',False,sycl_experts='mmvq-soa-v1')
    source=tmp_path/'config.cpp';source.write_text(r'''
#include "engine/strata/sycl/stage/wiring/common.hpp"
#include <cassert>
using strata::wiring::Config;
int main(){std::string base="pack=/absent\nmax_context=128\n";assert(Config::parse(base.c_str()).warm_sycl_experts=="serial-v1");bool bad=false;try{Config::parse((base+"warm_sycl_experts=mmvq-soa-v1\n").c_str());}catch(...){bad=true;}assert(bad);auto config=base+"hip_experts=pooled-layer-v1\nexpert_profile=/absent\nwarm_experts=1\nwarm_measure=1\nwarm_hip_fixed=1\nwarm_sycl_fixed=1\nwarm_packed_gu=1\nwarm_sycl_experts=mmvq-soa-v1\n";assert(Config::parse(config.c_str()).warm_sycl_experts=="mmvq-soa-v1");}
''')
    binary=tmp_path/'config';build(binary,source);assert run(binary).returncode==0


def test_reducer_rejects_incomplete_duplicate_fake_and_bad_timings(tmp_path,monkeypatch):
    import importlib.util
    import json
    monkeypatch.syspath_prepend(str(HERE))
    spec=importlib.util.spec_from_file_location('mmvq_summary',HERE/'mmvq_summary.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    rows=[dict(signature=s,layer=l,n=n,cache=c,mode=m,rep=r,host_ns=100+r,span_ns=90+r,gate_up_ns=40,down_ns=30) for l,s in enumerate(sorted(mod.SIGNATURES)) for n in (1,2,4,8) for c in ('warm','scrubbed') for m in ('exact-grouped','mmvq-soa') for r in range(5)]
    path=tmp_path/'raw'
    def write(data=rows,prefix='NATIVE device=TEST FIXTURE\n',suffix='PASS: MMVQ 6 batches; synthetic reducer fixture\n'):
        path.write_text(prefix+''.join('MMVQ_TIMING '+json.dumps(x)+'\n' for x in data)+suffix)
    write();result=mod.summarise(path);assert result['rows']==480 and len(result['comparisons'])==48
    for data in (rows[:-1],rows+[rows[0]],[dict(r,span_ns=0) for r in rows],[dict(r,gate_up_ns=80) for r in rows],[dict(r,host_ns=float('nan')) for r in rows]):
        write(data)
        with pytest.raises(ValueError):mod.summarise(path)
    for prefix,suffix in (('HOST_SIMULATION_ONLY\n','PASS: MMVQ 6 batches;\n'),('NATIVE device=TEST\n','')):
        write(prefix=prefix,suffix=suffix)
        with pytest.raises(ValueError):mod.summarise(path)


def test_wrong_selected_expert_is_rejected_by_numeric_gate(tmp_path):
    overlay=tmp_path/'overlay';overlay.mkdir()
    header=(HERE/'mmvq_project.hpp').read_text()
    assert header.count('expert=GateUp?part/2:part;')==1
    (overlay/'mmvq_project.hpp').write_text(header.replace('expert=GateUp?part/2:part;','expert=0;'))
    (overlay/'mmvq_v3.hpp').write_text((HERE/'mmvq_v3.hpp').read_text())
    (overlay/'mmvq_check.hpp').write_text((HERE/'mmvq_check.hpp').read_text())
    source=overlay/'expert_check.cpp';source.write_text((HERE/'expert_check.cpp').read_text())
    binary=tmp_path/'mutated';build(binary,source,primitives(),['-DSTRATA_EXPERT_MMVQ=1','-iquote',str(HERE)])
    result=run(binary)
    assert result.returncode==1 and 'MMVQ projection backward-error bound' in result.stderr,result.stdout+result.stderr
    print('EXPECTED WRONG-EXPERT REJECTION:',result.stderr)


def tiny_pack(path,homogeneous_iq1=True,extra_signatures=False):
    import random
    import struct
    path.mkdir()
    sigs=[('IQ2_XXS','IQ2_XXS','IQ2_XXS'),('IQ1_M','IQ1_M','IQ1_M'),('IQ2_XS','IQ2_XS','IQ1_M'),('IQ2_XXS','IQ2_XXS','IQ2_XS'),('IQ2_XS','IQ2_XS','IQ2_XXS'),('IQ2_XS','IQ2_XS','IQ2_XS'),('IQ2_XXS','IQ2_XXS','IQ2_XXS')]
    if not homogeneous_iq1:sigs[1]=sigs[0]
    if extra_signatures:sigs.extend([('IQ1_M','IQ1_M','IQ2_XXS'),('IQ2_XXS','IQ2_XXS','IQ1_M')])
    meta=dict(model_dir=path,format='gguf',vocab_size=2,hidden_size=256,num_hidden_layers=len(sigs),num_attention_heads=1,head_dim=128,qk_rope_head_dim=64,q_lora_rank=1,o_lora_rank=1,o_groups=1,n_routed_experts=8,num_experts_per_tok=8,moe_intermediate_size=256,sliding_window=1,hc_mult=1,compress_ratios=','.join(['0']*len(sigs)),swiglu_limit=1)
    (path/'ds4.txt').write_text(''.join(f'{k} {v}\n' for k,v in meta.items()))
    (path/'engram.bin').write_bytes(b'DSE1'+struct.pack('<6i',2,0,2,1,32,0)+struct.pack('<2i',0,1))
    rng=random.Random(6581);data=bytearray();lines=['file 0 fixture.gguf']
    for layer,sig in enumerate(sigs):
        for expert in range(8):
            for role,ty in zip((1,3,2),sig):
                bs={'IQ2_XXS':66,'IQ2_XS':74,'IQ1_M':56}[ty];raw=bytearray(rng.randbytes(256*bs))
                for row in range(256):
                    at=row*bs;half=0x3555 if row%2 else 0xb555
                    if ty!='IQ1_M':struct.pack_into('<H',raw,at,half)
                    else:
                        for word in range(4):
                            pos=at+48+2*word;sc=struct.unpack_from('<H',raw,pos)[0];struct.pack_into('<H',raw,pos,(sc&4095)|(((half>>(4*word))&15)<<12))
                lines.append(f't layers.{layer}.ffn.experts.{expert}.w{role}.weight {ty} 0 {len(data)} {len(raw)} 256 256');data.extend(raw)
    (path/'fixture.gguf').write_bytes(data);(path/'tensors.txt').write_text('\n'.join(lines)+'\n')
    captures=path/'synthetic-captures.bin'
    # A repeated signature layer must still consume its captured input.
    captures.write_bytes(b''.join(struct.pack('<II256f',l,256,*[float(i%17-8)/16 for i in range(256)]) for l in (0,6)))
    return captures


def test_native_pack_reader_and_captured_input_branch_on_cpu(tmp_path):
    binary=tmp_path/'native-accuracy-branch'
    build(binary,HERE/'expert_check.cpp',primitives(),['-DSTRATA_EXPERT_MMVQ=1','-DSTRATA_EXPERT_NATIVE_SIMULATION=1'])
    pack=tmp_path/'pack';captures=tiny_pack(pack)
    result=run(binary,'--pack',pack,'--activations',captures)
    assert result.returncode==0,result.stdout+result.stderr
    assert 'HOST_SIMULATION_ONLY' in result.stdout and 'PASS: MMVQ 46 batches;' in result.stdout
    assert result.stdout.count('input=captured')==2
    print(result.stdout)
