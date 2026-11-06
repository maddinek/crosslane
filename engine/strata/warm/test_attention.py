"""CPU conformance for the explicit subgroup attention candidate."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
import os
CXX=os.environ.get('CXX','c++')


def check_kernels(tmp_path,mutate=False):
    binary=tmp_path/'attention-subgroup'
    actual=ROOT/'engine/strata/sycl/attn/attn.cpp'
    source=actual
    if mutate:
        source=tmp_path/'wrong-tree.cpp'
        text=actual.read_text()
        old='for(int offset=16;offset;offset>>=1)'
        assert text.count(old)==2
        text=text.replace(old,'for(int offset=1;offset<32;offset<<=1)')
        source.write_text(text)
    command=[CXX,'-std=c++20','-O2','-ffp-contract=off','-Wno-attributes',
             '-DSTRATA_ROW34_HOST_API_TEST',
             '-I'+str(ROOT/'engine/strata/fixtures/row34/host_api'),
             '-I'+str(ROOT/'engine/strata/sycl'),
             '-iquote',str(actual.parent),str(source),
             str(ROOT/'engine/strata/fixtures/row34/attention_subgroup.cpp'),'-o',str(binary)]
    result=subprocess.run(command,text=True,capture_output=True,timeout=60)
    assert result.returncode==0,result.stdout+result.stderr
    result=subprocess.run([str(binary)],text=True,capture_output=True,timeout=120)
    return result


def test_actual_scalar_and_subgroup_kernels_bitwise(tmp_path):
    result=check_kernels(tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    assert '26 production-kernel cases' in result.stdout
    print(result.stdout.strip())


def test_wrong_reduction_tree_fails_actual_kernel_comparison(tmp_path):
    result=check_kernels(tmp_path,mutate=True)
    assert result.returncode!=0 and 'changed bits' in result.stderr,result.stdout+result.stderr
    print('Expected wrong-tree refusal:',result.stderr.strip())


def test_candidate_is_explicit_and_frozen_config_keeps_scalar_default():
    import importlib.util,sys
    spec=importlib.util.spec_from_file_location('fw5_attention_driver',ROOT/'engine/strata/warm/warm_config.py')
    d=importlib.util.module_from_spec(spec);sys.modules[spec.name]=d;spec.loader.exec_module(d)
    assert 'warm_sycl_attention=' not in d.config('/profile',True)
    assert d.config('/profile',True,'subgroup32-v1')==d.config('/profile',True)+'warm_sycl_attention=subgroup32-v1\n'
    import pytest
    with pytest.raises(ValueError,match='unadmitted'):d.config('/profile',True,'guess')


def test_actual_config_parser_rejects_unbound_candidate(tmp_path):
    source=tmp_path/'config.cpp';binary=tmp_path/'config'
    source.write_text(r'''
#include "engine/strata/sycl/stage/wiring/common.hpp"
#include <cassert>
using strata::wiring::Config;
int main(){
 const std::string base="pack=/absent\nmax_context=128\n";
 assert(Config::parse(base.c_str()).warm_sycl_attention=="scalar-v1");
 for(const std::string mode:{"subgroup32-v1","other"}){
  bool refused=false;try{Config::parse((base+"warm_sycl_attention="+mode+"\n").c_str());}catch(...){refused=true;}assert(refused);
 }
 const auto warm=base+"hip_experts=pooled-layer-v1\nexpert_profile=/absent\nwarm_experts=1\nwarm_measure=1\nwarm_hip_fixed=1\nwarm_sycl_fixed=1\nwarm_sycl_attention=subgroup32-v1\n";
 assert(Config::parse(warm.c_str()).warm_sycl_attention=="subgroup32-v1");
}
''')
    subprocess.run([CXX,'-std=c++20','-DSTRATA_FORK_WARM=1','-I'+str(ROOT),str(source),'-o',str(binary)],check=True,capture_output=True,timeout=60)
    subprocess.run([str(binary)],check=True,capture_output=True,timeout=10)
