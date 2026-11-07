"""Exercise standalone entrypoints without a GPU, external checkout or network."""
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    spec=importlib.util.spec_from_file_location('public_'+name,ROOT/'tools'/(name+'.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def stream(rates=(2,4,8,16)):
    text='READY\n';counter=0
    for i,rate in enumerate(rates):
        text+=f'BEGIN {i} {1000+i} {counter}\nTOKEN 1\nTOKEN 2\nEND 2 1000000 {int(2e9/rate)} eos {counter} {counter+2}\n'
        counter+=2
    return text


def test_native_stream_warmup_excluded_and_bad_records_rejected(tmp_path):
    m=module('bench');path=tmp_path/'raw';text=stream();path.write_text(text)
    rows,stats=m.parse_stream(path)
    assert stats=={'median':8.,'min':4.,'max':16.,'n':3}
    assert [r['phase'] for r in rows]==['warmup','measured','measured','measured']
    for bad in (text.replace('BEGIN 1 1001','BEGIN 1 1000'),text.replace('END 2','END 3'),
                text.replace('eos 0 2','eos 1 3'),text[:-1].rsplit('END',1)[0],text+'READY\n',
                text.replace('1000000 1000000000','1000000 0')):
        path.write_text(bad)
        with pytest.raises(ValueError):m.parse_stream(path)


@pytest.mark.parametrize('preset',['baseline','a16','a17'])
def test_benchmark_entrypoint_raw_answers_and_summary(tmp_path,preset):
    pack=tmp_path/'pack';pack.mkdir();tok=pack/'tokenizer';tok.mkdir()
    for name,data in {'vocab.json':{'x':0,'y':1,'z':2},'tokenizer.json':{'pre':'deepseek3','special_ids':{}},'token_type.json':[1,3,4]}.items():
        (tok/name).write_text(json.dumps(data))
    (tok/'merges.txt').write_text('')
    layers,experts=(40,384) if preset in ('a16','a17') else (2,8)
    (pack/'ds4.txt').write_text(f'num_hidden_layers {layers}\nn_routed_experts {experts}\n')
    for name in ('tensors.txt','engram.bin'):(pack/name).write_bytes(b'fixture')
    upstream=tmp_path/'upstream';(upstream/'tools').mkdir(parents=True)
    ids=json.loads((ROOT/'prompts/code.json').read_text())['ids']
    (upstream/'tools/strata_tokenizer.py').write_text('''class Tokenizer:
 def __init__(self,tokens,merges,token_types,pre,special_ids):
  assert pre=='deepseek3' and token_types==[1,3,4]
 def encode(self,text,parse_special):return '''+repr(ids)+'''
 def decode(self,ids):return 'fixture answer'
''')
    package=tmp_path/'package';package.mkdir();(package/'manifest.json').write_text('{}')
    binary=package/'strata-warm-run';binary.write_text('#!'+sys.executable+'''
import json,os,sys
from pathlib import Path
Path(sys.argv[sys.argv.index('--chosen-plan')+1]).write_text(json.dumps({'argv':sys.argv,'env':dict(os.environ)}))
print('''+repr(stream())+''',end="")
''');binary.chmod(0o755)
    profile=tmp_path/'site.json';profile.write_text('{}')
    out=tmp_path/'out'
    env={**os.environ,'CROSSLANE_PYTHON':sys.executable,'CROSSLANE_HOST_PROFILE':str(profile),'CROSSLANE_LOCK':str(tmp_path/'gpu.lock')}
    cmd=['bash',str(ROOT/'tools/bench.sh'),'--pack',str(pack),'--package',str(package),'--upstream',str(upstream),'--out',str(out)]
    if preset in ('a16','a17'):
        cmd+=['--preset',preset]
        env.update(DS4_OTHER_COMBINE_SIMD='0',DS4_CPU_AFFINITY='9999',DS4_WARM_HOST_PARTITION='1',
                   AIOPT_ENGRAM_QD='1',ROCR_VISIBLE_DEVICES='GPU-fixture',LD_LIBRARY_PATH='/ambient/unused')
    if preset=='a17':
        # Mock only validation of the large vendor ELFs; verify it separately below.
        script="import sys;sys.path.insert(0,sys.argv.pop(1));import bench;bench.rocm_runtime=lambda p:p;bench.main()"
        cmd=[sys.executable,'-c',script,str(ROOT/'tools'),*cmd[2:]]
    r=subprocess.run(cmd,env=env,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    assert 'median 8.000, min 4.000, max 16.000' in r.stdout
    assert (out/'engine.stdout').read_text()==stream()
    assert (out/'answer-03.txt').read_text()=='fixture answer'
    assert json.loads((out/'summary.json').read_text())['tok_s']['n']==3
    assert (out/'requests.txt').read_text().count('8192 325 ')==4
    assert len((out/'profile.bin').read_bytes())==layers*experts*4
    record=json.loads((out/'command.json').read_text())
    assert record['preset']==preset
    if preset in ('a16','a17'):
        # Independent fixture extracted from the real freeze, not from bench.py.
        expected=json.loads((ROOT/('tests/fixtures/a17-runtime.json' if preset=='a17' else 'tests/fixtures/a16r-runtime.json')).read_text())
        def relocate(value):
            for name,path in (('@OUT@',out),('@PACK@',pack),('@PACKAGE@',package),('@ROCM@',ROOT/'build/rocm-7.2.1')):
                value=value.replace(name,str(path))
            return value
        actual=json.loads((out/'chosen-plan.json').read_text())
        assert (out/'warm.conf').read_text()==relocate(expected['warm_conf'])
        assert dict(zip(actual['argv'][1::2],actual['argv'][2::2]))=={k:relocate(v) for k,v in expected['argv_options'].items()}
        engine_env={k:v for k,v in actual['env'].items() if k.startswith(('DS4_','AIOPT_ENGRAM_')) or k in ('LD_LIBRARY_PATH','HIP_VISIBLE_DEVICES','ONEAPI_DEVICE_SELECTOR')}
        assert engine_env=={k:relocate(v) for k,v in expected['environment'].items()}
        assert record['environment']=={**engine_env,'ROCR_VISIBLE_DEVICES':'GPU-fixture'}
        for name,key in (('profile.bin','profile_sha256'),('placement.txt','placement_sha256'),('requests.txt','requests_sha256')):
            assert hashlib.sha256((out/name).read_bytes()).hexdigest()==expected[key]
        assert record['profile_kind']==('a17-text-counts' if preset=='a17' else 'a16r-text-counts')
        assert record['placement_sha256']==expected['placement_sha256']
    else:
        cfg=module('bench').config(out/'profile.bin',True,dense_owner='hip-v1',sycl_attention='subgroup32-v1',sycl_experts='packed-strip-v1')
        assert (out/'warm.conf').read_text()==cfg
        assert record['profile_kind']=='uniform' and record['placement_sha256'] is None
    assert subprocess.run(cmd,env=env,capture_output=True).returncode!=0


def test_a16_refuses_wrong_dimensions_corrupt_text_and_conflicting_inputs(tmp_path,monkeypatch):
    import shutil
    m=module('bench');pack=tmp_path/'pack';pack.mkdir()
    (pack/'ds4.txt').write_text('num_hidden_layers 2\nn_routed_experts 8\n')
    with pytest.raises(ValueError,match='dimensions'):m.a16_inputs(pack)
    (pack/'ds4.txt').write_text('num_hidden_layers 40\nn_routed_experts 384\n')
    copied=tmp_path/'preset';shutil.copytree(m.A16,copied);monkeypatch.setattr(m,'A16',copied)
    counts=copied/'profile-counts.txt';counts.write_text(counts.read_text()+'\n')
    with pytest.raises(ValueError,match='text changed'):m.a16_inputs(pack)
    for extra in (['--profile','missing'],['--placement','missing'],['--sycl-experts','mmvq-soa-v1']):
        r=subprocess.run([sys.executable,str(ROOT/'tools/bench.py'),'--pack',str(pack),'--preset','a16',*extra],capture_output=True,text=True)
        assert r.returncode==2 and 'a16 binds' in r.stderr


def test_entrypoint_help():
    for command in (['bash',str(ROOT/'tools/build.sh'),'--help'],['bash',str(ROOT/'tools/bench.sh'),'--help'],
                    [sys.executable,str(ROOT/'tools/pack.py'),'--help']):
        r=subprocess.run(command,env={**os.environ,'CROSSLANE_PYTHON':sys.executable},capture_output=True,text=True)
        assert r.returncode==0,r.stderr


def test_pack_wrapper_uses_pinned_upstream_and_preserves_paths(tmp_path,monkeypatch):
    m=module('pack');calls=[]
    monkeypatch.setattr(m.subprocess,'check_output',lambda argv,**kw:(ROOT/'engine/strata/upstream.rev').read_text())
    monkeypatch.setattr(m.subprocess,'run',lambda argv,**kw:calls.append(argv))
    monkeypatch.setattr(sys,'argv',['pack','--upstream',str(tmp_path/'source'),'--model',str(tmp_path/'model shards'),'--out',str(tmp_path/'pack'),'--engram',str(tmp_path/'Engram')])
    m.main();assert len(calls)==1
    assert calls[0][1].endswith('tools/dsv41/pack_gguf.py')
    assert str(tmp_path/'model shards') in calls[0] and '--engram' in calls[0]
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'wrong\n')
    with pytest.raises(SystemExit):m.main()


@pytest.mark.parametrize('phase',['hip','sycl','host'])
def test_build_phases_use_warm_flag_j2_and_overridable_toolchains(tmp_path,phase):
    root=tmp_path/'source';(root/'tools').mkdir(parents=True)
    (root/'engine/strata').mkdir(parents=True);(root/'engine/strata/upstream.rev').write_text('pin\n')
    (root/'tools/build.sh').write_bytes((ROOT/'tools/build.sh').read_bytes())
    # Only external process boundaries are mocked. This executes each actual shell phase.
    (root/'tools/build_support.py').write_text('import pathlib,sys\npathlib.Path(sys.argv[-1]).write_text("{}")\n')
    (root/'tools/collect_runtime.py').write_text('')
    bindir=tmp_path/'bin';bindir.mkdir();log=tmp_path/'argv.jsonl'
    cmake=bindir/'cmake';cmake.write_text('#!'+sys.executable+'\nimport os,sys,json\nfrom pathlib import Path\nwith open(os.environ["CALL_LOG"],"a") as f:f.write(json.dumps(sys.argv[1:])+"\\n")\nif "--build" in sys.argv:\n p=Path(sys.argv[sys.argv.index("--build")+1]);p.mkdir(parents=True,exist_ok=True)\n t=p/"strata-row2-pool-fixture";t.write_text("#!/bin/sh\\nexit 0\\n");t.chmod(0o755)\n');cmake.chmod(0o755)
    compiler=bindir/'compiler';compiler.write_text('#!/bin/sh\necho /opt/fixture/lib/libsvml.so\n');compiler.chmod(0o755)
    env={**os.environ,'PATH':str(bindir)+':'+os.environ['PATH'],'CALL_LOG':str(log),
         'CROSSLANE_PYTHON':sys.executable,'CROSSLANE_HIP_CXX':str(compiler),
         'CROSSLANE_SYCL_CXX':str(compiler),'CROSSLANE_HOST_CXX':str(compiler),
         'CROSSLANE_BUILD_DIR':str(tmp_path/'build')}
    r=subprocess.run(['bash',str(root/'tools/build.sh'),'_phase',phase],env=env,capture_output=True,text=True)
    assert r.returncode==0,r.stdout+r.stderr
    calls=[json.loads(l) for l in log.read_text().splitlines()]
    assert any('-DSTRATA_FORK_WARM=ON' in c for c in calls)
    assert any('--build' in c and c[c.index('-j')+1]=='2' for c in calls)
    assert any('-DCMAKE_CXX_COMPILER='+str(compiler) in c for c in calls)


def test_exported_warm_host_links_without_vendor_sdk(tmp_path):
    binary=tmp_path/'strata-warm-run'
    cmd=[os.environ.get('CXX','c++'),'-std=c++20','-O1','-DSTRATA_FORK_WARM=1',
         str(ROOT/'engine/strata/warm/main.cpp'),str(ROOT/'engine/strata/sycl/stage/wiring/pack_source.cpp'),
         '-ldl','-o',str(binary)]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
    assert r.returncode==0,r.stdout+r.stderr
    r=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
    assert r.returncode==1 and 'missing --ctx' in r.stderr
