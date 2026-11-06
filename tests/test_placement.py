"""Critical-path, co-selection, capacity and provenance controls."""
import copy
import importlib.util
import json
from pathlib import Path
import pytest

HERE=Path(__file__).resolve().parents[1]/'engine/strata/warm'
spec=importlib.util.spec_from_file_location('fw6_rebalance',HERE/'rebalance.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def case(tmp_path,trace=False):
    profile=tmp_path/'decode.bin';counts=[1,1,1,1]*2 if trace else [1,1,10,0]*2
    profile.write_bytes(b''.join(x.to_bytes(4,'little') for x in counts))
    plan={'layers':2,'sycl_footprint':28_500_000_000,'hip_footprint':30_450_647_040,
          'warm_inventory':[[l,e,2**21,2**21,'IQ2_XS','IQ2_XS','IQ1_M'] for l in range(2) for e in range(4)],
          'hip_resident':{'experts':[[l,0,0,2**21] for l in range(2)]},
          'warm_sycl':{'members':[[l,1] for l in range(2)],'reservation_bytes':2*2**21,'profile_tag':99},
          'warm_hip_room_bytes':2*2**21}
    costs={'schema':'strata-warm-costs-v1','scope':'measured-decode-excluding-warmup','evidence':{'run_revision':'test-fixture'},
           'decode_tokens':2 if trace else 12,'profile_sha256':r.sha(profile),
           'layers':[{name:{'per_expert_ns':cost,'issue_ns':0} for name,cost in zip(r.LANES,(10,1,2))} for _ in range(2)]}
    if trace:costs['samples']=[{'weight':1,'layers':[[0,1],[0,1]]},{'weight':1,'layers':[[2,3],[2,3]]}]
    return plan,profile,costs


def test_real_objective_moves_hot_cpu_and_reconstructs_full_assignment(tmp_path):
    plan,profile,costs=case(tmp_path)
    assigned,report=r.optimise(plan,profile,costs,min_payload=0)
    assert report['after_ns']<report['before_ns']
    assert any(assigned[l,2]!=0 for l in range(2))
    assert report['hip_driver_charge_bytes']<=plan['warm_hip_room_bytes']
    assert report['sycl_allocated_bytes']<=plan['warm_sycl']['reservation_bytes']
    assert all(report['b65_experts_per_layer']) and 'surrogate' in report['objective']
    assert r.optimise(plan,profile,costs,min_payload=0)==(assigned,report)
    text=r.assignment_text(assigned,plan)
    assert len(text.splitlines())==9 and text.startswith('STRATA_WARM_PLACEMENT_V1 2 4 99\n')
    assert sum(report['per_layer_before_ns'])==report['before_ns']


def test_co_selection_changes_mean_of_max_and_binds_exact_histogram(tmp_path):
    plan,profile,costs=case(tmp_path,trace=True)
    _,traced=r.optimise(plan,profile,costs,min_payload=0)
    assert traced['before_ns']==22
    marginal=copy.deepcopy(costs);marginal.pop('samples')
    _,surrogate=r.optimise(plan,profile,marginal,min_payload=0)
    assert surrogate['before_ns']==20
    assert 'measured-co-selection' in traced['objective']
    costs['samples'][0]['layers'][0]=[0,2]
    with pytest.raises(ValueError,match='exact decode profile'):r.optimise(plan,profile,costs,min_payload=0)


@pytest.mark.parametrize('tamper',['hash','scope','tokens','nan','signature','sycl-cap','hip-cap','driver-rounding','trace-duplicate'])
def test_fail_closed_cost_profile_and_capacity_controls(tmp_path,tamper):
    plan,profile,costs=case(tmp_path)
    if tamper=='hash':costs['profile_sha256']='0'*64
    elif tamper=='scope':costs['scope']='shipped-profile'
    elif tamper=='tokens':costs['decode_tokens']=0
    elif tamper=='nan':costs['layers'][0]['cpu']['per_expert_ns']=float('nan')
    elif tamper=='signature':costs['layers'][0]['sycl'].pop('per_expert_ns')
    elif tamper=='sycl-cap':plan['sycl_footprint']+=1
    elif tamper=='hip-cap':plan['hip_footprint']+=1
    elif tamper=='driver-rounding':plan['warm_hip_room_bytes']=2*2**21-1
    elif tamper=='trace-duplicate':costs['samples']=[{'weight':12,'layers':[[0,0],[0,0]]}]
    with pytest.raises(ValueError):r.optimise(plan,profile,costs,min_payload=0)


def test_b65_fixed_workspace_budget_is_not_expert_bank_capacity(tmp_path):
    plan,profile,costs=case(tmp_path)
    plan['warm_sycl']['reservation_bytes']=2**21-1
    with pytest.raises(ValueError,match='coverage|violates'):r.optimise(plan,profile,costs,min_payload=0)


def test_timing_inventory_and_simulation_cannot_be_mistaken_for_native(tmp_path):
    spec=importlib.util.spec_from_file_location('fw6_timing_summary',HERE/'timing_summary.py')
    t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
    rows=[]
    for sig in ('IQ2_XS/IQ2_XS/IQ1_M/','IQ2_XXS/IQ2_XXS/IQ2_XS/','IQ2_XS/IQ2_XS/IQ2_XXS/','IQ2_XS/IQ2_XS/IQ2_XS/'):
        for cache in ('warm','scrubbed'):
            for packed in (False,True):
                for rep in range(5):rows.append('TIMING '+json.dumps({'signature':sig,'layer':0,'expert':0,'cache':cache,'packed':packed,'rep':rep,'chain_event_ns':100+rep,'host_ns':200+rep,'gate_up_event_ns':30,'down_event_ns':20}))
    path=tmp_path/'timing.stdout';path.write_text('NATIVE device=fixture driver=fixture\n'+'\n'.join(rows)+'\nPASS: fixture\n')
    report=t.summarise(path);assert len(report['cells'])==16
    assert report['cells'][0]['metrics']['chain_event_ns']['sample_sd']>0
    original=path.read_text();path.write_text('HOST_SIMULATION_ONLY\n'+original)
    with pytest.raises(ValueError,match='simulation'):t.summarise(path)
    path.write_text(original.replace(rows[0],rows[1]))
    with pytest.raises(ValueError,match='repetition'):t.summarise(path)


def test_cost_builder_subtracts_prefill_excludes_warmup_and_uses_host_service(tmp_path,monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(HERE))
    spec=importlib.util.spec_from_file_location('fw6_make_costs',HERE/'make_costs.py')
    c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
    plan,profile,_=case(tmp_path)
    profile.write_bytes(b''.join(n.to_bytes(4,'little') for n in [3,0,3,0]*2))
    lines=[]
    for i in range(1,5):
        for event in ('request_begin','prefill_end','decode_end','request_end'):
            lines.append('store_diag_v1 '+json.dumps({'request':i,'event':event,'interval_delta':{'model_fed_tokens':1}}))
            if event not in ('prefill_end','request_end'):continue
            for l in range(2):
                d=0 if event=='prefill_end' else (1000 if i==1 else 1)
                lines.append(f'AIOPT_WARM_LAYER_TOTALS request={i} layer={l} cpu_service_ns={5000+100*d} hip_service_ns={1000+20*d} sycl_service_ns={100+2*d} hip_calls={5+d} sycl_calls=0')
    stderr=tmp_path/'engine.stderr';stderr.write_text('\n'.join(lines))
    stdout=tmp_path/'native.stdout';stdout.write_text('fixture')
    monkeypatch.setattr(c,'summarise',lambda path:{'cells':[{'packed':True,'cache':'warm','signature':'IQ2_XS/IQ2_XS/IQ1_M/','metrics':{'host_ns':{'median':35},'chain_event_ns':{'median':3}}}]})
    result=c.build(stderr,profile,plan,stdout,'a'*40,'b'*40)
    assert result['decode_tokens']==3
    assert result['layers'][0]['cpu']['per_expert_ns']==100
    assert result['layers'][0]['hip']['per_expert_ns']==20
    assert result['layers'][0]['sycl']['per_expert_ns']==35
    assert result['evidence']['layer_deltas'][0]['calls']=={'hip':3,'sycl':0,'cpu':3}
