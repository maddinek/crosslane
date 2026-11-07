"""Warm stage configuration shared by research and standalone entrypoints."""
SYCL_FIXED_BYTES=3*2**30
HIP_FIXED_BYTES=1529365504

def fixed_budgets(hip=HIP_FIXED_BYTES,sycl=SYCL_FIXED_BYTES):
    # Explicit paired allocation probes; physical caps/reserves stay unchanged.
    if type(hip) is not int or hip not in (HIP_FIXED_BYTES,HIP_FIXED_BYTES-(512<<20)):
        raise ValueError('unadmitted HIP fixed budget')
    if type(sycl) is not int or sycl not in tuple(SYCL_FIXED_BYTES-i*(512<<20) for i in range(5)):
        raise ValueError('unadmitted SYCL fixed budget')
    return hip,sycl

def cost_modes(fields,cpu_layer_batch=False,cpu_affinity='',cpu_prefetch_rows=None,cpu_spin_us=None,cpu_store_thp=None,cpu_timing=False,engram_row_cache=None,engram_io=None,other_options=None):
    """Service-cost provenance excludes diagnostic timers, includes execution."""
    defaults={'warm_dense_owner':'split-v1','warm_hip_schedule':'baseline-v1',
              'warm_hip_gemv':'off','warm_hip_dense':'off','warm_hip_fuse':'off','warm_engram':'off',
              'warm_sycl_attention':'scalar-v1','warm_sycl_experts':'serial-v1',
              'warm_hip_experts':'serial-v1','hip_expert_math':'per32-v1','cpu_threads':'16'}
    result={**{k:fields.get(k,v) for k,v in defaults.items()},
            'cpu_layer_batch':cpu_layer_batch,'cpu_affinity':cpu_affinity}
    # Preserve historical default cost records, but never reuse them across a
    # changed memory/scheduling mode. Timer diagnostics are not execution modes.
    for k,v in (('cpu_prefetch_rows',cpu_prefetch_rows),('cpu_spin_us',cpu_spin_us),('cpu_store_thp',cpu_store_thp)):
        if v is not None:result[k]=v
    if engram_row_cache is not None:result['engram_row_cache']=dict(engram_row_cache)
    if engram_io is not None:result['engram_io']=dict(engram_io)
    if other_options is not None:result['other_options']=dict(other_options)
    if fields.get("warm_expert_issue","off")!="off":result["warm_expert_issue"]=fields["warm_expert_issue"]
    for key,default in (('warm_hip_fixed',HIP_FIXED_BYTES),('warm_sycl_fixed',SYCL_FIXED_BYTES)):
        if int(fields.get(key,default))!=default:result[key]=int(fields[key])
    for key in ('warm_hip_swiglu','warm_engram_priority'):
        if fields.get(key,'off')!='off':result[key]=fields[key]
    return result

def config(profile,packed_gu=False,sycl_attention='scalar-v1',dense_owner='split-v1',hip_schedule='baseline-v1',kernel_timing=False,sycl_experts='serial-v1',hip_dense='off',hip_fuse='off',engram='off',hip_experts='serial-v1',hip_gemv='off',cpu_timing=False,expert_issue='off',expert_timeline_limit=0,warm_hip_fixed=HIP_FIXED_BYTES,warm_sycl_fixed=SYCL_FIXED_BYTES,hip_swiglu="off",engram_priority="off",engram_trace_limit=0):
    if hip_swiglu not in ('off','batch-v1') or (hip_swiglu!='off' and hip_experts!='layer-batch-v1'):raise ValueError('batch SwiGLU requires AoS layer batching')
    if engram_priority not in ('off','first-v1') or type(engram_trace_limit) is not int or not 0<=engram_trace_limit<=2000:raise ValueError('invalid Engram scheduling')
    if (engram_priority!='off' or engram_trace_limit) and engram!='read-ahead-v1':raise ValueError('Engram scheduling requires read-ahead')
    fixed_budgets(warm_hip_fixed,warm_sycl_fixed)
    if (warm_hip_fixed,warm_sycl_fixed)!=(HIP_FIXED_BYTES,SYCL_FIXED_BYTES) and dense_owner!='hip-v1':
        raise ValueError('reduced fixed budgets require HIP cut0 bank-only B65')
    if expert_issue not in ('off','early-v1') or type(expert_timeline_limit) is not int or not 0<=expert_timeline_limit<=2000:raise ValueError('invalid expert issue/timeline')
    if (expert_issue!='off' or expert_timeline_limit) and (dense_owner!='hip-v1' or hip_schedule!='queued-v1' or hip_dense!='off'):raise ValueError('expert issue requires HIP cut0 queued and graphs off')
    if type(cpu_timing) is not bool:raise ValueError('invalid CPU timing')
    if hip_gemv not in ('off','q5-sums-v1','q5-sums-w4-v1') or (hip_gemv!='off' and (dense_owner!='hip-v1' or hip_schedule!='queued-v1' or hip_dense!='off')):raise ValueError('HIP dense GEMV requires HIP dense, queued-v1 and graphs off')
    if hip_experts not in ('serial-v1','layer-batch-v1','soa-v2') or (hip_experts!='serial-v1' and (hip_schedule!='queued-v1' or hip_dense!='off')):raise ValueError('HIP layer batching requires queued-v1 and graphs off')
    if sycl_experts not in ('serial-v1','grouped-v1','packed-strip-v1','mmvq-soa-v1','mmvq-soa-v2','mmvq-soa-v3') or (sycl_experts!='serial-v1' and not packed_gu):raise ValueError('unadmitted warm SYCL expert path')
    if engram not in ('off','read-ahead-v1') or (engram!='off' and dense_owner!='hip-v1'):raise ValueError('Engram read-ahead requires HIP dense ownership')
    if dense_owner not in ("split-v1","hip-v1"):raise ValueError("unknown dense owner")
    if hip_schedule not in ('baseline-v1','queued-v1'):raise ValueError('unknown HIP schedule')
    if hip_fuse not in ('off','small-v1','attention-v1','all-v1') or (hip_fuse!='off' and (hip_schedule!='queued-v1' or hip_dense!='off')):raise ValueError('HIP fusion requires queued-v1 and graphs off')
    if hip_dense not in ('off','graphs-v1') or (hip_dense!='off' and (hip_schedule!='queued-v1' or kernel_timing)):raise ValueError('HIP graphs require queued-v1 and no per-kernel timing')
    if sycl_attention not in ('scalar-v1','subgroup32-v1'):raise ValueError('unadmitted warm SYCL attention path')
    return ('expert_placement=all-cpu\ncpu_threads=16\ncpu_dispatch=pool-v1\ncpu_timing='+str(int(cpu_timing))+'\n'
            'hip_experts=pooled-layer-v1\nhip_expert_math=per32-v1\n'
            'expert_profile='+str(profile)+'\nwarm_experts=1\nwarm_measure=1\nwarm_hip_bytes=0\nwarm_sycl_bytes=0\nwarm_hip_fixed='+str(warm_hip_fixed)+'\nwarm_sycl_fixed='+str(warm_sycl_fixed)+'\n'
            'warm_packed_gu='+str(int(packed_gu))+'\n'+
            ('warm_sycl_attention='+sycl_attention+'\n' if sycl_attention!='scalar-v1' else '')+
            ('warm_dense_owner='+dense_owner+'\n' if dense_owner!='split-v1' else '')+
            ('warm_hip_experts='+hip_experts+'\n' if hip_experts!='serial-v1' else '')+
            ('warm_expert_issue='+expert_issue+'\n' if expert_issue!='off' else '')+
            ('warm_expert_timeline_limit='+str(expert_timeline_limit)+'\n' if expert_timeline_limit else '')+
            ('warm_hip_swiglu='+hip_swiglu+'\n' if hip_swiglu!='off' else '')+
            ('warm_engram_priority='+engram_priority+'\n' if engram_priority!='off' else '')+
            ('warm_engram_trace_limit='+str(engram_trace_limit)+'\n' if engram_trace_limit else '')+
            ('warm_engram='+engram+'\n' if engram!='off' else '')+
            ('warm_hip_schedule='+hip_schedule+'\n' if hip_schedule!='baseline-v1' else '')+
            ('warm_kernel_timing=1\n' if kernel_timing else '')+
            ('warm_hip_gemv='+hip_gemv+'\n' if hip_gemv!='off' else '')+
            ('warm_hip_dense='+hip_dense+'\n' if hip_dense!='off' else '')+
            ('warm_hip_fuse='+hip_fuse+'\n' if hip_fuse!='off' else '')+
            ('warm_sycl_experts='+sycl_experts+'\n' if sycl_experts!='serial-v1' else ''))
