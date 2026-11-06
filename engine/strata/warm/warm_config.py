"""Warm stage configuration shared by research and standalone entrypoints."""
SYCL_FIXED_BYTES=3*2**30

def config(profile,packed_gu=False,sycl_attention='scalar-v1',dense_owner='split-v1',hip_schedule='baseline-v1',kernel_timing=False,sycl_experts='serial-v1',hip_dense='off'):
    if sycl_experts not in ('serial-v1','grouped-v1','packed-strip-v1','mmvq-soa-v1') or (sycl_experts!='serial-v1' and not packed_gu):raise ValueError('unadmitted warm SYCL expert path')
    if dense_owner not in ("split-v1","hip-v1"):raise ValueError("unknown dense owner")
    if hip_schedule not in ('baseline-v1','queued-v1'):raise ValueError('unknown HIP schedule')
    if hip_dense not in ('off','graphs-v1') or (hip_dense!='off' and (hip_schedule!='queued-v1' or kernel_timing)):raise ValueError('HIP graphs require queued-v1 and no per-kernel timing')
    if sycl_attention not in ('scalar-v1','subgroup32-v1'):raise ValueError('unadmitted warm SYCL attention path')
    return ('expert_placement=all-cpu\ncpu_threads=16\ncpu_dispatch=pool-v1\ncpu_timing=0\n'
            'hip_experts=pooled-layer-v1\nhip_expert_math=per32-v1\n'
            'expert_profile='+str(profile)+'\nwarm_experts=1\nwarm_measure=1\nwarm_hip_bytes=0\nwarm_sycl_bytes=0\nwarm_hip_fixed=1529365504\nwarm_sycl_fixed='+str(SYCL_FIXED_BYTES)+'\n'
            'warm_packed_gu='+str(int(packed_gu))+'\n'+
            ('warm_sycl_attention='+sycl_attention+'\n' if sycl_attention!='scalar-v1' else '')+
            ('warm_dense_owner='+dense_owner+'\n' if dense_owner!='split-v1' else '')+
            ('warm_hip_schedule='+hip_schedule+'\n' if hip_schedule!='baseline-v1' else '')+
            ('warm_kernel_timing=1\n' if kernel_timing else '')+
            ('warm_hip_dense='+hip_dense+'\n' if hip_dense!='off' else '')+
            ('warm_sycl_experts='+sycl_experts+'\n' if sycl_experts!='serial-v1' else ''))

