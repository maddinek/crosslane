// MIT; same arithmetic as pinned ds4_gguf q8_quant and ds4_iq mv_kernel.
// No fast math, reassociation or contraction; explicit FMA is intentional.
#include "hip_experts.hpp"
#include <hip/hip_fp16.h>
#include <cmath>
#include <algorithm>
#define HIPEX_FN __device__ __forceinline__
#include "hip_experts_core.hpp"
namespace strata::wiring::hipex {
namespace {
struct DeviceMath {
    __device__ static int dot4(uint32_t a,uint32_t b,int sum){
        // Upstream compatibility wrapper selects signed sudot4 on gfx1201.
        return __dp4a(int(a),int(b),sum);
    }
};
__global__ void __launch_bounds__(256) quant_kernel(const float* x,int k,int8_t* q,float* d){
    const int r=blockIdx.y,b=blockIdx.x*8+(threadIdx.x>>5),lane=threadIdx.x&31;
    if(b*32>=k)return;
    const float v=x[size_t(r)*k+size_t(b)*32+lane];float am=fabsf(v);
#pragma unroll
    for(int o=16;o>0;o>>=1)am=fmaxf(am,__shfl_xor(am,o,32));
    const float sc=am/127.0f;
    q[size_t(r)*k+size_t(b)*32+lane]=int8_t(am>0.0f?__float2int_rn(v/sc):0);
    if(lane==0)d[size_t(r)*(k/32)+b]=sc;
}
template<int Type> __global__ void __launch_bounds__(128) dot_kernel(Jobs jobs,const uint32_t* table){
    const auto b=jobs.j[blockIdx.y];
    const int row=int(blockIdx.x)*4+int(threadIdx.x>>5),lane=threadIdx.x&31;
    if(row>=b.n)return;
    float acc=0.0f;const int nsb=b.k/32;const size_t sub=size_t(b.n)*nsb;
    for(int kb=lane;kb<nsb;kb+=32){
        const auto v=subdot<Type,DeviceMath>(b.w,sub,size_t(row)*nsb+kb,b.x+size_t(kb)*32,table);
        const float scale=__half2float(__ushort_as_half(v.half_scale))*0.125f;
        const float dot=scale*float(v.integer); // rounded before the activation FMA
        acc=fmaf(b.scale[kb],dot,acc);
    }
#pragma unroll
    for(int o=16;o>0;o>>=1)acc+=__shfl_xor(acc,o,32);
    if(lane==0)b.y[row]=acc+0.0f; // upstream alpha=1, bias=null; preserve signed zero
}
}
bool quantize(const float* x,int rows,int k,int8_t* q,float* d,hipStream_t s){
    if(!x||!q||!d||rows<=0||k<=0||k%32)return false;
    hipLaunchKernelGGL(quant_kernel,dim3((k/32+7)/8,rows),dim3(256),0,s,x,k,q,d);
    return hipGetLastError()==hipSuccess;
}
bool project(Jobs a,const uint32_t* tables,hipStream_t s){
    if(a.count<=0||a.count>max_jobs||!tables)return false;
    for(int i=0;i<a.count;++i){const auto& j=a.j[i];
        if(!supported(j.type)||j.n<=0||j.k<=0||j.k%256||!j.w||!j.x||!j.scale||!j.y)return false;}
    // Mixed signatures are grouped by dtype; each job retains its own shape,
    // input and output. Every output is computed by exactly one warp.
    for(int type:{16,17,29}){Jobs group{};int rows=0;
        for(int i=0;i<a.count;++i)if(a.j[i].type==type){group.j[group.count++]=a.j[i];rows=std::max(rows,a.j[i].n);}
        if(!group.count)continue;
        const dim3 grid((rows+3)/4,group.count);
        if(type==16){hipLaunchKernelGGL(dot_kernel<0>,grid,dim3(128),0,s,group,tables);}
        else if(type==17){hipLaunchKernelGGL(dot_kernel<1>,grid,dim3(128),0,s,group,tables);}
        else {hipLaunchKernelGGL(dot_kernel<2>,grid,dim3(128),0,s,group,tables);}
        if(hipGetLastError()!=hipSuccess)return false;
    }
    return true;
}
}
