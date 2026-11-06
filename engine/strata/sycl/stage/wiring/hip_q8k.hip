// MIT; first CPU-arithmetic resident HIP kernel. Build this TU without fast math.
#include <hip/hip_runtime.h>
#include "hip_q8k.hpp"
#define Q8K_FN __device__ __forceinline__
#define Q8K_TABLE static __device__ __constant__ const
#include "q8k_core.hpp"
namespace strata::wiring::q8k {
__global__ void quant_kernel(const float* x,Block* q,int nb){
    const int b=int(blockIdx.x*blockDim.x+threadIdx.x);if(b<nb)quant_block(x+256*b,q[b]);
}
__global__ void dot_kernel(int type,const uint8_t* w,const Block* q,float* y,int n,int nb){
    const int tid=int(blockIdx.x*blockDim.x+threadIdx.x),row=tid/8,lane=tid%8;
    // Full subgroups execute shuffles; last rows of the final group are masked.
    Sums s;if(row<n)s=dot_lane(type,w+size_t(row)*nb*block_bytes(type),q,nb,lane);
    for(int d=4;d;d/=2){s.main+=__shfl_down(s.main,d,8);s.delta+=__shfl_down(s.delta,d,8);}
    if(lane==0&&row<n)y[row]=type==29?s.main+0.125f*s.delta:0.125f*s.main;
}
__global__ void activate_kernel(const float* g,const float* u,float* h,int n,float limit){
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);if(i<n)h[i]=swiglu(g[i],u[i],limit);
}
bool supported_type(int t){return t==16||t==17||t==29;}
size_t activation_bytes(int k){return k>0&&k%256==0?size_t(k/256)*sizeof(Block):0;}
bool quantize(const float* x,void* q,int k,hipStream_t s){
    if(!x||!q||!activation_bytes(k))return false;
    hipLaunchKernelGGL(quant_kernel,dim3((k/256+63)/64),dim3(64),0,s,x,static_cast<Block*>(q),k/256);
    return hipGetLastError()==hipSuccess;
}
bool mv(int t,const void* w,const void* q,float* y,int n,int k,hipStream_t s){
    if(!supported_type(t)||!w||!q||!y||n<=0||!activation_bytes(k))return false;
    hipLaunchKernelGGL(dot_kernel,dim3((n+15)/16),dim3(128),0,s,t,static_cast<const uint8_t*>(w),static_cast<const Block*>(q),y,n,k/256);
    return hipGetLastError()==hipSuccess;
}
bool activate(const float* g,const float* u,float* h,int n,float limit,hipStream_t s){
    if(!g||!u||!h||n<=0||!std::isfinite(limit))return false;
    hipLaunchKernelGGL(activate_kernel,dim3((n+127)/128),dim3(128),0,s,g,u,h,n,limit);
    return hipGetLastError()==hipSuccess;
}
}
