// MIT. Calls the exact pinned upstream finish implementation, no Model or weights.
#include "ds4_kernels.cuh"
#include <hip/hip_runtime.h>
#include "fp8_control_io.hpp"
#include <cstdio>
static void ck(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
int main(int argc,char** argv) try {
    auto x=fp8_input(argc,argv); int count=0; ck(hipGetDeviceCount(&count));
    if(count!=1)throw std::runtime_error("one selected HIP device required");
    ck(hipSetDevice(0)); float* p=nullptr; void* reserve=nullptr;
    ck(hipMalloc(&reserve,128*1024*1024)); ck(hipMemset(reserve,0,128*1024*1024));
    ck(hipMalloc(reinterpret_cast<void**>(&p),x.size()*4));
    ck(hipMemcpy(p,x.data(),x.size()*4,hipMemcpyHostToDevice));
    strata::ds4::Finish f; f.quant=1; // rd=0, no norm: native quantizer control only.
    strata::ds4::finish(p,int(x.size()/512),512,512,f,nullptr);
    ck(hipDeviceSynchronize()); ck(hipMemcpy(x.data(),p,x.size()*4,hipMemcpyDeviceToHost));
    fp8_output(argv[2],x); ck(hipFree(p)); ck(hipFree(reserve)); return 0;
} catch(const std::exception& e){std::fprintf(stderr,"FP8 control: %s\n",e.what());return 1;}
