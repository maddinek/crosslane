// MIT. Calls the same finish object as the SYCL stage, no Model or weights.
#include "numerics.hpp"
#include "device.hpp"
#include "fp8_control_io.hpp"
#include <cstdio>
int main(int argc,char** argv) try {
    auto x=fp8_input(argc,argv);
    sycl::queue q(strata::intel_level_zero_device(0),sycl::property::queue::in_order{});
    float* p=sycl::malloc_device<float>(x.size(),q); if(!p)throw std::runtime_error("allocation failed");
    q.memcpy(p,x.data(),x.size()*4).wait_and_throw();
    strata::ds4::sycl_port::Finish f; f.quant=1;
    strata::ds4::sycl_port::finish(q,p,int(x.size()/512),512,512,f).wait_and_throw();
    q.memcpy(x.data(),p,x.size()*4).wait_and_throw();
    fp8_output(argv[2],x); sycl::free(p,q); return 0;
} catch(const std::exception& e){std::fprintf(stderr,"FP8 control: %s\n",e.what());return 1;}
