// Row2 diagnostic primitive (additive; P1 wiring signatures remain unchanged).
#pragma once
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
namespace strata::ds4::moe {
// q8_quant_kernel's bytes and F32 scales, compact [T,k] and [T,k/32].
sycl::event quantize(sycl::queue&,const float* x,size_t stride,int k,int T,int8_t* q,float* d);
}
