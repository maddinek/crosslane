// MIT; opt-in resident arithmetic API. No ownership or implicit allocation.
#pragma once
#include <cstddef>
namespace strata::wiring::q8k {
// hipStream_t is provided by the caller's HIP runtime (or host contract shim).
bool supported_type(int type);
size_t activation_bytes(int k);
bool quantize(const float* x,void* q,int k,hipStream_t stream);
bool mv(int type,const void* w,const void* q,float* y,int n,int k,hipStream_t stream);
bool activate(const float* g,const float* u,float* h,int n,float limit,hipStream_t stream);
}
