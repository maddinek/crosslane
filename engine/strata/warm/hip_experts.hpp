// MIT. Opt-in exact-order HIP SoA experts; no per-token allocation.
#pragma once
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
namespace strata::wiring::hipex {
constexpr int max_jobs=16; // two jobs per selected expert; fail closed above this bound
struct Job {const uint8_t* w=nullptr;int type=0,n=0,k=0;const int8_t* x=nullptr;const float* scale=nullptr;float* y=nullptr;};
struct Jobs {Job j[max_jobs]{};int count=0;};
inline bool supported(int type){return type==16||type==17||type==29;}
bool quantize(const float* x,int rows,int k,int8_t* q,float* d,hipStream_t stream);
bool project(Jobs jobs,const uint32_t* tables,hipStream_t stream);
// Optional checker-only phase boundaries. Null in serving; callbacks enqueue
// events on the same stream, never synchronize or change arithmetic.
using Mark=void(*)(void*,const char*,hipStream_t);
struct Trace {Mark mark=nullptr;void* state=nullptr;
    void operator()(const char* name,hipStream_t s)const {if(mark)mark(state,name,s);}};
// Enabled only around checker calls. The original q8 kernel bodies and launch
// helper are compiled verbatim; these marks separate quantization from dots.
inline thread_local Trace primitive_trace{};
}
