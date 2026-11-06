// HIP stream-local timing events; distinct from transport's disable-timing completion events.
#pragma once
#include "decode_timing.hpp"
#include <hip/hip_runtime.h>
#include <cmath>
namespace strata::wiring {
class HipDecodeClock {
    hipEvent_t first_{},last_{};
    static void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error("HIP decode timing event failed");}
public:
    HipDecodeClock(){
        check(hipEventCreate(&first_));
        try{check(hipEventCreate(&last_));}catch(...){hipEventDestroy(first_);throw;}
    }
    ~HipDecodeClock(){hipEventDestroy(last_);hipEventDestroy(first_);}
    void begin(hipStream_t& s){check(hipEventRecord(first_,s));}
    void mark_end(hipStream_t& s){check(hipEventRecord(last_,s));}
    void end(hipStream_t& s){mark_end(s);check(hipEventSynchronize(last_));}
    uint64_t elapsed_ns(){
        float ms=0;check(hipEventElapsedTime(&ms,first_,last_));
        if(!std::isfinite(ms)||ms<0)throw std::runtime_error("invalid HIP decode event duration");
        return uint64_t(std::llround(double(ms)*1e6));
    }
};
}
