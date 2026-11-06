// MIT; crosslane contributors. Owned, counted HIP allocations; CPU-tested via transport shim.
#pragma once
#include "hip_resident.hpp"
#include <hip/hip_runtime.h>

namespace strata::wiring {
class HipArena {
    struct Allocation { void* pointer; uint64_t bytes; };
    int device_; hipStream_t stream_;
    std::vector<Allocation> pointers_;
    static void check(hipError_t error) { require(error==hipSuccess,"HIP arena API failure"); }
public:
    uint64_t bytes=0;
    bool release_ok=true;
    HipArena(int d,hipStream_t s):device_(d),stream_(s){}
    ~HipArena(){clear();}
    uint64_t driver_used()const {
        size_t free=0,total=0;check(hipMemGetInfo(&free,&total));
        require(free<=total,"HIP driver memory extent");return total-free;
    }
    template<class T>T* dev(size_t n){
        require(n>0,"zero-size device allocation");
        const auto b=expert_align(strata::ds4::moe::product(n,sizeof(T)));
        const auto charged=stage::add(bytes,b);
        check(hipSetDevice(device_));
        // Reserve the owner record BEFORE hipMalloc, so a host allocation
        // exception cannot strand a device allocation (including failed free).
        pointers_.push_back({nullptr,b});
        void* p=nullptr;const auto error=hipMalloc(&p,size_t(b));
        if(error!=hipSuccess){pointers_.pop_back();check(error);}
        pointers_.back().pointer=p;
        bytes=charged;
        check(hipMemsetAsync(p,0,size_t(b),stream_));check(hipStreamSynchronize(stream_));
        return static_cast<T*>(p);
    }
    template<class T>T* upload(const std::vector<T>& v){
        auto* p=dev<T>(v.size());
        check(hipMemcpyAsync(p,v.data(),v.size()*sizeof(T),hipMemcpyHostToDevice,stream_));
        check(hipStreamSynchronize(stream_));return p;
    }
    bool drain()noexcept{
        release_ok=hipSetDevice(device_)==hipSuccess && hipStreamSynchronize(stream_)==hipSuccess;
        return release_ok;
    }
    void clear()noexcept{
        if(!drain())return; // keep owners and counters if outstanding work cannot be drained
        size_t keep=0;uint64_t remaining=0;
        for(auto a:pointers_){
            if(hipFree(a.pointer)!=hipSuccess){pointers_[keep++]=a;remaining+=a.bytes;release_ok=false;}
        }
        pointers_.resize(keep);bytes=remaining;
    }
};
} // namespace strata::wiring
