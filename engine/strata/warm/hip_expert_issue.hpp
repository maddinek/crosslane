// MIT. Reusable route fence; only the coordinator submits until shared drains.
#pragma once
#include "expert_timeline.hpp"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cstdlib>
namespace strata::wiring {
class HipExpertIssue {
    hipEvent_t ready_{};
    hipStream_t stream_{};
    bool pending_=false;
    static void check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error("HIP expert issue fence failed");}
public:
    ExpertTimeline timeline;
    HipExpertIssue(){check(hipEventCreateWithFlags(&ready_,hipEventDisableTiming));}
    HipExpertIssue(const HipExpertIssue&)=delete;
    void drain() noexcept {
        if(pending_){
            // An uncertain drain cannot release callback/readback owners. Fail
            // the process closed rather than unwind into a use-after-free.
            if(hipStreamSynchronize(stream_)!=hipSuccess)std::abort();
            pending_=false;
        }
    }
    ~HipExpertIssue(){drain();hipEventDestroy(ready_);}
    void begin(hipStream_t s,uint64_t r,uint64_t p,uint32_t l,bool trace,bool early){
        drain();stream_=s;timeline.begin(r,p,l,trace,early);pending_=true;
    }
    void route(){
        if(timeline.active)check(hipLaunchHostFunc(stream_,ExpertTimeline::route,&timeline));
        check(hipEventRecord(ready_,stream_));
    }
    void shared(){if(timeline.active)check(hipLaunchHostFunc(stream_,ExpertTimeline::shared,&timeline));}
    void wait_route(){check(hipEventSynchronize(ready_));}
    // Caller has successfully synchronized the stream; all callbacks are done.
    void completed() noexcept {pending_=false;}
};
struct HipIssueGuard {
    HipExpertIssue* issue;
    ~HipIssueGuard(){if(issue)issue->drain();}
};
}
