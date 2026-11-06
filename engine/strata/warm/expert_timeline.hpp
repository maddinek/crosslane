// MIT. Service intervals on one steady clock; HIP callbacks mark completed work.
#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
namespace strata::wiring {
struct ExpertTimeline {
    uint64_t request=0,position=0;uint32_t layer=0;
    bool active=false,early=false;
    std::atomic<uint64_t> route_ready{0},shared_end{0};
    std::array<uint64_t,3> start{},end{};
    std::array<size_t,3> counts{};
    uint64_t combine_start=0;
    static uint64_t now() noexcept {
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }
    void begin(uint64_t r,uint64_t p,uint32_t l,bool enabled,bool issue) {
        request=r;position=p;layer=l;active=enabled;early=issue;
        route_ready=0;shared_end=0;start={};end={};counts={};combine_start=0;
    }
    // No HIP calls, allocation or I/O on the runtime callback thread.
    static void route(void* p) {static_cast<ExpertTimeline*>(p)->route_ready.store(now(),std::memory_order_release);}
    static void shared(void* p) {static_cast<ExpertTimeline*>(p)->shared_end.store(now(),std::memory_order_release);}
    void emit() const {
        if(!active)return;
        const auto origin=route_ready.load(std::memory_order_acquire);
        auto relative=[&](uint64_t n){return (unsigned long long)(n?n-origin:0);};
        std::fprintf(stderr,"AIOPT_EXPERT_TIMELINE {\"request\":%llu,\"position\":%llu,\"layer\":%u,\"mode\":\"%s\",\"route_ready_ns\":%llu,\"shared_end_ns\":%llu,\"combine_start_ns\":%llu,\"counts\":[%zu,%zu,%zu],\"start_ns\":[%llu,%llu,%llu],\"end_ns\":[%llu,%llu,%llu]}\n",
            (unsigned long long)request,(unsigned long long)position,layer,early?"early-v1":"off",
            (unsigned long long)origin,relative(shared_end.load(std::memory_order_acquire)),relative(combine_start),
            counts[0],counts[1],counts[2],relative(start[0]),relative(start[1]),relative(start[2]),
            relative(end[0]),relative(end[1]),relative(end[2]));
    }
};
}
