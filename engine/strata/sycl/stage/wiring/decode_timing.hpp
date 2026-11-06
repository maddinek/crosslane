// MIT; opt-in decode diagnostics. Host waits overlap device spans: never add them.
#pragma once
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace strata::wiring {
struct CpuPhases {
    uint64_t completed_experts=0, logical_read_bytes=0, read_ns=0, quant1_ns=0,
        gate_up_ns=0, quant2_ns=0, down_ns=0, expert_ns=0,
        row_thread_ns=0, dispatch_outside_envelope_ns=0, worker_completion_wait_ns=0;
    CpuPhases operator-(const CpuPhases& b) const {
        return {completed_experts-b.completed_experts,logical_read_bytes-b.logical_read_bytes,
            read_ns-b.read_ns,quant1_ns-b.quant1_ns,gate_up_ns-b.gate_up_ns,
            quant2_ns-b.quant2_ns,down_ns-b.down_ns,expert_ns-b.expert_ns,
            row_thread_ns-b.row_thread_ns,dispatch_outside_envelope_ns-b.dispatch_outside_envelope_ns,
            worker_completion_wait_ns-b.worker_completion_wait_ns};
    }
};
inline uint64_t decode_clock_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
enum class DecodePart { attention_shared, resident, boundary, other };
struct DecodeTiming {
    bool active=false;
    uint64_t request=0,position=0; int32_t input=0;
    uint64_t attention_shared_ns=0,resident_ns=0,boundary_ns=0,other_ns=0,sync_wait_ns=0,spans=0,resident_experts=0;
    uint64_t engram_read_ns=0,embedding_read_ns=0,routing_combine_host_ns=0,route_io_wall_ns=0,publication_wall_ns=0;
    uint64_t stamp()const{return active?decode_clock_ns():0;}
    uint64_t elapsed(uint64_t tick)const{return active?decode_clock_ns()-tick:0;}
    template<class F>void host(uint64_t& field,F f){const auto tick=stamp();f();field+=elapsed(tick);}
    void begin(int64_t start,uint64_t req,uint64_t pos,int32_t id) {
        *this={}; active=start>=0 && pos>=uint64_t(start);
        if(active){request=req;position=pos;input=id;}
    }
    template<class F> void wait(F f) {
        if(!active){f();return;}
        const auto tick=decode_clock_ns();f();sync_wait_ns+=decode_clock_ns()-tick;
    }
    void add(DecodePart p,uint64_t ns) {
        if(p==DecodePart::attention_shared)attention_shared_ns+=ns;
        else if(p==DecodePart::resident){resident_ns+=ns;++resident_experts;}
        else if(p==DecodePart::boundary)boundary_ns+=ns;
        else other_ns+=ns;
        ++spans;
    }
    void emit(const char* backend,uint32_t first,uint32_t end,const CpuPhases& c) const {
        if(!active)return;
        // Fixed keys and bounded scalar fields: < 2 KiB per stage/step, no lifetime totals.
        const int n=std::fprintf(stderr,
            "AIOPT_DECODE_TIMING {\"schema\":\"strata-decode-phases-v1\",\"backend\":\"%s\",\"request\":%llu,\"position\":%llu,\"input\":%d,\"first\":%u,\"end\":%u,\"completed_experts\":%llu,\"logical_read_bytes\":%llu,\"read_ns\":%llu,\"quant1_ns\":%llu,\"gate_up_ns\":%llu,\"quant2_ns\":%llu,\"down_ns\":%llu,\"expert_ns\":%llu,\"row_thread_ns\":%llu,\"dispatch_outside_envelope_ns\":%llu,\"worker_completion_wait_ns\":%llu,\"attention_shared_event_span_ns\":%llu,\"resident_event_span_ns\":%llu,\"boundary_event_span_ns\":%llu,\"other_event_span_ns\":%llu,\"explicit_sync_wait_ns\":%llu,\"event_spans\":%llu,\"resident_experts\":%llu,\"engram_read_ns\":%llu,\"embedding_read_ns\":%llu,\"routing_combine_host_ns\":%llu,\"route_io_wall_ns\":%llu,\"publication_wall_ns\":%llu}\n",
            backend,(unsigned long long)request,(unsigned long long)position,input,first,end,
            (unsigned long long)c.completed_experts,(unsigned long long)c.logical_read_bytes,
            (unsigned long long)c.read_ns,(unsigned long long)c.quant1_ns,(unsigned long long)c.gate_up_ns,
            (unsigned long long)c.quant2_ns,(unsigned long long)c.down_ns,(unsigned long long)c.expert_ns,
            (unsigned long long)c.row_thread_ns,(unsigned long long)c.dispatch_outside_envelope_ns,(unsigned long long)c.worker_completion_wait_ns,
            (unsigned long long)attention_shared_ns,(unsigned long long)resident_ns,
            (unsigned long long)boundary_ns,(unsigned long long)other_ns,
            (unsigned long long)sync_wait_ns,(unsigned long long)spans,(unsigned long long)resident_experts,
            (unsigned long long)engram_read_ns,(unsigned long long)embedding_read_ns,
            (unsigned long long)routing_combine_host_ns,(unsigned long long)route_io_wall_ns,(unsigned long long)publication_wall_ns);
        if(n<=0)throw std::runtime_error("decode timing write failed");
    }
};
// Clock owns two reusable native events. No event allocation, submit, wait or query when off.
template<class Clock> class DecodeEvents {
    DecodeTiming& timing_; std::unique_ptr<Clock> clock_;
    DecodePart part_{}; bool open_=false;
public:
    explicit DecodeEvents(DecodeTiming& t):timing_(t){}
    void abandon() noexcept {open_=false;} // caller drains before retry/reusing the events
    template<class Queue> void begin(DecodePart part,Queue& q) {
        if(!timing_.active)return;
        if(open_)throw std::runtime_error("nested decode event span");
        if(!clock_)clock_=std::make_unique<Clock>();
        part_=part;clock_->begin(q);open_=true;
    }
    template<class Queue> void end(Queue& q) {
        if(!timing_.active)return;
        if(!open_)throw std::runtime_error("missing decode event span");
        timing_.wait([&]{clock_->end(q);});
        timing_.add(part_,clock_->elapsed_ns());open_=false;
    }
};
} // namespace strata::wiring
