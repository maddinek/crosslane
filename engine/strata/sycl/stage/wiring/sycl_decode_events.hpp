// In-order profiling queue markers delimit spans, including queue-idle gaps between markers.
#pragma once
#include "decode_timing.hpp"
#include <sycl/sycl.hpp>
namespace strata::wiring {
class SyclDecodeClock {
    sycl::event first_,last_;
public:
    void begin(sycl::queue& q){first_=q.single_task([]{});}
    void mark_end(sycl::queue& q){last_=q.single_task([]{});}
    void end(sycl::queue& q){mark_end(q);last_.wait_and_throw();}
    uint64_t elapsed_ns(){
        const auto a=first_.get_profiling_info<sycl::info::event_profiling::command_end>();
        const auto b=last_.get_profiling_info<sycl::info::event_profiling::command_start>();
        if(b<a)throw std::runtime_error("invalid SYCL decode event duration");
        return b-a;
    }
};
}
