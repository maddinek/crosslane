// MIT. Reuses the accepted native event clocks; spans include queue-idle gaps.
#pragma once
#include <memory>
#include <cstdint>
#include <vector>
namespace strata::wiring {
template<class Clock>struct WarmDeviceTiming {
 std::unique_ptr<Clock> clock;std::vector<double> ns;std::vector<uint64_t> calls;
 explicit WarmDeviceTiming(size_t layers):ns(layers+1,0),calls(layers+1,0){}
 template<class Queue>void begin(Queue& q){if(!clock)clock=std::make_unique<Clock>();clock->begin(q);}
 template<class Queue>uint64_t end(size_t layer,Queue& q){clock->end(q);auto n=clock->elapsed_ns();ns.at(layer)+=double(n);++calls.at(layer);return n;}
 template<class Queue>void mark_end(Queue& q){clock->mark_end(q);}
 uint64_t collect(size_t layer){auto n=clock->elapsed_ns();ns.at(layer)+=double(n);++calls.at(layer);return n;}
 std::vector<double> means()const{auto result=ns;for(size_t l=0;l<result.size();++l)result[l]=calls[l]?result[l]/double(calls[l]):0;return result;}
};
}
