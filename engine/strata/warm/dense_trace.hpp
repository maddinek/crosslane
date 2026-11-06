// MIT. Bounded native launch spans. Read only after the caller's existing dense-end wait.
// A batched API can launch several kernels; spans include dispatch/queue-idle gaps.
#pragma once
#include <cstdio>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>
#include <stdexcept>
namespace strata::wiring {
template<class Clock>class DenseTrace {
 struct Span {std::unique_ptr<Clock> clock;const char* name=nullptr;};
 std::vector<Span> spans_;size_t used_=0;bool active_=false;
 uint64_t request_=0,position_=0;uint32_t layer_=0;
public:
 void begin(bool enabled,uint64_t request,uint32_t layer,uint64_t position){
  used_=0;request_=request;layer_=layer;position_=position;
  active_=enabled&&(position==31||position==127||position==324||(position>=1024&&position%1024==0));
 }
 template<class Queue,class F>auto run(const char* name,Queue& q,F fn){
  if(!active_)return fn();
  if(used_>=128)throw std::runtime_error("dense trace exceeds 128 launch spans");
  if(used_==spans_.size())spans_.push_back({std::make_unique<Clock>(),nullptr});
  auto& span=spans_[used_++];span.name=name;span.clock->begin(q);
  if constexpr(std::is_void_v<std::invoke_result_t<F>>){fn();span.clock->mark_end(q);}
  else {auto value=fn();span.clock->mark_end(q);return value;}
 }
 void collect(const char* backend){
  if(!active_)return;
  for(size_t i=0;i<used_;++i)std::fprintf(stderr,"AIOPT_WARM_KERNEL request=%llu backend=%s layer=%u position=%llu index=%zu name=%s event_ns=%llu scope=launch-span-including-queue-gaps\n",
   (unsigned long long)request_,backend,layer_,(unsigned long long)position_,i,spans_[i].name,(unsigned long long)spans_[i].clock->elapsed_ns());
  active_=false;
 }
};
}
