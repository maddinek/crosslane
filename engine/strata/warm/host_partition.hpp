// MIT. Decode-only exclusive host wall partition; no device events or waits.
#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
namespace strata::wiring {
enum class HostPart : size_t { other, sample, runtime, embedding, dense, prepare, submit, join, combine, completion, head, planner, readback, output, result_h2d, ffn_wait, sample_readback, count };
class HostPartition {
 static constexpr size_t parts=size_t(HostPart::count);
 static constexpr const char* names[parts]={"host_other","sample","runtime_other","embedding","dense","prepare","submit","join","combine","completion","head","planner","readback","output","result_h2d","ffn_wait","sample_readback"};
 struct Row {uint32_t position,layer,n;std::array<int,32> ids{};std::array<float,32> weights{};
             std::array<uint32_t,3> counts{},batches{};std::array<uint64_t,3> service{};uint64_t join;};
 bool enabled_=false,active_=false;size_t limit_=0,layers_=0;std::string prefix_;
 uint64_t request_=0,seed_=0,prompt_=0,start_=0,last_=0,position_=0,runtime_ns_=0;
 HostPart part_=HostPart::other;
 std::array<std::array<uint64_t,parts>,2> ns_{};
 std::vector<Row> rows_;
 static void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
 void account(uint64_t tick){need(tick>=last_,"host clock reversed");ns_[(prompt_+position_)%2][size_t(part_)]+=tick-last_;last_=tick;}
public:
 static uint64_t now(){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
 // Explicit host activation: default-off does not read a clock or allocate.
 void configure(size_t layers){
  const char* flag=std::getenv("DS4_WARM_HOST_PARTITION");
  need(!flag||std::string(flag)=="1","host partition must be unset or 1");enabled_=flag;
  const char* prefix=std::getenv("DS4_WARM_ORDERED_PREFIX");
  const char* count=std::getenv("DS4_WARM_ORDERED_LIMIT");
  need(bool(prefix)==bool(count),"ordered prefix/limit must be paired");
  if(prefix){char* end=nullptr;auto n=std::strtoul(count,&end,10);
   need(prefix[0]=='/'&&end&&!*end&&n>0&&n<=2000&&layers>0&&layers<=4096&&layers*n<=131072,"ordered trace bound");
   limit_=n;layers_=layers;prefix_=prefix;rows_.reserve(layers*n);
  }
 }
 bool on()const{return enabled_||limit_;}
 bool active()const{return active_&&enabled_;}
 bool recording()const{return active_&&limit_&&position_<limit_;}
 void begin(uint64_t request,uint64_t seed,uint64_t prompt,uint64_t tick){
  if(!on())return;need(!active_,"nested decode partition");active_=true;request_=request;seed_=seed;prompt_=prompt;
  start_=last_=tick;position_=0;runtime_ns_=0;part_=HostPart::other;ns_={};rows_.clear();
 }
 void position(uint64_t p){if(!on()||!active_)return;if(enabled_)account(now());position_=p;}
 HostPart enter(HostPart next){auto old=part_;if(active()){account(now());part_=next;}return old;}
 void leave(HostPart old){if(active()){account(now());part_=old;}}
 void runtime(uint64_t ns){if(active())runtime_ns_+=ns;}
 void record(uint32_t layer,std::span<const int> ids,std::span<const float> weights,
             std::array<uint32_t,3> counts,std::array<uint32_t,3> batches,
             std::array<uint64_t,3> service,uint64_t join){
  if(!active_||!limit_||position_>=limit_)return;
  need(layer<layers_&&rows_.size()==position_*layers_+layer&&ids.size()<=32&&ids.size()>0&&ids.size()==weights.size(),"ordered trace extent/order");
  need(counts[0]+counts[1]+counts[2]==ids.size(),"ordered trace lane census");
  Row r{};r.position=uint32_t(position_);r.layer=layer;r.n=uint32_t(ids.size());r.counts=counts;r.batches=batches;r.service=service;r.join=join;
  for(size_t i=0;i<ids.size();++i){need(ids[i]>=0&&std::isfinite(weights[i]),"ordered trace route");r.ids[i]=ids[i];r.weights[i]=weights[i];}
  rows_.push_back(r);
 }
 // Called with the SAME end tick used by END. Formatting/I/O follows timing.
 void end(uint64_t tick,uint64_t outputs,uint64_t steps){
  if(!on())return;need(active_&&tick>=start_,"host partition end");if(enabled_)account(tick);active_=false;
  if(enabled_){uint64_t sum=0;for(auto& parity:ns_)for(auto v:parity)sum+=v;need(sum==tick-start_,"host partition does not sum to wall");
   std::fprintf(stderr,"AIOPT_HOST_PARTITION {\"request\":%llu,\"seed\":%llu,\"prompt_tokens\":%llu,\"outputs\":%llu,\"decode_steps\":%llu,\"wall_ns\":%llu,\"runtime_inclusive_ns\":%llu,\"parity_ns\":[",(unsigned long long)request_,(unsigned long long)seed_,(unsigned long long)prompt_,(unsigned long long)outputs,(unsigned long long)steps,(unsigned long long)sum,(unsigned long long)runtime_ns_);
   for(size_t p=0;p<2;++p){std::fprintf(stderr,"%s{",p?",":"");for(size_t k=0;k<parts;++k)std::fprintf(stderr,"%s\"%s\":%llu",k?",":"",names[k],(unsigned long long)ns_[p][k]);std::fprintf(stderr,"}");}std::fprintf(stderr,"]}\n");
  }
  if(limit_){need(rows_.size()==size_t(std::min<uint64_t>(steps,limit_))*layers_,"incomplete ordered trace");
   auto path=prefix_+"-request-"+std::to_string(request_)+".jsonl";auto f=std::fopen(path.c_str(),"wx");need(f,"ordered trace output exists/open failed");
   for(const auto& r:rows_){std::fprintf(f,"{\"request\":%llu,\"seed\":%llu,\"position\":%u,\"model_position\":%llu,\"layer\":%u,\"ids\":[",(unsigned long long)request_,(unsigned long long)seed_,r.position,(unsigned long long)(prompt_+r.position),r.layer);
    for(size_t i=0;i<r.n;++i)std::fprintf(f,"%s%d",i?",":"",r.ids[i]);std::fprintf(f,"],\"weights\":[");
    for(size_t i=0;i<r.n;++i)std::fprintf(f,"%s%.9g",i?",":"",double(r.weights[i]));
    auto ints=[&](const char* name,const auto& values){std::fprintf(f,"],\"%s\":[",name);for(size_t i=0;i<3;++i)std::fprintf(f,"%s%llu",i?",":"",(unsigned long long)values[i]);};
    ints("counts",r.counts);ints("batches",r.batches);ints("service_ns",r.service);std::fprintf(f,"],\"join_ns\":%llu}\n",(unsigned long long)r.join);
   }bool ok=!std::ferror(f);ok=(std::fclose(f)==0)&&ok;need(ok,"ordered trace write failed");
   std::fprintf(stderr,"AIOPT_ORDERED_TRACE request=%llu records=%zu positions=%zu limit=%zu lanes=cpu,sycl,hip\n",(unsigned long long)request_,rows_.size(),rows_.size()/layers_,limit_);
  }
 }
};
struct HostScope {
 HostPartition* p;HostPart old;uint64_t start=0;bool runtime_;
 HostScope(HostPartition* host,HostPart part):p(host),old(p?p->enter(part):HostPart::other),runtime_(part==HostPart::runtime){if(p&&p->active()&&runtime_)start=HostPartition::now();}
 ~HostScope(){if(p){if(start)p->runtime(HostPartition::now()-start);p->leave(old);}}
};
extern "C" HostPartition* strata_warm_host_partition_v1();
}
