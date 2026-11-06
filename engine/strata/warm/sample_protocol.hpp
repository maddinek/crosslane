// MIT. No device dependency: used by the real host and CPU schedule checks.
#pragma once
#include <cstdint>
#include <istream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace strata::forkwarm {
struct SampleProtocol {
 bool sampled=false;
 uint64_t measured=3,max_tokens=8192;
 static constexpr uint64_t context=12288,prompt_tokens=325,checkpoint=8192;
 static constexpr uint64_t host_bound=context-prompt_tokens+1;
 uint64_t request_count()const{return measured+1;}
 void check(bool ok,const char* why)const{if(!ok)throw std::runtime_error(why);}
 static SampleProtocol quality(uint64_t n,uint64_t cap){
  SampleProtocol p{true,n,cap};
  p.check(n>=1&&n<=1000,"quality samples must be in 1..1000");
  p.check(cap>=checkpoint&&cap<=host_bound,"quality cap must be in 8192..11964");return p;
 }
 struct Request {uint64_t seed,counter_start,max_tokens;std::vector<int32_t> ids;};
 std::vector<Request> read(std::istream& in)const{
  std::vector<Request> rows;
  for(uint64_t i=0;i<request_count();++i){
   Request r{};uint64_t count=0;
   check(bool(in>>r.seed),"short warm request inventory");
   if(sampled)check(bool(in>>r.counter_start)&&r.counter_start==0,"quality counter start must be frozen zero");
   check(bool(in>>r.max_tokens>>count)&&r.seed==1000+i&&count==prompt_tokens&&r.max_tokens==max_tokens,"noncanonical warm schedule");
   check(count+r.max_tokens-1<=context,"warm generation exceeds context");
   r.ids.resize(count);for(auto& id:r.ids)check(bool(in>>id)&&id>=0,"short or negative warm IDs");
   rows.push_back(std::move(r));
  }
  in>>std::ws;check(in.eof(),"extra warm requests or trailing schedule bytes");return rows;
 }
};
}
