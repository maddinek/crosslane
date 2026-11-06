// MIT. Included after MmvqFixture. Separate v3 grid leaves legacy grids intact.
// Timings are paired on the same losslessly reordered tensors and activations.
#pragma once
static bool mmvq_v3_profile=true;
struct MmvqV3Timing {
 uint64_t host,span,h2d,input,gate_up,swiglu,hidden_quant,down,d2h;
};
static MmvqV3Timing mmvq_v3_run(MmvqFixture& f,int count,bool candidate,const std::vector<float>& x,float lim){
 auto& q=f.q;auto& w=*f.w;auto a=f.descriptor(count,true);
 std::vector<float> out(size_t(count)*w.dim);
 try{
  const auto begin=now();auto start=q.single_task([]{});
  auto h2d=q.memcpy(w.x,x.data(),x.size()*sizeof(float));sycl::event swiglu;
  auto e=candidate?mq::v3::enqueue(q,w,a,count,f.refs[0]->a[0].type,f.refs[0]->a[2].type,lim,f.table.p):
   mq::enqueue(q,w,a,count,f.refs[0]->a[0].type,f.refs[0]->a[2].type,lim,f.table.p,true,&swiglu);
  auto d2h=q.memcpy(out.data(),w.y,out.size()*sizeof(float));
  auto end=q.single_task([]{});end.wait_and_throw();const auto host=now()-begin;
  for(float v:out)need(std::isfinite(v),"MMVQ v3 timing nonfinite");
  if(!mmvq_v3_profile)return {host,0,0,0,0,0,0,0,0};
  auto elapsed=[](const sycl::event& ev){return ev.get_profiling_info<sycl::info::event_profiling::command_end>()-ev.get_profiling_info<sycl::info::event_profiling::command_start>();};
  return {host,end.get_profiling_info<sycl::info::event_profiling::command_start>()-start.get_profiling_info<sycl::info::event_profiling::command_end>(),
   elapsed(h2d),elapsed(e.input),elapsed(e.gate_up),candidate?0:elapsed(swiglu),elapsed(e.hidden_quant),elapsed(e.down),elapsed(d2h)};
 }catch(...){try{q.wait_and_throw();}catch(...){}throw;}
}
static void mmvq_v3_timing(MmvqFixture& f,const std::vector<float>& x,float lim,int layer,const std::string& sig){
 Device<uint32_t> scrub(f.q,64*1024*1024);auto ptr=scrub.p;
 for(int n:{1,2,3,4,5,6})for(int cache=0;cache<2;++cache){
  for(int warm=0;warm<3;++warm){mmvq_v3_run(f,n,false,x,lim);mmvq_v3_run(f,n,true,x,lim);}
  for(int rep=0;rep<5;++rep)for(int arm=0;arm<2;++arm){
   const bool candidate=(arm+rep)%2;
   if(cache)f.q.parallel_for(sycl::range<1>(scrub.n),[=](sycl::id<1> i){ptr[i[0]]=uint32_t(i[0])*1664525u+1013904223u;}).wait_and_throw();
   auto t=mmvq_v3_run(f,n,candidate,x,lim);
   std::printf("MMVQ_V3_TIMING {\"signature\":\"%s\",\"layer\":%d,\"n\":%d,\"cache\":\"%s\",\"mode\":\"%s\",\"rep\":%d,\"profile\":%s,\"host_ns\":%llu,\"span_ns\":%llu,\"h2d_ns\":%llu,\"input_ns\":%llu,\"gate_up_ns\":%llu,\"swiglu_ns\":%llu,\"hidden_quant_ns\":%llu,\"down_ns\":%llu,\"d2h_ns\":%llu}\n",
    sig.c_str(),layer,n,cache?"scrubbed":"warm",candidate?"mmvq-soa-v3":"mmvq-soa-v2",rep,mmvq_v3_profile?"true":"false",
    (unsigned long long)t.host,(unsigned long long)t.span,(unsigned long long)t.h2d,(unsigned long long)t.input,
    (unsigned long long)t.gate_up,(unsigned long long)t.swiglu,(unsigned long long)t.hidden_quant,(unsigned long long)t.down,(unsigned long long)t.d2h);
  }
 }
}
