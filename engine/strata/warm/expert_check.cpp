// MIT. Exact Q8/dot CPU oracle plus original scalar-device chain on native USM.
// Transcendental stages use the same-device reference, never host libm.
// Also builds as a CPU control-flow simulation.
#include "packed_project.hpp"
#include "../sycl/stage/wiring/sycl_row2.hpp"
#include "../sycl/stage/wiring/weight_source.hpp"
#include "../sycl/moe/core.hpp"
#include "../sycl/moe/moe.hpp"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <random>
#include <numeric>
namespace r=strata::ds4::row2;namespace m=strata::ds4::moe;
using namespace strata::wiring;
struct Math {
 static float fma(float a,float b,float c){return std::fma(a,b,c);}
 static float exp(float a){return std::exp(a);}
 static float min(float a,float b){return std::fmin(a,b);}
 static float max(float a,float b){return std::fmax(a,b);}
};
static void need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
static uint64_t now(){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
template<class T>struct Device {
 sycl::queue& q;T* p;size_t n;
 Device(sycl::queue& queue,size_t count):q(queue),p(sycl::malloc_device<T>(count,q)),n(count){if(!p)throw std::bad_alloc();}
 Device(sycl::queue& queue,const std::vector<T>& v):Device(queue,v.size()){q.memcpy(p,v.data(),v.size()*sizeof(T)).wait_and_throw();}
 ~Device(){try{q.wait_and_throw();}catch(...){}sycl::free(p,q);}
 Device(const Device&)=delete;
 std::vector<T> read(){std::vector<T> v(n);q.memcpy(v.data(),p,n*sizeof(T)).wait_and_throw();return v;}
};
template<class T>static void exact(const std::vector<T>& a,const std::vector<T>& b,const char* stage){
 need(a.size()==b.size(),"comparison extent");
 if constexpr(std::is_same_v<T,float>)for(size_t i=0;i<a.size();++i)need(std::isfinite(a[i])&&std::isfinite(b[i]),"nonfinite result");
 if(std::memcmp(a.data(),b.data(),a.size()*sizeof(T))){
  for(size_t i=0;i<a.size();++i)if(std::memcmp(&a[i],&b[i],sizeof(T))){
   if constexpr(std::is_same_v<T,float>)std::fprintf(stderr,"MISMATCH stage=%s index=%zu candidate=%a reference=%a\n",stage,i,a[i],b[i]);
   else std::fprintf(stderr,"MISMATCH stage=%s index=%zu\n",stage,i);break;}
  throw std::runtime_error(std::string(stage)+" changed bits");
 }
}
static void quant(const std::vector<float>& x,std::vector<int8_t>& q,std::vector<float>& d){
 for(size_t b=0;b<d.size();++b){float am=0;for(int j=0;j<32;++j)am=std::fmax(am,std::fabs(x[b*32+j]));d[b]=am/127.0f;
  for(int j=0;j<32;++j)q[b*32+j]=int8_t(am>0?int(std::nearbyint(x[b*32+j]/d[b])):0);}
}
struct Matrix {std::string type;int n,k;size_t ld;std::vector<uint8_t> bytes;};
static std::vector<float> reference(const Matrix& a,const std::vector<int8_t>& q,const std::vector<float>& d){
 std::vector<float> y(size_t(a.n));auto type=m::type(a.type);
 for(int row=0;row<a.n;++row){std::array<float,32> lane;
  for(int l=0;l<32;++l)lane[l]=m::lane<Math>(type,a.bytes.data()+size_t(row)*a.ld,a.k,nullptr,q.data(),d.data(),l,true);
  for(int o=16;o;o>>=1){auto old=lane;for(int l=0;l<32;++l)lane[l]=old[l]+old[l^o];}y[size_t(row)]=1.0f*lane[0]+0.0f;
 }return y;
}
struct Tables {
 Device<uint16_t> xx,xs,im;
 Tables(sycl::queue& q):xx(q,std::vector<uint16_t>(r::iq2::compact_codebook.begin(),r::iq2::compact_codebook.end())),
 xs(q,std::vector<uint16_t>(p1k_iq2xs::compact.begin(),p1k_iq2xs::compact.end())),
 im(q,std::vector<uint16_t>(p1k_down::iq1_compact.begin(),p1k_down::iq1_compact.end())){}
};
struct Chain {
 sycl::queue& q;std::array<Matrix,3> a;Tables& t;
 sycl::event gate_event,up_event,down_event;
 Device<uint8_t> gate,up,down;Device<float> x,g,u,h,y,d;Device<int8_t> codes;
 Chain(sycl::queue& queue,std::array<Matrix,3> matrices,Tables& tables):q(queue),a(std::move(matrices)),t(tables),
 gate(q,a[0].bytes),up(q,a[1].bytes),down(q,a[2].bytes),x(q,size_t(a[0].k)),g(q,size_t(a[0].n)),u(q,size_t(a[1].n)),h(q,size_t(a[0].n)),y(q,size_t(a[2].n)),d(q,size_t(std::max(a[0].k,a[2].k)/32)),codes(q,size_t(std::max(a[0].k,a[2].k))){}
 sycl::event project(int role,bool packed,float* out){
  auto& z=a[size_t(role)];uint8_t* w=role==0?gate.p:role==1?up.p:down.p;
  if(packed)return warm_packed_project(q,z.type,role!=2,w,z.ld,z.n,z.k,out,codes.p,d.p,t.xx.p,t.xs.p,t.im.p);
  r::DenseMat mat{"control",z.type,z.n,z.k,z.ld,w};r::MvJob job{&mat,role==2?h.p:x.p,out,1,size_t(z.k),size_t(z.n),1};return r::mv(q,std::span(&job,1));
 }
 std::pair<sycl::event,sycl::event> run(bool packed,float lim){
  auto start=m::quantize(q,x.p,size_t(a[0].k),a[0].k,1,codes.p,d.p);
  gate_event=project(0,packed,g.p);up_event=project(1,packed,u.p);r::swiglu(q,g.p,u.p,a[0].n,lim,h.p);
  m::quantize(q,h.p,size_t(a[2].k),a[2].k,1,codes.p,d.p);down_event=project(2,packed,y.p);down_event.wait_and_throw();return {start,down_event};
 }
 void check(const std::vector<float>& input,float lim){
  q.memcpy(x.p,input.data(),input.size()*4).wait_and_throw();
  std::vector<int8_t> rq(input.size());std::vector<float> rd(input.size()/32);quant(input,rq,rd);
  m::quantize(q,x.p,input.size(),a[0].k,1,codes.p,d.p).wait_and_throw();
  auto cq=codes.read();cq.resize(rq.size());auto cd=d.read();cd.resize(rd.size());exact(cq,rq,"input quant bytes");exact(cd,rd,"input quant scale");
  project(0,true,g.p).wait_and_throw();auto rg=reference(a[0],rq,rd);exact(g.read(),rg,"gate");
  project(1,true,u.p).wait_and_throw();auto ru=reference(a[1],rq,rd);exact(u.read(),ru,"up");
  r::swiglu(q,g.p,u.p,a[0].n,lim,h.p).wait_and_throw();auto ph=h.read();
  // CPU Q8/dot controls consume the actual device SwiGLU output. Host libm
  // exp may differ by ulps from device exp even with identical gate/up bits.
  rq.resize(ph.size());rd.resize(ph.size()/32);quant(ph,rq,rd);
  m::quantize(q,h.p,ph.size(),a[2].k,1,codes.p,d.p).wait_and_throw();
  cq=codes.read();cq.resize(rq.size());cd=d.read();cd.resize(rd.size());exact(cq,rq,"hidden quant bytes CPU");exact(cd,rd,"hidden quant scale CPU");
  project(2,true,y.p).wait_and_throw();auto py=y.read();exact(py,reference(a[2],rq,rd),"down CPU (device hidden input)");
  // Preserve every candidate stage, then run the original production scalar
  // device chain independently from the same input. No candidate intermediates
  // feed that reference. Both paths use the same queue, device and SwiGLU.
  run(false,lim);
  exact(rg,g.read(),"gate scalar-device");exact(ru,u.read(),"up scalar-device");
  exact(ph,h.read(),"SwiGLU scalar-device");
  auto sq=codes.read();sq.resize(rq.size());auto sd=d.read();sd.resize(rd.size());
  exact(cq,sq,"hidden quant bytes scalar-device");exact(cd,sd,"hidden quant scale scalar-device");
  exact(py,y.read(),"down scalar-device full chain");
 }
};
static std::array<Matrix,3> synthetic(const std::array<std::string,3>& types,std::mt19937& rng,int dim=256,int ff=256){
 std::array<Matrix,3> out;
 for(int role=0;role<3;++role){auto& a=out[size_t(role)];a.type=types[size_t(role)];a.n=role==2?dim:ff;a.k=role==2?ff:dim;a.ld=m::row_bytes(m::type(a.type),a.k);a.bytes.resize(size_t(a.n)*a.ld);
  for(auto& v:a.bytes)v=uint8_t(rng());
  for(int row=0;row<a.n;++row)for(int b=0;b<a.k/256;++b){auto p=a.bytes.data()+size_t(row)*a.ld+size_t(b)*m::block_bytes(m::type(a.type));uint16_t half=uint16_t(row%3==0?0xb555:0x3555);
   if(a.type!="IQ1_M")std::memcpy(p,&half,2);
   else for(int word=0;word<4;++word){uint16_t sc=uint16_t((m::u16(p+48+word*2)&0xfffu)|(((half>>(4*word))&15u)<<12));std::memcpy(p+48+word*2,&sc,2);}
  }
 }return out;
}
#ifdef STRATA_EXPERT_GROUPED
#include "grouped_check.hpp"
#endif
#ifdef STRATA_EXPERT_MMVQ
#include "mmvq_check.hpp"
#endif
int main(int argc,char** argv)try{
#ifdef STRATA_EXPERT_MMVQ
 return mmvq_main(argc,argv);
#else
#ifdef STRATA_EXPERT_GROUPED
 return grouped_main(argc,argv);
#else
 std::string pack;need(argc==1||argc==3,"usage: strata-expert-{bitcheck,timing} [--pack PATH]");
 if(argc==3){need(std::string(argv[1])=="--pack","unknown argument");pack=argv[2];}
 sycl::queue queue(sycl::gpu_selector_v,sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}});
#ifndef STRATA_IQ2_HOST_TEST
 const auto device=queue.get_device();need(device.is_gpu()&&device.get_info<sycl::info::device::vendor_id>()==0x8086,"requires selected Intel GPU");
 std::printf("NATIVE device=%s driver=%s\n",device.get_info<sycl::info::device::name>().c_str(),device.get_info<sycl::info::device::driver_version>().c_str());
#else
 std::puts("HOST_SIMULATION_ONLY: no oneAPI or device claim");
#endif
 std::mt19937 rng(0x723aa119);Tables tables(queue);size_t chains=0;
 const std::array<std::string,3> signatures[]={ {"IQ2_XS","IQ2_XS","IQ1_M"},{"IQ2_XXS","IQ2_XXS","IQ2_XS"},{"IQ2_XS","IQ2_XS","IQ2_XXS"},{"IQ2_XS","IQ2_XS","IQ2_XS"} };
#ifndef STRATA_EXPERT_TIMING
 for(const auto& sig:signatures)for(int expert=0;expert<3;++expert){Chain c(queue,synthetic(sig,rng),tables);
  for(int probe=0;probe<3;++probe)for(float lim:{0.0f,1.0f}){std::vector<float> x(size_t(c.a[0].k));
   for(size_t i=0;i<x.size();++i)x[i]=probe==1?(i%2?-0.0f:0.0f):probe==2?(i%32==0?127.0f:i%4==0?1.5f:i%4==1?2.5f:i%4==2?-1.5f:-2.5f):float(int(rng()%255)-127)/32.0f;
   c.check(x,lim);++chains;}}
 // Real-K rows, local4 tails and every input alignment are covered separately
 // by the exhaustive CPU proof; here exercise actual-K/tail native launches.
 for(int k:{2304,5120})for(int n:{1,5}){Matrix a=synthetic(signatures[3],rng,k,256)[0];a.n=n;a.bytes.resize(size_t(n)*a.ld);Device<uint8_t> w(queue,a.bytes);Device<int8_t> q(queue,size_t(k));Device<float> d(queue,size_t(k/32)),y(queue,size_t(n));
  std::vector<int8_t> codes(size_t(k),int8_t(-127));std::vector<float> scales(size_t(k/32),0.125f);
  queue.memcpy(q.p,codes.data(),codes.size());queue.memcpy(d.p,scales.data(),scales.size()*4);
  warm_packed_project(queue,a.type,false,w.p,a.ld,n,k,y.p,q.p,d.p,tables.xx.p,tables.xs.p,tables.im.p).wait_and_throw();exact(y.read(),reference(a,codes,scales),"native actual-K/tail rows");}
#endif
#if !defined(STRATA_IQ2_HOST_TEST) || defined(STRATA_EXPERT_NATIVE_SIMULATION)
 if(!pack.empty()){
  auto ws=open_pack(pack);std::set<std::string> found;const char* roles[]={"w1","w3","w2"};
  for(int l=0;l<ws->meta().nl;++l){std::array<Matrix,3> a;std::string sig;
   for(int role=0;role<3;++role){auto name="layers."+std::to_string(l)+".ffn.experts.0."+roles[role]+".weight";auto t=ws->require_tensor(name);auto& z=a[size_t(role)];need(t.shape.size()==2&&t.shape[0]>0&&t.shape[1]>0,"real matrix geometry");z.type=t.dtype;z.n=int(t.shape.at(0));z.k=int(t.shape.at(1));z.ld=m::row_bytes(m::type(z.type),z.k);need(t.bytes==size_t(z.n)*z.ld,"real matrix byte extent");sig+=z.type+"/";}
   bool requested=false;for(const auto& s:signatures)requested|=sig==s[0]+"/"+s[1]+"/"+s[2]+"/";
   if(!requested||!found.insert(sig).second)continue;
   for(int role=0;role<3;++role){auto name="layers."+std::to_string(l)+".ffn.experts.0."+roles[role]+".weight";auto raw=ws->read_all(name);a[size_t(role)].bytes.resize(raw.size());std::memcpy(a[size_t(role)].bytes.data(),raw.data(),raw.size());std::printf("TENSOR %s bytes=%zu\n",name.c_str(),raw.size());}
   need(a[0].n==a[1].n&&a[0].k==a[1].k&&a[2].n==a[0].k&&a[2].k==a[0].n,"mixed expert role geometry");
   Chain c(queue,std::move(a),tables);std::vector<float> x(size_t(c.a[0].k));for(auto& v:x)v=float(int(rng()%255)-127)/32.0f;
   c.check(x,ws->meta().lim);++chains;std::fill(x.begin(),x.end(),0.0f);c.check(x,ws->meta().lim);++chains;
#ifdef STRATA_EXPERT_TIMING
   for(auto& v:x)v=float(int(rng()%255)-127)/32.0f;queue.memcpy(c.x.p,x.data(),x.size()*4).wait_and_throw();
   Device<uint32_t> scrub(queue,64*1024*1024);auto* scrub_ptr=scrub.p;
   for(int cache=0;cache<2;++cache){for(bool packed:{false,true})for(int i=0;i<3;++i)c.run(packed,ws->meta().lim);
    for(int rep=0;rep<5;++rep)for(bool packed:{false,true}){
     if(cache)queue.parallel_for(sycl::range<1>(scrub.n),[=](sycl::id<1> i){scrub_ptr[i[0]]=uint32_t(i[0]);}).wait_and_throw();
     std::vector<float> output(size_t(c.a[2].n));auto begin=now();queue.memcpy(c.x.p,x.data(),x.size()*4);auto events=c.run(packed,ws->meta().lim);queue.memcpy(output.data(),c.y.p,output.size()*4).wait_and_throw();auto wall=now()-begin;
     for(float v:output)need(std::isfinite(v),"nonfinite timing output");
     auto duration=events.second.get_profiling_info<sycl::info::event_profiling::command_end>()-events.first.get_profiling_info<sycl::info::event_profiling::command_start>();
     auto elapsed=[](const sycl::event& e){return e.get_profiling_info<sycl::info::event_profiling::command_end>()-e.get_profiling_info<sycl::info::event_profiling::command_start>();};
     const auto gu=elapsed(c.gate_event)+elapsed(c.up_event),dn=elapsed(c.down_event);
     std::printf("TIMING {\"signature\":\"%s\",\"layer\":%d,\"expert\":0,\"cache\":\"%s\",\"packed\":%s,\"rep\":%d,\"chain_event_ns\":%llu,\"host_ns\":%llu,\"gate_up_event_ns\":%llu,\"down_event_ns\":%llu}\n",sig.c_str(),l,cache?"scrubbed":"warm",packed?"true":"false",rep,(unsigned long long)duration,(unsigned long long)wall,(unsigned long long)gu,(unsigned long long)dn);
    }}
#endif
  }
  need(found.size()==4,"pack does not cover all four mixed signatures");
 }
#ifdef STRATA_EXPERT_TIMING
 need(!pack.empty(),"timing requires real --pack and successful CPU-dot/scalar-device bitcheck");
#endif
#endif
 std::printf("PASS: %zu bit-exact expert chains; four mixed signatures; CPU reference Q8/gate/up/hidden-Q8/down with device hidden input; scalar-device reference SwiGLU/hidden-Q8/down full chain\n",chains);
#endif
#endif
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}
