// MIT. Included by expert_check.cpp: reuse its ORIGINAL Chain as the comparator.
// Both targets execute this body on real USM; host API shim is labeled separately.
#pragma once
#include "grouped_project.hpp"
#include <algorithm>
struct GroupFixture {
 uint64_t last_gate_up=0,last_down=0;
 sycl::queue& q;Tables& t;std::vector<std::unique_ptr<Chain>> refs;
 std::unique_ptr<WarmGroupedWorkspace> grouped;
 explicit GroupFixture(sycl::queue& queue,Tables& tables):q(queue),t(tables){}
 void add(std::array<Matrix,3> a){
  if(!refs.empty())for(int r=0;r<3;++r)need(a[r].type==refs[0]->a[r].type&&a[r].n==refs[0]->a[r].n&&a[r].k==refs[0]->a[r].k,"nonuniform group fixture");
  need(a[0].type==a[1].type,"gate/up types differ");
  refs.push_back(std::make_unique<Chain>(q,std::move(a),t));
 }
 void ready(){need(!refs.empty()&&refs.size()<=size_t(kPicks),"fixture capacity");grouped=std::make_unique<WarmGroupedWorkspace>(q,refs[0]->a[0].k,refs[0]->a[0].n,int(refs.size()));
#ifdef STRATA_EXPERT_STRIP
 grouped->fast=true;
#endif
 }
 // Reverse order ensures slot position is not accidentally interpreted as expert ID.
 WarmPackedBatch descriptor(int count){WarmPackedBatch b;for(int i=0;i<count;++i){auto& c=*refs[refs.size()-1-size_t(i)];
  b.gate[i]=c.gate.p;b.up[i]=c.up.p;b.down[i]=c.down.p;b.gate_ld[i]=c.a[0].ld;b.up_ld[i]=c.a[1].ld;b.down_ld[i]=c.a[2].ld;
 }return b;}
 template<class T>std::vector<T> read(const T* p,size_t n){std::vector<T> out(n);q.memcpy(out.data(),p,n*sizeof(T)).wait_and_throw();return out;}
 template<class T>void slice(const std::vector<T>& batch,int i,const std::vector<T>& single,const char* stage){
  const size_t n=single.size();exact(std::vector<T>(batch.begin()+size_t(i)*n,batch.begin()+(size_t(i)+1)*n),single,stage);
 }
 void cpu_dot(const std::vector<float>& batch,int i,const Matrix& a,const std::vector<int8_t>& codes,const std::vector<float>& scales,const char* stage){
  if(a.n<=256){slice(batch,i,reference(a,codes,scales),stage);return;}
  // Native full-size fixtures compare EVERY word to the serial-device chain;
  // the independent CPU oracle samples rows to keep the guarded check bounded.
  for(int row:{0,a.n/2,a.n-1}){Matrix one{a.type,1,a.k,a.ld,{}};
   one.bytes.assign(a.bytes.begin()+size_t(row)*a.ld,a.bytes.begin()+size_t(row+1)*a.ld);
   exact(std::vector<float>{batch[size_t(i)*a.n+row]},reference(one,codes,scales),stage);
  }
 }
 void check(int count,const std::vector<float>& x,float lim){
  auto& w=*grouped;auto a=descriptor(count);q.memcpy(w.x,x.data(),x.size()*4).wait_and_throw();
  w.enqueue(a,count,refs[0]->a[0].type,refs[0]->a[2].type,lim,t.xx.p,t.xs.p,t.im.p).down.wait_and_throw();
  const auto gate=read(w.gate,size_t(count)*w.ff),up=read(w.up,size_t(count)*w.ff),hidden=read(w.hidden,size_t(count)*w.ff),y=read(w.y,size_t(count)*w.dim);
  const auto codes=read(w.quant,size_t(count)*w.ff);const auto scales=read(w.scale,size_t(count)*w.ff/32);
  for(int i=0;i<count;++i){auto& c=*refs[refs.size()-1-size_t(i)];
   q.memcpy(c.x.p,x.data(),x.size()*4).wait_and_throw();c.run(true,lim); // current production packed path, same device exp
   slice(gate,i,c.g.read(),"grouped gate");slice(up,i,c.u.read(),"grouped up");slice(hidden,i,c.h.read(),"grouped SwiGLU");slice(y,i,c.y.read(),"grouped down");
   auto cq=c.codes.read();cq.resize(size_t(w.ff));auto cd=c.d.read();cd.resize(size_t(w.ff/32));
   slice(codes,i,cq,"grouped hidden Q8");slice(scales,i,cd,"grouped hidden scales");
   std::vector<int8_t> iq(x.size());std::vector<float> id(x.size()/32);quant(x,iq,id);
   cpu_dot(gate,i,c.a[0],iq,id,"grouped gate CPU dot");cpu_dot(up,i,c.a[1],iq,id,"grouped up CPU dot");
   std::vector<float> h(hidden.begin()+size_t(i)*w.ff,hidden.begin()+size_t(i+1)*w.ff);
   std::vector<int8_t> hq(h.size());std::vector<float> hd(h.size()/32);quant(h,hq,hd);
   slice(codes,i,hq,"grouped hidden CPU Q8");slice(scales,i,hd,"grouped hidden CPU scale");cpu_dot(y,i,c.a[2],hq,hd,"grouped down CPU dot");
  }
 }
 uint64_t run(int mode,int count,const std::vector<float>& x,float lim){
  const bool batch=mode!=0;grouped->fast=mode==2;
  uint64_t busy=0;last_gate_up=last_down=0;
  auto elapsed=[](const sycl::event& e){return e.get_profiling_info<sycl::info::event_profiling::command_end>()-e.get_profiling_info<sycl::info::event_profiling::command_start>();};
  std::vector<float> out(size_t(grouped->dim)*count);
  try{
   if(batch){auto& w=*grouped;q.single_task([]{});q.memcpy(w.x,x.data(),x.size()*4);
    const auto e=w.enqueue(descriptor(count),count,refs[0]->a[0].type,refs[0]->a[2].type,lim,t.xx.p,t.xs.p,t.im.p);
    q.memcpy(out.data(),w.y,out.size()*4);q.single_task([]{}).wait_and_throw();last_gate_up=elapsed(e.gate_up);last_down=elapsed(e.down);busy=last_gate_up+last_down;
   }else for(int i=0;i<count;++i){auto& c=*refs[refs.size()-1-size_t(i)];auto& scratch=*refs[0];
    // Match Bank::warm_compute: one reused scratch set, no down wait before
    // readback, and the same two profiling markers/one completion per expert.
    q.single_task([]{});q.memcpy(scratch.x.p,x.data(),x.size()*4);
    m::quantize(q,scratch.x.p,x.size(),grouped->dim,1,scratch.codes.p,scratch.d.p);
    const auto ge=warm_packed_project(q,c.a[0].type,true,c.gate.p,c.a[0].ld,c.a[0].n,c.a[0].k,scratch.g.p,scratch.codes.p,scratch.d.p,t.xx.p,t.xs.p,t.im.p);
    const auto ue=warm_packed_project(q,c.a[1].type,true,c.up.p,c.a[1].ld,c.a[1].n,c.a[1].k,scratch.u.p,scratch.codes.p,scratch.d.p,t.xx.p,t.xs.p,t.im.p);
    r::swiglu(q,scratch.g.p,scratch.u.p,grouped->ff,lim,scratch.h.p);
    m::quantize(q,scratch.h.p,size_t(grouped->ff),grouped->ff,1,scratch.codes.p,scratch.d.p);
    const auto de=warm_packed_project(q,c.a[2].type,false,c.down.p,c.a[2].ld,c.a[2].n,c.a[2].k,scratch.y.p,scratch.codes.p,scratch.d.p,t.xx.p,t.xs.p,t.im.p);
    q.memcpy(out.data()+size_t(i)*grouped->dim,scratch.y.p,size_t(grouped->dim)*4);q.single_task([]{}).wait_and_throw();last_gate_up+=elapsed(ge)+elapsed(ue);last_down+=elapsed(de);busy=last_gate_up+last_down;
   }
  }catch(...){try{q.wait_and_throw();}catch(...){}throw;}
  for(float v:out)need(std::isfinite(v),"nonfinite timed output");return busy;
 }
};
// Actual model input widths plus partial four-row groups; distinct activations
// for down catch expert-stride bugs that symmetric tiny chains could conceal.
static void grouped_actual_k(sycl::queue& q,Tables& tables,std::mt19937& rng){
#ifdef STRATA_EXPERT_STRIP
 constexpr bool fast=true;
#else
 constexpr bool fast=false;
#endif
 size_t cases=0;
 for(const std::string type:{"IQ2_XXS","IQ2_XS","IQ1_M"})for(int k:{2304,5120})for(int n:{1,5,31,33}){
  std::vector<std::unique_ptr<Device<uint8_t>>> weights;std::vector<Matrix> matrices;WarmPackedBatch b;
  for(int role=0;role<3;++role)for(int expert=0;expert<3;++expert){
   Matrix a{type,n,k,m::row_bytes(m::type(type),k),{}};a.bytes.resize(size_t(n)*a.ld);
   for(auto& v:a.bytes)v=uint8_t(rng());
   for(int row=0;row<n;++row)for(int block=0;block<k/256;++block){auto p=a.bytes.data()+size_t(row)*a.ld+size_t(block)*m::block_bytes(m::type(type));const uint16_t half=0x3555;
    if(type!="IQ1_M")std::memcpy(p,&half,2);
    else for(int word=0;word<4;++word){uint16_t sc=uint16_t((m::u16(p+48+word*2)&0xfffu)|(((half>>(4*word))&15u)<<12));std::memcpy(p+48+word*2,&sc,2);}
   }
   weights.push_back(std::make_unique<Device<uint8_t>>(q,a.bytes));matrices.push_back(std::move(a));
   auto raw=weights.back()->p;const auto ld=matrices.back().ld;
   if(role==0){b.gate[expert]=raw;b.gate_ld[expert]=ld;}else if(role==1){b.up[expert]=raw;b.up_ld[expert]=ld;}else{b.down[expert]=raw;b.down_ld[expert]=ld;}
  }
  std::vector<int8_t>x(size_t(k)*3);std::vector<float>d(size_t(k/32)*3);
  for(size_t i=0;i<x.size();++i)x[i]=int8_t(int(rng()%255)-127);
  for(size_t i=0;i<d.size();++i)d[i]=(i%2?-1.0f:1.0f)*float(i%13+1)/128.0f;
  Device<int8_t> dx(q,x);Device<float> dd(q,d),y(q,size_t(n)*3),up(q,size_t(n)*3);
  for(int count:{1,3}){
   warm_grouped_detail::dispatch<true,fast>(q,type,b,count,n,k,y.p,up.p,dx.p,dd.p,tables.xx.p,tables.xs.p,tables.im.p).wait_and_throw();
   const auto gy=y.read(),gu=up.read();std::vector<int8_t> qx(x.begin(),x.begin()+k);std::vector<float> ds(d.begin(),d.begin()+k/32);
   for(int e=0;e<count;++e){exact(std::vector<float>(gy.begin()+e*n,gy.begin()+(e+1)*n),reference(matrices[e],qx,ds),"actual-K grouped gate");
    exact(std::vector<float>(gu.begin()+e*n,gu.begin()+(e+1)*n),reference(matrices[3+e],qx,ds),"actual-K grouped up");}++cases;
   warm_grouped_detail::dispatch<false,fast>(q,type,b,count,n,k,y.p,nullptr,dx.p,dd.p,tables.xx.p,tables.xs.p,tables.im.p).wait_and_throw();
   const auto dn=y.read();for(int e=0;e<count;++e){std::vector<int8_t> qh(x.begin()+e*k,x.begin()+(e+1)*k);std::vector<float> dh(d.begin()+e*(k/32),d.begin()+(e+1)*(k/32));
    exact(std::vector<float>(dn.begin()+e*n,dn.begin()+(e+1)*n),reference(matrices[6+e],qh,dh),"actual-K grouped down");}++cases;
  }
 }
 std::printf("ACTUAL_K_CHECKS %zu projection batches: K2304/5120, N1/5/31/33, selected1/3, three formats\n",cases);
}
static int grouped_main(int argc,char** argv){
 std::string pack;need(argc==1||argc==3,"usage: strata-grouped-{bitcheck,timing} [--pack PATH]");
 if(argc==3){need(std::string(argv[1])=="--pack","unknown argument");pack=argv[2];}
 sycl::queue q(sycl::gpu_selector_v,sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}});
#ifndef STRATA_IQ2_HOST_TEST
 const auto dev=q.get_device();need(dev.is_gpu()&&dev.get_info<sycl::info::device::vendor_id>()==0x8086,"requires selected Intel GPU");
 std::printf("NATIVE device=%s driver=%s\n",dev.get_info<sycl::info::device::name>().c_str(),dev.get_info<sycl::info::device::driver_version>().c_str());
#else
 std::puts("HOST_SIMULATION_ONLY: no oneAPI/device/timing claim");
#endif
 std::mt19937 rng(0x603bad65);Tables tables(q);size_t checks=0;
#ifndef STRATA_GROUPED_TIMING
 grouped_actual_k(q,tables,rng);
 const std::array<std::string,3> signatures[]={ {"IQ2_XXS","IQ2_XXS","IQ2_XXS"},{"IQ1_M","IQ1_M","IQ1_M"},
 {"IQ2_XS","IQ2_XS","IQ1_M"},{"IQ2_XXS","IQ2_XXS","IQ2_XS"},{"IQ2_XS","IQ2_XS","IQ2_XXS"},{"IQ2_XS","IQ2_XS","IQ2_XS"} };
 for(const auto& sig:signatures){GroupFixture f(q,tables);for(int i=0;i<8;++i)f.add(synthetic(sig,rng));f.ready();
  // Shrink after full capacity and reuse without clearing; stale slots must not leak.
  for(int n:{1,2,4,8,1})for(int probe=0;probe<2;++probe){std::vector<float> x(256);
   for(size_t i=0;i<x.size();++i)x[i]=probe?(i%32==0?127.0f:i%4==0?1.5f:i%4==1?2.5f:i%4==2?-1.5f:-2.5f):float(int(rng()%255)-127)/32.0f;
   f.check(n,x,probe?1.0f:0.0f);++checks;
  }
  std::vector<float> zero(256,-0.0f);f.check(8,zero,0.0f);++checks;
 }
#endif
#if !defined(STRATA_IQ2_HOST_TEST) || defined(STRATA_EXPERT_NATIVE_SIMULATION)
 if(!pack.empty()){
  auto ws=open_pack(pack);std::set<std::string> found;need(ws->meta().n_exp>=8,"native group fixture needs eight distinct experts");
  const char* roles[]={"w1","w3","w2"};
  for(int l=0;l<ws->meta().nl;++l){std::array<std::string,3> types;std::string sig;
   for(int r=0;r<3;++r){types[r]=ws->require_tensor(layer_prefix(l)+"ffn.experts.0."+roles[r]+".weight").dtype;sig+=types[r]+"/";}
   if(!warm_packed_type(types[0])||!warm_packed_type(types[2])||types[0]!=types[1]||!found.insert(sig).second)continue;
   std::printf("REAL_SIGNATURE layer=%d signature=%s\n",l,sig.c_str());
   GroupFixture f(q,tables);
   for(int e=0;e<8;++e){std::array<Matrix,3>a;
    for(int r=0;r<3;++r){const auto name=layer_prefix(l)+"ffn.experts."+std::to_string(e)+"."+roles[r]+".weight";const auto z=ws->require_tensor(name);
     need(z.shape.size()==2&&z.shape[0]>0&&z.shape[0]<=INT32_MAX&&z.shape[1]>0&&z.shape[1]<=INT32_MAX,"real grouped shape");
     auto& mtx=a[r];mtx.type=z.dtype;mtx.n=int(z.shape[0]);mtx.k=int(z.shape[1]);mtx.ld=m::row_bytes(m::type(z.dtype),mtx.k);
     need(z.bytes==size_t(mtx.n)*mtx.ld,"real grouped byte extent");auto raw=ws->read_all(name);mtx.bytes.resize(raw.size());std::memcpy(mtx.bytes.data(),raw.data(),raw.size());
     std::printf("TENSOR %s bytes=%zu\n",name.c_str(),raw.size());
    }f.add(std::move(a));
   }f.ready();std::vector<float>x(size_t(f.grouped->dim));for(auto& v:x)v=float(int(rng()%255)-127)/32.0f;
   for(int n:{1,2,4,8}){f.check(n,x,ws->meta().lim);++checks;}
#ifdef STRATA_GROUPED_TIMING
   Device<uint32_t> scrub(q,64*1024*1024);auto* ptr=scrub.p;

#ifdef STRATA_EXPERT_STRIP
   constexpr int modes=3;const char* marker="STRIP_TIMING";
#else
   constexpr int modes=2;const char* marker="GROUP_TIMING";
#endif
   for(int n:{1,2,4,8})for(int cache=0;cache<2;++cache){
    for(int i=0;i<3;++i)for(int mode=0;mode<modes;++mode)f.run(mode,n,x,ws->meta().lim);
    for(int rep=0;rep<5;++rep)for(int arm=0;arm<modes;++arm){const int mode=(arm+rep)%modes;
     if(cache)q.parallel_for(sycl::range<1>(scrub.n),[=](sycl::id<1> i){ptr[i[0]]=uint32_t(i[0]);}).wait_and_throw();
     const auto start=q.single_task([]{});const auto begin=now();const auto busy=f.run(mode,n,x,ws->meta().lim);const auto wall=now()-begin;
     auto end=q.single_task([]{});end.wait_and_throw();
     const auto span=end.get_profiling_info<sycl::info::event_profiling::command_start>()-start.get_profiling_info<sycl::info::event_profiling::command_end>();
#ifdef STRATA_EXPERT_STRIP
     std::printf("%s {\"signature\":\"%s\",\"layer\":%d,\"n\":%d,\"cache\":\"%s\",\"mode\":\"%s\",\"rep\":%d,\"host_ns\":%llu,\"span_ns\":%llu,\"matrix_ns\":%llu,\"gate_up_ns\":%llu,\"down_ns\":%llu}\n",marker,sig.c_str(),l,n,cache?"scrubbed":"warm",mode==2?"packed-strip":mode==1?"grouped":"serial",rep,(unsigned long long)wall,(unsigned long long)span,(unsigned long long)busy,(unsigned long long)f.last_gate_up,(unsigned long long)f.last_down);
#else
     std::printf("%s {\"signature\":\"%s\",\"layer\":%d,\"n\":%d,\"cache\":\"%s\",\"mode\":\"%s\",\"rep\":%d,\"host_ns\":%llu,\"span_ns\":%llu,\"matrix_ns\":%llu}\n",marker,sig.c_str(),l,n,cache?"scrubbed":"warm",mode==2?"packed-strip":mode==1?"grouped":"serial",rep,(unsigned long long)wall,(unsigned long long)span,(unsigned long long)busy);
#endif
    }
   }
#endif
  }
  need(!found.empty(),"pack has no packed-supported expert signature");std::printf("REAL_SIGNATURES %zu\n",found.size());
 }
#ifdef STRATA_GROUPED_TIMING
 need(!pack.empty(),"native timing requires real pack");
#endif
#endif
 std::printf("PASS: %zu grouped batch checks; serial packed same-device stages and CPU Q8/dot exact; reversed distinct selections\n",checks);return 0;
}
