// MIT. Native/CPU-shim execution of the actual SoA kernels and bank arithmetic.
#pragma once
#include "mmvq_project.hpp"
#include "mmvq_metrics.hpp"
#include <fstream>
namespace mq=strata::wiring::mmvq;
static void metric_line(const char* stage,const mq::Metrics& v){
 std::printf("MMVQ_ACCURACY {\"stage\":\"%s\",\"count\":%llu,\"nrmse\":%.12g,\"max_abs\":%.12g,\"peak\":%.12g,\"max_relative_away_from_zero\":%.12g,\"max_ulp\":%llu}\n",stage,(unsigned long long)v.count,v.nrmse(),v.max_abs,v.peak,v.max_relative,(unsigned long long)v.max_ulp);
}
static void projection(const Matrix& a,std::span<const float> candidate,const std::vector<int8_t>& x,const std::vector<float>& d,const char* name){
 const auto ref=reference(a,x,d);need(candidate.size()==ref.size(),"MMVQ projection size");
 for(int row=0;row<a.n;++row){double absolute_sum=0;for(int kb=0;kb<a.k/32;++kb){auto z=m::decode(m::type(a.type),a.bytes.data()+size_t(row)*a.ld,kb);int v=0,hi=0;for(int j=0;j<16;++j){v+=z.q[j]*x[32*kb+j];hi+=z.q[j+16]*x[32*kb+j+16];}absolute_sum+=std::abs(double(z.d)*double(z.s[0]*v+z.s[1]*hi)*double(d[kb]));}
  need(std::isfinite(candidate[row])&&std::abs(double(candidate[row])-ref[row])<=2e-5*absolute_sum+1e-7,"MMVQ projection backward-error bound");
 }metric_line(name,mq::metrics(candidate,ref));
}
struct MmvqFixture {
 sycl::queue& q;Tables& exact_tables;Device<uint32_t> table;
 std::vector<std::unique_ptr<Chain>> refs;std::vector<std::array<std::unique_ptr<Device<uint8_t>>,3>> reordered;
 std::unique_ptr<WarmGroupedWorkspace> w;
 MmvqFixture(sycl::queue& queue,Tables& t):q(queue),exact_tables(t),table(q,mq::table_words){auto host=mq::codebooks();q.memcpy(table.p,host.data(),sizeof(host)).wait_and_throw();}
 void add(std::array<Matrix,3> a){
  if(!refs.empty())for(int role=0;role<3;++role)need(a[role].type==refs[0]->a[role].type&&a[role].n==refs[0]->a[role].n&&a[role].k==refs[0]->a[role].k,"MMVQ mixed batch geometry");
  need(a[0].n==a[1].n&&a[0].k==a[1].k&&a[2].k==a[0].n&&a[2].n==a[0].k&&a[0].type==a[1].type,"MMVQ role geometry");
  std::array<std::unique_ptr<Device<uint8_t>>,3> s;
  for(int role=0;role<3;++role){auto& z=a[role];auto out=mq::reorder(mq::kind(z.type),z.n,z.k,z.bytes);need(out.size()==z.bytes.size(),"MMVQ payload expansion");need(mq::restore(mq::kind(z.type),z.n,z.k,out)==z.bytes,"MMVQ layout roundtrip");s[role]=std::make_unique<Device<uint8_t>>(q,out);}
  reordered.push_back(std::move(s));refs.push_back(std::make_unique<Chain>(q,std::move(a),exact_tables));
 }
 void ready(){need(!refs.empty()&&refs.size()<=size_t(kPicks),"MMVQ fixture count");w=std::make_unique<WarmGroupedWorkspace>(q,refs[0]->a[0].k,refs[0]->a[0].n,int(refs.size()));}
 WarmPackedBatch descriptor(int count,bool soa){WarmPackedBatch b;for(int i=0;i<count;++i){size_t slot=refs.size()-1-size_t(i);auto& c=*refs[slot];b.gate[i]=soa?reordered[slot][0]->p:c.gate.p;b.up[i]=soa?reordered[slot][1]->p:c.up.p;b.down[i]=soa?reordered[slot][2]->p:c.down.p;b.gate_ld[i]=c.a[0].ld;b.up_ld[i]=c.a[1].ld;b.down_ld[i]=c.a[2].ld;}return b;}
 template<class T>std::vector<T> read(const T* p,size_t n){std::vector<T> v(n);q.memcpy(v.data(),p,n*sizeof(T)).wait_and_throw();return v;}
 template<class T>static std::vector<T> slice(const std::vector<T>& v,size_t slot,size_t n){return {v.begin()+slot*n,v.begin()+(slot+1)*n};}
 WarmGroupedWorkspace::Events enqueue(int count,bool soa,float lim){auto a=descriptor(count,soa);return soa?mq::enqueue(q,*w,a,count,refs[0]->a[0].type,refs[0]->a[2].type,lim,table.p):w->enqueue(a,count,refs[0]->a[0].type,refs[0]->a[2].type,lim,exact_tables.xx.p,exact_tables.xs.p,exact_tables.im.p);}
 void check(int count,const std::vector<float>& x,float lim){
  need(count>0&&count<=w->capacity&&x.size()==size_t(w->dim),"MMVQ check input geometry");for(float v:x)need(std::isfinite(v),"MMVQ check nonfinite input");
  q.memcpy(w->x,x.data(),x.size()*4).wait_and_throw();enqueue(count,true,lim).down.wait_and_throw();
  auto gate=read(w->gate,size_t(count)*w->ff),up=read(w->up,size_t(count)*w->ff),hidden=read(w->hidden,size_t(count)*w->ff),out=read(w->y,size_t(count)*w->dim);
  auto hq=read(w->quant,size_t(count)*w->ff);auto hd=read(w->scale,size_t(count)*w->ff/32);
  std::vector<int8_t> xq(x.size());std::vector<float> xd(x.size()/32);quant(x,xq,xd);
  for(int i=0;i<count;++i){auto& c=*refs[refs.size()-1-size_t(i)];auto g=slice(gate,i,w->ff),u=slice(up,i,w->ff),h=slice(hidden,i,w->ff),y=slice(out,i,w->dim);
   projection(c.a[0],g,xq,xd,"gate CPU");projection(c.a[1],u,xq,xd,"up CPU");
   std::vector<int8_t> cq(h.size());std::vector<float> cd(h.size()/32);quant(h,cq,cd);exact(slice(hq,i,w->ff),cq,"MMVQ hidden quant bytes");exact(slice(hd,i,w->ff/32),cd,"MMVQ hidden scale");projection(c.a[2],y,cq,cd,"down same input CPU");
   // Independent CPU chain; no candidate intermediate feeds this oracle.
   auto cg=reference(c.a[0],xq,xd),cu=reference(c.a[1],xq,xd);std::vector<float> ch(h.size());for(size_t j=0;j<ch.size();++j)ch[j]=m::swiglu<Math>(cg[j],cu[j],lim);
   quant(ch,cq,cd);auto cy=reference(c.a[2],cq,cd);auto cpu_metric=mq::metrics(y,cy);metric_line("whole expert CPU",cpu_metric);need(cpu_metric.expert_ok(),"MMVQ complete CPU expert gate");
   q.memcpy(c.x.p,x.data(),x.size()*4).wait_and_throw();c.run(true,lim);auto device_metric=mq::metrics(y,c.y.read());metric_line("whole expert exact device",device_metric);need(device_metric.expert_ok(),"MMVQ complete exact-device expert gate");
   auto eq=c.codes.read();size_t flips=0;for(int j=0;j<w->ff;++j)flips+=eq[j]!=hq[size_t(i)*w->ff+j];std::printf("MMVQ_HIDDEN_Q8 selected=%d flips=%zu total=%d\n",i,flips,w->ff);
  }
 }
 void prefixes(const std::vector<float>& x,float lim){
  q.memcpy(w->x,x.data(),x.size()*4).wait_and_throw();enqueue(8,true,lim).down.wait_and_throw();auto all=read(w->y,size_t(8)*w->dim);
  for(int count:{1,2,4,8,1}){enqueue(count,true,lim).down.wait_and_throw();auto got=read(w->y,size_t(count)*w->dim);exact(got,std::vector<float>(all.begin(),all.begin()+got.size()),"MMVQ native batch prefix");}
 }
 struct Timing {uint64_t host,span,gate_up,down;};
 Timing run(int count,bool soa,const std::vector<float>& x,float lim){std::vector<float> out(size_t(count)*w->dim);try{auto begin=now();auto start=q.single_task([]{});q.memcpy(w->x,x.data(),x.size()*4);auto events=enqueue(count,soa,lim);q.memcpy(out.data(),w->y,out.size()*4);auto end=q.single_task([]{});end.wait_and_throw();const auto host=now()-begin;
  auto elapsed=[](const sycl::event& e){return e.get_profiling_info<sycl::info::event_profiling::command_end>()-e.get_profiling_info<sycl::info::event_profiling::command_start>();};for(float v:out)need(std::isfinite(v),"MMVQ timing nonfinite");return {host,end.get_profiling_info<sycl::info::event_profiling::command_start>()-start.get_profiling_info<sycl::info::event_profiling::command_end>(),elapsed(events.gate_up),elapsed(events.down)};
  }catch(...){try{q.wait_and_throw();}catch(...){}throw;}
 }
};
static void mmvq_actual_k(sycl::queue& q,Tables& t,std::mt19937& rng){
 auto host=mq::codebooks();Device<uint32_t> table(q,mq::table_words);q.memcpy(table.p,host.data(),sizeof(host)).wait_and_throw();size_t cases=0;
 for(const std::string type:{"IQ2_XXS","IQ2_XS","IQ1_M"})for(int k:{256,2304,5120})for(int n:{1,5}){
  auto a=synthetic({type,type,type},rng,k,256)[0];a.n=n;a.bytes.resize(size_t(n)*a.ld);auto soa=mq::reorder(mq::kind(type),n,k,a.bytes);need(mq::restore(mq::kind(type),n,k,soa)==a.bytes,"actual-K SoA roundtrip");Device<uint8_t> w(q,soa);WarmPackedBatch b;b.down[0]=w.p;
  std::vector<int8_t>x(size_t(k),0);std::vector<float>d(size_t(k/32));for(auto& v:x)v=int8_t(int(rng()%255)-127);for(size_t i=0;i<d.size();++i)d[i]=float(i%17+1)/128;
  Device<int8_t> dx(q,x);Device<float> dd(q,d),y(q,size_t(n));mq::dispatch<false>(q,type,b,1,n,k,y.p,nullptr,dx.p,dd.p,table.p).wait_and_throw();projection(a,y.read(),x,d,"actual K/tail CPU");++cases;
 }
 std::printf("MMVQ_ACTUAL_K %zu\n",cases);
}
static int mmvq_main(int argc,char** argv){
 std::string pack,activation_path;for(int i=1;i<argc;i+=2){need(i+1<argc,"missing MMVQ argument");std::string flag=argv[i];if(flag=="--pack"&&pack.empty())pack=argv[i+1];else if(flag=="--activations"&&activation_path.empty())activation_path=argv[i+1];else throw std::invalid_argument("unknown/duplicate MMVQ argument");}
#ifdef STRATA_MMVQ_TIMING
 need(!pack.empty(),"native MMVQ timing requires real pack");
#endif
 sycl::queue q(sycl::gpu_selector_v,sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}});
#ifndef STRATA_IQ2_HOST_TEST
 auto dev=q.get_device();need(dev.is_gpu()&&dev.get_info<sycl::info::device::vendor_id>()==0x8086,"MMVQ requires Intel GPU");std::printf("NATIVE device=%s driver=%s\n",dev.get_info<sycl::info::device::name>().c_str(),dev.get_info<sycl::info::device::driver_version>().c_str());
#else
 std::puts("HOST_SIMULATION_ONLY: no device, compiler lowering or speed claim");
#endif
 std::mt19937 rng(0x65a1);Tables tables(q);size_t checks=0;
 const std::array<std::string,3> signatures[]={{"IQ2_XXS","IQ2_XXS","IQ2_XXS"},{"IQ1_M","IQ1_M","IQ1_M"},{"IQ2_XS","IQ2_XS","IQ1_M"},{"IQ2_XXS","IQ2_XXS","IQ2_XS"},{"IQ2_XS","IQ2_XS","IQ2_XXS"},{"IQ2_XS","IQ2_XS","IQ2_XS"}};
#ifndef STRATA_MMVQ_TIMING
 mmvq_actual_k(q,tables,rng);
 for(const auto& sig:signatures){MmvqFixture f(q,tables);for(int i=0;i<2;++i)f.add(synthetic(sig,rng));f.ready();for(int n:{2,1})for(int probe=0;probe<3;++probe){std::vector<float>x(256);for(size_t i=0;i<x.size();++i)x[i]=probe==1?-0.0f:probe==2?(i%32==0?127.0f:i%2?-2.5f:1.5f):float(int(rng()%255)-127)/32;f.check(n,x,probe==2?1.0f:0.0f);++checks;}}
 {MmvqFixture f(q,tables);for(int i=0;i<8;++i)f.add(synthetic(signatures[2],rng));f.ready();std::vector<float>x(256,0.125f);f.check(8,x,1);++checks;}
#endif
#if !defined(STRATA_IQ2_HOST_TEST) || defined(STRATA_EXPERT_NATIVE_SIMULATION)
 if(!pack.empty()){
  auto ws=open_pack(pack);need(ws->meta().n_exp>=8,"MMVQ real pack needs eight experts");std::set<std::string> found;const char* roles[]={"w1","w3","w2"};
  // Optional binary records: uint32 layer, uint32 dim, then dim float32 values.
  // Repeated layer records exercise actual captured inputs; reject mismatches.
  std::map<int,std::vector<std::vector<float>>> captures;
  if(!activation_path.empty()){std::ifstream file(activation_path,std::ios::binary);need(bool(file),"MMVQ activation open");size_t records=0;while(file.peek()!=EOF){uint32_t l=0,n=0;file.read(reinterpret_cast<char*>(&l),4);file.read(reinterpret_cast<char*>(&n),4);need(bool(file)&&l<uint32_t(ws->meta().nl)&&n==uint32_t(ws->meta().dim)&&++records<=4096,"MMVQ activation header");std::vector<float>x(n);file.read(reinterpret_cast<char*>(x.data()),std::streamsize(n)*4);need(bool(file),"truncated MMVQ activation");for(float v:x)need(std::isfinite(v),"nonfinite MMVQ activation");captures[int(l)].push_back(std::move(x));}need(!captures.empty(),"empty MMVQ captures");}
  for(int l=0;l<ws->meta().nl;++l){std::array<std::string,3> ty;std::string sig;for(int r=0;r<3;++r){ty[r]=ws->require_tensor(layer_prefix(l)+"ffn.experts.0."+roles[r]+".weight").dtype;sig+=ty[r]+"/";}
   if(!warm_packed_type(ty[0])||!warm_packed_type(ty[2])||ty[0]!=ty[1])continue;
   bool first=found.insert(sig).second;if(first)std::printf("REAL_SIGNATURE layer=%d signature=%s\n",l,sig.c_str());if(!first&&!captures.contains(l))continue;
   MmvqFixture f(q,tables);for(int e=0;e<8;++e){std::array<Matrix,3>a;for(int r=0;r<3;++r){auto name=layer_prefix(l)+"ffn.experts."+std::to_string(e)+"."+roles[r]+".weight";auto z=ws->require_tensor(name);need(z.shape.size()==2&&z.shape[0]>0&&z.shape[1]>0&&z.shape[0]<=INT32_MAX&&z.shape[1]<=INT32_MAX,"MMVQ pack shape");auto& mtx=a[r];mtx.type=z.dtype;mtx.n=int(z.shape[0]);mtx.k=int(z.shape[1]);mtx.ld=m::row_bytes(m::type(z.dtype),mtx.k);auto raw=ws->read_all(name);need(raw.size()==size_t(mtx.n)*mtx.ld,"MMVQ pack extent");mtx.bytes.resize(raw.size());std::memcpy(mtx.bytes.data(),raw.data(),raw.size());std::printf("TENSOR %s bytes=%zu\n",name.c_str(),raw.size());}f.add(std::move(a));}f.ready();
   std::vector<float>x(size_t(f.w->dim));for(auto& v:x)v=float(int(rng()%255)-127)/32;std::printf("MMVQ_CASE layer=%d signature=%s input=synthetic\n",l,sig.c_str());f.check(8,x,ws->meta().lim);f.prefixes(x,ws->meta().lim);++checks;
   if(captures.contains(l)){for(const auto& input:captures.at(l)){std::printf("MMVQ_CASE layer=%d signature=%s input=captured\n",l,sig.c_str());f.check(8,input,ws->meta().lim);++checks;}captures.erase(l);}
#ifdef STRATA_MMVQ_TIMING
   if(first){Device<uint32_t> scrub(q,64*1024*1024);auto ptr=scrub.p;for(int n:{1,2,4,8})for(int cache=0;cache<2;++cache){for(int warm=0;warm<3;++warm){f.run(n,false,x,ws->meta().lim);f.run(n,true,x,ws->meta().lim);}for(int rep=0;rep<5;++rep)for(int arm=0;arm<2;++arm){bool soa=(arm+rep)%2;if(cache)q.parallel_for(sycl::range<1>(scrub.n),[=](sycl::id<1> i){ptr[i[0]]=uint32_t(i[0])*1664525u+1013904223u;}).wait_and_throw();auto tm=f.run(n,soa,x,ws->meta().lim);std::printf("MMVQ_TIMING {\"signature\":\"%s\",\"layer\":%d,\"n\":%d,\"cache\":\"%s\",\"mode\":\"%s\",\"rep\":%d,\"host_ns\":%llu,\"span_ns\":%llu,\"gate_up_ns\":%llu,\"down_ns\":%llu}\n",sig.c_str(),l,n,cache?"scrubbed":"warm",soa?"mmvq-soa":"exact-grouped",rep,(unsigned long long)tm.host,(unsigned long long)tm.span,(unsigned long long)tm.gate_up,(unsigned long long)tm.down);}}}
#endif
  }
  need(!found.empty(),"MMVQ pack has no packed-supported signatures");need(captures.empty(),"unconsumed MMVQ captured layers");
  std::printf("REAL_SIGNATURES %zu\n",found.size());
 }else need(activation_path.empty(),"MMVQ captures require pack");
#endif
 std::printf("PASS: MMVQ %zu batches; proposed numerical screens only; full-model quality still required\n",checks);return 0;
}
