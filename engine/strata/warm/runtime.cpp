// MIT. One upstream ExpertStore and redirect registry in libstrata-row2-cpu.so.
#include "runtime.hpp"
#include "host_partition.hpp"
#include "worker.hpp"
#include "other_combine.hpp"
#include "diagnostic.hpp"
#include "ds4_store.hpp"
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <set>
#include <numeric>
#include "../sycl/moe/cpu_options.hpp"
#include <sys/mman.h>
#include <cerrno>
namespace strata::wiring {
extern "C" HostPartition* strata_warm_host_partition_v1(){static HostPartition value;return &value;}
namespace {
using Store=strata::ds4::ExpertStore;
void check(bool x,const char* why){if(!x)throw std::runtime_error(why);}
uint64_t number(const char* key,uint64_t fallback){
 const char* s=std::getenv(key);if(!s)return fallback;
 std::string v(s);size_t end=0;auto n=std::stoull(v,&end);
 check(!v.empty()&&v[0]!='-'&&end==v.size(),"invalid warm numeric environment");return n;
}
WarmRedirect* redirect=nullptr;WarmRedirect* hip_redirect=nullptr;
struct ServiceWorkers {WarmWorker cpu,sycl,hip;};std::unique_ptr<ServiceWorkers> service_workers;
std::vector<std::vector<int>> gpu_sycl,gpu_hip;
std::vector<uint32_t> decode_profile,measured_profile;
size_t profile_experts=0,request_number=0;bool profile_active=false;
std::vector<uint64_t> joins,join_ns;std::array<std::vector<uint64_t>,3> service_ns;std::array<std::vector<uint64_t>,2> gpu_ns,gpu_calls;
void profile_add(uint32_t l,int e){if(profile_active){check(e>=0&&size_t(e)<profile_experts&&size_t(l)*profile_experts+size_t(e)<decode_profile.size(),"warm profile range");auto& n=decode_profile[size_t(l)*profile_experts+size_t(e)];check(n<UINT32_MAX,"warm profile overflow");++n;}}
void save_profile(const std::string& path,const std::vector<uint32_t>& words){
 check(!std::ifstream(path).good(),"warm decode profile output exists");std::ofstream f(path,std::ios::binary);check(bool(f),"warm decode profile open");
 for(uint32_t word:words){uint8_t bytes[]={uint8_t(word),uint8_t(word>>8),uint8_t(word>>16),uint8_t(word>>24)};f.write(reinterpret_cast<const char*>(bytes),4);}check(bool(f),"warm profile write");
}
bool on_gpu(size_t l,int e){return (l<gpu_sycl.size()&&size_t(e)<gpu_sycl[l].size()&&gpu_sycl[l][size_t(e)]>=0)||(l<gpu_hip.size()&&size_t(e)<gpu_hip[l].size()&&gpu_hip[l][size_t(e)]>=0);}
struct OwnedStore final:WarmStore {
 std::vector<strata::ds4::os::fd_t> direct,buffered;
 std::unique_ptr<Store> store;
 std::vector<int> ids;std::vector<char> ok;uint32_t layer=0;
 explicit OwnedStore(const WeightSource& ws) {
  namespace os=strata::ds4::os;
  const auto paths=ws.shard_paths();check(!paths.empty(),"warm store needs actual shard paths");
  try{
   for(const auto& p:paths){auto fd=os::open_ro(p);check(fd!=os::kBadFd,"warm buffered shard open failed");buffered.push_back(fd);direct.push_back(os::open_direct(p));}
   std::vector<Store::LayerGeo> geo(size_t(ws.meta().nl));
   const char* roles[]={"w1","w3","w2"};
   for(int l=0;l<ws.meta().nl;++l){auto& g=geo[size_t(l)];g.n_exp=ws.meta().n_exp;
    uint64_t offset=0;
    for(int r=0;r<3;++r){auto name=[&](int e){return layer_prefix(uint32_t(l))+"ffn.experts."+std::to_string(e)+"."+roles[r]+".weight";};
     const auto a=ws.require_tensor(name(0));check(a.bytes>0,"empty warm matrix");
     g.file[r]=a.file;g.off[r]=a.offset;g.bytes[r]=a.bytes;
     check(a.file>=0&&size_t(a.file)<paths.size(),"warm shard index");
     const auto b=g.n_exp>1?ws.require_tensor(name(1)):a;
     check(b.offset>=a.offset,"warm tensor stride order");g.stride[r]=b.offset-a.offset;
     for(int e=0;e<g.n_exp;++e){auto t=ws.require_tensor(name(e));check(t.file==a.file&&t.bytes==a.bytes&&t.offset==a.offset+uint64_t(e)*g.stride[r],"nonuniform warm store tensor stride");}
     g.tr[r]=offset;offset+=a.bytes+Store::kSlack;
    }g.blob=offset;
   }
   profile_experts=size_t(ws.meta().n_exp);decode_profile.assign(size_t(ws.meta().nl)*profile_experts,0);measured_profile=decode_profile;joins.assign(size_t(ws.meta().nl),0);join_ns=joins;for(auto& row:service_ns)row=joins;for(auto& row:gpu_ns)row=joins;for(auto& row:gpu_calls)row=joins;
   Store::Cfg cfg;const auto threads=number("DS4_STORE_THREADS",32);
   check(threads>0&&threads<=256,"warm store thread bound");cfg.threads=int(threads);
   cfg.budget_bytes=number("DS4_STORE_BYTES",0);check(cfg.budget_bytes>0,"warm store requires explicit DS4_STORE_BYTES");
   cfg.floor_bytes=number("DS4_STORE_FLOOR_BYTES",8ull<<30);cfg.reserve=0;
   cfg.ram_avail=[](){std::ifstream f("/proc/meminfo");std::string key,unit;uint64_t n;while(f>>key>>n>>unit)if(key=="MemAvailable:")return n*1024;return uint64_t(0);};
   store=std::make_unique<Store>();check(store->init(cfg,geo,direct,buffered),"warm store initialization failed");
   // The arena is a private anonymous mapping, not malloc memory or a model
   // file. Advice covers our own reservation before the preload first-touch.
   // No global THP policy changes, huge pool, pinning or blocking collapse.
   if(std::getenv("DS4_STORE_THP")) {
    const auto thp=cpu_option("DS4_STORE_THP",1);
    check(!store->gpu_mapped(),"CPU THP candidate requires an unmapped store");
    const auto lo=store->arena_lo(),hi=store->arena_hi();
    const int rc=::madvise(reinterpret_cast<void*>(uintptr_t(lo)),size_t(hi-lo),thp?MADV_HUGEPAGE:MADV_NOHUGEPAGE);
    const int saved=rc?errno:0;
    std::fprintf(stderr,"AIOPT_CPU_THP requested=%u lo=%llx hi=%llx advice_rc=%d errno=%d actual_pages=see_smaps\n",thp,lo,hi,rc,saved);
    check(rc==0,"CPU store THP advice failed");
   }
   if(cpu_option("DS4_CPU_MEMORY_REPORT",1))std::fprintf(stderr,"AIOPT_CPU_ARENA lo=%llx hi=%llx\n",store->arena_lo(),store->arena_hi());
   std::fprintf(stderr,"AIOPT_WARM_STORE threads=%d budget=%llu slot_bytes=%zu slots=%zu floor=%zu\n",cfg.threads,(unsigned long long)cfg.budget_bytes,store->slot_bytes(),store->slots_max(),cfg.floor_bytes);
   if(number("DS4_STORE_PRELOAD",0)==1){
    size_t needed=0;for(size_t l=0;l<geo.size();++l)for(int e=0;e<geo[l].n_exp;++e)needed+=!on_gpu(l,e);
    check(needed<=store->slots_max(),"RAM slots do not cover all non-GPU experts");
    for(size_t l=0;l<geo.size();++l){std::vector<int> es;for(int e=0;e<geo[l].n_exp;++e)if(!on_gpu(l,e))es.push_back(e);
     std::vector<char> ready;if(!es.empty()){bool loaded=store->ensure(int(l),es.data(),int(es.size()),ready);store->release(int(l),es.data(),int(es.size()),ready);check(loaded,"warm preload failed");}}
    std::fprintf(stderr,"AIOPT_WARM_PRELOAD experts=%zu slots=%zu allocated_bytes=%zu complete=1\n",needed,store->slots_allocated(),store->slots_allocated()*store->slot_bytes());
   }
  }catch(...){store.reset();close_files();throw;}
 }
 void close_files()noexcept{for(auto fd:direct)if(fd!=strata::ds4::os::kBadFd)strata::ds4::os::close_fd(fd);for(auto fd:buffered)strata::ds4::os::close_fd(fd);direct.clear();buffered.clear();}
 ~OwnedStore(){finish();store.reset();close_files();}
 void prepare(uint32_t l,std::span<const int> selected)override{
  finish();layer=l;std::set<int> unique;
  for(int e:selected){check(e>=0&&size_t(e)<profile_experts,"warm expert range");if((!redirect||!redirect->warm_contains(l,e))&&unique.insert(e).second)ids.push_back(e);}
  if(!ids.empty()&&!store->ensure(int(l),ids.data(),int(ids.size()),ok)){finish();throw std::runtime_error("warm store demand failed/no slot");}
 }
 std::array<const uint8_t*,3> matrices(uint32_t l,int e)override{
  if(l!=layer||std::find(ids.begin(),ids.end(),e)==ids.end())prepare(l,std::span(&e,1));
  int slot=store->find(int(l),e);check(slot>=0,"warm demand not ready");return {store->mat(slot,0),store->mat(slot,1),store->mat(slot,2)};
 }
 void finish()noexcept override{if(store&&!ids.empty())store->release(int(layer),ids.data(),int(ids.size()),ok);ids.clear();ok.clear();}
};
std::weak_ptr<OwnedStore> shared_store;std::string shared_identity;
strata::forkwarm::StoreDiagnostic diagnostic;
uint64_t cpu_count=0,sycl_count=0,hip_count=0,cpu_ns=0,sycl_ns=0;
}
void warm_set_residency(const std::vector<std::vector<int>>& sycl,const std::vector<std::vector<int>>& hip){gpu_sycl=sycl;gpu_hip=hip;}
void warm_register(WarmRedirect* p,bool hip){auto& slot=hip?hip_redirect:redirect;check(p&&(!slot||slot==p),"duplicate warm GPU bank");slot=p;}
void warm_unregister(WarmRedirect* p)noexcept{if(redirect==p)redirect=nullptr;if(hip_redirect==p)hip_redirect=nullptr;}
WarmRedirect* warm_redirect(bool hip){return hip?hip_redirect:redirect;}
std::shared_ptr<WarmStore> warm_store(const WeightSource& ws){
 if(number("DS4_FORK_WARM",0)!=1)return {};
 if(auto p=shared_store.lock()){check(shared_identity==ws.identity(),"warm store model identity differs");return p;}
 auto p=std::make_shared<OwnedStore>(ws);shared_identity=ws.identity();shared_store=p;return p;
}
void warm_note_cpu(uint32_t l,int e,uint64_t ns){++cpu_count;cpu_ns+=ns;profile_add(l,e);}
void warm_note_sycl(uint32_t l,int e,uint64_t ns){++sycl_count;sycl_ns+=ns;profile_add(l,e);}
void warm_note_hip(uint32_t l,int e){++hip_count;profile_add(l,e);}
void warm_gpu_span(bool hip,uint32_t l,uint64_t ns){gpu_ns[size_t(hip)].at(l)+=ns;++gpu_calls[size_t(hip)].at(l);}
std::vector<float> warm_execute(CpuLane& cpu,uint32_t layer,std::span<const int> ids,std::span<const float> weights,std::span<const float> x,std::span<const float> shared,const std::function<void()>& before_hip,ExpertTimeline* timeline){
 auto* host=strata_warm_host_partition_v1();HostScope host_scope(host,HostPart::prepare);
 static const auto other_options=OtherOptions::environment();
 check(ids.size()==weights.size()&&x.size()==shared.size(),"warm layer shape");
 if(!service_workers)service_workers=std::make_unique<ServiceWorkers>();
 auto& cpu_worker=service_workers->cpu;auto& sycl_worker=service_workers->sycl;auto& hip_worker=service_workers->hip;
 std::array<std::vector<size_t>,3> picks;std::set<int> unique;
 for(size_t i=0;i<ids.size();++i){check(ids[i]>=0&&unique.insert(ids[i]).second&&std::isfinite(weights[i]),"warm invalid/duplicate route");
  size_t tier=hip_redirect&&hip_redirect->warm_contains(layer,ids[i])?2:redirect&&redirect->warm_contains(layer,ids[i])?1:0;picks[tier].push_back(i);}
 std::vector<int> cpu_ids;for(auto i:picks[0])cpu_ids.push_back(ids[i]);cpu.prepare(layer,cpu_ids);
 struct Finish{CpuLane& cpu;~Finish(){cpu.finish();}} finish{cpu};
 using Results=std::vector<std::pair<size_t,std::vector<float>>>;
 if(timeline&&timeline->active)for(size_t k=0;k<3;++k)timeline->counts[k]=picks[k].size();
 auto work=[&](size_t tier){
 if(timeline&&timeline->active&&!picks[tier].empty())timeline->start[tier]=ExpertTimeline::now();
 struct End {ExpertTimeline* t;size_t tier;~End(){if(t&&t->active&&t->counts[tier])t->end[tier]=ExpertTimeline::now();}} end{timeline,tier};
 const auto service_begin=std::chrono::steady_clock::now();Results result;
 if(tier==0&&cpu.layer_batch()&&!cpu_ids.empty()){
  auto rows=cpu.experts(layer,cpu_ids,x);check(rows.size()==cpu_ids.size(),"CPU batch output count");
  for(size_t j=0;j<rows.size();++j){auto& y=rows[j];
   check(y.size()==x.size()&&std::all_of(y.begin(),y.end(),[](float v){return std::isfinite(v);}),"CPU batch output shape/nonfinite");
   result.emplace_back(picks[0][j],std::move(y));
  }
 }else if(tier>0&&(tier==1?redirect:hip_redirect)&&(tier==1?redirect:hip_redirect)->warm_grouped()&&!picks[tier].empty()){
  std::vector<int> selected;for(auto i:picks[tier])selected.push_back(ids[i]);
  const auto begin=std::chrono::steady_clock::now();auto rows=(tier==1?redirect:hip_redirect)->warm_compute_many(layer,selected,x);
  const auto ns=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count());
  check(rows.size()==selected.size(),"warm grouped output count");
  for(size_t j=0;j<rows.size();++j){auto& y=rows[j];
   check(y.size()==x.size()&&std::all_of(y.begin(),y.end(),[](float v){return std::isfinite(v);}),"warm grouped output shape/nonfinite");
   // Count every selection/profile hit; charge the batch duration once, never N times.
   if(tier==1)warm_note_sycl(layer,selected[j],j==0?ns:0);else warm_note_hip(layer,selected[j]);result.emplace_back(picks[tier][j],std::move(y));
  }
 }else for(auto i:picks[tier]){auto begin=std::chrono::steady_clock::now();
  auto y=tier==0?cpu.expert(layer,ids[i],x):(tier==1?redirect:hip_redirect)->warm_compute(layer,ids[i],x);
  auto ns=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count());
  if(tier==1)warm_note_sycl(layer,ids[i],ns);if(tier==2)warm_note_hip(layer,ids[i]);
  check(y.size()==x.size()&&std::all_of(y.begin(),y.end(),[](float v){return std::isfinite(v);}),"warm output shape/nonfinite");result.emplace_back(i,std::move(y));}service_ns[tier].at(layer)+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-service_begin).count());return result;};
 std::array<uint64_t,3> service_before{};if(host->recording())for(size_t k=0;k<3;++k)service_before[k]=service_ns[k].at(layer);
 host->enter(HostPart::submit);
 std::future<Results> a,b,c;try{a=cpu_worker.submit([&]{return work(0);});b=sycl_worker.submit([&]{return work(1);});
 // Shared completion transfers exclusive HIP stream/scratch ownership to the worker.
 // On failure, drain CPU/B65 before their borrowed inputs/store pins leave scope.
 if(before_hip){HostScope shared_drain(host,HostPart::ffn_wait);before_hip();}
 c=hip_worker.submit([&]{return work(2);});}catch(...){if(a.valid())a.wait();if(b.valid())b.wait();if(c.valid())c.wait();throw;}
 host->enter(HostPart::join);
 const auto start=std::chrono::steady_clock::now();std::array<Results,3> results;std::exception_ptr error;
 // Always join every lane even if another lane fails; inputs stay pinned until all drain.
 try{results[0]=a.get();}catch(...){error=std::current_exception();}
 try{results[1]=b.get();}catch(...){if(!error)error=std::current_exception();}
 try{results[2]=c.get();}catch(...){if(!error)error=std::current_exception();}
 const auto wait=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
 ++joins.at(layer);join_ns.at(layer)+=uint64_t(wait);
 if(error)std::rethrow_exception(error);
 if(timeline&&timeline->active)timeline->combine_start=ExpertTimeline::now();
 host->enter(HostPart::combine);
 if(host->recording()){std::array<uint32_t,3> counts{},batches{};std::array<uint64_t,3> elapsed{};
 for(size_t k=0;k<3;++k){counts[k]=uint32_t(picks[k].size());bool grouped=k==0?cpu.layer_batch():(k==1?redirect:hip_redirect)&&(k==1?redirect:hip_redirect)->warm_grouped();batches[k]=grouped?uint32_t(counts[k]>0):counts[k];elapsed[k]=counts[k]?service_ns[k].at(layer)-service_before[k]:0;}
 host->record(layer,ids,weights,counts,batches,elapsed,uint64_t(wait));}
 if(other_options.simd){
  auto out=other_combine(ids,weights,shared,results,other_options);
  if(timeline)timeline->emit();return out;
 }
 std::vector<std::vector<float>> rows(ids.size());for(auto& tier:results)for(auto& item:tier)rows[item.first]=std::move(item.second);
 std::vector<size_t> order(ids.size());std::iota(order.begin(),order.end(),0);std::sort(order.begin(),order.end(),[&](auto i,auto j){return ids[i]<ids[j];});
 std::vector<float> out(x.size(),0);for(auto i:order)for(size_t k=0;k<x.size();++k)out[k]=std::fma(weights[i],rows[i][k],out[k]);
 for(size_t k=0;k<out.size();++k)out[k]+=shared[k];
 if(timeline)timeline->emit();return out;
}
extern "C" void strata_warm_stop_workers_v1(){service_workers.reset();}
extern "C" void strata_warm_reset_v1(){check(shared_store.expired()&&!redirect&&!hip_redirect,"warm reset requires drained calibration");cpu_count=sycl_count=hip_count=cpu_ns=sycl_ns=0;request_number=0;profile_active=false;}
extern "C" void strata_warm_boundary_v1(const char* event,uint64_t fed,uint64_t prompt,uint64_t outputs){
 if(std::string_view(event)=="begin"){profile_active=false;++request_number;std::fill(decode_profile.begin(),decode_profile.end(),0);}
 else if(std::string_view(event)=="prefill")profile_active=true;
 else if(std::string_view(event)=="end"){
  profile_active=false;
  if(const char* prefix=std::getenv("DS4_WARM_PROFILE_OUTPUT")){
   save_profile(std::string(prefix)+"-request-"+std::to_string(request_number)+".bin",decode_profile);
   if(request_number>1)for(size_t i=0;i<decode_profile.size();++i){check(measured_profile[i]<=UINT32_MAX-decode_profile[i],"measured profile overflow");measured_profile[i]+=decode_profile[i];}
   if(request_number==number("DS4_WARM_REQUESTS",4))save_profile(std::string(prefix)+"-measured.bin",measured_profile);
  }
 }
 if(number("DS4_STORE_DIAGNOSTIC",0)!=1)return;
 strata::forkwarm::StoreDiagnosticStats x;auto& m=x.model;
 m.tokens=int64_t(fed);m.experts=int64_t(cpu_count+sycl_count+hip_count);m.cpu_experts=int64_t(cpu_count);
 m.stream_experts=int64_t(sycl_count);m.cpu_ms=double(cpu_ns)/1e6;m.gpu_wait_ms=double(sycl_ns)/1e6;
 if(auto p=shared_store.lock()){const auto& s=p->store->stats();
  m.store_hits=s.hits.load();m.store_misses=s.misses.load();m.store_prefetched=s.prefetched.load();m.store_pf_used=s.prefetch_used.load();m.store_pf_wasted=s.prefetch_wasted.load();m.store_evictions=s.evictions.load();m.store_bytes=s.bytes_read.load();m.store_fails=s.fails.load();m.store_wait_ms=double(s.wait_ns.load())/1e6;m.store_busy_ms=p->store->busy_ms();m.store_slots=int64_t(p->store->slots_allocated());m.store_slots_max=int64_t(p->store->slots_max());
  x.no_slot=s.no_slot.load();x.cancelled=s.cancelled.load();x.put=s.put.load();x.slot_bytes=int64_t(p->store->slot_bytes());
 }
 if(std::string_view(event)=="begin")diagnostic.begin(x,int64_t(prompt));
 else if(std::string_view(event)=="prefill")diagnostic.prefill_end(x);
 else if(std::string_view(event)=="end")diagnostic.end(x,int64_t(outputs),true);
 else throw std::runtime_error("unknown warm boundary");
 for(size_t l=0;l<joins.size();++l)if(joins[l])std::fprintf(stderr,"AIOPT_WARM_LAYER_TOTALS request=%zu layer=%zu joins=%llu join_ns=%llu cpu_service_ns=%llu sycl_service_ns=%llu hip_service_ns=%llu sycl_event_ns=%llu hip_event_ns=%llu sycl_calls=%llu hip_calls=%llu\n",request_number,l,(unsigned long long)joins[l],(unsigned long long)join_ns[l],(unsigned long long)service_ns[0][l],(unsigned long long)service_ns[1][l],(unsigned long long)service_ns[2][l],(unsigned long long)gpu_ns[0][l],(unsigned long long)gpu_ns[1][l],(unsigned long long)gpu_calls[0][l],(unsigned long long)gpu_calls[1][l]);
 {std::ifstream io("/proc/self/io");std::string key;uint64_t bytes=0,value=0;while(io>>key>>value)if(key=="read_bytes:")bytes=value;
  std::fprintf(stderr,"AIOPT_WARM_IO request=%zu event=%s read_bytes=%llu scope=whole-process-including-Engram-and-runtime\n",request_number,event,(unsigned long long)bytes);}
 std::fprintf(stderr,"AIOPT_WARM_SELECTIONS cpu=%llu sycl=%llu hip=%llu\n",(unsigned long long)cpu_count,(unsigned long long)sycl_count,(unsigned long long)hip_count);
}
}
