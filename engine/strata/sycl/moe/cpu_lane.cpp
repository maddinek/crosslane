// Extracted one-expert decode body of Strata c62d5e3 ds4_model.cu cpu_experts_iq.
// Copyright (c) 2026 Peasant Smith / Project Maya contributors, MIT; licenses/.
#include "../stage/wiring/cpu_lane.hpp"
#include "format.hpp"
#include "row_team.hpp"
#include "../../warm/runtime.hpp"
#include "ds4_gguf_cpu.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <fstream>
#include <mutex>
#include <set>
#include <thread>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <climits>
namespace strata::wiring {
namespace {
namespace gg=strata::ds4::gguf;namespace m=strata::ds4::moe;
void need(bool b,const char* s){if(!b)throw std::invalid_argument(s);}
// Match the API's physical-core default; an explicit count remains bounded by this host.
int physical_cores(){
    std::ifstream in("/proc/cpuinfo");std::string line,physical="0",core;std::set<std::pair<std::string,std::string>> found;
    while(std::getline(in,line)){auto pos=line.find(':');if(pos==std::string::npos)continue;
        if(line.starts_with("physical id"))physical=line.substr(pos+1);
        if(line.starts_with("core id")){core=line.substr(pos+1);found.emplace(physical,core);}}
    return found.empty()?1:int(found.size());
}
class Lane final:public CpuLane {
    const WeightSource& ws_;uint32_t first_,end_;int workers_;
    std::string dispatch_;bool timing_,batch_=false;
    unsigned prefetch_rows_=0;
    std::unique_ptr<RowTeam> team_;
#ifdef STRATA_FORK_WARM
    std::shared_ptr<WarmStore> store_;
#endif
    uint64_t calls_=0,read_bytes_=0,read_ns_=0,q1_ns_=0,gu_ns_=0,q2_ns_=0,down_ns_=0,total_ns_=0;
    uint64_t kernel_thread_ns_=0,dispatch_outer_ns_=0,spawns_=0,completion_wait_ns_=0;
    static uint64_t clock_ns(){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}
    bool decode_active_=false; CpuPhases decode_start_{};
    CpuPhases counters()const{return {calls_,read_bytes_,read_ns_,q1_ns_,gu_ns_,q2_ns_,down_ns_,total_ns_,kernel_thread_ns_,dispatch_outer_ns_,completion_wait_ns_};}
    uint64_t now()const{return timing_||decode_active_?clock_ns():0;}
    std::mutex mutex_;
    static void prefetch(const uint8_t* w,size_t stride,int first,int end) {
        for(int r=first;r<end;++r)for(size_t b=0;b<stride;b+=64)
            __builtin_prefetch(w+size_t(r)*stride+b,0,3);
    }
    void gate_up(const gg::IqCpu& f,const uint8_t* g,const uint8_t* u,const void* const* a,float* const* h,int lo,int hi,float limit) {
        if(!prefetch_rows_){gg::iq_cpu_gate_up(f,g,u,a,1,h,lo,hi,limit);return;}
        // Bounded lookahead within this worker's assigned rows. Four-row
        // calls keep each dot/fold/libm operation identical; no new layout.
        const int lead=int(prefetch_rows_);
        prefetch(g,f.gu_row,lo,std::min(hi,lo+lead));prefetch(u,f.gu_row,lo,std::min(hi,lo+lead));
        for(int r=lo;r<hi;r+=4){
            prefetch(g,f.gu_row,std::min(hi,r+lead),std::min(hi,r+lead+4));
            prefetch(u,f.gu_row,std::min(hi,r+lead),std::min(hi,r+lead+4));
            gg::iq_cpu_gate_up(f,g,u,a,1,h,r,std::min(hi,r+4),limit);
        }
    }
    void down(const gg::IqCpu& f,const uint8_t* d,const void* const* a,float* const* y,int lo,int hi) {
        if(!prefetch_rows_){gg::iq_cpu_down(f,d,a,1,y,lo,hi);return;}
        const int lead=int(prefetch_rows_);prefetch(d,f.d_row,lo,std::min(hi,lo+lead));
        for(int r=lo;r<hi;r+=4){prefetch(d,f.d_row,std::min(hi,r+lead),std::min(hi,r+lead+4));
            gg::iq_cpu_down(f,d,a,1,y,r,std::min(hi,r+4));}
    }
    template<class F>void dispatch_rows(int n,F f){
        if(team_){team_->rows(n,f,(timing_||decode_active_)?&completion_wait_ns_:nullptr);return;}
        constexpr int RB=64;int jobs=(n+RB-1)/RB,nt=std::min(workers_,jobs);
        // Prime upstream lazy ISA policy on the owner thread before parallel row calls.
        f(0,std::min(n,RB));if(jobs==1)return;
        std::atomic<int> next{1};std::exception_ptr error;std::mutex err_mutex;
        auto run=[&]{try{for(int j;(j=next.fetch_add(1))<jobs;)f(j*RB,std::min(n,(j+1)*RB));}catch(...){std::lock_guard lock(err_mutex);if(!error)error=std::current_exception();next=jobs;}};
        std::vector<std::thread> threads;
        try{for(int i=1;i<nt;++i){threads.emplace_back(run);++spawns_;}}catch(...){next=jobs;for(auto& t:threads)t.join();throw;}
        run();const auto tick=now();for(auto& t:threads)t.join();completion_wait_ns_+=now()-tick;
        if(error)std::rethrow_exception(error);
    }
    template<class F>void rows(int n,F f){
        if(!timing_&&!decode_active_){dispatch_rows(n,f);return;}
        std::atomic<uint64_t> sum{0},first{UINT64_MAX},last{0};
        const auto begin=clock_ns();
        dispatch_rows(n,[&](int a,int b){
            const auto start=clock_ns();f(a,b);const auto finish=clock_ns();
            sum.fetch_add(finish-start,std::memory_order_relaxed);
            auto old=first.load();while(start<old&&!first.compare_exchange_weak(old,start)){}
            old=last.load();while(finish>old&&!last.compare_exchange_weak(old,finish)){}
        });
        const auto end=clock_ns();
        kernel_thread_ns_+=sum.load();
        // Only before-first/after-last work. Internal scheduling gaps are NOT attributed here.
        dispatch_outer_ns_+=(end-begin)-(last.load()-first.load());
    }
    TensorInfo info(uint32_t l,int e,const char* role,int n,int k){
        auto t=ws_.require_tensor(layer_prefix(l)+"ffn.experts."+std::to_string(e)+"."+role+".weight");
        need(t.shape==std::vector<int64_t>{n,k},"row2 CPU expert shape");auto ty=m::type(t.dtype);m::ggml_type(ty);
        need(t.bytes==m::product(size_t(n),m::row_bytes(ty,k)),"row2 CPU expert byte extent");return t;
    }
public:
    Lane(const WeightSource& ws,uint32_t first,uint32_t end,int threads,const std::string& dispatch,bool timing):ws_(ws),first_(first),end_(end),workers_(threads?threads:physical_cores()),dispatch_(dispatch),timing_(timing){
        need(dispatch_=="spawn-v1"||dispatch_=="pool-v1","unknown CPU dispatch version");
        if(const char* v=std::getenv("DS4_CPU_LAYER_BATCH")) {
            need(std::string_view(v)=="0"||std::string_view(v)=="1","DS4_CPU_LAYER_BATCH must be 0 or 1");
            batch_=std::string_view(v)=="1";
        }
        need(!batch_||dispatch_=="pool-v1","CPU layer batch requires pool-v1");
        prefetch_rows_=cpu_option("DS4_CPU_PREFETCH_ROWS",32);
        need(!prefetch_rows_||prefetch_rows_%4==0,"CPU prefetch lead must be a multiple of four rows");
        need(!cpu_option("DS4_CPU_SPIN_US",200)||dispatch_=="pool-v1","CPU spin requires pool-v1");
        need(first<end&&end<=uint32_t(ws.meta().nl)&&threads>=0,"row2 CPU layer/thread range");need(workers_>0&&workers_<=int(std::max(1u,std::thread::hardware_concurrency())),"row2 CPU thread count exceeds host");
        need(ws.meta().dim>0&&ws.meta().ff>0&&ws.meta().n_exp>0&&std::isfinite(ws.meta().lim),"row2 CPU model geometry/limit");
        // Admission checks every expert before any execution; no eagerly loaded multi-GB weight store.
        for(uint32_t l=first;l<end;++l)for(int e=0;e<ws.meta().n_exp;++e){
            auto g=info(l,e,"w1",ws.meta().ff,ws.meta().dim),u=info(l,e,"w3",ws.meta().ff,ws.meta().dim),d=info(l,e,"w2",ws.meta().dim,ws.meta().ff);
            need(g.dtype==u.dtype,"row2 CPU gate/up formats differ");gg::IqCpu f;std::string err;
            if(!gg::iq_cpu_fmt(m::ggml_type(m::type(g.dtype)),m::ggml_type(m::type(d.dtype)),ws.meta().dim,ws.meta().ff,f,err))throw std::runtime_error("row2 CPU admission: "+err);
            need(f.gu_row==g.bytes/size_t(ws.meta().ff)&&f.d_row==d.bytes/size_t(ws.meta().dim),"row2 CPU ggml row-layout mismatch");
        }
        auto affinity=cpu_affinity(workers_);
        need(affinity.empty()||dispatch_=="pool-v1","CPU affinity requires pool-v1");
        if(dispatch_=="pool-v1"){team_=std::make_unique<RowTeam>(workers_,affinity);spawns_=team_->background_threads();}
        if(!affinity.empty())std::fprintf(stderr,"AIOPT_CPU_AFFINITY cpus=%s scope=row-team owner_restore=1\n",std::getenv("DS4_CPU_AFFINITY"));
#ifdef STRATA_FORK_WARM
        store_=warm_store(ws_);
#endif
        std::fprintf(stderr,"AIOPT_CPU_DISPATCH version=%s first=%u end=%u workers=%d persistent_threads=%zu timing=%d\n",dispatch_.c_str(),first_,end_,workers_,team_?team_->background_threads():size_t(0),int(timing_));
        if(batch_)std::fprintf(stderr,"AIOPT_CPU_LAYER_BATCH version=1 shared_input_quant=1 row_block=64 workers=%d\n",workers_);
        if(prefetch_rows_||cpu_option("DS4_CPU_SPIN_US",200))std::fprintf(stderr,"AIOPT_CPU_CANDIDATE prefetch_rows=%u spin_us=%u\n",prefetch_rows_,cpu_option("DS4_CPU_SPIN_US",200));
    }
    ~Lane()override{
        if(timing_)std::fprintf(stderr,"AIOPT_CPU_TIMING {\"schema\":\"cpu-lane-timing-v1\",\"dispatch\":\"%s\",\"first\":%u,\"end\":%u,\"workers\":%d,\"scope\":\"lane-lifetime-including-prefill\",\"completed_experts\":%llu,\"thread_spawns\":%llu,\"logical_expert_read_bytes\":%llu,\"read_ns\":%llu,\"quant1_ns\":%llu,\"gate_up_wall_ns\":%llu,\"quant2_ns\":%llu,\"down_wall_ns\":%llu,\"row_kernel_thread_sum_ns\":%llu,\"dispatch_outside_kernel_envelope_ns\":%llu,\"expert_wall_ns\":%llu}\n",
            dispatch_.c_str(),first_,end_,workers_,(unsigned long long)calls_,(unsigned long long)spawns_,(unsigned long long)read_bytes_,(unsigned long long)read_ns_,(unsigned long long)q1_ns_,(unsigned long long)gu_ns_,(unsigned long long)q2_ns_,(unsigned long long)down_ns_,(unsigned long long)kernel_thread_ns_,(unsigned long long)dispatch_outer_ns_,(unsigned long long)total_ns_);
    }
    std::vector<float> expert(uint32_t l,int e,std::span<const float> x)override{
        std::lock_guard lock(mutex_);const auto& g=ws_.meta();need(l>=first_&&l<end_&&e>=0&&e<g.n_exp&&x.size()==size_t(g.dim),"row2 CPU expert request");
        for(float v:x)need(std::isfinite(v),"row2 CPU nonfinite activation");
#ifdef STRATA_FORK_WARM
        const auto warm_begin=clock_ns();
        if(auto* remote=warm_redirect();remote&&remote->warm_contains(l,e)){
            auto y=remote->warm_compute(l,e,x);warm_note_sycl(l,e,clock_ns()-warm_begin);return y;
        }
#endif
        const auto begin=now();
        auto gate=info(l,e,"w1",g.ff,g.dim),up=info(l,e,"w3",g.ff,g.dim),down=info(l,e,"w2",g.dim,g.ff);
        gg::IqCpu f;std::string err;if(!gg::iq_cpu_fmt(m::ggml_type(m::type(gate.dtype)),m::ggml_type(m::type(down.dtype)),g.dim,g.ff,f,err))throw std::runtime_error(err);
        auto tick=now();std::vector<std::byte> gb,ub,db;
        std::array<const uint8_t*,3> weights{};
#ifdef STRATA_FORK_WARM
        if(store_)weights=store_->matrices(l,e);
        else
#endif
        {gb=ws_.read_all(gate.name);ub=ws_.read_all(up.name);db=ws_.read_all(down.name);
         weights={reinterpret_cast<const uint8_t*>(gb.data()),reinterpret_cast<const uint8_t*>(ub.data()),reinterpret_cast<const uint8_t*>(db.data())};}
        read_ns_+=now()-tick;read_bytes_+=gate.bytes+up.bytes+down.bytes;
        // max_align_t aligned storage for ggml SIMD loads, including activation-block metadata.
        std::vector<std::max_align_t> aq((f.act_bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t)),hq((f.h_bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));
        std::vector<float> h(size_t(g.ff)),y(size_t(g.dim));tick=now();gg::iq_cpu_quant(f.gu_act,x.data(),g.dim,aq.data());q1_ns_+=now()-tick;
        const void* ap[1]={aq.data()};float* hp[1]={h.data()};
        tick=now();
        rows(g.ff,[&](int a,int b){gate_up(f,weights[0],weights[1],ap,hp,a,b,g.lim);});
        gu_ns_+=now()-tick;tick=now();
        gg::iq_cpu_quant(f.d_act,h.data(),g.ff,hq.data());q2_ns_+=now()-tick;const void* qp[1]={hq.data()};float* yp[1]={y.data()};
        tick=now();
        rows(g.dim,[&](int a,int b){this->down(f,weights[2],qp,yp,a,b);});
        down_ns_+=now()-tick;
        for(float v:y)need(std::isfinite(v),"row2 CPU nonfinite output");
        ++calls_;total_ns_+=now()-begin;
#ifdef STRATA_FORK_WARM
        if(store_)warm_note_cpu(l,e,clock_ns()-warm_begin);
#endif
        return y;
    }
    bool layer_batch()const override{return batch_;}
    std::vector<std::vector<float>> experts(uint32_t l,std::span<const int> ids,std::span<const float> x)override {
        // No quantization/dispatch can be shared for a singleton; retain the
        // original body instead of paying batch metadata and scratch setup.
        if(!batch_||ids.size()==1)return CpuLane::experts(l,ids,x);
        std::lock_guard lock(mutex_);
        const auto& g=ws_.meta();
        need(l>=first_&&l<end_&&x.size()==size_t(g.dim)&&ids.size()<=size_t(g.n_exp),"CPU batch shape");
        std::set<int> unique;
        for(int e:ids)need(e>=0&&e<g.n_exp&&unique.insert(e).second,"CPU batch invalid/duplicate expert");
        for(float v:x)need(std::isfinite(v),"CPU batch nonfinite input");
        if(ids.empty())return {};
        const auto begin=now();
#ifdef STRATA_FORK_WARM
        const auto warm_begin=clock_ns();
#endif
        using Aligned=std::vector<std::max_align_t>;
        auto aligned=[](size_t bytes){return Aligned((bytes+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));};
        struct Job {
            gg::IqCpu f;std::array<const uint8_t*,3> w{};
            std::array<std::vector<std::byte>,3> owned;
            Aligned hq;std::vector<float> h;const void* aq=nullptr;
        };
        std::vector<Job> jobs(ids.size());std::vector<std::vector<float>> out(ids.size());
        // Activation types may differ between experts. Reuse only identical
        // (type, byte extent) quantizers within this invocation, never across x.
        std::map<std::pair<int,size_t>,Aligned> inputs;
        for(size_t i=0;i<ids.size();++i){
            auto& j=jobs[i];const int e=ids[i];
#ifdef STRATA_FORK_WARM
            need(!warm_redirect()||!warm_redirect()->warm_contains(l,e),"CPU batch contains GPU expert");
            need(!warm_redirect(true)||!warm_redirect(true)->warm_contains(l,e),"CPU batch contains HIP expert");
#endif
            auto gate=info(l,e,"w1",g.ff,g.dim),up=info(l,e,"w3",g.ff,g.dim),down=info(l,e,"w2",g.dim,g.ff);
            std::string err;need(gg::iq_cpu_fmt(m::ggml_type(m::type(gate.dtype)),m::ggml_type(m::type(down.dtype)),g.dim,g.ff,j.f,err),"CPU batch format");
            auto tick=now();
#ifdef STRATA_FORK_WARM
            if(store_)j.w=store_->matrices(l,e);else
#endif
            {j.owned={ws_.read_all(gate.name),ws_.read_all(up.name),ws_.read_all(down.name)};
             for(size_t r=0;r<3;++r)j.w[r]=reinterpret_cast<const uint8_t*>(j.owned[r].data());}
            read_ns_+=now()-tick;read_bytes_+=gate.bytes+up.bytes+down.bytes;
            auto key=std::make_pair(j.f.gu_act,j.f.act_bytes);auto it=inputs.find(key);
            if(it==inputs.end()){
                auto buffer=aligned(j.f.act_bytes);tick=now();
                gg::iq_cpu_quant(j.f.gu_act,x.data(),g.dim,buffer.data());q1_ns_+=now()-tick;
                it=inputs.emplace(key,std::move(buffer)).first;
            }
            j.aq=it->second.data();j.hq=aligned(j.f.h_bytes);j.h.resize(size_t(g.ff));out[i].resize(size_t(g.dim));
        }
        // Prime each expert's format on the owner before any parallel call:
        // upstream's ISA policy has lazy process-wide state. Rows >=64 are
        // interleaved across experts; each row keeps its original dot order.
        auto phase=[&](int n,auto fn){
            CpuPin pin(team_->owner_cpu());
            for(size_t i=0;i<jobs.size();++i){
                if(timing_||decode_active_){const auto start=clock_ns();fn(i,0,std::min(n,64));kernel_thread_ns_+=clock_ns()-start;}
                else fn(i,0,std::min(n,64));
            }
            const size_t chunks=size_t((n-1)/64),tasks=chunks*jobs.size();
            need(tasks<=size_t(INT_MAX/64),"CPU batch row-job overflow");
            if(tasks)rows(int(tasks*64),[&](int a,int){
                const size_t task=size_t(a/64),i=task%jobs.size();
                const int lo=int(1+task/jobs.size())*64;fn(i,lo,std::min(n,lo+64));
            });
        };
        auto tick=now();phase(g.ff,[&](size_t i,int a,int b){auto& j=jobs[i];const void* ap[]={j.aq};float* hp[]={j.h.data()};gate_up(j.f,j.w[0],j.w[1],ap,hp,a,b,g.lim);});gu_ns_+=now()-tick;
        tick=now();for(auto& j:jobs)gg::iq_cpu_quant(j.f.d_act,j.h.data(),g.ff,j.hq.data());q2_ns_+=now()-tick;
        tick=now();phase(g.dim,[&](size_t i,int a,int b){auto& j=jobs[i];const void* qp[]={j.hq.data()};float* yp[]={out[i].data()};down(j.f,j.w[2],qp,yp,a,b);});down_ns_+=now()-tick;
        for(const auto& row:out)for(float v:row)need(std::isfinite(v),"CPU batch nonfinite output");
        calls_+=ids.size();total_ns_+=now()-begin;
#ifdef STRATA_FORK_WARM
        if(store_){const auto ns=clock_ns()-warm_begin;for(size_t i=0;i<ids.size();++i)warm_note_cpu(l,ids[i],i?0:ns);}
#endif
        return out;
    }
#ifdef STRATA_FORK_WARM
    void prepare(uint32_t l,std::span<const int> ids)override{if(store_)store_->prepare(l,ids);}
    void finish()noexcept override{if(store_)store_->finish();}
#endif
    void drain()noexcept override{} // synchronous, no pending jobs after expert() returns
    void begin_decode_step(bool active)override{decode_active_=active;if(active)decode_start_=counters();}
    CpuPhases decode_phases()const override{return decode_active_?counters()-decode_start_:CpuPhases{};}
};
}
std::unique_ptr<CpuLane> make_cpu_lane(const WeightSource& ws,uint32_t first,uint32_t end,int threads,const std::string& dispatch,bool timing){return std::make_unique<Lane>(ws,first,end,threads,dispatch,timing);}
}
