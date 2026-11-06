// MIT. Included by strata-cpu-lane-bench after the historical fixture helpers.
// No shard copies: a contiguous subset aliases original pack extents into layer 0.
#pragma once
#include "cpu_pmu.hpp"
#include "worker.hpp"
#include "../sycl/moe/row_team.hpp"
#include <sys/resource.h>
#include <cmath>
#include <iomanip>
#include <numeric>
class RealBank final:public sw::WeightSource {
    const sw::WeightSource& pack_;sw::ModelMeta meta_;
    std::map<std::string,std::pair<sw::TensorInfo,std::string>> info_;
public:
    RealBank(const sw::WeightSource& p,int layer,int first,int count):pack_(p),meta_(p.meta()) {
        need(layer>=0&&layer<meta_.nl&&first>=0&&count>=1&&first<=meta_.n_exp-count,"real bank bounds");
        meta_.nl=1;meta_.n_exp=count;
        for(int e=0;e<count;++e)for(const char* role:{"w1","w3","w2"}){
            const auto orig=sw::layer_prefix(uint32_t(layer))+"ffn.experts."+std::to_string(first+e)+"."+role+".weight";
            auto t=p.require_tensor(orig);t.name="layers.0.ffn.experts."+std::to_string(e)+"."+role+".weight";
            info_[t.name]={t,orig};
        }
    }
    const sw::ModelMeta& meta()const override{return meta_;}
    std::string identity()const override{return pack_.identity()+":"+info_.begin()->second.second+":"+std::to_string(meta_.n_exp);}
    std::vector<std::string> shard_paths()const override{return pack_.shard_paths();}
    std::optional<sw::TensorInfo> find(std::string_view n)const override {auto it=info_.find(std::string(n));if(it==info_.end())return {};return it->second.first;}
    std::vector<std::string> names()const override{std::vector<std::string> v;for(const auto& [n,t]:info_){(void)t;v.push_back(n);}return v;}
    void read(std::string_view n,uint64_t off,std::span<std::byte> dst)const override{pack_.read(info_.at(std::string(n)).second,off,dst);}
    int engram_format(int)const override{throw std::runtime_error("unused");}
    void engram_rows(int,std::span<const uint64_t>,std::span<std::byte>)const override{throw std::runtime_error("unused");}
    void embedding_bf16(int32_t,std::span<uint16_t>)const override{throw std::runtime_error("unused");}
};
inline gg::IqCpu real_format(const sw::WeightSource& ws,int e) {
    const auto p="layers.0.ffn.experts."+std::to_string(e)+".";
    const auto g=ws.require_tensor(p+"w1.weight"),d=ws.require_tensor(p+"w2.weight");gg::IqCpu f;std::string err;
    need(gg::gg_cpu_fmt(type(g.dtype),type(d.dtype),ws.meta().dim,ws.meta().ff,f,err),err.c_str());return f;
}
using CpuAligned=std::vector<std::max_align_t>;
inline CpuAligned cpu_aligned(size_t n){return CpuAligned((n+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t));}
inline std::vector<float> real_oracle(const sw::WeightSource& ws,int e,const std::vector<float>& x) {
    const auto f=real_format(ws,e);const auto p="layers.0.ffn.experts."+std::to_string(e)+".";
    auto g=ws.read_all(p+"w1.weight"),u=ws.read_all(p+"w3.weight"),d=ws.read_all(p+"w2.weight");
    auto aq=cpu_aligned(f.act_bytes),hq=cpu_aligned(f.h_bytes);std::vector<float> h(size_t(f.n_ff)),y(size_t(f.n_embd));
    gg::gg_cpu_quant(f.gu_act,x.data(),f.n_embd,aq.data());const void* ap[]={aq.data()};float* hp[]={h.data()};
    gg::gg_cpu_gate_up(f,reinterpret_cast<const uint8_t*>(g.data()),reinterpret_cast<const uint8_t*>(u.data()),ap,1,hp,0,f.n_ff,ws.meta().lim);
    gg::gg_cpu_quant(f.d_act,h.data(),f.n_ff,hq.data());const void* qp[]={hq.data()};float* yp[]={y.data()};
    gg::gg_cpu_down(f,reinterpret_cast<const uint8_t*>(d.data()),qp,1,yp,0,f.n_embd);return y;
}
inline void cpu_snapshot(const std::filesystem::path& out,const char* suffix) {
    for(const char* name:{"smaps","status","sched","io"}) {
        std::ifstream in(std::string("/proc/self/")+name);need(bool(in),"proc snapshot open");
        std::ofstream file(out/(std::string(name)+"-"+suffix+".txt"));file<<in.rdbuf();need(bool(file),"proc snapshot write");
    }
}
inline void cpu_components(sw::WeightSource& source,sw::CpuLane& lane,sw::WarmStore& store,
        const std::vector<float>& x,const std::vector<std::vector<float>>& refs,int workers,int rounds,bool hot) {
    const int bank=source.meta().n_exp;std::vector<int> all(size_t(bank),0);std::iota(all.begin(),all.end(),0);lane.prepare(0,all);
    sw::RowTeam team(workers,sw::cpu_affinity(workers));
    for(int rep=-1;rep<3;++rep){uint64_t q1=0,q2=0,gu=0,dn=0;std::atomic<uint64_t> dot{0},silu{0};
        for(int r=0;r<rounds;++r){const int e=hot?0:r%bank;const auto f=real_format(source,e);const auto w=store.matrices(0,e);
            auto aq=cpu_aligned(f.act_bytes),hq=cpu_aligned(f.h_bytes);std::vector<float> g(size_t(f.n_ff)),u(g.size()),h(g.size()),y(size_t(f.n_embd));
            auto t=sw::decode_clock_ns();gg::iq_cpu_quant(f.gu_act,x.data(),f.n_embd,aq.data());q1+=sw::decode_clock_ns()-t;
            const void* ap[]={aq.data()};float* gp[]={g.data()};float* up[]={u.data()};
            auto gf=f;gf.td=f.tg;gf.n_ff=f.n_embd;gf.d_row=f.gu_row;
            t=sw::decode_clock_ns();team.rows(f.n_ff,[&](int a,int b){
                const auto begin=sw::decode_clock_ns();gg::gg_cpu_down(gf,w[0],ap,1,gp,a,b);gg::gg_cpu_down(gf,w[1],ap,1,up,a,b);
                const auto middle=sw::decode_clock_ns();
                for(int j=a;j<b;++j){float gate=g[size_t(j)],v=u[size_t(j)];const auto limit=source.meta().lim;
                    if(limit>0){gate=gate<limit?gate:limit;v=v< -limit? -limit:(v>limit?limit:v);}
                    h[size_t(j)]=gate/(1.0f+std::exp(-gate))*v;}
                const auto end=sw::decode_clock_ns();dot.fetch_add(middle-begin,std::memory_order_relaxed);silu.fetch_add(end-middle,std::memory_order_relaxed);
            });gu+=sw::decode_clock_ns()-t;
            t=sw::decode_clock_ns();gg::iq_cpu_quant(f.d_act,h.data(),f.n_ff,hq.data());q2+=sw::decode_clock_ns()-t;
            const void* qp[]={hq.data()};float* yp[]={y.data()};t=sw::decode_clock_ns();team.rows(f.n_embd,[&](int a,int b){gg::iq_cpu_down(f,w[2],qp,1,yp,a,b);});dn+=sw::decode_clock_ns()-t;
            need(y.size()==refs[size_t(e)].size()&&std::memcmp(y.data(),refs[size_t(e)].data(),y.size()*4)==0,"component split differs from ggml oracle");
        }
        if(rep>=0)std::printf("CPU_COMPONENT rep=%d rounds=%d access=%s quant1_ns=%llu quant2_ns=%llu gate_up_wall_ns=%llu down_wall_ns=%llu dot_thread_ns=%llu silu_thread_ns=%llu bit_equal=1\n",rep,rounds,hot?"hot":"rotate",(unsigned long long)q1,(unsigned long long)q2,(unsigned long long)gu,(unsigned long long)dn,(unsigned long long)dot.load(),(unsigned long long)silu.load());
    }lane.finish();
}
inline int real_bench(int argc,char** argv) {
    need(argc==14,"--real PACK LAYER FIRST BANK WORKERS REPS ROUNDS SELECTED CPU_TIMING hot|rotate FRESH_OUT X_F32_FILE_OR_DASH");
    auto integer=[&](int i){std::string v(argv[i]);size_t end=0;int n=std::stoi(v,&end);need(end==v.size(),"integer trailing bytes");return n;};
    const int layer=integer(3),first=integer(4),bank=integer(5),workers=integer(6),reps=integer(7),rounds=integer(8),selected=integer(9),timers=integer(10);
    need(bank>=1&&bank<=128&&workers>=1&&workers<=32&&reps>=3&&reps<=10&&rounds>=1&&rounds<=1024&&selected>=1&&selected<=6&&selected<=bank&&(timers==0||timers==1),"real bench bounds");
    const std::string access(argv[11]);need(access=="hot"||access=="rotate","unknown bank access");const bool hot=access=="hot";
    const std::filesystem::path out(argv[12]);need(!std::filesystem::exists(out),"fresh output directory required");std::filesystem::create_directories(out);
    auto pack=sw::open_pack(argv[2]);RealBank source(*pack,layer,first,bank);const auto& meta=source.meta();
    std::vector<float> x(size_t(meta.dim));const bool captured=std::string(argv[13])!="-";
    if(captured){x=read<float>(argv[13]);need(x.size()==size_t(meta.dim),"real activation dimension");}
    else for(size_t i=0;i<x.size();++i)x[i]=float(int((i*37+11)%257)-128)/128.0f;
    for(float v:x)need(std::isfinite(v),"nonfinite real-bench activation");
    std::ofstream xf(out/"input.f32",std::ios::binary);xf.write(reinterpret_cast<const char*>(x.data()),std::streamsize(x.size()*4));need(bool(xf),"activation archive");xf.close();
    std::ofstream inventory(out/"weights.tsv");inventory<<"name\tdtype\tfile\toffset\tbytes\n";
    uint64_t payload=0;for(const auto& n:source.names()){const auto t=source.require_tensor(n);payload+=t.bytes;inventory<<n<<'\t'<<t.dtype<<'\t'<<t.file<<'\t'<<t.offset<<'\t'<<t.bytes<<'\n';}inventory.close();
    // Force the actual ggml reference (all native row overrides off), then
    // restore automatic Strata ISA selection before any timed candidate call.
    gg::gg_cpu_set_iq256(0);std::vector<std::vector<float>> refs;
    for(int e=0;e<bank;++e)refs.push_back(real_oracle(source,e,x));gg::gg_cpu_set_iq256(2);
    setenv("DS4_FORK_WARM","1",1);const std::string budget=std::to_string(payload+uint64_t(bank)*131072);
    setenv("DS4_STORE_BYTES",budget.c_str(),1);setenv("DS4_STORE_FLOOR_BYTES","0",1);setenv("DS4_STORE_THREADS","2",1);setenv("DS4_STORE_PRELOAD","1",1);
    setenv("DS4_CPU_MEMORY_REPORT","1",1);
    // PMU events must predate the row/service worker threads. All I/O/oracle
    // work above, and store preload below, is outside the enable interval.
    sw::CpuPmu pmu;auto lane=sw::make_cpu_lane(source,0,1,workers,"pool-v1",bool(timers));auto store=sw::warm_store(source);
    std::unique_ptr<sw::WarmWorker> service;if(sw::cpu_option("DS4_CPU_BENCH_SERVICE",1))service=std::make_unique<sw::WarmWorker>();
    cpu_snapshot(out,"loaded");
    std::printf("CPU_REAL_BANK layer=%d first=%d bank=%d bytes=%llu workers=%d selected=%d cpu_timing=%d kernel_timing=not_applicable access=%s activation=%s batch=%d service=%d\n",layer,first,bank,(unsigned long long)payload,workers,selected,timers,access.c_str(),captured?"captured":"deterministic",int(lane->layer_batch()),int(bool(service)));
    for(int rep=-1;rep<reps;++rep){uint64_t wall=0,prepare=0,finish=0,wakeup=0;sw::CpuPhases phases{};rusage before{},after{};getrusage(RUSAGE_SELF,&before);if(rep>=0)pmu.start();
        for(int r=0;r<rounds;++r){std::vector<int> ids;for(int i=0;i<selected;++i)ids.push_back((hot?i:(r*selected+i)%bank));
            auto t=sw::decode_clock_ns();lane->prepare(0,ids);prepare+=sw::decode_clock_ns()-t;lane->begin_decode_step(bool(timers));
            const auto start=sw::decode_clock_ns();std::vector<std::vector<float>> got;
            if(service){auto pending=service->submit([&]{wakeup+=sw::decode_clock_ns()-start;return lane->experts(0,ids,x);});got=pending.get();}
            else got=lane->experts(0,ids,x);wall+=sw::decode_clock_ns()-start;
            const auto p=lane->decode_phases();
            phases.completed_experts+=p.completed_experts;phases.logical_read_bytes+=p.logical_read_bytes;phases.read_ns+=p.read_ns;phases.quant1_ns+=p.quant1_ns;
            phases.gate_up_ns+=p.gate_up_ns;phases.quant2_ns+=p.quant2_ns;phases.down_ns+=p.down_ns;phases.expert_ns+=p.expert_ns;
            phases.row_thread_ns+=p.row_thread_ns;phases.dispatch_outside_envelope_ns+=p.dispatch_outside_envelope_ns;phases.worker_completion_wait_ns+=p.worker_completion_wait_ns;
            t=sw::decode_clock_ns();lane->finish();finish+=sw::decode_clock_ns()-t;
            need(got.size()==ids.size(),"real output count");for(size_t i=0;i<ids.size();++i){const auto& ref=refs[size_t(ids[i])];need(got[i].size()==ref.size()&&std::memcmp(got[i].data(),ref.data(),ref.size()*4)==0,"real bank differs from ggml FFN oracle");}
        }
        if(rep>=0)pmu.stop("prepare_compute_finish_validate",rep);getrusage(RUSAGE_SELF,&after);
        if(rep<0)continue;
        const auto usec=[](timeval t){return uint64_t(t.tv_sec)*1000000+uint64_t(t.tv_usec);};
        std::printf("CPU_REAL_BENCH rep=%d rounds=%d selected=%d wall_ns=%llu prepare_ns=%llu finish_ns=%llu wakeup_ns=%llu logical_bytes=%llu user_us=%llu sys_us=%llu voluntary_cs=%ld involuntary_cs=%ld minor_faults=%ld major_faults=%ld bit_equal=1\n",rep,rounds,selected,(unsigned long long)wall,(unsigned long long)prepare,(unsigned long long)finish,(unsigned long long)wakeup,(unsigned long long)(payload/uint64_t(bank)*uint64_t(selected)*uint64_t(rounds)),(unsigned long long)(usec(after.ru_utime)-usec(before.ru_utime)),(unsigned long long)(usec(after.ru_stime)-usec(before.ru_stime)),after.ru_nvcsw-before.ru_nvcsw,after.ru_nivcsw-before.ru_nivcsw,after.ru_minflt-before.ru_minflt,after.ru_majflt-before.ru_majflt);
        if(timers){sw::DecodeTiming t;t.active=true;t.position=uint64_t(rep);t.emit("cpu-real",0,1,phases);}
    }
    service.reset();
    if(sw::cpu_option("DS4_CPU_COMPONENTS",1))cpu_components(source,*lane,*store,x,refs,workers,rounds,hot);
    cpu_snapshot(out,"end");return 0;
}
inline int cpu_candidate_check(int argc,char** argv) {
    need(argc==3,"--check-a2 FIXTURE_DIR");Source source(argv[2]);auto seed=read<float>(std::filesystem::path(argv[2])/"fx.bin");
    unsetenv("DS4_FORK_WARM");unsetenv("DS4_CPU_AFFINITY");
    std::vector<std::vector<float>> inputs{seed,seed};for(float& v:inputs[1])v=-v*.875f;
    std::vector<std::vector<std::vector<float>>> refs(2);
    gg::gg_cpu_set_iq256(0);for(size_t v=0;v<inputs.size();++v)for(int e=0;e<6;++e)refs[v].push_back(real_oracle(source,e,inputs[v]));gg::gg_cpu_set_iq256(2);
    size_t outputs=0;
    for(int workers:{1,2,4})for(auto [prefetch,spin]:std::vector<std::pair<int,int>>{{0,0},{4,0},{16,0},{32,0},{0,50},{4,200}})for(bool timing:{false,true}){
        setenv("DS4_CPU_PREFETCH_ROWS",std::to_string(prefetch).c_str(),1);setenv("DS4_CPU_SPIN_US",std::to_string(spin).c_str(),1);setenv("DS4_CPU_LAYER_BATCH","1",1);
        auto lane=sw::make_cpu_lane(source,0,1,workers,"pool-v1",timing);
        for(size_t v=0;v<inputs.size();++v)for(int count:{1,2,6}){
            std::vector<int> ids;for(int i=0;i<count;++i)ids.push_back(5-i);
            lane->begin_decode_step(timing);auto got=lane->experts(0,ids,inputs[v]);
            for(size_t i=0;i<ids.size();++i){const auto& ref=refs[v][size_t(ids[i])];need(got[i].size()==ref.size()&&std::memcmp(got[i].data(),ref.data(),ref.size()*4)==0,"a2 candidate differs from ggml reference");++outputs;}
        }
    }
    std::printf("CPU_A2_CHECK outputs=%zu formats=6 inputs=2 workers=1,2,4 selected=1,2,6 prefetch=0,4,16,32 spin=0,50,200 timers=off,on bit_equal=1\n",outputs);
    return 0;
}
