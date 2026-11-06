// CPU-only native target: real ExpertStore hit path and original quant/dot kernels.
// Synthetic weights from the committed six-format fixture. Never a model speed claim.
#define CPU_LANE_FIXTURE_NO_MAIN
#include "../fixtures/moe/cpu_lane_fixture.cpp"
#include "runtime.hpp"
#include <chrono>
class BankSource final:public sw::WeightSource {
    sw::ModelMeta meta_;std::map<std::string,sw::TensorInfo> info_;std::string path_;
public:
    BankSource(const Source& source,int kind,int count,const std::string& path):meta_(source.meta()),path_(path) {
        need(!std::filesystem::exists(path_),"fresh bank file required");meta_.n_exp=count;
        std::ofstream file(path_,std::ios::binary);need(bool(file),"bank open");uint64_t offset=0;
        for(const char* role:{"w1","w3","w2"}) {
            const auto key="layers.0.ffn.experts."+std::to_string(kind)+"."+role+".weight";
            auto t=source.require_tensor(key);auto bytes=source.read_all(key);
            for(int e=0;e<count;++e){auto row=t;row.name="layers.0.ffn.experts."+std::to_string(e)+"."+role+".weight";row.file=0;row.offset=offset;info_[row.name]=row;
                file.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));offset+=bytes.size();}
        }need(bool(file),"bank write");
    }
    const sw::ModelMeta& meta()const override{return meta_;}
    std::string identity()const override{return path_;}
    std::vector<std::string> shard_paths()const override{return {path_};}
    std::optional<sw::TensorInfo> find(std::string_view n)const override{auto it=info_.find(std::string(n));if(it==info_.end())return {};return it->second;}
    std::vector<std::string> names()const override{std::vector<std::string> out;for(const auto& [n,t]:info_){(void)t;out.push_back(n);}return out;}
    void read(std::string_view,uint64_t,std::span<std::byte>)const override{throw std::runtime_error("bench must read through ExpertStore");}
    int engram_format(int)const override{throw std::runtime_error("unused");}
    void engram_rows(int,std::span<const uint64_t>,std::span<std::byte>)const override{throw std::runtime_error("unused");}
    void embedding_bf16(int32_t,std::span<uint16_t>)const override{throw std::runtime_error("unused");}
};
int main(int argc,char** argv)try {
    need(argc==9,"strata-cpu-lane-bench FIXTURE_DIR FRESH_BANK_FILE KIND BANK_EXPERTS WORKERS REPS ROUNDS TIMERS");
    const int kind=std::stoi(argv[3]),bank=std::stoi(argv[4]),workers=std::stoi(argv[5]),reps=std::stoi(argv[6]),rounds=std::stoi(argv[7]),timers=std::stoi(argv[8]);
    need(kind>=0&&kind<6&&bank>=6&&bank<=64&&workers>0&&workers<=32&&reps>=3&&reps<=10&&rounds>=1&&rounds<=1024&&(timers==0||timers==1),"bench bounds");
    Source seed(argv[1]);auto x=read<float>(std::filesystem::path(argv[1])/"fx.bin");auto ref=original(seed,kind,x);
    BankSource source(seed,kind,bank,argv[2]);
    setenv("DS4_FORK_WARM","1",1);setenv("DS4_STORE_BYTES","1073741824",1);setenv("DS4_STORE_FLOOR_BYTES","0",1);setenv("DS4_STORE_THREADS","2",1);setenv("DS4_STORE_PRELOAD","1",1);
    setenv("DS4_CPU_LAYER_BATCH","0",1);auto serial=sw::make_cpu_lane(source,0,1,workers,"pool-v1",bool(timers));
    setenv("DS4_CPU_LAYER_BATCH","1",1);auto batch=sw::make_cpu_lane(source,0,1,workers,"pool-v1",bool(timers));
    uint64_t bytes=0;for(const char* role:{"w1","w3","w2"})bytes+=source.require_tensor(std::string("layers.0.ffn.experts.0.")+role+".weight").bytes;
    for(int count:{1,2,4,6})for(int rep=-1;rep<reps;++rep)for(int arm=0;arm<2;++arm){
        const bool batching=(arm+(rep>=0?rep:0))%2;auto& lane=batching?batch:serial;
        uint64_t ns=0;sw::CpuPhases phases{};
        for(int r=0;r<rounds;++r){std::vector<int> ids;for(int i=0;i<count;++i)ids.push_back((r*count+i)%bank);
            lane->prepare(0,ids);lane->begin_decode_step(bool(timers));
            const auto start=sw::decode_clock_ns();auto got=lane->experts(0,ids,x);ns+=sw::decode_clock_ns()-start;
            const auto p=lane->decode_phases();
            phases.completed_experts+=p.completed_experts;phases.logical_read_bytes+=p.logical_read_bytes;phases.read_ns+=p.read_ns;phases.quant1_ns+=p.quant1_ns;
            phases.gate_up_ns+=p.gate_up_ns;phases.quant2_ns+=p.quant2_ns;phases.down_ns+=p.down_ns;phases.expert_ns+=p.expert_ns;
            phases.row_thread_ns+=p.row_thread_ns;phases.dispatch_outside_envelope_ns+=p.dispatch_outside_envelope_ns;phases.worker_completion_wait_ns+=p.worker_completion_wait_ns;
            lane->finish();need(got.size()==ids.size(),"bench output count");
            for(const auto& y:got)need(y.size()==ref.size()&&std::memcmp(y.data(),ref.data(),ref.size()*4)==0,"bench CPU output mismatch");
        }
        if(rep<0)continue;
        std::printf("CPU_LANE_BENCH kind=%d bank=%d workers=%d selected=%d rep=%d batch=%d timers=%d rounds=%d wall_ns=%llu logical_bytes=%llu bit_equal=1\n",kind,bank,workers,count,rep,int(batching),timers,rounds,(unsigned long long)ns,(unsigned long long)(bytes*count*rounds));
        if(timers){sw::DecodeTiming t;t.active=true;t.position=uint64_t(rep);t.request=uint64_t(count);t.emit(batching?"cpu-batch":"cpu-serial",0,1,phases);}
    }
    batch.reset();serial.reset();std::filesystem::remove(argv[2]);
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
