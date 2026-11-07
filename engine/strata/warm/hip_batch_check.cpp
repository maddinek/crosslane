// MIT. Real-pack, same-kernel layer batch discriminator. Never a CPU/GPU tolerance test.
#include "../sycl/stage/wiring/hip_resident_bank.hpp"
#include "../sycl/stage/wiring/hip_arena.hpp"
#include "../sycl/stage/wiring/hip_decode_events.hpp"
#include <chrono>
#include <iostream>
#include <map>
#include <sstream>
using namespace strata::wiring;
namespace gg=strata::ds4::gguf;
namespace x3=strata::dsv41k::exl3;
static void ck(hipError_t e){require(e==hipSuccess,"HIP batch checker API error");}
static std::string signature(const ResidentExpert& e){
    std::ostringstream s;for(const auto& t:e.matrices){s<<t.dtype;for(auto n:t.shape)s<<'-'<<n;s<<'/';}return s.str();
}
static void require_bits(std::span<const float> a,std::span<const float> b,const std::string& label){
    require(a.size()==b.size(),"checker extent");
    for(size_t i=0;i<a.size();++i){
        require(std::isfinite(a[i])&&std::isfinite(b[i]),"checker nonfinite "+label);
        if(std::memcmp(&a[i],&b[i],4)){
            char detail[180];std::snprintf(detail,sizeof(detail)," element=%zu old=%a new=%a",i,double(a[i]),double(b[i]));
            throw std::runtime_error(label+detail);
        }
    }
}
#include "hip_experts_check.hpp"
#include "hip_overlap_micro.hpp"
int main(int argc,char**argv){try{
    if(argc==3&&std::string(argv[1])=="--overlap-micro")return hip_overlap_micro(argv[2]);
    if(argc==3){const std::string mode=argv[1];
        if(mode=="--experts-v2-identity"||mode=="--experts-v2-timing"||mode=="--experts-v2-profile")
            return hip_experts_check(argv[2],mode=="--experts-v2-timing",mode=="--experts-v2-profile");}
    const bool swiglu=argc==3&&(std::string(argv[1])=="--swiglu-identity"||std::string(argv[1])=="--swiglu-timing");
    require(argc==3&&(swiglu||std::string(argv[1])=="--identity"||std::string(argv[1])=="--timing"),
            "usage: strata-hip-batch-check --identity|--timing PACK");
    const bool timing=std::string(argv[1])=="--timing"||std::string(argv[1])=="--swiglu-timing";
    for(const char* key:{"DS4_QUANT_NODEDUP","DS4_QUANT_SINGLE","DS4_IQ_Q8K"})
        require(!std::getenv(key),std::string("unset ")+key);
    auto ws=open_pack(argv[2]);const auto& g=ws->meta();
    require(g.k_exp>=6&&g.n_exp>=6,"native 1..6 discriminator requires six selections/experts");
    // Inventory every real expert, then select six same-layer representatives
    // per distinct (gate/up/down dtype + full dimensions) signature. No guessed count.
    std::map<std::string,std::map<uint32_t,std::vector<int>>> inventory;
    for(int l=0;l<g.nl;++l)for(int e=0;e<g.n_exp;++e)
        inventory[signature(expert_metadata(*ws,uint32_t(l),e))][uint32_t(l)].push_back(e);
    require(!inventory.empty(),"no real signatures");
    ck(hipSetDevice(0));hipDeviceProp_t prop{};ck(hipGetDeviceProperties(&prop,0));
    std::cout<<"DEVICE name="<<prop.name<<" arch="<<prop.gcnArchName<<'\n';
    hipStream_t stream{};ck(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking));
    struct Stream{hipStream_t s;~Stream(){hipStreamSynchronize(s);hipStreamDestroy(s);}} stream_owner{stream};
    require(x3::prepare(stream)&&gg::prepare(stream),"HIP prepare failed");
    size_t cases=0;
    for(const auto& [sig,layers]:inventory){
        auto found=std::find_if(layers.begin(),layers.end(),[](const auto& p){return p.second.size()>=6;});
        require(found!=layers.end(),"real signature lacks six same-layer experts: "+sig);
        const auto layer=found->first;
        HipArena arena(0,stream);HipResidency r;r.workspace=kHipExpertWorkspaceReserve;r.policy="pooled-layer-v1";r.all_layers=true;
        r.slots.assign(size_t(g.nl),std::vector<int>(size_t(g.n_exp),-1));
        for(int i=0;i<6;++i){auto e=expert_metadata(*ws,layer,found->second[size_t(i)]);e.slot=i;
            r.slots[layer][size_t(e.expert)]=i;r.payload+=e.payload;r.allocated+=e.allocated;r.experts.push_back(e);}
        r.layer_allocated.assign(size_t(g.nl),0);r.layer_driver_charge.assign(size_t(g.nl),0);
        r.layer_allocated[layer]=r.allocated;r.driver_charge=expert_driver_charge(r.allocated);r.layer_driver_charge[layer]=r.driver_charge;
        uint64_t offset=0;for(auto& e:r.experts)for(size_t role=0;role<3;++role){e.offsets[role]=offset;offset+=expert_align(e.matrices[role].bytes);}
        r.layout_tag=residency_layout_tag(r);
        HipResidentBank bank;bank.load(*ws,r,arena,stream,true);
        auto* dx=arena.dev<float>(size_t(g.dim));auto* dy=arena.dev<float>(size_t(g.dim));
        std::vector<float> x(size_t(g.dim)),old(size_t(g.dim)*6),ref_stage(size_t(g.ff)*18),got_stage(ref_stage.size());
        struct Drain{hipStream_t s;~Drain(){hipStreamSynchronize(s);}} drain{stream};
        HipDecodeClock clock;uint64_t expected_hits=0;const auto allocated=arena.bytes,uploaded=bank.uploaded();
        auto input=[&](int round){for(size_t j=0;j<x.size();++j)x[j]=round==0?0.0f:float(int((j*37+size_t(round)*19)%257)-128)*float(round%3+1)/128.0f;};
        std::cout<<"SIGNATURE case="<<cases<<" value="<<sig<<" layer="<<layer<<" experts=";
        for(const auto& e:r.experts)std::cout<<e.expert<<',';std::cout<<" workspace="<<bank.workspace()<<'\n';
        for(int n=1;n<=6;++n){
            std::vector<int> slots(size_t(n),0);uint64_t event_ns=0;
            auto run=[&](bool batch,bool snapshot){
                expected_hits+=uint64_t(n);
                if(batch||swiglu){bank.set_swiglu_batch(swiglu&&batch);clock.begin(stream);bank.enqueue_layer(layer,slots,x,g,stream);clock.end(stream);event_ns+=clock.elapsed_ns();
                    if(snapshot){
                        if(!batch)std::copy(bank.layer_output().begin(),bank.layer_output().end(),old.begin());
                        auto ptr=bank.probe_intermediates();auto* stage=batch?got_stage.data():ref_stage.data();
                        for(size_t p=0;p<3;++p)ck(hipMemcpyAsync(stage+p*size_t(n)*g.ff,ptr[p],size_t(n)*g.ff*4,hipMemcpyDeviceToHost,stream));
                        ck(hipStreamSynchronize(stream));}
                }else for(int i=0;i<n;++i){clock.begin(stream);
                    ck(hipMemcpyAsync(dx,x.data(),x.size()*4,hipMemcpyHostToDevice,stream));bank.expert(layer,slots[size_t(i)],dx,dy,g,stream);
                    ck(hipMemcpyAsync(old.data()+size_t(i)*g.dim,dy,size_t(g.dim)*4,hipMemcpyDeviceToHost,stream));clock.end(stream);event_ns+=clock.elapsed_ns();
                    if(snapshot){auto ptr=bank.probe_intermediates();for(size_t p=0;p<3;++p)
                        ck(hipMemcpyAsync(ref_stage.data()+(p*size_t(n)+size_t(i))*g.ff,ptr[p],size_t(g.ff)*4,hipMemcpyDeviceToHost,stream));ck(hipStreamSynchronize(stream));}
                }
            };
            for(int round=0;round<8;++round){input(round);for(int i=0;i<n;++i)slots[size_t(i)]=(round+5-i)%6;
                run(false,true);run(true,true);
                const auto label="case="+std::to_string(cases)+" n="+std::to_string(n)+" round="+std::to_string(round);
                require_bits(std::span(old.data(),size_t(n)*g.dim),bank.layer_output(),label+" output");
                require_bits(std::span(ref_stage.data(),size_t(n)*g.ff*3),std::span(got_stage.data(),size_t(n)*g.ff*3),label+" gate/up/hidden");
            }
            std::printf("IDENTITY case=%zu selections=%d rounds=8 PASS\n",cases,n);
            if(timing)for(int rep=0;rep<5;++rep)for(int arm=0;arm<2;++arm){
                const bool batch=(rep+arm)%2;input(3);for(int i=0;i<n;++i)slots[size_t(i)]=5-i;
                for(int i=0;i<8;++i)run(batch,false);event_ns=0;
                const auto start=std::chrono::steady_clock::now();for(int i=0;i<64;++i)run(batch,false);
                const double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/64;
                std::printf("TIMING case=%zu selections=%d rep=%d batch=%d iterations=64 event_us=%.6f wall_us=%.6f\n",cases,n,rep,int(batch),double(event_ns)/64000,wall);
            }
        }
        require(arena.bytes==allocated&&bank.uploaded()==uploaded&&bank.hits()==expected_hits,"per-layer allocation/weight upload");
        ck(hipGetLastError());ck(hipStreamSynchronize(stream));bank.reset();arena.clear();require(arena.bytes==0&&arena.release_ok,"checker teardown");++cases;
    }
    require(gg::launch_errors()==0,"upstream launch failure");
    std::printf("%s PASS signatures=%zu selections=1..6 rounds=8\n",swiglu?"HIP_SWIGLU_NATIVE":"HIP_BATCH_NATIVE",cases);return 0;
}catch(const std::exception&e){std::fprintf(stderr,"HIP_BATCH_NATIVE FAIL %s\n",e.what());return 1;}}
