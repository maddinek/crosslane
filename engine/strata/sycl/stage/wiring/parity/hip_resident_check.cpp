// MIT; P1 native equal-input probe. No model generation, no speed claim.
#include "../hip_resident.hpp"
#include "../cpu_lane.hpp"
#include <hip/hip_runtime.h>
#include "../hip_resident_bank.hpp"
#include "../hip_arena.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include "probe_metrics.hpp"
using namespace strata::wiring;
namespace gg = strata::ds4::gguf;
namespace k4 = strata::ds4;
namespace x3 = strata::dsv41k::exl3;
void ck(hipError_t e) { require(e == hipSuccess, "HIP probe API failure"); }
int main(int argc, char** argv) {
    try {
        require(argc >= 6 && argc <= 8,
            "usage: strata-hip-resident-check PACK LAYER EXPERT INPUT.f32 OUT_PREFIX [cpu-q8k-v1] [pooled-layer-v1]");
        bool cpu_q8k = false, pooled_probe = false;
        for (int i=6;i<argc;++i) {
            const std::string option=argv[i];
            if(option=="cpu-q8k-v1" && !cpu_q8k) cpu_q8k=true;
            else if(option=="pooled-layer-v1" && !pooled_probe) pooled_probe=true;
            else throw std::invalid_argument("unknown or duplicate probe option");
        }
        auto ws = open_pack(argv[1]); const auto& g = ws->meta();
        const uint32_t layer = uint32_t(std::stoul(argv[2])); const int e = std::stoi(argv[3]);
        require(layer > 0 && layer < uint32_t(g.nl) && e >= 0 && e < g.n_exp, "probe layer/expert range");
        std::vector<float> input(size_t(g.dim));
        std::ifstream file(argv[4], std::ios::binary); file.read(reinterpret_cast<char*>(input.data()), std::streamsize(input.size()*4));
        require(bool(file) && file.peek() == std::char_traits<char>::eof(), "probe input extent");
        for (float v : input) require(std::isfinite(v), "nonfinite input");
        ck(hipSetDevice(0)); hipStream_t stream{}; ck(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
        struct StreamOwner { hipStream_t stream;~StreamOwner(){hipSetDevice(0);hipStreamDestroy(stream);} } owner{stream};
        require(x3::prepare(stream) && gg::prepare(stream), "probe prepare");
        auto save=[&](const char* role,const std::vector<float>& values){
            std::ofstream out(std::string(argv[5])+"."+role+".f32",std::ios::binary);
            out.write(reinterpret_cast<const char*>(values.data()),std::streamsize(values.size()*4));
            out.close();require(bool(out),"probe output write");
        };
        HipArena arena(0,stream);
        HipResidency r; r.cpu_q8k=cpu_q8k; r.slots.assign(size_t(g.nl), std::vector<int>(size_t(g.n_exp), -1));
        auto meta = expert_metadata(*ws, layer, e); r.slots[layer][size_t(e)] = 0;
        r.allocated = meta.allocated; r.payload = meta.payload; r.workspace = kHipExpertWorkspaceReserve;
        r.experts.push_back(meta);
        if(pooled_probe) {
            r.policy="pooled-layer-v1";
            r.layer_allocated.assign(size_t(g.nl),0);r.layer_driver_charge.assign(size_t(g.nl),0);
            r.layer_allocated[layer]=r.allocated;r.driver_charge=expert_driver_charge(r.allocated);
            r.layer_driver_charge[layer]=r.driver_charge;
            uint64_t offset=0;for(size_t role=0;role<3;++role){r.experts[0].offsets[role]=offset;offset=strata::stage::add(offset,expert_align(meta.matrices[role].bytes));}
            r.layout_tag=residency_layout_tag(r);
        }
        HipResidentBank bank; bank.load(*ws, r, arena, stream);
        auto snapshot=[&](const char* prefix,std::array<const float*,3> pointers){
            const char* names[]={"gate","up","hidden"};
            for(size_t i=0;i<3;++i){std::vector<float> values(size_t(g.ff));
                ck(hipMemcpyAsync(values.data(),pointers[i],values.size()*4,hipMemcpyDeviceToHost,stream));ck(hipStreamSynchronize(stream));
                save((std::string(prefix)+"-"+names[i]).c_str(),values);
            }
        };
        auto* x = arena.dev<float>(size_t(g.dim)); auto* y = arena.dev<float>(size_t(g.dim));
        ck(hipMemcpyAsync(x, input.data(), input.size()*4, hipMemcpyHostToDevice, stream));
        std::vector<float> first(input.size()), second(input.size());
        const auto uploaded = bank.uploaded(), allocated = arena.bytes;
        bank.expert(layer, 0, x, y, g, stream);
        ck(hipMemcpyAsync(first.data(), y, first.size()*4, hipMemcpyDeviceToHost, stream)); ck(hipStreamSynchronize(stream));
        bank.expert(layer, 0, x, y, g, stream);
        ck(hipMemcpyAsync(second.data(), y, second.size()*4, hipMemcpyDeviceToHost, stream)); ck(hipStreamSynchronize(stream));
        save("hip",first);save("repeat",second);
        if(pooled_probe)snapshot("hip",bank.probe_intermediates());
        require(std::memcmp(first.data(), second.data(), first.size()*4) == 0, "repeated HIP hit not exact");
        require(bank.uploaded() == uploaded && arena.bytes == allocated && bank.hits() == 2, "per-hit allocation/weight upload");
        // Independent upstream direct path, fresh raw buffers and descriptor setup.
        // This checks bank layout/order exactly, without assuming CPU activation math matches.
        std::array<x3::Mat,3> direct{};
        for (size_t role=0; role<3; ++role) {
            const auto& t=meta.matrices[role]; auto raw=ws->read_all(t.name);
            auto* ptr=arena.dev<uint8_t>(raw.size());
            ck(hipMemcpyAsync(ptr,raw.data(),raw.size(),hipMemcpyHostToDevice,stream));
            ck(hipStreamSynchronize(stream));
            auto& m=direct[role];m.K=strata::ds4::moe::ggml_type(strata::ds4::moe::type(t.dtype));
            m.fmt=gg::kFmtGG;m.raw=ptr;m.k=int(t.shape[1]);m.n=int(t.shape[0]);
        }
        auto* dg=arena.dev<float>(size_t(g.ff));auto* du=arena.dev<float>(size_t(g.ff));auto* dh=arena.dev<float>(size_t(g.ff));
        auto job=[](const x3::Mat& m,const float* input,float* output){
            x3::MvJob j{};j.m=&m;j.xf=input;j.y=output;j.alpha=1.0f;j.ncol=1;j.xf_cs=size_t(m.k);j.y_cs=size_t(m.n);return j;
        };
        x3::MvJob gu[]={job(direct[0],x,dg),job(direct[1],x,du)};
        if(cpu_q8k) {
            auto* q=arena.dev<uint8_t>(q8k::activation_bytes(std::max(g.dim,g.ff)));
            auto saveq=[&](const char* role,int k){
                std::vector<uint8_t> bytes(q8k::activation_bytes(k));
                ck(hipMemcpyAsync(bytes.data(),q,bytes.size(),hipMemcpyDeviceToHost,stream));ck(hipStreamSynchronize(stream));
                std::ofstream out(std::string(argv[5])+"."+role+".q8k",std::ios::binary);
                out.write(reinterpret_cast<const char*>(bytes.data()),std::streamsize(bytes.size()));out.close();require(bool(out),"probe quant write");
            };
            require(q8k::quantize(x,q,g.dim,stream),"direct Q8_K quant1 failed");saveq("x",g.dim);
            require(q8k::mv(direct[0].K,direct[0].raw,q,dg,g.ff,g.dim,stream) &&
                    q8k::mv(direct[1].K,direct[1].raw,q,du,g.ff,g.dim,stream),"direct Q8_K gate/up failed");
            require(q8k::activate(dg,du,dh,g.ff,g.lim,stream),"direct Q8_K activation failed");
            require(q8k::quantize(dh,q,g.ff,stream),"direct Q8_K quant2 failed");saveq("h",g.ff);
            require(q8k::mv(direct[2].K,direct[2].raw,q,y,g.dim,g.ff,stream),"direct Q8_K down failed");
        }else{
            require(gg::mv(gu,2,stream),"direct gate/up failed");k4::swiglu(dg,du,g.ff,g.lim,dh,stream);
            auto down=job(direct[2],dh,y);require(gg::mv(&down,1,stream),"direct down failed");
        }
        ck(hipMemcpyAsync(second.data(),y,second.size()*4,hipMemcpyDeviceToHost,stream));ck(hipStreamSynchronize(stream));
        save("direct",second);
        for (auto [role,ptr] : {std::pair{"g",dg},std::pair{"u",du},std::pair{"h",dh}}) {
            std::vector<float> values(size_t(g.ff));
            ck(hipMemcpyAsync(values.data(),ptr,values.size()*4,hipMemcpyDeviceToHost,stream));
            ck(hipStreamSynchronize(stream));
            for(float v:values) require(std::isfinite(v),"nonfinite direct intermediate");
            save(role,values);
        }

        if(pooled_probe)snapshot("direct",{dg,du,dh});
        require(std::memcmp(first.data(),second.data(),first.size()*4)==0,"retained vs direct HIP differs");
        auto cpu = make_cpu_lane(*ws, layer, layer+1, 1, "pool-v1");
        auto cy = cpu->expert(layer, e, input); cpu->drain();save("cpu",cy);
        const bool zero = std::all_of(input.begin(), input.end(), [](float v) { return v == 0.0f; });
        // CPU block256 Q8_K vs HIP per32 activation math: norm errors are
        // diagnostics, not P1 admission. Exactness/max_abs remain gating.
        const auto metrics=strata::wiring::probe_metrics(first,cy);
        const bool passed=metrics.passed(zero) && (!cpu_q8k || std::sqrt(metrics.nmse_cpu())<=1e-5);
        ck(hipStreamSynchronize(stream)); bank.reset(); arena.clear();
        require(arena.release_ok && arena.bytes==0 && bank.uploaded()==0 && bank.workspace()==0,"probe teardown accounting");
        std::cout<<std::setprecision(17)<<(passed?"PASS":"FAIL tolerance")<<" zero="<<zero<<" exact="<<metrics.exact
                 <<" nrmse="<<std::sqrt(metrics.nmse_cpu())<<" nmse_cpu="<<metrics.nmse_cpu()
                 <<" nmse_hip="<<metrics.nmse_hip()<<" nrmse_hip="<<std::sqrt(metrics.nmse_hip())<<" max_abs="<<metrics.max_abs
                 <<" startup_weight_h2d="<<uploaded<<" repeated_hit_weight_h2d=0 teardown_bytes=0\n";
        return passed?0:1;
    } catch(const std::exception& e) { std::cerr<<"FAIL "<<e.what()<<'\n'; return 1; }
}
