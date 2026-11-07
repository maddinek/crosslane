// MIT; crosslane contributors. Retained HIP matrices + serial expert workspace.
// gg::mv and swiglu are upstream's DS4_SYNC path. Arena owns all device lifetimes.
#pragma once
#include "hip_resident.hpp"
#include <climits>
#include "ds4_gguf.cuh"
#include "ds4_kernels.cuh"
#include "hip_q8k.hpp"
#include "../../../warm/hip_experts.hpp"
#include "../../../warm/mmvq_layout.hpp"

namespace strata::wiring {
class HipResidentBank {
    using Mat = strata::dsv41k::exl3::Mat;
    using Job = strata::dsv41k::exl3::MvJob;
    std::vector<std::vector<std::array<Mat, 3>>> matrices_;
    std::vector<std::byte> staging_; // survives a failed async-copy drain until reset
    float *gate_ = nullptr, *up_ = nullptr, *hidden_ = nullptr;
    void* quant_ = nullptr;
    size_t batch_capacity_ = 0;
    float *batch_x_ = nullptr, *batch_y_ = nullptr;
    // Retained through failed drains, just like weight staging_. No async copy
    // ever refers to a temporary host vector owned by warm_compute_many.
    std::vector<float> batch_input_, batch_output_;
    size_t batch_count_ = 0;
    bool soa_ = false, swiglu_batch_ = false;
    int8_t* soa_quant_ = nullptr;
    float* soa_scale_ = nullptr;
    uint32_t* soa_tables_ = nullptr;
    static Job job(const Mat& m, const float* x, float* y) {
        Job j{};j.m=&m;j.xf=x;j.y=y;j.alpha=1.0f;j.ncol=1;
        j.xf_cs=size_t(m.k);j.y_cs=size_t(m.n);return j;
    }
    bool cpu_q8k_ = false;
    uint64_t uploaded_ = 0, hits_ = 0, workspace_ = 0;
    bool ready_ = false;
public:
    template<class Arena> void load(const WeightSource& ws, const HipResidency& r, Arena& arena, hipStream_t stream, bool layer_batch = false, bool soa = false, bool swiglu_batch = false) {
        namespace gg = strata::ds4::gguf;
        require(matrices_.empty(), "resident bank already loaded");
        const bool pooled=pooled_policy(r.policy);
        if(pooled)validate_pooled_layout(r); // before workspace or any device allocation
        matrices_.resize(r.slots.size());
        cpu_q8k_ = r.cpu_q8k;
        require(!soa || (layer_batch && ws.meta().k_exp<=hipex::max_jobs/2), "SoA requires layer batch within job capacity");
        require(!swiglu_batch || (layer_batch && !soa),"batch SwiGLU requires AoS layer batching");
        require(!swiglu_batch || (ws.meta().k_exp>0 && ws.meta().ff>0 && ws.meta().ff <= INT_MAX/ws.meta().k_exp),"batch SwiGLU extent overflow");
        swiglu_batch_=swiglu_batch;
        soa_ = soa;
        if(soa_)for(const char* key:{"DS4_QUANT_NODEDUP","DS4_QUANT_SINGLE","DS4_IQ_Q8K"})
            require(!std::getenv(key),std::string("HIP SoA requires unset ")+key);
        require(!layer_batch || (!cpu_q8k_ && ws.meta().k_exp > 0), "layer batch requires per32 math and positive selections");
        batch_capacity_ = layer_batch ? size_t(ws.meta().k_exp) : 0;
        if (r.experts.empty()) {ready_=true;return;}
        // Device-format admission before the first weight allocation/upload.
        for (const auto& e : r.experts) for (const auto& t : e.matrices) {
            if(soa_)require(hipex::supported(strata::ds4::moe::ggml_type(strata::ds4::moe::type(t.dtype))) &&
                    mmvq::extent(mmvq::kind(t.dtype),int(t.shape[0]),int(t.shape[1]))==t.bytes,"unsupported HIP SoA matrix");
            require(gg::type_on_device(strata::ds4::moe::ggml_type(strata::ds4::moe::type(t.dtype))), "resident type has no HIP kernel");
            if (cpu_q8k_) require(q8k::supported_type(strata::ds4::moe::ggml_type(strata::ds4::moe::type(t.dtype))) &&
                q8k::activation_bytes(int(t.shape[1])) > 0, "resident type/geometry has no Q8_K kernel");
        }
        const auto lanes = std::max(size_t(1), batch_capacity_);
        const auto row = expert_align(strata::ds4::moe::product(strata::ds4::moe::product(size_t(ws.meta().ff), lanes), sizeof(float)));
        const auto input_bytes = layer_batch ? expert_align(strata::ds4::moe::product(size_t(ws.meta().dim),sizeof(float))) : 0;
        const auto output_bytes = layer_batch ? expert_align(strata::ds4::moe::product(strata::ds4::moe::product(size_t(ws.meta().dim),lanes),sizeof(float))) : 0;
        const auto quant_bytes = cpu_q8k_ ? q8k::activation_bytes(std::max(ws.meta().dim,ws.meta().ff)) : 0;
        const auto soa_values = soa_ ? std::max(size_t(ws.meta().dim),strata::ds4::moe::product(size_t(ws.meta().ff),lanes)) : 0;
        const auto soa_bytes = soa_ ? stage::add(expert_align(soa_values),stage::add(expert_align(soa_values/32*sizeof(float)),expert_align(mmvq::table_words*sizeof(uint32_t)))) : 0;
        require(stage::add(soa_bytes,stage::add(strata::ds4::moe::product(size_t(row), size_t(3)),stage::add(expert_align(quant_bytes),stage::add(input_bytes,output_bytes)))) <= r.workspace,
            "resident workspace reservation too small");
        const auto before = arena.bytes;
        gate_ = arena.template dev<float>(size_t(ws.meta().ff)*lanes);
        up_ = arena.template dev<float>(size_t(ws.meta().ff)*lanes);
        hidden_ = arena.template dev<float>(size_t(ws.meta().ff)*lanes);
        if (cpu_q8k_) quant_ = arena.template dev<uint8_t>(quant_bytes);
        if(soa_) {
            soa_quant_=arena.template dev<int8_t>(soa_values);
            soa_scale_=arena.template dev<float>(soa_values/32);
            soa_tables_=arena.template dev<uint32_t>(mmvq::table_words);
            const auto table=mmvq::codebooks();staging_.resize(sizeof(table));
            std::memcpy(staging_.data(),table.data(),sizeof(table));
            require(hipMemcpyAsync(soa_tables_,staging_.data(),staging_.size(),hipMemcpyHostToDevice,stream)==hipSuccess &&
                    hipStreamSynchronize(stream)==hipSuccess,"HIP SoA codebook upload failed");
            std::vector<std::byte>().swap(staging_);
        }
        if(layer_batch) {
            batch_x_=arena.template dev<float>(size_t(ws.meta().dim));
            batch_y_=arena.template dev<float>(size_t(ws.meta().dim)*lanes);
            batch_input_.resize(size_t(ws.meta().dim));
            batch_output_.resize(size_t(ws.meta().dim)*lanes);
        }
        workspace_ = arena.bytes - before;
        require(workspace_ <= r.workspace, "resident workspace exceeds reservation");
        const auto start = arena.bytes;
        uint32_t current=UINT32_MAX;
        uint8_t* slab=nullptr;
        for (const auto& e : r.experts) {
            if(pooled && current!=e.layer){
                current=e.layer;
                const auto driver_before=arena.driver_used();
                slab=arena.template dev<uint8_t>(size_t(r.layer_allocated.at(e.layer)));
                require(reinterpret_cast<uintptr_t>(slab)%256==0,"pooled slab base alignment");
                require(r.layer_allocated[e.layer]<=UINTPTR_MAX-reinterpret_cast<uintptr_t>(slab),"pooled slab pointer extent overflow");
                std::fprintf(stderr,"AIOPT_HIP_SLAB layer=%u requested_bytes=%llu driver_charge_bytes=%llu arena_bytes=%llu driver_before_bytes=%llu driver_after_bytes=%llu\n",
                    e.layer,(unsigned long long)r.layer_allocated[e.layer],
                    (unsigned long long)r.layer_driver_charge[e.layer],(unsigned long long)arena.bytes,
                    (unsigned long long)driver_before,(unsigned long long)arena.driver_used());
            }
            auto& layer = matrices_.at(e.layer);
            require(size_t(e.slot) == layer.size(), "resident slots are not contiguous");
            layer.push_back({});
            for (size_t role = 0; role < 3; ++role) {
                const auto& t = e.matrices[role];
                auto& m = layer.back()[role];
                staging_.resize(size_t(t.bytes));
                ws.read(t.name,0,staging_); // only startup; one matrix staging vector
                auto& raw=staging_;
                require(raw.size() == t.bytes, "resident weight read extent");
                if(soa_) {
                    auto reordered=mmvq::reorder(mmvq::kind(t.dtype),int(t.shape[0]),int(t.shape[1]),
                        {reinterpret_cast<const uint8_t*>(raw.data()),raw.size()});
                    std::memcpy(raw.data(),reordered.data(),raw.size());
                }
                m = Mat{}; m.K = strata::ds4::moe::ggml_type(strata::ds4::moe::type(t.dtype));
                m.fmt = gg::kFmtGG; m.k = int(t.shape[1]); m.n = int(t.shape[0]);
                // Same view as Impl::set_vm, without repacking/dequantizing weights.
                // Arena owns slab bases only; Mat stores a nonowning slice.
                if(pooled)require(e.offsets[role]<=UINTPTR_MAX-reinterpret_cast<uintptr_t>(slab),"pooled pointer overflow");
                auto* ptr = pooled ? slab+e.offsets[role] : arena.template dev<uint8_t>(raw.size()); m.raw = ptr;
                if (hipMemcpyAsync(ptr, raw.data(), raw.size(), hipMemcpyHostToDevice, stream) != hipSuccess ||
                    hipStreamSynchronize(stream) != hipSuccess) throw std::runtime_error("resident weight H2D failed");
                uploaded_ += raw.size();
                std::vector<std::byte>().swap(staging_); // copy completed; release this matrix
            }
        }
        require(arena.bytes - start == r.allocated && uploaded_ == r.payload, "resident allocation/upload accounting differs");
        if(soa_)std::fprintf(stderr,"AIOPT_HIP_EXPERT_KERNEL layout=soa-v2 weight_bytes=%llu workspace_bytes=%llu codebook_bytes=%zu\n",
            (unsigned long long)r.allocated,(unsigned long long)workspace_,mmvq::table_words*sizeof(uint32_t));
        if(swiglu_batch_)std::fprintf(stderr,"AIOPT_HIP_SWIGLU mode=batch-v1\n");
        ready_=true; // no partially loaded bank can serve a hit
    }
    void expert(uint32_t layer, int slot, const float* x, float* y, const ModelMeta& g, hipStream_t stream) {
        namespace gg = strata::ds4::gguf;
        require(ready_ && slot >= 0 && !soa_, "unready/SoA bank or negative resident slot");
        const auto& m = matrices_.at(layer).at(size_t(slot));
        if (cpu_q8k_) {
            require(q8k::quantize(x, quant_, g.dim, stream), "resident Q8_K input quantization failed");
            require(q8k::mv(m[0].K,m[0].raw,quant_,gate_,m[0].n,m[0].k,stream) &&
                    q8k::mv(m[1].K,m[1].raw,quant_,up_,m[1].n,m[1].k,stream), "resident Q8_K gate/up failed");
            require(q8k::activate(gate_,up_,hidden_,g.ff,g.lim,stream), "resident Q8_K activation failed");
            require(q8k::quantize(hidden_,quant_,g.ff,stream) &&
                    q8k::mv(m[2].K,m[2].raw,quant_,y,m[2].n,m[2].k,stream), "resident Q8_K down failed");
            ++hits_; return;
        }
        auto job = [](const Mat& mat, const float* input, float* output) {
            Job j{}; j.m = &mat; j.xf = input; j.y = output; j.alpha = 1.0f;
            j.ncol = 1; j.xf_cs = size_t(mat.k); j.y_cs = size_t(mat.n); return j;
        };
        Job gu[] = {job(m[0], x, gate_), job(m[1], x, up_)};
        require(gg::mv(gu, 2, stream), "resident gate/up HIP GEMV failed");
        strata::ds4::swiglu(gate_, up_, g.ff, g.lim, hidden_, stream);
        Job down = job(m[2], hidden_, y);
        require(gg::mv(&down, 1, stream), "resident down HIP GEMV failed");
        ++hits_; // submitted; caller synchronizes and only then publishes a completed step
    }
    // The caller owns stream ordering: queued-v1 drains dense/shared before
    // dispatch and joins this lane before uploading the unchanged ordered sum.
    // No change to kernels, ncol, reduction order, SwiGLU or weight bytes.
    void enqueue_layer(uint32_t layer, std::span<const int> slots, std::span<const float> x,
                       const ModelMeta& g, hipStream_t stream, hipex::Trace trace = {}) {
        namespace gg = strata::ds4::gguf;
        constexpr size_t jobs_max = strata::dsv41k::exl3::kMaxMvJobs;
        static_assert(jobs_max >= 2);
        require(ready_ && batch_capacity_ && !cpu_q8k_ && !slots.empty() &&
                slots.size()<=batch_capacity_ && x.size()==batch_input_.size(), "HIP batch extent/mode");
        std::set<int> unique;
        for(int slot:slots) {
            require(slot>=0 && unique.insert(slot).second, "HIP batch invalid/duplicate slot");
            const auto& m=matrices_.at(layer).at(size_t(slot));
            require(m[0].k==g.dim && m[1].k==g.dim && m[0].n==g.ff && m[1].n==g.ff &&
                    m[2].k==g.ff && m[2].n==g.dim, "HIP batch matrix shape");
        }
        batch_count_=slots.size();
        std::copy(x.begin(),x.end(),batch_input_.begin());
        trace("start",stream);
        require(hipMemcpyAsync(batch_x_,batch_input_.data(),x.size_bytes(),hipMemcpyHostToDevice,stream)==hipSuccess,
                "HIP batch input copy failed");
        trace("h2d",stream);
        if(soa_) {
            require(hipex::quantize(batch_x_,1,g.dim,soa_quant_,soa_scale_,stream),"HIP SoA input quantization failed");
            trace("input_quant",stream);
            hipex::Jobs gu{},dn{};
            for(size_t i=0;i<slots.size();++i){const auto& m=matrices_[layer][size_t(slots[i])];
                for(size_t role=0;role<2;++role)gu.j[gu.count++]={static_cast<const uint8_t*>(m[role].raw),m[role].K,m[role].n,m[role].k,
                    soa_quant_,soa_scale_,(role?up_:gate_)+i*size_t(g.ff)};
                dn.j[dn.count++]={static_cast<const uint8_t*>(m[2].raw),m[2].K,m[2].n,m[2].k,
                    soa_quant_+i*size_t(g.ff),soa_scale_+i*size_t(g.ff/32),batch_y_+i*size_t(g.dim)};
            }
            require(hipex::project(gu,soa_tables_,stream),"HIP SoA gate/up failed");trace("gate_up",stream);
            // Independent scalar elements; call the original upstream kernel.
            strata::ds4::swiglu(gate_,up_,g.ff*int(slots.size()),g.lim,hidden_,stream);trace("swiglu",stream);
            require(hipex::quantize(hidden_,int(slots.size()),g.ff,soa_quant_,soa_scale_,stream),"HIP SoA hidden quantization failed");trace("hidden_quant",stream);
            require(hipex::project(dn,soa_tables_,stream),"HIP SoA down failed");trace("down",stream);
        } else {
        // gg::mv quant_add shares an identical (pointer,stride,ncol,k) input.
        // Eight upstream jobs allow four gate/up pairs per call, without any
        // scratch/ABI/kernel change. Six selections need two quantizations.
        for(size_t first=0;first<slots.size();first+=jobs_max/2) {
            Job jobs[jobs_max]{};size_t n=0;
            for(size_t i=first;i<std::min(slots.size(),first+jobs_max/2);++i) {
                const auto& m=matrices_[layer][size_t(slots[i])];
                jobs[n++]=job(m[0],batch_x_,gate_+i*size_t(g.ff));
                jobs[n++]=job(m[1],batch_x_,up_+i*size_t(g.ff));
            }
            require(gg::mv(jobs,int(n),stream), "HIP batch gate/up failed");
        }
        trace("gate_up_including_quant",stream);
        if(swiglu_batch_)
            strata::ds4::swiglu(gate_,up_,g.ff*int(slots.size()),g.lim,hidden_,stream);
        else for(size_t i=0;i<slots.size();++i)
            strata::ds4::swiglu(gate_+i*size_t(g.ff),up_+i*size_t(g.ff),g.ff,g.lim,hidden_+i*size_t(g.ff),stream);
        trace("swiglu",stream);
        for(size_t first=0;first<slots.size();first+=jobs_max) {
            Job jobs[jobs_max]{};size_t n=0;
            for(size_t i=first;i<std::min(slots.size(),first+jobs_max);++i)
                jobs[n++]=job(matrices_[layer][size_t(slots[i])][2],hidden_+i*size_t(g.ff),batch_y_+i*size_t(g.dim));
            require(gg::mv(jobs,int(n),stream), "HIP batch down failed");
        }
        trace("down_including_quant",stream);
        }
        require(hipMemcpyAsync(batch_output_.data(),batch_y_,slots.size()*x.size_bytes(),hipMemcpyDeviceToHost,stream)==hipSuccess,
                "HIP batch output copy failed");
        trace("d2h",stream);
        hits_+=slots.size();
    }
    // Call only after the queued D2H has completed successfully.
    std::span<const float> layer_output() const {
        return {batch_output_.data(),batch_count_*batch_input_.size()};
    }
    // Checker toggles the identical loaded bank; production sets it only at load.
    void set_swiglu_batch(bool value) {
        require(ready_ && batch_capacity_ && !soa_ && !cpu_q8k_,"SwiGLU checker mode");swiglu_batch_=value;
    }
    uint64_t uploaded() const { return uploaded_; }
    uint64_t hits() const { return hits_; }
    uint64_t workspace() const { return workspace_; }
    size_t staging_bytes()const{return staging_.size();}
    std::array<const float*,3> probe_intermediates()const {
        require(ready_,"unready resident probe");return {gate_,up_,hidden_};
    }
    // Caller MUST drain stream before reset/arena clear, including partial-load failure.
    void reset() noexcept { ready_=false;batch_capacity_=batch_count_=0;batch_x_=batch_y_=nullptr;batch_input_.clear();batch_output_.clear();matrices_.clear();std::vector<std::byte>().swap(staging_); gate_ = up_ = hidden_ = nullptr; quant_ = nullptr; soa_=false;swiglu_batch_=false;soa_quant_=nullptr;soa_scale_=nullptr;soa_tables_=nullptr; cpu_q8k_ = false; uploaded_ = hits_ = workspace_ = 0; }
};
} // namespace strata::wiring
