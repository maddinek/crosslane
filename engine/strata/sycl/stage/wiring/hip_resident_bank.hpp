// MIT; crosslane contributors. Retained HIP matrices + serial expert workspace.
// gg::mv and swiglu are upstream's DS4_SYNC path. Arena owns all device lifetimes.
#pragma once
#include "hip_resident.hpp"
#include "ds4_gguf.cuh"
#include "ds4_kernels.cuh"
#include "hip_q8k.hpp"

namespace strata::wiring {
class HipResidentBank {
    using Mat = strata::dsv41k::exl3::Mat;
    using Job = strata::dsv41k::exl3::MvJob;
    std::vector<std::vector<std::array<Mat, 3>>> matrices_;
    std::vector<std::byte> staging_; // survives a failed async-copy drain until reset
    float *gate_ = nullptr, *up_ = nullptr, *hidden_ = nullptr;
    void* quant_ = nullptr;
    bool cpu_q8k_ = false;
    uint64_t uploaded_ = 0, hits_ = 0, workspace_ = 0;
    bool ready_ = false;
public:
    template<class Arena> void load(const WeightSource& ws, const HipResidency& r, Arena& arena, hipStream_t stream) {
        namespace gg = strata::ds4::gguf;
        require(matrices_.empty(), "resident bank already loaded");
        const bool pooled=pooled_policy(r.policy);
        if(pooled)validate_pooled_layout(r); // before workspace or any device allocation
        matrices_.resize(r.slots.size());
        cpu_q8k_ = r.cpu_q8k;
        if (r.experts.empty()) {ready_=true;return;}
        // Device-format admission before the first weight allocation/upload.
        for (const auto& e : r.experts) for (const auto& t : e.matrices) {
            require(gg::type_on_device(strata::ds4::moe::ggml_type(strata::ds4::moe::type(t.dtype))), "resident type has no HIP kernel");
            if (cpu_q8k_) require(q8k::supported_type(strata::ds4::moe::ggml_type(strata::ds4::moe::type(t.dtype))) &&
                q8k::activation_bytes(int(t.shape[1])) > 0, "resident type/geometry has no Q8_K kernel");
        }
        const auto row = expert_align(strata::ds4::moe::product(size_t(ws.meta().ff), sizeof(float)));
        const auto quant_bytes = cpu_q8k_ ? q8k::activation_bytes(std::max(ws.meta().dim,ws.meta().ff)) : 0;
        require(stage::add(strata::ds4::moe::product(size_t(row), size_t(3)),expert_align(quant_bytes)) <= r.workspace,
            "resident workspace reservation too small");
        const auto before = arena.bytes;
        gate_ = arena.template dev<float>(size_t(ws.meta().ff));
        up_ = arena.template dev<float>(size_t(ws.meta().ff));
        hidden_ = arena.template dev<float>(size_t(ws.meta().ff));
        if (cpu_q8k_) quant_ = arena.template dev<uint8_t>(quant_bytes);
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
        ready_=true; // no partially loaded bank can serve a hit
    }
    void expert(uint32_t layer, int slot, const float* x, float* y, const ModelMeta& g, hipStream_t stream) {
        namespace gg = strata::ds4::gguf;
        require(ready_ && slot >= 0, "unready bank/negative resident slot");
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
    uint64_t uploaded() const { return uploaded_; }
    uint64_t hits() const { return hits_; }
    uint64_t workspace() const { return workspace_; }
    size_t staging_bytes()const{return staging_.size();}
    std::array<const float*,3> probe_intermediates()const {
        require(ready_,"unready resident probe");return {gate_,up_,hidden_};
    }
    // Caller MUST drain stream before reset/arena clear, including partial-load failure.
    void reset() noexcept { ready_=false;matrices_.clear();std::vector<std::byte>().swap(staging_); gate_ = up_ = hidden_ = nullptr; quant_ = nullptr; cpu_q8k_ = false; uploaded_ = hits_ = workspace_ = 0; }
};
} // namespace strata::wiring
