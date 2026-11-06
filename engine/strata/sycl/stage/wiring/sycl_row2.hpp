

// The optional row2_stub.cpp still fails closed. Contracts are upstream's:
// gg::mv (ds4_gguf.cuh), gemv_f32 / gemv_f16_rows / swiglu (ds4_kernels.cuh), cpu_experts (CPU lane).
// In-order queue, caller-owned USM, T == 1 is the only admitted window here.
#pragma once
#include "../../numerics.hpp"
#include "../../route/route.hpp"
#include "weight_source.hpp"
#include "decode_timing.hpp"
#include "../../moe/iq2_policy.hpp"
#include <memory>
#include <span>

namespace strata::wiring {class CpuLane;}
namespace strata::ds4::row2 {
// A dense matrix as gg::mv takes it: out = n rows, in = k values. Bytes are the pack's, uploaded once.
struct DenseMat {
    std::string name, dtype;   // tensors.txt dtype; row 2 rejects any it has no exact kernel for
    int n = 0, k = 0;
    size_t ld = 0;             // bytes a row (BF16: 2k; Q8_0: k/32*34; ggml block types: type_row_bytes)
    void* raw = nullptr;       // USM device, owned by the stage arena
};
// Uploads `name` (whole matrix; no tensor-parallel split). Throws for formats row 2 does not admit.
DenseMat upload(sycl::queue&, const wiring::WeightSource&, const std::string& name);
void release(sycl::queue&, DenseMat&) noexcept;
// y[c*y_cs + r] = alpha * sum_j W[r][j] * x[c*x_cs + j] for c < ncol (gg::mv / x3::MvJob semantics,
// identical summation order to the HIP kernel for each admitted format).
struct MvJob { const DenseMat* m = nullptr; const float* x = nullptr; float* y = nullptr; int ncol = 1;
               size_t x_cs = 0, y_cs = 0; float alpha = 1.0f; };
sycl::event mv(sycl::queue&, std::span<const MvJob>, IqPath = IqPath::control);
// Router logits in F32, warp-per-row (ds4_kernels.cu gemv_f32_kernel order).
sycl::event gemv_f32(sycl::queue&, const float* W, const float* x, int n, int k, int T, float* y, int ldy);
// Indexer weights_proj, W F16 bits [n][k] (gemv_f16_rows: bit-identical to gemv_f16 per row).
sycl::event gemv_f16_rows(sycl::queue&, const uint16_t* W, const float* x, size_t ldx, int n, int k, int T,
                          float alpha, float* y, size_t ldy);
// h = silu(min(g, lim)) * clamp(u, -lim, lim)  (lim <= 0: no clamp), ds4_kernels.cu swiglu_kernel.
sycl::event swiglu(sycl::queue&, const float* g, const float* u, int n, float lim, float* h);

// Routed experts of layers [first, end) for route_port::ExpertService. device()/cpu() compute ONE expert's
// FFN output (gate, up, swiglu, down) for one token row; publish() is supplied by the layer driver.
// The CPU lane MUST be upstream's (ds4_gg_cpu.cpp / ds4_iq_cpu.cpp / ds4_gguf_cpu.cpp, compiled as host
// code by the reference CXX compiler; P2 uses a shared host DSO) so the AMD-only reference,
// the HIP tail and this SYCL prefix share CPU bits.
struct ExpertCompute {
    virtual ~ExpertCompute() = default;
    virtual bool resident(size_t layer, int expert) const = 0;
    virtual bool store_contains(size_t layer, int expert) const = 0;
    virtual void stage(size_t layer, int expert) = 0;
    virtual std::vector<float> device(size_t layer, int expert, std::span<const float> x) = 0;
    virtual std::vector<float> cpu(size_t layer, int expert, std::span<const float> x) = 0;
    virtual void drain() noexcept = 0;
    virtual uint64_t device_bytes(uint32_t layer) const = 0;  // exact VRAM of this layer's resident experts
    virtual void begin_decode_step(bool) {}
    virtual wiring::CpuPhases decode_phases() const { return {}; }
    virtual wiring::CpuLane* host_lane() { return nullptr; }
};
// placement "all-cpu": no resident experts (first parity gate). cpu_threads 0: physical cores.
std::unique_ptr<ExpertCompute> make_experts(sycl::queue&, const wiring::WeightSource&, uint32_t first, uint32_t end,
                                            const std::string& placement, int cpu_threads,
                                            const std::string& dispatch = "spawn-v1", bool timing = false, IqPath = IqPath::control);
}  // namespace strata::ds4::row2
