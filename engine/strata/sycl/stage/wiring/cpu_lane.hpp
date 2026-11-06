// MIT; crosslane contributors. Host-only. Row 2 CPU expert lane shared by
// BOTH stage DSOs. It must be upstream's own CPU arithmetic (Strata c62d5e3 src/dsv41/ds4_gg_cpu.cpp,
// ds4_iq_cpu.cpp, ds4_gguf_cpu.cpp: ggml-cpu dots incl. their activation quantization), compiled once
// into libstrata-row2-cpu.so with the reference build's host flags, so the AMD-only reference, the HIP tail and
// the SYCL prefix produce identical CPU-lane bits. Upstream entry: Impl::cpu_experts (ds4_model.cu:6816),
// which is an Impl member today; row 2 extracts its per-expert body (gate, up, swiglu, down) into a free
// function taking (layer tensors, x) and binds it here.
#pragma once
#include "weight_source.hpp"
#include "decode_timing.hpp"
#include <memory>
#include <span>

namespace strata::wiring {
class CpuLane {
public:
    virtual ~CpuLane() = default;
    // y = down(swiglu(gate(x), up(x))) of routed expert `e` of layer `layer`, F32 [dim]; upstream's c_yd row.
    virtual std::vector<float> expert(uint32_t layer, int e, std::span<const float> x) = 0;
    virtual void drain() noexcept = 0;
    // Warm serving may issue one layer's cache misses together. Default parity
    // lane remains synchronous and unchanged.
    virtual void prepare(uint32_t, std::span<const int>) {}
    virtual void finish() noexcept {}
    virtual void begin_decode_step(bool) {}
    virtual CpuPhases decode_phases() const { return {}; }
    virtual bool layer_batch() const { return false; }
    virtual std::vector<std::vector<float>> experts(uint32_t layer, std::span<const int> ids,
                                                   std::span<const float> x) {
        std::vector<std::vector<float>> out;
        for (int e : ids) out.push_back(expert(layer, e, x));
        return out;
    }
};
// Tensors: layers.L.ffn.experts.E.{w1 (gate), w3 (up), w2 (down)}.weight, ggml block types (MXFP4, IQ*, Q*_K).
std::unique_ptr<CpuLane> make_cpu_lane(const WeightSource&, uint32_t first, uint32_t end, int threads,
                                       const std::string& dispatch = "spawn-v1", bool timing = false);
}  // namespace strata::wiring
