// MIT; crosslane contributors. Host-only (plain g++): builds the frozen stage::Plan and the
// coordinator's EngramNgram from the pack, for the split runner and the parity harness.
#pragma once
#include "../engram_ngram.hpp"
#include "weight_source.hpp"
#include "hip_resident.hpp"
#include "../../../warm/frozen_placement.hpp"
#include "codec.hpp"

namespace strata::wiring {

// Per-layer VRAM estimate for admission: the layer's dense tensors as stored (F32 conversions of BF16
// norms are rounded up by 2x), its ring, compressor state and the source caches it owns. Excluded:
//  * routed experts (base plan: read by CPU; P1 adds aligned resident bytes below);
//  * the Engram tables `layers.L.engram.embed.weight` (~55.3 GB each on this pack): never uploaded. Both
//    stages pread the cols selected rows per token on the host (page cache, no device or RAM row cache)
//    and stage them through the per-stage Engram scratch, which is counted on the Engram layer below.
// begin() measures the exact device bytes (arena + dense matrices) and rejects a stage whose measurement
// exceeds the frozen Plan footprint for its range (review 2026-11-02 #3).
inline bool device_resident(const std::string& n) {
    return n.find(".ffn.experts.") == std::string::npos && n.find(".engram.embed.") == std::string::npos;
}
inline std::vector<uint64_t> layer_bytes(const WeightSource& ws, const Topology& t) {
    const auto& g = ws.meta();
    std::vector<uint64_t> out(t.layers.size(), 0);
    for (const auto& n : ws.names()) {
        if (n.rfind("layers.", 0) != 0 || !device_resident(n)) continue;
        const auto il = uint32_t(std::stoul(n.substr(7, n.find('.', 7) - 7)));
        if (il >= out.size()) continue;
        const auto ti = *ws.find(n);
        out[il] += ti.dtype == "F32" || ti.dtype == "F16" || ti.dtype == "BF16" ? 2 * ti.bytes : ti.bytes;
    }
    const uint64_t cols = uint64_t(std::max(0, g.eg_cols())), ehd = uint64_t(std::max(0, g.eg_hd));
    const uint64_t eg_stage = cols * (ehd + ehd / 32) + cols * ehd * 4 + uint64_t(g.hc + 1) * uint64_t(g.dim) * 4;
    for (const auto& l : t.layers) {
        out[l.il] += uint64_t(g.win) * g.win_rb() + (l.kv_src && l.ratio > 1 ? 2ull * uint64_t(l.ratio) * uint64_t(g.hd) * 4 : 0);
        for (const auto& s : t.sources)
            if (s.layer == l.il) out[l.il] += s.cap * (g.ckv_rb() + (l.ix_src ? g.ik_rb() : 0));
        if (l.eg >= 0) out[l.il] += eg_stage;
    }
    return out;
}

inline std::vector<stage::EngramLayer> engram_tables(const ModelMeta& g) {
    std::vector<stage::EngramLayer> v;
    const size_t cols = size_t(g.eg_cols());
    for (int li = 0; li < g.eg_nlayers; ++li) {
        strata::ds4::engram::HashTable h;
        h.ngram = g.eg_ngram; h.heads = g.eg_heads; h.pad = g.eg_pad;
        h.multipliers.assign(g.eg_mult.begin() + li * g.eg_ngram, g.eg_mult.begin() + (li + 1) * g.eg_ngram);
        h.primes.assign(g.eg_primes.begin() + ptrdiff_t(li * cols), g.eg_primes.begin() + ptrdiff_t((li + 1) * cols));
        h.offsets.assign(g.eg_off.begin() + ptrdiff_t(li * cols), g.eg_off.begin() + ptrdiff_t((li + 1) * cols));
        v.push_back({uint32_t(g.eg_layers[size_t(li)]), std::move(h)});
    }
    return v;
}
inline std::unique_ptr<stage::EngramNgram> make_ngram(const ModelMeta& g) {
    return std::make_unique<stage::EngramNgram>(g.eg_map, engram_tables(g));
}

// The split plan: SYCL [0,cut), HIP [cut,n) + head. cut == 0 lets stage::choose_cut pick the smallest
// fitting SYCL prefix under the frozen caps (capacity rule, not a speed claim).
inline stage::Plan split_plan(const WeightSource& ws, int64_t max_context, uint32_t cut, const StageCaps& caps,
                              const stage::Ngram& ngram, bool hip_dense_only=false) {
    const auto& g = ws.meta();
    const Topology t = topology(g, max_context);
    const auto lb = layer_bytes(ws, t);
    auto lay = ngram.layout();
    stage::Plan p = make_plan(g, t, ws.identity(), schema_for(max_context), lb, lb, caps,
                              std::vector<stage::EngramSlice>(lay.begin(), lay.end()));
    if(hip_dense_only){
        require(cut==0,"HIP dense ownership requires explicit cut0");
        p.schema_sha+=";warm-dense=hip-v1;";p.cut=0;p.hip.mirrors=0;p.validate();return p;
    }
    if (cut == 0) return stage::choose_cut(p);
    p.cut = cut;
    p.hip.mirrors = p.crossing_bytes();
    p.validate();
    return p;
}

inline stage::Plan configured_split_plan(const WeightSource& ws, const Config& cfg, uint32_t cut,
                                         const StageCaps& caps, const stage::Ngram& ngram,
                                         HipResidency* selected = nullptr, WarmResidency* warm_selected = nullptr) {
    require(!cfg.warm_experts || (caps.hip_fixed==cfg.warm_hip_fixed && caps.sycl_fixed==cfg.warm_sycl_fixed),"warm base fixed budgets differ from config");
    auto p = split_plan(ws, cfg.max_context, cut, caps, ngram, cfg.warm_dense_owner=="hip-v1");
    // P1's workspace in HipStage::begin supports only the complete tail.
    // Refuse during plan-only/freeze, before selecting banks or loading a device.
    require(!cfg.warm_experts || p.cut == 1 || p.hip_dense_only(), "warm P1 requires the complete HIP tail at cut1");
    auto r = cfg.warm_experts ? select_warm_hip(ws,cfg,p,{}) : select_hip_residency(ws, cfg, p);
    const auto prefix_used=p.footprint(p.sycl,0,p.cut);
    require(!cfg.warm_experts || (prefix_used<=p.sycl.cap && kWarmSyclWorkspace<p.sycl.cap-prefix_used),"no B65 bank room");
    auto sycl = cfg.warm_experts ? select_warm_sycl(ws,cfg,r,p.sycl.cap-prefix_used-kWarmSyclWorkspace) : WarmResidency{};
    require(!cfg.warm_experts || sycl.payload>=20000000000ull,"warm B65 bank below 20GB");
    if(cfg.warm_experts)apply_frozen_warm_placement(ws,cfg,r,sycl);
    apply_hip_residency(p, r);
    apply_warm_sycl(p,cfg,sycl);
    if (selected) *selected = std::move(r);
    if (warm_selected) *warm_selected = std::move(sycl);
    return p;
}
// Both DSOs independently reconstruct P1's full plan BEFORE device allocation.
// No ABI extension: schema identity binds profile + slots; budgets bind bytes.
inline HipResidency check_hip_residency_plan(const WeightSource& ws, const Config& cfg, const stage::Plan& p, WarmResidency* warm_selected = nullptr) {
    if (cfg.hip_experts == "off") {
        require(p.schema_sha == schema_for(cfg.max_context)+(cfg.warm_dense_owner=="hip-v1"?";warm-dense=hip-v1;":""), "plan schema/context identity mismatch");
        return {};
    }
    if(cfg.warm_experts){
        auto ng=make_ngram(ws.meta());HipResidency r;
        const auto expected=configured_split_plan(ws,cfg,p.cut,{p.sycl.cap,p.hip.cap,cfg.warm_sycl_fixed,cfg.warm_hip_fixed},*ng,&r,warm_selected);
        require(codec::encode(p)==codec::encode(expected),"warm complementary bank plan differs");return r;
    }
    constexpr auto reserve = kHipExpertWorkspaceReserve + kHipDriverBaseReserve + kHipDriverSafetyMargin;
    require(p.hip.fixed >= reserve+(cfg.warm_experts?cfg.warm_hip_bytes:0), "missing HIP driver/workspace/bank reservation");
    require(p.sycl.fixed >= warm_sycl_reservation(cfg),"missing warm SYCL reservation");
    auto ngram = make_ngram(ws.meta());
    HipResidency r;
    const auto expected = configured_split_plan(ws, cfg, p.cut,
        {p.sycl.cap, p.hip.cap, p.sycl.fixed-warm_sycl_reservation(cfg), p.hip.fixed-reserve-(cfg.warm_experts?cfg.warm_hip_bytes:0)}, *ngram, &r);
    require(codec::encode(p) == codec::encode(expected), "P1 plan/profile/slots/accounting differs");
    return r;
}

// Single-layer parity plan: cut = L, dependencies restricted to edges whose consumer is L (so the stage
// [L, L+1) only allocates what layer L reads). Budgets are set to the estimate (no admission claim).
inline stage::Plan layer_plan(const WeightSource& ws, int64_t max_context, uint32_t L, const stage::Ngram& ngram) {
    const auto& g = ws.meta();
    const Topology t = topology(g, max_context);
    require(L < uint32_t(g.nl), "parity layer must be in [0, n_layers)");
    const auto lb = layer_bytes(ws, t);
    uint64_t total = 0; for (auto b : lb) total += b;
    auto lay = ngram.layout();
    stage::Plan p = make_plan(g, t, ws.identity(), schema_for(max_context), lb, lb, {total, total, 0, 0},
                              std::vector<stage::EngramSlice>(lay.begin(), lay.end()));
    std::vector<stage::Edge> keep;
    for (const auto& e : p.dependencies) if (e.consumer == L) keep.push_back(e);
    p.dependencies = keep;
    p.cut = L == 0 ? 1 : L; // L0 executes the real prefix, with no incoming boundary.
    p.hip.cap = UINT64_MAX / 4;  // test plan: no device admission is claimed
    p.hip.mirrors = p.crossing_bytes();
    p.validate();
    return p;
}
}  // namespace strata::wiring
