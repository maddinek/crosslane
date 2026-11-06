// MIT; crosslane contributors. Host-only, toolchain-neutral (g++, icpx, hipcc host pass).
// Geometry, layer roles, per-position plan and the complete dependency inventory of DeepSeek V4.1
// as pinned Strata c62d5e3 src/dsv41/ds4_model.cu derives them. Derived from Project Maya / Strata
// (MIT); preserve third_party/project-maya/{LICENSE,NOTICE.md} with distributions.
// Every fact comes from pack metadata (ds4.txt) or tensors.txt: no model constants here.
#pragma once
#include "../stage.hpp"
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::wiring {
using stage::require;
inline void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }

constexpr int kWin = 16;       // ds4_kernels.cuh kWin (T <= 16; this driver runs T == 1)
constexpr int kPicks = 8;      // ds4_kernels.cuh kPicks: ridx/rw row stride
constexpr int kMixStride = 32, kPreStride = 8, kCombStride = 64, kGlogStride = 1024, kIwStride = 64;
// ds4_kernels.cuh cache row formats (same numbering in sycl_port::row_bytes).
constexpr int kKvBf16 = 0, kKvFp8 = 1, kKvFp4 = 2, kKvFp4p = 3;
inline size_t kv_row_bytes(int fmt, int d) {
    return fmt == kKvBf16  ? size_t(d) * 2
           : fmt == kKvFp8 ? (size_t(d) + size_t(d) / 32 + 15) / 16 * 16
           : fmt == kKvFp4 ? (size_t(d) / 2 + size_t(d) / 16 + 15) / 16 * 16
                           : (size_t(d) / 2 + size_t(d) / 32 + 15) / 16 * 16;
}

// Mirrors Geo (ds4_model.cu:501-546) plus the Engram header of engram.bin (DSE1).
struct ModelMeta {
    int vocab = 0, dim = 0, nl = 0, heads = 0, hd = 0, rd = 0, q_lora = 0, o_lora = 0, o_groups = 0, n_exp = 0,
        k_exp = 0, ff = 0, win = 0, hc = 0, sinkhorn = 0, ix_heads = 0, ix_hd = 0, ix_topk = 0, cand_src = -1,
        cand_topk = 0, cand_blk = 0, orig_ctx = 0, eos = 1;
    float eps = 1e-20f, hc_eps = 1e-6f, route_scale = 1.0f, lim = 0.0f;
    double rope_theta = 1e4, comp_theta = 1e4, factor = 1.0, beta_fast = 32, beta_slow = 1;
    std::vector<int> ratios, kv_src, ix_src, eg_layers;
    // Engram (engram.bin): vocab of the compressed map, layers, ngram, heads, row width, pad token.
    int eg_vocab = 0, eg_nlayers = 0, eg_ngram = 0, eg_heads = 0, eg_hd = 0, eg_pad = 0;
    std::vector<int32_t> eg_map;
    std::vector<int64_t> eg_mult, eg_primes, eg_off;
    int eg_cols() const { return (eg_ngram - 1) * eg_heads; }
    size_t topk_stride() const { return size_t(std::max(ix_topk, 32)); }       // ds4_model.cu:6141
    size_t cand_stride() const { return size_t(std::max(cand_topk, 32)); }     // :6142
    size_t bm_stride(int64_t max_ctx) const { return size_t(max_ctx + 64) / size_t(std::max(1, cand_blk)) + 64; }
    size_t win_rb() const { return kv_row_bytes(kKvFp8, hd); }
    size_t ckv_rb() const { return kv_row_bytes(kKvFp4, hd); }
    size_t ik_rb() const { return kv_row_bytes(kKvFp4p, ix_hd); }
    void validate() const {
        require(vocab > 0 && dim > 0 && nl > 1 && heads > 0 && hd > 0 && rd > 0 && rd <= hd && q_lora > 0 &&
                    o_lora > 0 && o_groups > 0 && o_groups <= 16 && n_exp > 0 && k_exp > 0 && k_exp <= kPicks &&
                    ff > 0 && win > 0 && hc > 0 && hc <= 8 && (2 + hc) * hc <= kMixStride && sinkhorn >= 0,
                "invalid dsv41 geometry");
        require(int(ratios.size()) >= nl, "compress_ratios shorter than layer count");
        require(heads * hd % o_groups == 0, "heads*hd not divisible by o_groups");
        require(rd / 2 <= 64, "rope frequency table exceeds 64 floats (ds4_model.cu dalloc freqs 64)");
    }
};

// Per-layer roles exactly as Impl::load assigns them (ds4_model.cu:2392-2412).
struct LayerRole {
    uint32_t il = 0;
    int ratio = 0;
    bool kv_src = false, ix_src = false, owns_k = false, cand_src = false, uses_cand = false;
    int src = -1, ik_src = -1;  // index into Topology::sources
    int eg = -1;                // Engram table index, -1 none
};
struct Source {
    uint32_t layer = 0;
    int ratio = 1;
    uint64_t cap = 0;  // rows: (max_context + ratio - 1) / ratio + 1
};
struct Topology {
    std::vector<LayerRole> layers;
    std::vector<Source> sources;
    int64_t max_context = 0;
};
inline bool contains(const std::vector<int>& v, int x) { return std::find(v.begin(), v.end(), x) != v.end(); }
inline Topology topology(const ModelMeta& g, int64_t max_context) {
    g.validate();
    require(max_context > 0, "max_context");
    Topology t;
    t.max_context = max_context;
    int last_src = -1;
    for (int il = 0; il < g.nl; ++il) {
        LayerRole l;
        l.il = uint32_t(il);
        l.ratio = g.ratios[size_t(il)];
        l.kv_src = contains(g.kv_src, il);
        l.ix_src = contains(g.ix_src, il);
        l.owns_k = l.kv_src;
        l.cand_src = il == g.cand_src;
        l.uses_cand = g.cand_src >= 0 && g.cand_src < il && l.ix_src;
        if (l.kv_src) {
            require(l.ratio > 0, "KV source without compression");
            t.sources.push_back({uint32_t(il), l.ratio, uint64_t((max_context + l.ratio - 1) / l.ratio + 1)});
            last_src = int(t.sources.size()) - 1;
        }
        if (l.ratio > 0) {
            require(last_src >= 0 && t.sources[size_t(last_src)].ratio == l.ratio,
                    "compress ratio without a KV source of that ratio before it");
            l.src = last_src;
            l.ik_src = last_src;
        }
        auto eg = std::find(g.eg_layers.begin(), g.eg_layers.end(), il);
        if (eg != g.eg_layers.end()) l.eg = int(eg - g.eg_layers.begin());
        t.layers.push_back(l);
    }
    return t;
}

// DeepSeek precompute_freqs_cis (YaRN when original > 0), F32 as torch: ds4_model.cu:548-566, verbatim.
inline std::vector<float> rope_freqs(int dim, int original, double base, double factor, double beta_fast,
                                     double beta_slow) {
    std::vector<float> f(size_t(dim) / 2);
    for (int i = 0; i < dim / 2; ++i) {
        const float e = float(2 * i) / float(dim);
        f[size_t(i)] = 1.0f / std::pow(float(base), e);
    }
    if (original > 0) {
        auto corr = [&](double rot) {
            return dim * std::log(original / (rot * 2 * 3.14159265358979323846)) / (2 * std::log(base));
        };
        const double low = std::max(std::floor(corr(beta_fast)), 0.0);
        const double high = std::min(std::ceil(corr(beta_slow)), double(dim) - 1);
        const float den = float(std::max(high - low, 1e-3));
        for (int i = 0; i < dim / 2; ++i) {
            float ramp = (float(i) - float(low)) / den;
            ramp = std::min(1.0f, std::max(0.0f, ramp));
            const float smooth = 1.0f - ramp;
            f[size_t(i)] = f[size_t(i)] / float(factor) * (1.0f - smooth) + f[size_t(i)] * smooth;
        }
    }
    return f;
}

// plan_window (ds4_model.cu:9566-9597) for T == 1. The carried cur_topk_n / cur_cand_n persist across
// positions AND requests exactly as upstream (reset() does not clear them). Every stage advances its
// own planner over ALL layers, so both runtimes derive identical per-layer counts from the position.
class StepPlanner {
    int cur_topk_ = 0, cur_cand_ = 0;
public:
    std::vector<int> topk, cand;  // per layer, valid after advance()
    void advance(const ModelMeta& g, const Topology& t, uint64_t p) {
        topk.assign(t.layers.size(), 0);
        cand.assign(t.layers.size(), 0);
        for (const auto& l : t.layers) {
            if (l.ratio <= 0) continue;
            const int n = int((p + 1) / uint64_t(l.ratio)), k = std::min(g.ix_topk, n);
            if (l.ix_src) {
                cand[l.il] = cur_cand_;
                if (n > 0 && l.cand_src) cur_cand_ = std::min(g.cand_topk, (n + g.cand_blk - 1) / g.cand_blk);
                cur_topk_ = k;
            }
            topk[l.il] = cur_topk_;
        }
    }
};

// Complete dependency inventory for stage::Plan. Producers/consumers follow the layer loop above:
//   compressed_kv : source layer s -> every ratio>0 layer reading srcs[l.src]            (accumulated, ratio R)
//   index_rows    : owns_k&&ix_src source layer -> every ix_src layer with ik_src == s   (accumulated, ratio R)
//   selected_topk : last ix_src layer before c -> ratio>0 non-ix_src layer c             (current_step)
//   candidates    : cand_src layer -> uses_cand layers                                   (current_step)
//   hc_pre        : layer l -> l+1 (d.pre = f_pre; ds4_model.cu:10306,10567)              (current_step)
// visible_ring is never emitted: every window ring is private to its layer (Layer::win).
inline std::vector<stage::Edge> dependencies(const ModelMeta& g, const Topology& t) {
    std::vector<stage::Edge> out;
    const auto& L = t.layers;
    const uint64_t pre_bytes = kPreStride * sizeof(float);
    for (uint32_t il = 0; il + 1 < L.size(); ++il)
        out.push_back({il, il + 1, stage::Kind::hc_pre, pre_bytes, pre_bytes, 1, stage::RowMode::current_step});
    const uint64_t ckv = g.ckv_rb(), ik = g.ik_rb();
    for (const auto& l : L) {
        if (l.ratio <= 0) continue;
        const auto& s = t.sources[size_t(l.src)];
        if (s.layer != l.il)
            out.push_back({s.layer, l.il, stage::Kind::compressed_kv, s.cap * ckv, ckv, uint64_t(s.ratio),
                           stage::RowMode::accumulated});
        if (l.ix_src && !l.owns_k) {
            const auto& k = t.sources[size_t(l.ik_src)];
            // A non-indexing source never writes index keys: upstream reads its zeroed buffer; so do both stages.
            if (L[k.layer].ix_src) out.push_back({k.layer, l.il, stage::Kind::index_rows, k.cap * ik, ik, uint64_t(k.ratio),
                           stage::RowMode::accumulated});
        }
        if (!l.ix_src) {
            int last = -1;
            for (int j = int(l.il) - 1; j >= 0; --j)
                if (L[size_t(j)].ix_src) { last = j; break; }
            // No earlier indexer: cur_topk_n stays 0, so nsel == 0 and the zeroed buffer is never read. No edge.
            if (last < 0) continue;
            const uint64_t tb = g.topk_stride() * sizeof(int32_t);
            out.push_back({uint32_t(last), l.il, stage::Kind::selected_topk, tb, tb, 1, stage::RowMode::current_step});
        }
        if (l.uses_cand) {
            const uint64_t cb = g.cand_stride() * sizeof(int32_t);
            out.push_back({uint32_t(g.cand_src), l.il, stage::Kind::candidates, cb, cb, 1,
                           stage::RowMode::current_step});
        }
    }
    return out;
}

// Upstream dense_bytes (ds4_model.cu:2440-2452) is a placement estimate; the backends report their
// exact allocations instead. This only carries caller-measured numbers into stage::Budget.
struct StageCaps { uint64_t sycl_cap = 0, hip_cap = 0, sycl_fixed = 0, hip_fixed = 0; };
inline stage::Plan make_plan(const ModelMeta& g, const Topology& t, std::string model_sha, std::string schema_sha,
                             const std::vector<uint64_t>& sycl_layer_bytes, const std::vector<uint64_t>& hip_layer_bytes,
                             const StageCaps& caps, std::vector<stage::EngramSlice> engram_layout) {
    stage::Plan p;
    p.model_sha = std::move(model_sha);
    p.schema_sha = std::move(schema_sha);
    p.layers = uint32_t(g.nl);
    p.hc = uint32_t(g.hc);
    p.dim = uint32_t(g.dim);
    p.activation_bytes = uint64_t(g.hc) * uint64_t(g.dim) * sizeof(float);
    p.sycl = {caps.sycl_cap, caps.sycl_fixed, 0, sycl_layer_bytes};
    p.hip = {caps.hip_cap, caps.hip_fixed, 0, hip_layer_bytes};
    p.dependencies = dependencies(g, t);
    p.dependencies_complete = true;
    p.engram_layout = std::move(engram_layout);
    return p;
}
// Schema identity of this driver's boundary layout; bump when the packet format or edge rules change.
// The context capacity is part of it: source snapshot extents are cap(ctx) rows, so a plan built for one
// max_context is rejected by a stage configured for another (review 2026-11-02 #1).
inline const char* kSchema = "strata-dsv41-wiring-v2:residual-f32,hc_pre-8f32,topk-i32,cand-i32,ckv-fp4,ik-fp4p";
inline std::string schema_for(int64_t max_context) { return std::string(kSchema) + ";max_context=" + std::to_string(max_context); }
}  // namespace strata::wiring
