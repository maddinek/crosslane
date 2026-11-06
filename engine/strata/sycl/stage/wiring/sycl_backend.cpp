// MIT; crosslane contributors. SYCL (B65, Level Zero) layer driver: stage::Backend for layers
// [first,end) of DeepSeek V4.1, T == 1, synchronous host-routed MoE (upstream DS4_SYNC=1 path,
// ds4_model.cu:10414-10571). The op order below is ds4_model.cu Impl::layer (9961-10571) line by line;
// each call names the row it comes from. Derived from Project Maya / Strata c62d5e3 (MIT).
// No numerics live here: every arithmetic op is a row function (hc, attn, numerics, engram, route, row2).
#include "common.hpp"
#include "host_plan.hpp"
#include "sycl_row2.hpp"
#include "../../../warm/device_timing.hpp"
#include "../../../warm/dense_trace.hpp"
#ifdef STRATA_FORK_WARM
#include "../../../warm/sycl_bank.hpp"
#include "../../../warm/runtime.hpp"
#endif
#include "sycl_decode_events.hpp"
#include "../../device.hpp"
#include "../../hc/hc.hpp"
#include "../../attn/attn.hpp"
#include "../../engram/engram.hpp"
#include "../../route/route_sycl.hpp"
#include <atomic>

namespace strata::wiring {
namespace sp = strata::ds4::sycl_port;
namespace eg = strata::ds4::engram;
namespace rp = strata::ds4::route_port;
namespace r2 = strata::ds4::row2;
namespace {

class Arena {  // every USM allocation of the stage; zeroed at allocation (dalloc semantics)
    sycl::queue& q_;
    std::vector<void*> p_;
public:
    uint64_t bytes = 0;
    explicit Arena(sycl::queue& q) : q_(q) {}
    ~Arena() { clear(); }
    template <class T> T* dev(size_t n) {
        require(n > 0, "zero-size device allocation");
        auto* p = sycl::malloc_device<T>(n, q_);
        if (!p) throw std::bad_alloc();
        p_.push_back(p);
        q_.memset(p, 0, n * sizeof(T)).wait_and_throw();
        bytes += n * sizeof(T);
        return p;
    }
    template <class T> T* upload(const std::vector<T>& v) {
        T* p = dev<T>(v.size());
        q_.memcpy(p, v.data(), v.size() * sizeof(T)).wait_and_throw();
        return p;
    }
    void clear() noexcept {
        try { q_.wait(); } catch (...) {}
        for (void* p : p_) try { sycl::free(p, q_); } catch (...) {}
        p_.clear(); bytes = 0;
    }
};

struct LayerW {
    LayerRole role;
    r2::DenseMat wq_a, wq_b, wkv, wo_a[16], wo_b, s1, s2, s3, c_wkv, c_wgate, i_wq_b, i_wk, eg_wkv;
    float *q_norm = nullptr, *kv_norm = nullptr, *attn_norm = nullptr, *ffn_norm = nullptr, *c_norm = nullptr,
          *ik_norm = nullptr, *sink = nullptr, *hca_fn = nullptr, *hca_scale = nullptr, *hca_base = nullptr,
          *hcf_fn = nullptr, *hcf_scale = nullptr, *hcf_base = nullptr, *gate_w = nullptr, *gate_bias = nullptr,
          *eg_w = nullptr;
    uint16_t* i_wproj = nullptr;
    uint8_t* win = nullptr;                     // window ring, g.win rows of kKvFp8
    float *c_kv = nullptr, *c_score = nullptr;  // compressor partial group (kv_src && ratio > 1)
};
struct SourceBuf { uint8_t* ckv = nullptr; uint8_t* ik = nullptr; bool mirror = false; };

// Window scratch (Dev, ds4_model.cu:568-616) for T == 1.
struct Scratch {
    float *h, *h2, *x, *qr, *q, *kvr, *ckr, *cgr, *latent, *ikr, *iq, *iw, *scores, *bm, *o, *oa, *ao, *mix, *a_pre,
        *a_post, *a_comb, *f_pre, *f_post, *f_comb, *pre, *glog, *rw, *moe_out, *sg, *su, *sh, *sh_out, *cpu_part,
        *eg_x, *eg_kv, *freqs_win, *freqs_comp, *route_scratch;
    int *ridx, *topk, *cand, *cpu_flag;
    uint16_t *emb_rows, *ext;
    uint8_t* eg_rows;
};

// ExpertBackend = row2 compute + this driver's publication into moe_out (route_port::publish_response).
class Publishing final : public rp::ExpertBackend {
    sycl::queue& q_; r2::ExpertCompute& c_; Scratch& s_;
    DecodeTiming& timing_;
public:
    Publishing(sycl::queue& q, r2::ExpertCompute& c, Scratch& s,DecodeTiming& timing) : q_(q), c_(c), s_(s),timing_(timing) {}
    bool resident(size_t l, int e) const override { return c_.resident(l, e); }
    bool store_contains(size_t l, int e) const override { return c_.store_contains(l, e); }
    void stage(size_t l, int e) override { c_.stage(l, e); }
    std::vector<float> device(size_t l, int e, std::span<const float> x) override { return c_.device(l, e, x); }
    std::vector<float> cpu(size_t l, int e, std::span<const float> x) override { return c_.cpu(l, e, x); }
    void publish(const rp::Resp& r) override {
        timing_.host(timing_.publication_wall_ns,[&]{rp::publish_response(q_, r, s_.cpu_flag, s_.cpu_part, s_.moe_out);});
    }
    void drain() noexcept override { c_.drain(); try { q_.wait(); } catch (...) {} }
};

class SyclStage final : public stage::Backend {
    Config cfg_;
    std::unique_ptr<WeightSource> ws_;
    ModelMeta g_;
    Topology topo_;
    sycl::queue q_;
    Arena arena_;
    Range r_{};
    stage::Plan plan_;
    BoundaryLayout bl_;
    std::vector<LayerW> L_;
    std::vector<SourceBuf> src_;
    Scratch s_{};
    std::unique_ptr<r2::ExpertCompute> experts_;
    WarmResidency warm_residency_;
    std::unique_ptr<Publishing> pub_;
    std::unique_ptr<rp::ExpertService> service_;
    StepPlanner planner_;
    SubDump dump_;
    uint64_t request_ = 0, epoch_ = 0, seq_ = 0, executed_step_ = UINT64_MAX, prepared_ = UINT64_MAX;
    uint64_t dense_bytes_ = 0;
    bool valid_ = false, imported_ = false;
    bool loaded_ = false;  // weights/state for r_ fully initialized; only then may begin() reuse them
    Fault fault_;
    std::string report_ = "strata-dsv41-sycl-stage";
    DecodeTiming timing_;
#ifdef STRATA_FORK_WARM
    std::unique_ptr<WarmDeviceTiming<SyclDecodeClock>> warm_dense_;
    DenseTrace<SyclDecodeClock> dense_trace_;
#endif
    DecodeEvents<SyclDecodeClock> events_{timing_};
    void timing_begin(const stage::Token& t) {
        timing_.begin(cfg_.decode_timing_start,t.request,t.position,t.id);
        experts_->begin_decode_step(timing_.active);
    }
    void sync(){timing_.wait([&]{q_.wait_and_throw();});}

    void dump(const char* n, uint32_t il, uint64_t p, int T, const void* dev, size_t bytes) {
        if (!dump_.on(il, n)) return;
        std::vector<uint8_t> v(bytes);
        q_.memcpy(v.data(), dev, bytes).wait_and_throw();
        dump_.write(n, il, p, T, v.data(), bytes);
    }
    // Ridx/rw in upstream's kPicks-strided, zero-padded layout (dalloc zeroed; route writes k slots).
    void dump_route(uint32_t il, uint64_t p, const rp::Req& req) {
        if (dump_.on(il, "ridx")) {
            std::vector<int32_t> v(kPicks, 0);
            for (int i = 0; i < g_.k_exp; ++i) v[size_t(i)] = req.route.ids[size_t(i)];
            dump_.write("ridx", il, p, 1, v.data(), v.size() * 4);
        }
        if (dump_.on(il, "rw")) {
            std::vector<float> v(kPicks, 0.0f);
            for (int i = 0; i < g_.k_exp; ++i) v[size_t(i)] = req.route.weights[size_t(i)];
            dump_.write("rw", il, p, 1, v.data(), v.size() * 4);
        }
    }
    r2::DenseMat mat(const std::string& n) {
        auto m = r2::upload(q_, *ws_, n);
        dense_bytes_ += uint64_t(m.n) * uint64_t(m.ld);  // row2::upload allocates exactly n*ld (= tensor bytes)
        return m;
    }
    float* f32(const std::string& n, size_t want = 0) { return arena_.upload(read_as_f32(*ws_, n, want)); }

    void load_layer(LayerW& l) {
        const auto& g = g_;
        const std::string P = layer_prefix(l.role.il), A = P + "attn.";
        l.wq_a = mat(A + "wq_a"); l.wq_b = mat(A + "wq_b"); l.wkv = mat(A + "wkv"); l.wo_b = mat(A + "wo_b");
        l.q_norm = f32(A + "q_norm.weight", size_t(g.q_lora)); l.kv_norm = f32(A + "kv_norm.weight", size_t(g.hd));
        l.sink = f32(A + "attn_sink", size_t(g.heads));
        l.attn_norm = f32(P + "attn_norm.weight", size_t(g.dim)); l.ffn_norm = f32(P + "ffn_norm.weight", size_t(g.dim));
        l.hca_fn = f32(P + "hc_attn_fn"); l.hca_scale = f32(P + "hc_attn_scale"); l.hca_base = f32(P + "hc_attn_base");
        l.hcf_fn = f32(P + "hc_ffn_fn"); l.hcf_scale = f32(P + "hc_ffn_scale"); l.hcf_base = f32(P + "hc_ffn_base");
        l.gate_w = f32(P + "ffn.gate.weight", size_t(g.n_exp) * size_t(g.dim));
        l.gate_bias = f32(P + "ffn.gate.bias", size_t(g.n_exp));
        l.s1 = mat(P + "ffn.shared_experts.w1"); l.s2 = mat(P + "ffn.shared_experts.w2"); l.s3 = mat(P + "ffn.shared_experts.w3");
        for (int gi = 0; gi < g.o_groups; ++gi) l.wo_a[gi] = mat(A + "wo_a.slice." + std::to_string(gi));
        if (l.role.kv_src) {
            const std::string C = A + "compressor.";
            l.c_wkv = mat(C + "wkv"); l.c_norm = f32(C + "norm.weight");
            if (l.role.ratio > 1) {
                l.c_wgate = mat(C + "wgate");
                l.c_kv = arena_.dev<float>(size_t(l.role.ratio) * size_t(g.hd));
                l.c_score = arena_.dev<float>(size_t(l.role.ratio) * size_t(g.hd));
            }
        }
        if (l.role.ix_src) {
            const std::string I = A + "indexer.";
            l.i_wq_b = mat(I + "wq_b");
            l.i_wproj = arena_.upload(read_as_f16(*ws_, I + "weights_proj.weight", f16_from_f32_rne));
            if (l.role.owns_k) { l.i_wk = mat(I + "wk"); l.ik_norm = f32(I + "k_norm.weight"); }
        }
        if (l.role.eg >= 0) {
            l.eg_wkv = mat(P + "engram.wkv");
            if (ws_->find(P + "engram.gate.weight")) l.eg_w = f32(P + "engram.gate.weight", size_t(g.hc) * size_t(g.dim));
            else {  // q_weight * k_weight folded on the host (ds4_model.cu:4352-4362)
                auto a = read_as_f32(*ws_, P + "engram.q_weight"), b = read_as_f32(*ws_, P + "engram.k_weight");
                require(a.size() == b.size() && a.size() == size_t(g.hc) * size_t(g.dim), "Engram q/k weight shape");
                for (size_t i = 0; i < a.size(); ++i) a[i] *= b[i];
                l.eg_w = arena_.upload(a);
            }
        }
        l.win = arena_.dev<uint8_t>(size_t(g.win) * g.win_rb());
    }
    void alloc_scratch() {
        const auto& g = g_;
        const size_t D = size_t(g.dim), H = size_t(g.hc), cap = size_t(cfg_.max_context) + 64;
        auto& s = s_;
        auto F = [&](size_t n) { return arena_.dev<float>(n); };
        s.h = F(H * D); s.h2 = F(H * D); s.x = F(D); s.qr = F(size_t(g.q_lora)); s.q = F(size_t(g.heads) * size_t(g.hd));
        s.kvr = F(size_t(g.hd)); s.ckr = F(size_t(g.hd)); s.cgr = F(size_t(g.hd)); s.latent = F(size_t(g.hd));
        s.ikr = F(size_t(g.ix_hd)); s.iq = F(size_t(g.ix_heads) * size_t(g.ix_hd)); s.iw = F(kIwStride);
        s.scores = F(cap); s.bm = F(g.bm_stride(cfg_.max_context)); s.o = F(size_t(g.heads) * size_t(g.hd));
        s.oa = F(size_t(g.o_groups) * size_t(g.o_lora)); s.ao = F(D); s.mix = F(kMixStride);
        s.a_pre = F(kPreStride); s.a_post = F(kPreStride); s.a_comb = F(kCombStride);
        s.f_pre = F(kPreStride); s.f_post = F(kPreStride); s.f_comb = F(kCombStride); s.pre = F(kPreStride);
        s.glog = F(kGlogStride); s.rw = F(kPicks); s.moe_out = F(D); s.sg = F(size_t(g.ff)); s.su = F(size_t(g.ff));
        s.sh = F(size_t(g.ff)); s.sh_out = F(D); s.cpu_part = F(D);
        s.route_scratch = F(2 * size_t(g.n_exp));
        s.ridx = arena_.dev<int>(kPicks); s.cpu_flag = arena_.dev<int>(1);
        s.topk = arena_.dev<int>(g.topk_stride()); s.cand = arena_.dev<int>(g.cand_stride());
        s.emb_rows = arena_.dev<uint16_t>(D); s.ext = arena_.dev<uint16_t>((size_t(g.win) + kWin) * size_t(g.hd));
        if (!g.eg_layers.empty()) {
            const size_t cols = size_t(g.eg_cols()), hd = size_t(g.eg_hd);
            s.eg_rows = arena_.dev<uint8_t>(cols * (hd + hd / 32)); s.eg_x = F(cols * hd); s.eg_kv = F((H + 1) * D);
        }
        s.freqs_win = arena_.upload(rope_freqs(g.rd, 0, g.rope_theta, g.factor, g.beta_fast, g.beta_slow));
        s.freqs_comp = arena_.upload(rope_freqs(g.rd, g.orig_ctx, g.comp_theta, g.factor, g.beta_fast, g.beta_slow));
    }
    // Owned sources are produced here; sources produced earlier and read by an owned layer are mirrors that
    // import_boundary fills. Full capacity, zeroed so uncommitted snapshot bytes are deterministic.
    void alloc_sources() {
        src_.assign(topo_.sources.size(), {});
        for (size_t si = 0; si < topo_.sources.size(); ++si) {
            const auto& s = topo_.sources[si];
            bool read = false, read_ik = false;
            for (const auto& l : topo_.layers)
                if (r_.owns(l.il) && l.src == int(si)) { read = true; read_ik = read_ik || l.ix_src; }
            const bool own = r_.owns(s.layer);
            if (!own && !read) continue;
            src_[si].mirror = !own;
            src_[si].ckv = arena_.dev<uint8_t>(size_t(s.cap) * g_.ckv_rb());
            if (topo_.layers[s.layer].ix_src || read_ik) src_[si].ik = arena_.dev<uint8_t>(size_t(s.cap) * g_.ik_rb());
        }
    }
    uint64_t edge_extent(const stage::Edge& e) const {  // bytes this stage allocated for the edge (0: none)
        switch (e.kind) {
        case stage::Kind::hc_pre: return kPreStride * 4;
        case stage::Kind::selected_topk: return g_.topk_stride() * 4;
        case stage::Kind::candidates: return g_.cand_stride() * 4;
        case stage::Kind::compressed_kv:
        case stage::Kind::index_rows:
            for (size_t si = 0; si < topo_.sources.size(); ++si)
                if (topo_.sources[si].layer == e.producer) {
                    const bool kv = e.kind == stage::Kind::compressed_kv;
                    if (!(kv ? src_[si].ckv : src_[si].ik)) return 0;
                    return topo_.sources[si].cap * (kv ? g_.ckv_rb() : g_.ik_rb());
                }
            return 0;
        default: return 0;
        }
    }
    void* edge_buffer(const stage::Edge& e) {
        require(edge_extent(e) == e.bytes, "boundary edge extent differs from this stage's allocation");
        switch (e.kind) {
        case stage::Kind::hc_pre: return s_.pre;
        case stage::Kind::selected_topk: return s_.topk;
        case stage::Kind::candidates: return s_.cand;
        case stage::Kind::compressed_kv:
        case stage::Kind::index_rows:
            for (size_t si = 0; si < topo_.sources.size(); ++si)
                if (topo_.sources[si].layer == e.producer)
                    return e.kind == stage::Kind::compressed_kv ? (void*) src_[si].ckv : (void*) src_[si].ik;
            break;
        default: break;
        }
        throw std::runtime_error("unsupported boundary edge kind in dsv41 wiring");
    }
    // Single teardown path: destruction, range replacement and failed begin() (review #5/#6).
    void teardown() noexcept {
        loaded_ = false; valid_ = false;
        try { pub_.reset(); } catch (...) {}
        try { service_.reset(); } catch (...) {}
        try { if (experts_) experts_->drain(); experts_.reset(); } catch (...) {}
        try { q_.wait(); } catch (...) {}
        for (auto& l : L_) {
            for (auto* m : {&l.wq_a, &l.wq_b, &l.wkv, &l.wo_b, &l.s1, &l.s2, &l.s3, &l.c_wkv, &l.c_wgate, &l.i_wq_b,
                            &l.i_wk, &l.eg_wkv}) r2::release(q_, *m);
            for (auto& m : l.wo_a) r2::release(q_, m);
        }
        L_.clear(); src_.clear(); s_ = {}; arena_.clear(); dense_bytes_ = 0;
        report_ = "strata-dsv41-sycl-stage";
    }
    uint64_t device_bytes() const {
        uint64_t n = arena_.bytes + dense_bytes_;
        if (experts_) for (uint32_t il = r_.first; il < std::max(r_.end,r_.first+1); ++il) n += experts_->device_bytes(il);
        return n;
    }
    void load_range(uint32_t first, uint32_t end) {
#ifdef STRATA_FORK_WARM
        if(first==end){
            if(cfg_.warm_experts)experts_=make_warm_sycl_experts(q_,*ws_,first,end,cfg_,warm_residency_);
            return; // expert-only B65: no dense scratch, state, CPU lane or publisher
        }
#endif
        alloc_scratch();
        for (uint32_t il = first; il < end; ++il) {
            L_.push_back({});  // pushed before loading so a partial layer is released by teardown()
            L_.back().role = topo_.layers[il];
            load_layer(L_.back());
            fault_.at("load_weights");
        }
        alloc_sources();
        fault_.at("cpu_lane");
#ifdef STRATA_FORK_WARM
        if(cfg_.warm_experts)experts_=make_warm_sycl_experts(q_,*ws_,first,end,cfg_,warm_residency_);
        else
#endif
        experts_ = r2::make_experts(q_, *ws_, first, end, cfg_.expert_placement, cfg_.cpu_threads, cfg_.cpu_dispatch, cfg_.cpu_timing);
        pub_ = std::make_unique<Publishing>(q_, *experts_, s_,timing_);
        service_ = std::make_unique<rp::ExpertService>(size_t(g_.n_exp), size_t(g_.k_exp));
    }
    void mv(std::initializer_list<r2::MvJob> j) { r2::mv(q_, std::span<const r2::MvJob>(j.begin(), j.size())); }
    static r2::MvJob job(const r2::DenseMat& m, const float* x, float* y) { r2::MvJob j; j.m = &m; j.x = x; j.y = y; return j; }

    template<class F>auto dense_launch(const char* name,F fn){
#ifdef STRATA_FORK_WARM
        return dense_trace_.run(name,q_,fn);
#else
        return fn();
#endif
    }
    void layer(LayerW& l, const stage::Token& t) {
#ifdef STRATA_FORK_WARM
        dense_trace_.begin(cfg_.warm_kernel_timing,t.request,l.role.il,t.position);
        if(cfg_.warm_measure){if(!warm_dense_)warm_dense_=std::make_unique<WarmDeviceTiming<SyclDecodeClock>>(size_t(g_.nl));warm_dense_->begin(q_);}
#endif
        events_.begin(DecodePart::other,q_);
        const auto& g = g_; const auto& role = l.role; const uint32_t il = role.il; const uint64_t p = t.position;
        const int D = g.dim, H = g.hc, HD = g.hd, NH = g.heads, HQ = NH * HD, OG = g.o_groups;
        auto& s = s_;
        const float* freqs = role.ratio ? s.freqs_comp : s.freqs_win;
        // ---- Engram (row5): host rows -> dequant -> eg_wkv (row2) -> gate
        if (role.eg >= 0) {
            const int cols = g.eg_cols(), ehd = g.eg_hd;
            const size_t per = size_t(cols) * size_t(ehd + ehd / 32);
            std::vector<std::byte> host(per);
            timing_.host(timing_.engram_read_ns,[&]{ws_->engram_rows(role.eg, engram_slice(plan_, t, il), host);});
            q_.memcpy(s.eg_rows, host.data(), per).wait_and_throw();
            const uint8_t* sc = s.eg_rows + size_t(cols) * size_t(ehd);
            eg::dequant(q_, s.eg_rows, sc, cols, ehd, s.eg_x, ws_->engram_format(role.eg) == 3 ? 0 : ws_->engram_format(role.eg));
            dump("eg_rows", il, p, 1, s.eg_rows, size_t(cols) * size_t(ehd));
            dump("eg_x", il, p, 1, s.eg_x, size_t(cols) * size_t(ehd) * 4);
            { r2::MvJob j = job(l.eg_wkv, s.eg_x, s.eg_kv); dense_launch("mv",[&]{return mv({j});}); }
            dump("eg_kv", il, p, 1, s.eg_kv, size_t(H + 1) * size_t(D) * 4);
            dump("eg_h0", il, p, 1, s.h, size_t(H) * size_t(D) * 4);
            dump("eg_w", il, p, 1, l.eg_w, size_t(H) * size_t(D) * 4);
            eg::gate(q_, s.h, s.eg_kv, l.eg_w, H, D, 1, g.eps);
            dump("eg_h", il, p, 1, s.h, size_t(H) * size_t(D) * 4);
        }
        events_.end(q_);
        events_.begin(DecodePart::attention_shared,q_);
        dump("h_in", il, p, 1, s.h, size_t(H) * size_t(D) * 4);
        dump("ring_in", il, p, 1, l.win, size_t(g.win) * g.win_rb());
        // ---- attention mHC (row3); pre = previous layer's f_pre (lagged pre-mix)
        dense_launch("sp::hc_mix_split",[&]{return sp::hc_mix_split(q_, s.h, l.hca_fn, H * D, (2 + H) * H, 1, g.eps, s.mix, l.hca_scale, l.hca_base, H, g.sinkhorn,
                         g.hc_eps, s.a_pre, s.a_post, s.a_comb);});
        dump("a_mix", il, p, 1, s.mix, kMixStride * 4); dump("a_pre", il, p, 1, s.a_pre, kPreStride * 4);
        dump("a_post", il, p, 1, s.a_post, kPreStride * 4); dump("a_comb", il, p, 1, s.a_comb, kCombStride * 4);
        dense_launch("sp::hc_pre_norm",[&]{return sp::hc_pre_norm(q_, s.h, s.pre, l.attn_norm, D, H, 1, g.eps, s.x);});
        dump("pre_in", il, p, 1, s.pre, kPreStride * 4); dump("attn_x", il, p, 1, s.x, size_t(D) * 4);
        // ---- projections (row2)
        {
            std::vector<r2::MvJob> j{job(l.wq_a, s.x, s.qr), job(l.wkv, s.x, s.kvr)};
            if (role.kv_src) j.push_back(job(l.c_wkv, s.x, s.ckr));
            if (role.kv_src && role.ratio > 1) j.push_back(job(l.c_wgate, s.x, s.cgr));
            dense_launch("r2::mv",[&]{return r2::mv(q_, j);});
        }
        dump("qr_raw", il, p, 1, s.qr, size_t(g.q_lora) * 4); dump("kvr_raw", il, p, 1, s.kvr, size_t(HD) * 4);
        { sp::Finish f; f.w = l.q_norm; f.eps = g.eps; dense_launch("sp::finish",[&]{return sp::finish(q_, s.qr, 1, g.q_lora, g.q_lora, f);}); }
        dump("qr", il, p, 1, s.qr, size_t(g.q_lora) * 4);
        { r2::MvJob j = job(l.wq_b, s.qr, s.q); dense_launch("mv",[&]{return mv({j});}); }
        { sp::Finish f; f.rd = g.rd; f.pos = int(p); f.vpp = NH; f.freqs = freqs; dense_launch("sp::finish",[&]{return sp::finish(q_, s.q, NH, HD, HD, f);}); }
        dump("q", il, p, 1, s.q, size_t(HQ) * 4);
        // ---- window ring (numerics): ext = ring rows before p, then this token's FP8-rounded BF16 row
        dense_launch("sp::ring_to_ext",[&]{return sp::ring_to_ext(q_, l.win, s.ext, int64_t(p), g.win, HD);});
        {
            sp::Finish f; f.w = l.kv_norm; f.eps = g.eps; f.rd = g.rd; f.pos = int(p); f.freqs = freqs; f.quant = 1;
            f.dst = s.ext + size_t(g.win) * size_t(HD); f.dst_stride = HD;
            dense_launch("sp::finish",[&]{return sp::finish(q_, s.kvr, 1, HD, HD, f);});
        }
        dense_launch("sp::ext_to_ring",[&]{return sp::ext_to_ring(q_, s.ext, l.win, int64_t(p), g.win, 1, HD);});
        dump("ext", il, p, 1, s.ext, (size_t(g.win) + 1) * size_t(HD) * 2);
        sp::AttnRows ar{};
        {
            const int first = int(std::max<int64_t>(int64_t(g.win) - int64_t(p), 1));
            ar.win0[0] = first; ar.nwin[0] = g.win - first + 1; ar.nsel[0] = 0;
        }
        const void* ckv = nullptr;
        // ---- compressor / indexer / selection (row4)
        if (role.ratio > 0) {
            auto& sb = src_[size_t(role.src)];
            const int R = role.ratio;
            const int64_t g_first = int64_t(p) / R, g_end = int64_t(p + 1) / R;
            const int ng = int(std::max<int64_t>(0, g_end - g_first));
            if (role.kv_src) {
                if (R == 1) {
                    sp::Finish f; f.w = l.c_norm; f.eps = g.eps; dense_launch("sp::finish",[&]{return sp::finish(q_, s.ckr, 1, HD, HD, f);});
                    q_.memcpy(s.latent, s.ckr, size_t(HD) * 4);
                } else {
                    dense_launch("sp::compress_win",[&]{return sp::compress_win(q_, s.ckr, s.cgr, l.c_kv, l.c_score, R, int(p), 1, HD, l.c_norm, g.eps, s.latent);});
                    dump("ckr", il, p, 1, s.ckr, size_t(HD) * 4); dump("cgr", il, p, 1, s.cgr, size_t(HD) * 4);
                }
            }
            sp::Rows lim{};
            lim.n[0] = int((p + 1) / uint64_t(R)); lim.k[0] = std::min(g.ix_topk, lim.n[0]);
            const int clen = lim.n[0];
            if (role.ix_src) {
                auto& ks = src_[size_t(role.ik_src)];
                if (role.owns_k && ng > 0) {
                    { r2::MvJob j = job(l.i_wk, s.latent, s.ikr); j.ncol = ng; dense_launch("mv",[&]{return mv({j});}); }
                    sp::Finish f; f.w = l.ik_norm; f.eps = g.eps; f.rd = g.rd; f.pos = int(g_first * R); f.pos_step = R;
                    f.freqs = freqs; f.quant = 3; f.pdst = ks.ik + size_t(g_first) * g.ik_rb(); f.pdst_stride = (long long) g.ik_rb();
                    dense_launch("sp::finish",[&]{return sp::finish(q_, s.ikr, ng, g.ix_hd, g.ix_hd, f);});
                }
                if (clen > 0) {
                    { r2::MvJob j = job(l.i_wq_b, s.qr, s.iq); dense_launch("mv",[&]{return mv({j});}); }
                    sp::Finish f; f.rd = g.rd; f.pos = int(p); f.vpp = g.ix_heads; f.freqs = freqs; f.quant = 3;
                    dense_launch("sp::finish",[&]{return sp::finish(q_, s.iq, g.ix_heads, g.ix_hd, g.ix_hd, f);});
                    dense_launch("r2::gemv_f16_rows",[&]{return r2::gemv_f16_rows(q_, l.i_wproj, s.x, size_t(D), g.ix_heads, D, 1,
                                      1.0f / std::sqrt(float(g.ix_hd)) / std::sqrt(float(g.ix_heads)), s.iw, kIwStride);});
                    const long long ss = (long long) cfg_.max_context + 64;
                    const bool restrict_ = (lim.n[0] + g.cand_blk - 1) / g.cand_blk > g.cand_topk;
                    if (role.uses_cand && restrict_) {
                        sp::Rows cl{}; cl.n[0] = lim.n[0]; cl.k[0] = planner_.cand[il];
                        const int nmax = planner_.cand[il] * g.cand_blk;
                        sp::index_score(q_, s.iq, s.iw, ks.ik, g.ix_heads, g.ix_hd, 1, nmax, s.cand, int(g.cand_stride()),
                                        g.cand_blk, cl, s.scores, ss);
                        sp::Rows tr{}; tr.n[0] = planner_.cand[il] * g.cand_blk; tr.k[0] = lim.k[0];
                        dense_launch("sp::topk",[&]{return sp::topk(q_, s.scores, ss, 1, tr, s.cand, int(g.cand_stride()), g.cand_blk, s.topk, int(g.topk_stride()));});
                    } else {
                        sp::index_score(q_, s.iq, s.iw, ks.ik, g.ix_heads, g.ix_hd, 1, clen, nullptr, 0, 1, lim, s.scores, ss);
                        if (role.cand_src) {
                            sp::Rows cr{}; cr.n[0] = lim.n[0]; cr.k[0] = g.cand_topk;
                            sp::cand_blocks(q_, s.scores, ss, 1, cr, g.cand_blk, s.bm, (long long) g.bm_stride(cfg_.max_context),
                                            s.cand, int(g.cand_stride()));
                        }
                        dense_launch("sp::topk",[&]{return sp::topk(q_, s.scores, ss, 1, lim, nullptr, 0, 1, s.topk, int(g.topk_stride()));});
                    }
                }
            }
            if (role.kv_src && ng > 0) {
                sp::Finish f; f.rd = g.rd; f.pos = int(g_first * R); f.pos_step = R; f.freqs = freqs; f.quant = 2;
                f.pdst = sb.ckv + size_t(g_first) * g.ckv_rb(); f.pdst_stride = (long long) g.ckv_rb();
                dense_launch("sp::finish",[&]{return sp::finish(q_, s.latent, ng, HD, HD, f);});
            }
            if (ng > 0) dump("latent", il, p, ng, s.latent, size_t(ng) * size_t(HD) * 4);
            dump("topk", il, p, 1, s.topk, g.topk_stride() * 4);
            ar.nsel[0] = lim.n[0] > 0 ? planner_.topk[il] : 0;
            ckv = sb.ckv;
        }
        // ---- sparse attention + inverse rope + FP8 (row4/numerics), output projection (row2)
        const auto attention_event=dense_launch("sp::sparse_attn",[&]{return (cfg_.warm_sycl_attention=="subgroup32-v1"?sp::sparse_attn_subgroup32:sp::sparse_attn)(
                        q_, s.q, s.ext, ckv, s.topk, int(g.topk_stride()), ar, l.sink, NH, HD, 1,
                        1.0f / std::sqrt(float(HD)), s.o, kKvFp4);});
        dump("o", il, p, 1, s.o, size_t(HQ) * 4);
        { sp::Finish f; f.rd = g.rd; f.pos = int(p); f.vpp = NH; f.freqs = freqs; f.inverse = 1; f.quant = 1;
          dense_launch("sp::finish",[&]{return sp::finish(q_, s.o, NH, HD, HD, f);}); }
        dump("o_fin", il, p, 1, s.o, size_t(HQ) * 4);
        {
            const int per = HQ / OG;
            std::vector<r2::MvJob> j;
            for (int gi = 0; gi < OG; ++gi) j.push_back(job(l.wo_a[gi], s.o + size_t(gi) * size_t(per), s.oa + size_t(gi) * size_t(g.o_lora)));
            dense_launch("r2::mv",[&]{return r2::mv(q_, j);});
            r2::MvJob jb = job(l.wo_b, s.oa, s.ao); dense_launch("mv",[&]{return mv({jb});});
        }
        dump("ao", il, p, 1, s.ao, size_t(D) * 4);
        dense_launch("sp::hc_post",[&]{return sp::hc_post(q_, s.ao, s.h, s.a_post, s.a_comb, D, H, 1, s.h2);});
        std::swap(s.h, s.h2);
        dump("h_attn", il, p, 1, s.h, size_t(H) * size_t(D) * 4);
        // ---- FFN mHC (row3); the FFN input uses a_pre (upstream's lagged pre-mix, ds4_model.cu:10306)
        dense_launch("sp::hc_mix_split",[&]{return sp::hc_mix_split(q_, s.h, l.hcf_fn, H * D, (2 + H) * H, 1, g.eps, s.mix, l.hcf_scale, l.hcf_base, H, g.sinkhorn,
                         g.hc_eps, s.f_pre, s.f_post, s.f_comb);});
        dense_launch("sp::hc_pre_norm",[&]{return sp::hc_pre_norm(q_, s.h, s.a_pre, l.ffn_norm, D, H, 1, g.eps, s.x);});
        dump("f_pre", il, p, 1, s.f_pre, kPreStride * 4); dump("f_post", il, p, 1, s.f_post, kPreStride * 4);
        dump("f_comb", il, p, 1, s.f_comb, kCombStride * 4); dump("ffn_x", il, p, 1, s.x, size_t(D) * 4);
        // ---- router (row2 gemv_f32 + row6 route), shared expert (row2), routed experts (row6 service + row2)
        dense_launch("r2::gemv_f32",[&]{return r2::gemv_f32(q_, l.gate_w, s.x, g.n_exp, D, 1, s.glog, kGlogStride);});
        rp::route_device(q_, s.glog, kGlogStride, l.gate_bias, size_t(g.n_exp), size_t(g.k_exp), 1, g.route_scale,
                         s.ridx, s.rw, s.route_scratch);
        { std::vector<r2::MvJob> j{job(l.s1, s.x, s.sg), job(l.s3, s.x, s.su)}; dense_launch("r2::mv",[&]{return r2::mv(q_, j);}); }
        dense_launch("r2::swiglu",[&]{return r2::swiglu(q_, s.sg, s.su, g.ff, g.lim, s.sh);});
        { r2::MvJob j = job(l.s2, s.sh, s.sh_out); dense_launch("mv",[&]{return mv({j});}); }
        events_.end(q_);
#ifdef STRATA_FORK_WARM
        if(cfg_.warm_measure)warm_dense_->end(il,q_);
        dense_trace_.collect("sycl");
        // Dense end has already waited. Read the kernel's own timestamps at a
        // bounded cadence without adding markers or a synchronization.
        if(cfg_.warm_measure && (p==31 || p==127 || p==324 || p%1024==0)){
            const auto start=attention_event.get_profiling_info<sycl::info::event_profiling::command_start>();
            const auto end=attention_event.get_profiling_info<sycl::info::event_profiling::command_end>();
            require(end>=start,"invalid warm attention kernel timestamps");
            std::fprintf(stderr,"AIOPT_WARM_ATTN layer=%u position=%llu path=%s kernel_ns=%llu\n",il,(unsigned long long)p,cfg_.warm_sycl_attention.c_str(),(unsigned long long)(end-start));
        }
#endif
        const auto route_tick=timing_.stamp();
        auto req = rp::read_request(q_, request_, epoch_, ++seq_, il, size_t(D), size_t(g.n_exp), size_t(g.k_exp), 1,
                                    s.ridx, s.rw, s.x, s.sh_out);
        timing_.route_io_wall_ns+=timing_.elapsed(route_tick);
        req.combine_path = rp::CombinePath::synchronous;  // the reference runs DS4_SYNC=1 (README, parity gate)
        dump("glog", il, p, 1, s.glog, kGlogStride * 4);
        dump_route(il, p, req);
        dump("sh_out", il, p, 1, s.sh_out, size_t(D) * 4);
        const auto cpu_before=timing_.active?experts_->decode_phases().expert_ns:0;
        const auto publish_before=timing_.publication_wall_ns;
        const auto service_tick=timing_.stamp();
        rp::Resp resp;
#ifdef STRATA_FORK_WARM
        if(cfg_.warm_experts){
            auto* host=experts_->host_lane();require(host!=nullptr,"warm prefix needs CPU lane");
            resp.request=req.request;resp.epoch=req.epoch;resp.seq=req.seq;resp.cpu={0};resp.part.assign(size_t(D),0);
            resp.combined=warm_execute(*host,il,req.route.ids,req.route.weights,req.x,req.shared);pub_->publish(resp);
        }else
#endif
        resp = service_->execute(req, *pub_);
        if(timing_.active&&!cfg_.warm_experts){
            const auto cpu=experts_->decode_phases().expert_ns-cpu_before;
            const auto publish=timing_.publication_wall_ns-publish_before;
            const auto wall=timing_.elapsed(service_tick);
            require(cpu+publish<=wall,"SYCL service timer partition");
            timing_.routing_combine_host_ns+=wall-cpu-publish;
        }
        events_.begin(DecodePart::other,q_);
        if (dump_.on(il, "cpu_part") && resp.cpu[0]) dump_.write("cpu_part", il, p, 1, resp.part.data(), resp.part.size() * 4);
        dump("moe_out", il, p, 1, s.moe_out, size_t(D) * 4);
        dense_launch("sp::hc_post",[&]{return sp::hc_post(q_, s.moe_out, s.h, s.f_post, s.f_comb, D, H, 1, s.h2);});
        std::swap(s.h, s.h2);
        dump("h_out", il, p, 1, s.h, size_t(H) * size_t(D) * 4);
        q_.memcpy(s.pre, s.f_pre, kPreStride * 4);
        events_.end(q_);
    }

public:
    SyclStage(uint32_t device, Config c)
        : cfg_(std::move(c)), ws_(open_pack(cfg_.pack, {cfg_.q5k_fma, cfg_.cpu_timing})), g_(ws_->meta()),
          topo_(topology(g_, cfg_.max_context)),
          q_(strata::intel_level_zero_device(device),
             [](sycl::exception_list errors) { for (const auto& e : errors) std::rethrow_exception(e); },
             (cfg_.decode_timing_start>=0 || cfg_.warm_measure)
                 ? sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}}
                 : sycl::property_list{sycl::property::queue::in_order{}}),
          arena_(q_) {
        fault_.where = cfg_.fault_inject;
        dump_.open(cfg_);
    }
    ~SyclStage() override { invalidate(); teardown(); }
    // describe(): name, range, measured device bytes (arena + row-2 dense matrices + resident experts) and
    // the Plan footprint it was admitted against.
    const char* name() const override { return report_.c_str(); }
#ifdef STRATA_FORK_WARM
    std::vector<double> warm_dense_costs()const override{return warm_dense_?warm_dense_->means():std::vector<double>{};}
#endif
    void begin(const stage::Plan& plan, uint32_t first, uint32_t end, uint64_t request) override {
        valid_ = false;
        timing_.active=false;events_.abandon();
        drain();
        plan.validate();
        require(plan.layers == uint32_t(g_.nl) && plan.hc == uint32_t(g_.hc) && plan.dim == uint32_t(g_.dim),
                "plan geometry differs from the pack");
        auto hip_residency=check_hip_residency_plan(*ws_, cfg_, plan,&warm_residency_);
#ifdef STRATA_FORK_WARM
        if(cfg_.warm_experts)warm_set_residency(warm_residency_.slots,hip_residency.slots);
#endif
        require(plan.model_sha == ws_->identity(),
                "plan model/schema/context identity mismatch");
        require((first < end || (first==0 && end==0 && plan.hip_dense_only())) && end <= plan.layers, "invalid stage range");
        require((first == 0 && end == plan.cut) || first == plan.cut, "stage range must be the plan's prefix or start at its cut");
        require(end < plan.layers, "the output head is HIP-only in this wiring");
        const bool same = loaded_ && r_.first == first && r_.end == end && plan_.model_sha == plan.model_sha;
        if (!same) {  // (re)load weights only when the range changes; otherwise reset request state below
            teardown();
            r_ = {first, end};
            try { load_range(first, end); loaded_ = true; }
            catch (...) { teardown(); throw; }
        }
        const BoundaryLayout bl = BoundaryLayout::of(plan);
        check_extents(bl, [&](const stage::Edge& e) { return edge_extent(e); });
        const stage::Budget& budget = first == 0 ? plan.sycl : plan.hip;
        const uint64_t measured = device_bytes(), admitted = plan.footprint(budget, first, end);
        report_ = "strata-dsv41-sycl-stage range=[" + std::to_string(first) + "," + std::to_string(end) +
                  ") device_bytes=" + std::to_string(measured) + " arena=" + std::to_string(arena_.bytes) +
                  " dense=" + std::to_string(dense_bytes_) + " plan_footprint=" + std::to_string(admitted) +
                  " cap=" + std::to_string(budget.cap);
        require(measured <= budget.cap, "measured device allocation exceeds the frozen stage cap: " + report_);
        plan_ = plan; bl_ = bl;
        // Request reset (Impl::reset, ds4_model.cu:5670-5689) plus zeroed sources/mirrors (deterministic snapshots).
        for (auto& l : L_) {
            q_.memset(l.win, 0, size_t(g_.win) * g_.win_rb());
            if (l.c_kv) q_.memset(l.c_kv, 0, size_t(l.role.ratio) * size_t(g_.hd) * 4);
            if (l.c_score) q_.memset(l.c_score, 0, size_t(l.role.ratio) * size_t(g_.hd) * 4);
        }
        for (size_t si = 0; si < src_.size(); ++si) {
            if (src_[si].ckv) q_.memset(src_[si].ckv, 0, size_t(topo_.sources[si].cap) * g_.ckv_rb());
            if (src_[si].ik) q_.memset(src_[si].ik, 0, size_t(topo_.sources[si].cap) * g_.ik_rb());
        }
        if(first!=end){
        q_.memset(s_.topk, 0, g_.topk_stride() * 4); q_.memset(s_.cand, 0, g_.cand_stride() * 4);
        sync();
        request_ = request; ++epoch_; seq_ = 0; executed_step_ = prepared_ = UINT64_MAX; imported_ = false;
        if(service_)service_->reset(request_, epoch_);
        }else{request_=request;executed_step_=prepared_=UINT64_MAX;imported_=false;}
        valid_ = true;
    }
    stage::HostMemory pinned(size_t n) override {
        void* p = sycl::malloc_host(n, q_);
        if (!p) throw std::bad_alloc();
        auto ctx = q_.get_context();
        return {p, n, [ctx](void* x) { try { sycl::free(x, ctx); } catch (...) {} }};
    }
    void execute(const stage::Token& t, const std::atomic_bool& cancel) override {
        require(valid_, "stage invalid; begin() required");
        require(t.request == request_ && t.step == t.position, "token request/position mismatch");
        // Upstream forward(): pos + T <= max_context (ds4_model.cu:9705), before ANY state change or submit.
        require(t.position < uint64_t(cfg_.max_context), "the context is full (position >= max_context)");
        require(t.id >= 0 && t.id < g_.vocab, "token out of range");
        if(r_.prefix())timing_begin(t);
        planner_.advance(g_, topo_, t.position);
        if (r_.prefix()) {  // embedding (numerics) + init_pre: [1,0,...] (ds4_kernels.cu init_pre)
            events_.begin(DecodePart::other,q_);
            std::vector<uint16_t> row(size_t(g_.dim));
            timing_.host(timing_.embedding_read_ns,[&]{ws_->embedding_bf16(t.id, row);});
            q_.memcpy(s_.emb_rows, row.data(), row.size() * 2);
            sp::embed(q_, s_.emb_rows, 1, s_.h, g_.dim, g_.hc);
            const float pre[kPreStride] = {1.0f};
            q_.memcpy(s_.pre, pre, sizeof(pre)).wait_and_throw();
            events_.end(q_);
        } else {
            require(imported_, "tail stage executed without an imported boundary");
        }
        for (auto& l : L_) {
            require(!cancel.load(std::memory_order_acquire), "request cancelled");
            layer(l, t);
        }
        sync();
        executed_step_ = t.step; imported_ = false;
    }
    std::unique_ptr<stage::Completion> export_boundary(const stage::Token& t, std::span<std::byte> host) override {
        require(valid_ && executed_step_ == t.step && host.size() == bl_.total, "export before execute / extent");
        events_.begin(DecodePart::boundary,q_);
        sync();
        q_.memcpy(host.data(), s_.h, size_t(bl_.residual));
        for (const auto& sl : bl_.slots) q_.memcpy(host.data() + sl.offset, edge_buffer(sl.edge), size_t(sl.edge.bytes));
        events_.end(q_);
        sync();
        struct Done final : stage::Completion { size_t n; explicit Done(size_t b) : n(b) {} void wait() override {}
                                                size_t bytes() const override { return n; } };
        return std::make_unique<Done>(host.size());
    }
    std::vector<uint64_t> boundary_committed_rows(const stage::Token& t) override { return bl_.rows(plan_, t.position); }
    std::unique_ptr<stage::Completion> import_boundary(const stage::Token& t, std::span<const std::byte> host,
                                                       const stage::Packet& pk) override {
        require(valid_ && !r_.prefix() && host.size() == bl_.total && pk.edges == plan_.crossing(), "import extent/schema");
        require(pk.committed_rows == bl_.rows(plan_, t.position), "imported valid rows differ from position");
        timing_begin(t);
        events_.begin(DecodePart::boundary,q_);
        q_.memcpy(s_.h, host.data(), size_t(bl_.residual));
        for (const auto& sl : bl_.slots) q_.memcpy(edge_buffer(sl.edge), host.data() + sl.offset, size_t(sl.edge.bytes));
        events_.end(q_);
        sync();
        imported_ = true;
        struct Done final : stage::Completion { size_t n; explicit Done(size_t b) : n(b) {} void wait() override {}
                                                size_t bytes() const override { return n; } };
        return std::make_unique<Done>(host.size());
    }
    std::vector<float> logits(const stage::Token&) override { throw std::runtime_error("SYCL stage has no head"); }
    void prepare_commit(uint64_t step) override {
        require(valid_ && executed_step_ == step, "prepare_commit for a step this stage did not execute");
        sync(); prepared_ = step;
        timing_.emit("sycl",r_.first,r_.end,experts_->decode_phases());
    }
    void commit(uint64_t step) override { require(valid_ && prepared_ == step, "commit without prepare"); }
    void invalidate() noexcept override { valid_ = false; if (service_) service_->cancel(); }
    void drain() noexcept override {
        if (experts_) experts_->drain();
        try { q_.wait(); } catch (...) {}
    }
};
}  // namespace

std::unique_ptr<stage::Backend> make_model_backend(uint32_t device, const Config& c) {
    return std::make_unique<SyclStage>(device, c);
}
}  // namespace strata::wiring
