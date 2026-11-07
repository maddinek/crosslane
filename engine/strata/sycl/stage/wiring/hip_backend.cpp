// MIT; crosslane contributors. HIP (R9700, gfx1201) layer driver: stage::Backend for layers
// [first,end) (+ final norm/head when end == n_layers) of DeepSeek V4.1, T == 1, synchronous host-routed
// MoE. It calls the UPSTREAM kernels at engine/strata/upstream.rev (src/dsv41/ds4_kernels.cuh, ds4_gguf.cuh).
// v0.2.10 mapping: docs/experiments/strata-v0210-port.md; original derivation/line references below are c62d5e3.
// in exactly Impl::layer's order (ds4_model.cu:9961-10571, DS4_SYNC=1 branch) so a HIP stage reproduces
// the AMD-only reference; the SYCL port is then compared against this. Compiled as LANGUAGE HIP with
// Strata's hip_compat/cuda_runtime.h force-included (cmake/hip_backend.cmake), like ds4_model.cu itself.
// Derived from Project Maya / Strata (MIT); preserve third_party/project-maya/{LICENSE,NOTICE.md}.
#ifdef STRATA_FORK_WARM
#include "../../../warm/hip_expert_issue.hpp"
#endif
#include "common.hpp"
#include "../../../warm/device_timing.hpp"
#include "../../../warm/dense_trace.hpp"
#include "../../../warm/engram_read_ahead.hpp"
#include "cpu_lane.hpp"
#include "host_plan.hpp"
#include "ds4_kernels.cuh"
#include "ds4_gguf.cuh"
#include "strata/dsv41/exl3.hpp"
#include "strata/kernels/f16_bits.hpp"
#include <hip/hip_runtime.h>
#include <atomic>
#include "hip_resident_bank.hpp"
#include "../../../warm/runtime.hpp"
#include "../../../warm/host_partition.hpp"
#ifdef STRATA_FORK_WARM
#include "strata/dsv41/sampler.hpp"
#endif
#include "hip_decode_events.hpp"
#include "hip_arena.hpp"
#ifdef STRATA_FORK_WARM
#include "../../../warm/hip_dense_graph.hpp"
#include "../../../warm/route_capture.hpp"
#include "../../../warm/hip_fuse.hpp"
#include "../../../warm/hip_dense_v2.hpp"
#include <array>
#endif

namespace strata::wiring {
namespace k4 = strata::ds4;          // upstream kernels (cudaStream_t == hipStream_t through hip_compat)
namespace gg = strata::ds4::gguf;
namespace x3 = strata::dsv41k::exl3;
namespace {
void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}
#define CK(x) ck((x), #x)

using Arena = HipArena;

struct LayerW {
    LayerRole role;
    x3::Mat wq_a, wq_b, wkv, wo_a[16], wo_b, s1, s2, s3, c_wkv, c_wgate, i_wq_b, i_wk, eg_wkv;
    float *q_norm = nullptr, *kv_norm = nullptr, *attn_norm = nullptr, *ffn_norm = nullptr, *c_norm = nullptr,
          *ik_norm = nullptr, *sink = nullptr, *hca_fn = nullptr, *hca_scale = nullptr, *hca_base = nullptr,
          *hcf_fn = nullptr, *hcf_scale = nullptr, *hcf_base = nullptr, *gate_w = nullptr, *gate_bias = nullptr,
          *eg_w = nullptr;
    uint16_t* i_wproj = nullptr;
    uint8_t* win = nullptr;
    float *c_kv = nullptr, *c_score = nullptr;
};
struct SourceBuf { uint8_t* ckv = nullptr; uint8_t* ik = nullptr; bool mirror = false; };
struct Scratch {
    float *h, *h2, *x, *qr, *q, *kvr, *ckr, *cgr, *latent, *ikr, *iq, *iw, *scores, *bm, *o, *oa, *ao, *mix, *a_pre,
        *a_post, *a_comb, *f_pre, *f_post, *f_comb, *pre, *glog, *rw, *moe_out, *sg, *su, *sh, *sh_out, *cpu_part,
        *eg_x, *eg_kv, *freqs_win, *freqs_comp, *ed, *logits;
    int *ridx, *topk, *cand, *sample_token;
    uint16_t *emb_rows, *ext;
    uint8_t* eg_rows;
};

class HipStage final : public stage::Backend
#ifdef STRATA_FORK_WARM
 ,public WarmRedirect
#endif
{
    Config cfg_;
    std::unique_ptr<WeightSource> ws_;
    ModelMeta g_;
    Topology topo_;
    int dev_;
    hipStream_t s_{};
    std::unique_ptr<Arena> arena_;
    Range r_{};
    stage::Plan plan_;
    BoundaryLayout bl_;
    std::vector<LayerW> L_;
    std::vector<SourceBuf> src_;
    Scratch d_{};
    x3::Mat head_;
    float* norm_w_ = nullptr;
    std::unique_ptr<CpuLane> cpu_;
    HipResidency residency_;
    HipResidentBank residents_;
    uint64_t resident_device_start_ = UINT64_MAX;
    uint64_t upstream_prepare_charge_ = 0; // observed HIP charge; upstream owns stream scratch for process lifetime
    uint64_t free_before_prepare_ = 0;
    StepPlanner planner_;
    SubDump dump_;
    std::vector<float> h_x_, h_part_, h_shared_;
    uint64_t request_ = 0, executed_step_ = UINT64_MAX, prepared_ = UINT64_MAX;
    bool valid_ = false, imported_ = false;
    bool loaded_ = false;  // weights/state for r_ fully initialized; only then may begin() reuse them
    Fault fault_;
    std::string report_ = "strata-dsv41-hip-stage";
    DecodeTiming timing_;
#ifdef STRATA_FORK_WARM
    std::unique_ptr<WarmDeviceTiming<HipDecodeClock>> warm_dense_;
    DenseTrace<HipDecodeClock> dense_trace_;
    std::unique_ptr<EngramReadAhead> engram_reads_;
    RouteCapture route_capture_;
    std::unique_ptr<HipExpertIssue> expert_issue_;
    uint64_t expert_issue_calls_=0;
    std::unique_ptr<HipDecodeClock> warm_expert_clock_;
    // Per layer, no shared capture across mutable weights or residual pointers.
    std::vector<std::unique_ptr<std::array<HipDenseGraph,4>>> dense_graphs_;
    k4::hipdense::Workspace dense_gemv_{};
    uint64_t dense_gemv_calls_=0,dense_gemv_fallbacks_=0;
    float* fuse_scores_ = nullptr;
    int fuse_score_stride_ = 0;
#endif
    DecodeEvents<HipDecodeClock> events_{timing_};

    bool fuse_small()const {
#ifdef STRATA_FORK_WARM
        return cfg_.warm_hip_fuse=="small-v1" || cfg_.warm_hip_fuse=="all-v1";
#else
        return false;
#endif
    }
    bool fuse_attention()const {
#ifdef STRATA_FORK_WARM
        return cfg_.warm_hip_fuse=="attention-v1" || cfg_.warm_hip_fuse=="all-v1";
#else
        return false;
#endif
    }
    void timing_begin(const stage::Token& t) {
        timing_.begin(cfg_.decode_timing_start,t.request,t.position,t.id);
        cpu_->begin_decode_step(timing_.active);
    }

    void sync() { timing_.wait([&]{CK(hipStreamSynchronize(s_));}); }
    void dump(const char* n, uint32_t il, uint64_t p, int T, const void* dev, size_t bytes) {
        if (!dump_.on(il, n)) return;
        std::vector<uint8_t> v(bytes);
        CK(hipMemcpyAsync(v.data(), dev, bytes, hipMemcpyDeviceToHost, s_)); sync();
        dump_.write(n, il, p, T, v.data(), bytes);
    }
    // mat_bf16 / mat_gg (ds4_model.cu:1600-1710) for whole GGUF dense matrices: BF16, Q8_0, Q4_K/Q5_K/Q6_K.
    // EXL3 trellis packs are not admitted by this driver (GGUF pack only).
    x3::Mat mat(const std::string& pre) {
        const TensorInfo t = ws_->require_tensor(pre + ".weight");
        require(t.shape.size() == 2, pre + ": not a matrix");
        const int n = int(t.shape[0]), k = int(t.shape[1]);
        x3::Mat m{};
        if (t.dtype == "BF16" || t.dtype == "Q8_0") {
            const bool q8 = t.dtype == "Q8_0";
            require(!q8 || k % 32 == 0, pre + ": Q8_0 row not whole blocks");
            const size_t rb = q8 ? size_t(k) / 32 * gg::kQ8Bytes : size_t(k) * 2;
            require(t.bytes == size_t(n) * rb, pre + ": byte count");
            m.fmt = q8 ? gg::kFmtQ8_0 : gg::kFmtBF16;
            m.ld = q8 ? int(rb) : k;
        } else if (t.dtype == "Q4_K" || t.dtype == "Q5_K" || t.dtype == "Q6_K") {
            int ggt = -1;
            for (int ty : {12, 13, 14}) if (t.dtype == gg::type_name(ty)) ggt = ty;  // gg_type_of, K-quants only
            require(ggt > 0 && k % 256 == 0 && t.bytes == size_t(n) * gg::type_row_bytes(ggt, k), pre + ": K-quant layout");
            m.fmt = gg::kFmtGG; m.K = ggt; m.ld = int(gg::type_row_bytes(ggt, k));
        } else throw std::runtime_error(pre + ": dense dtype " + t.dtype + " not admitted (EXL3/other: fail closed)");
#ifdef STRATA_FORK_WARM
        if(cfg_.warm_hip_gemv!="off" && m.fmt==gg::kFmtGG && m.K==gg::iq::kQ5_K)
            dense_gemv_.max_k=std::max(dense_gemv_.max_k,k);
#endif
        auto bytes = ws_->read_all(t.name);
        auto* dp = arena_->dev<uint8_t>(bytes.size());
        CK(hipMemcpyAsync(dp, bytes.data(), bytes.size(), hipMemcpyHostToDevice, s_)); sync();
        m.raw = dp; m.k = k; m.n = n;
        return m;
    }
    float* f32(const std::string& n, size_t want = 0) { return arena_->upload(read_as_f32(*ws_, n, want)); }
    static x3::MvJob job(const x3::Mat& m, const float* x, float* y, int T = 1, size_t x_cs = 0, size_t y_cs = 0) {
        x3::MvJob j; j.m = &m; j.xf = x; j.y = y; j.alpha = 1.0f; j.ncol = T; j.xf_cs = x_cs; j.y_cs = y_cs; return j;
    }
    void mv(std::initializer_list<x3::MvJob> jobs) {  // Impl::mv / mv_any for GGUF formats
        std::vector<x3::MvJob> j(jobs);
        mv_n(j.data(), int(j.size()));
    }
    void mv_n(x3::MvJob* j, int n) {
        for (int a = 0; a < n; a += x3::kMaxMvJobs) {
            const int count=std::min(x3::kMaxMvJobs,n-a);
#ifdef STRATA_FORK_WARM
            if(cfg_.warm_hip_gemv!="off") {
                if(k4::hipdense::eligible(j+a,count))++dense_gemv_calls_;else ++dense_gemv_fallbacks_;
                require(k4::hipdense::mv(j+a,count,dense_gemv_,k4::hipdense::mode(cfg_.warm_hip_gemv.c_str()),s_),"HIP dense v2 GEMV failed");
                continue;
            }
#endif
            require(gg::mv(j+a,count,s_), "gg::mv failed");
        }
    }

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
                l.c_kv = arena_->dev<float>(size_t(l.role.ratio) * size_t(g.hd));
                l.c_score = arena_->dev<float>(size_t(l.role.ratio) * size_t(g.hd));
            }
        }
        if (l.role.ix_src) {
            const std::string I = A + "indexer.";
            l.i_wq_b = mat(I + "wq_b");
            l.i_wproj = arena_->upload(read_as_f16(*ws_, I + "weights_proj.weight",
                                                   [](float v) { return strata::kernels::f16_from_f32(v); }));
            if (l.role.owns_k) { l.i_wk = mat(I + "wk"); l.ik_norm = f32(I + "k_norm.weight"); }
        }
        if (l.role.eg >= 0) {
            l.eg_wkv = mat(P + "engram.wkv");
            if (ws_->find(P + "engram.gate.weight")) l.eg_w = f32(P + "engram.gate.weight", size_t(g.hc) * size_t(g.dim));
            else {
                auto a = read_as_f32(*ws_, P + "engram.q_weight"), b = read_as_f32(*ws_, P + "engram.k_weight");
                require(a.size() == b.size() && a.size() == size_t(g.hc) * size_t(g.dim), "Engram q/k weight shape");
                for (size_t i = 0; i < a.size(); ++i) a[i] *= b[i];
                l.eg_w = arena_->upload(a);
            }
        }
        l.win = arena_->dev<uint8_t>(size_t(g.win) * g.win_rb());
    }
    void alloc_scratch() {  // ds4_model.cu:2638-2700 with W = 1
        const auto& g = g_;
        const size_t D = size_t(g.dim), H = size_t(g.hc), cap = size_t(cfg_.max_context) + 64;
        auto F = [&](size_t n) { return arena_->dev<float>(n); };
        auto& d = d_;
        d.h = F(H * D); d.h2 = F(H * D); d.x = F(D); d.qr = F(size_t(g.q_lora)); d.q = F(size_t(g.heads) * size_t(g.hd));
        d.kvr = F(size_t(g.hd)); d.ckr = F(size_t(g.hd)); d.cgr = F(size_t(g.hd)); d.latent = F(size_t(g.hd));
        d.ikr = F(size_t(g.ix_hd)); d.iq = F(size_t(g.ix_heads) * size_t(g.ix_hd)); d.iw = F(kIwStride);
        d.scores = F(cap); d.bm = F(g.bm_stride(cfg_.max_context)); d.o = F(size_t(g.heads) * size_t(g.hd));
        d.oa = F(size_t(g.o_groups) * size_t(g.o_lora)); d.ao = F(D); d.mix = F(kMixStride);
        d.a_pre = F(kPreStride); d.a_post = F(kPreStride); d.a_comb = F(kCombStride);
        d.f_pre = F(kPreStride); d.f_post = F(kPreStride); d.f_comb = F(kCombStride); d.pre = F(kPreStride);
        d.glog = F(kGlogStride); d.rw = F(kPicks); d.moe_out = F(D); d.sg = F(size_t(g.ff)); d.su = F(size_t(g.ff));
        d.sh = F(size_t(g.ff)); d.sh_out = F(D); d.cpu_part = F(D); d.ed = F(size_t(kPicks) * D);
        d.ridx = arena_->dev<int>(kPicks); d.topk = arena_->dev<int>(g.topk_stride()); d.cand = arena_->dev<int>(g.cand_stride());
        d.emb_rows = arena_->dev<uint16_t>(D); d.ext = arena_->dev<uint16_t>((size_t(g.win) + kWin) * size_t(g.hd));
        if (!g.eg_layers.empty()) {
            const size_t cols = size_t(g.eg_cols()), hd = size_t(g.eg_hd);
            d.eg_rows = arena_->dev<uint8_t>(cols * (hd + hd / 32)); d.eg_x = F(cols * hd); d.eg_kv = F((H + 1) * D);
        }
        // freqs: dalloc 64 floats each (ds4_model.cu:2648)
        auto fw = rope_freqs(g.rd, 0, g.rope_theta, g.factor, g.beta_fast, g.beta_slow); fw.resize(64, 0.0f);
        auto fc = rope_freqs(g.rd, g.orig_ctx, g.comp_theta, g.factor, g.beta_fast, g.beta_slow); fc.resize(64, 0.0f);
        d.freqs_win = arena_->upload(fw); d.freqs_comp = arena_->upload(fc);
        h_x_.assign(D, 0.0f); h_part_.assign(D, 0.0f); h_shared_.assign(D,0.0f);
    }
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
            src_[si].ckv = arena_->dev<uint8_t>(size_t(s.cap) * g_.ckv_rb());
            if (topo_.layers[s.layer].ix_src || read_ik) src_[si].ik = arena_->dev<uint8_t>(size_t(s.cap) * g_.ik_rb());
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
        case stage::Kind::hc_pre: return d_.pre;
        case stage::Kind::selected_topk: return d_.topk;
        case stage::Kind::candidates: return d_.cand;
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
    // Single teardown path: destruction, range replacement and failed begin() (review #5). Dense matrices
    // live in the arena on HIP, so clearing it releases them.
    void teardown() noexcept {
#ifdef STRATA_FORK_WARM
        if (engram_reads_) {
            engram_reads_->drain();
            for (int li=0;li<g_.eg_nlayers;++li) {
                const auto x=engram_reads_->stats(li);
                if (x.issued) std::fprintf(stderr,"AIOPT_ENGRAM_READAHEAD layer=%d issued=%llu consumed=%llu read_wall_ns=%llu demand_wait_ns=%llu scope=stage-lifetime-overlapping-host-times\n",
                    g_.eg_layers[size_t(li)],(unsigned long long)x.issued,(unsigned long long)x.consumed,
                    (unsigned long long)x.read_ns,(unsigned long long)x.wait_ns);
            }
        }
        engram_reads_.reset(); // joins before WeightSource, CPU lane or device storage can be released
        warm_unregister(this);
#endif
        loaded_ = false; valid_ = false;
        try { if (cpu_) cpu_->drain(); cpu_.reset(); } catch (...) {}
        if (arena_ && !arena_->drain()) {
            std::fprintf(stderr,"AIOPT_HIP_RESIDENT_DRAIN_FAILED remaining_arena_bytes=%llu release_ok=0\n",(unsigned long long)arena_->bytes);
            return; // retain views/owners until a later successful drain; audit fails closed
        }
        const bool had_residents = residency_.workspace != 0;
#ifdef STRATA_FORK_WARM
        if(dense_gemv_.q)std::fprintf(stderr,"AIOPT_HIP_GEMV_TOTAL mode=%s candidate_calls=%llu fallback_calls=%llu\n",
            cfg_.warm_hip_gemv.c_str(),(unsigned long long)dense_gemv_calls_,(unsigned long long)dense_gemv_fallbacks_);
        dense_gemv_={};dense_gemv_calls_=dense_gemv_fallbacks_=0;
        fuse_scores_=nullptr;fuse_score_stride_=0;
        if(expert_issue_calls_)std::fprintf(stderr,"AIOPT_EXPERT_ISSUE mode=early-v1 completed=%llu\n",(unsigned long long)expert_issue_calls_);
        expert_issue_calls_=0;expert_issue_.reset();
        dense_graphs_.clear(); // stream drained; destroy graph owners BEFORE arena/weights
#endif
        const auto uploaded = residents_.uploaded(), hits = residents_.hits();
        const auto owned = resident_device_start_ == UINT64_MAX ? 0 : arena_->bytes-resident_device_start_;
        residents_.reset(); residency_ = {};
        resident_device_start_ = UINT64_MAX;
        L_.clear(); src_.clear(); d_ = {}; head_ = x3::Mat{}; norm_w_ = nullptr; hccnt_ = nullptr; attn_scr_ = {};
        if (arena_) arena_->clear();
        if (had_residents)
            std::fprintf(stderr, "AIOPT_HIP_RESIDENT_RELEASE uploaded_bytes=%llu submitted_hits=%llu device_bytes=%llu remaining_arena_bytes=%llu release_ok=%d\n",
                (unsigned long long)uploaded, (unsigned long long)hits, (unsigned long long)owned,
                (unsigned long long)arena_->bytes, int(arena_->release_ok));
        report_ = "strata-dsv41-hip-stage";
    }
    void load_range(uint32_t first, uint32_t end) {
        alloc_scratch();
        hccnt_ = arena_->dev<unsigned>(kWin);
        attn_scr_.max_t = k4::kAttnScratchT; attn_scr_.cap = int(g_.topk_stride());
        attn_scr_.rows = arena_->dev<uint16_t>(size_t(k4::kAttnScratchT) * g_.topk_stride() * size_t(g_.hd));
#ifdef STRATA_FORK_WARM
        if(fuse_attention()){
            require(!std::getenv("DS4_ATTN_SPLIT") && !std::getenv("DS4_FAST_ATTN"), "unset alternate attention environment for HIP fusion");
            fuse_score_stride_=g_.win+int(g_.topk_stride());
            fuse_scores_=arena_->dev<float>(strata::ds4::moe::product(size_t(g_.heads),size_t(fuse_score_stride_)));
        }
#endif
        for (uint32_t il = first; il < end; ++il) {
            L_.push_back({}); L_.back().role = topo_.layers[il]; load_layer(L_.back());
            fault_.at("load_weights");
        }
        alloc_sources();
        if (end == uint32_t(g_.nl)) {
            head_ = mat("head");
            norm_w_ = f32("norm.weight", size_t(g_.dim));
            d_.logits = arena_->dev<float>(size_t(g_.vocab));
#ifdef STRATA_FORK_WARM
            if(cfg_.warm_experts)d_.sample_token=arena_->dev<int>(1);
#endif
        }
#ifdef STRATA_FORK_WARM
        if(cfg_.warm_hip_gemv!="off") {
            require(!std::getenv("DS4_QUANT_NODEDUP") && !std::getenv("DS4_QUANT_SINGLE"),"unset quantizer diagnostic environment for HIP dense v2");
            require(dense_gemv_.max_k>0,"HIP dense v2 has no eligible Q5 matrices");
            const size_t values=size_t(dense_gemv_.max_k)*x3::kMaxMvJobs;
            dense_gemv_.q=arena_->dev<int8_t>(values);
            dense_gemv_.d=arena_->dev<float>(values/32);
            dense_gemv_.sums=arena_->dev<k4::hipdense::Sums>(values/32);
            std::fprintf(stderr,"AIOPT_HIP_GEMV mode=%s max_k=%d workspace_bytes=%zu\n",cfg_.warm_hip_gemv.c_str(),dense_gemv_.max_k,
                values+(values/32)*(sizeof(float)+sizeof(k4::hipdense::Sums)));
        }
#endif
        fault_.at("cpu_lane");
        cpu_ = make_cpu_lane(*ws_, first, end, cfg_.cpu_threads, cfg_.cpu_dispatch, cfg_.cpu_timing);
        if (residency_.workspace) {
            resident_device_start_ = arena_->bytes;
            residents_.load(*ws_, residency_, *arena_, s_, cfg_.warm_hip_experts!="serial-v1",cfg_.warm_hip_experts=="soa-v2",cfg_.warm_hip_swiglu=="batch-v1");
#ifdef STRATA_FORK_WARM
            if(cfg_.warm_experts)warm_register(this,true);
#endif
            if(pooled_policy(residency_.policy))
                std::fprintf(stderr,"AIOPT_HIP_POOLED layout_tag=%llu base_arena_bytes=%llu bank_requested_bytes=%llu driver_charge_bytes=%llu workspace_bytes=%llu\n",
                    (unsigned long long)residency_.layout_tag,(unsigned long long)resident_device_start_,
                    (unsigned long long)residency_.allocated,(unsigned long long)residency_.driver_charge,
                    (unsigned long long)residents_.workspace());
        }
    }

    template<class F>auto dense_launch(const char* name,F fn){
#ifdef STRATA_FORK_WARM
        return dense_trace_.run(name,s_,fn);
#else
        return fn();
#endif
    }
    template<class F>void dense_group(uint32_t il,size_t part,F fn){
#ifdef STRATA_FORK_WARM
        if(cfg_.warm_hip_dense=="graphs-v1"){
            if(dense_graphs_.empty()){
                dense_graphs_.resize(size_t(g_.nl)+1);
                for(auto& g:dense_graphs_)g=std::make_unique<std::array<HipDenseGraph,4>>();
            }
            dense_graphs_.at(il)->at(part).run(true,s_,{d_.h,d_.h2},fn);
            return;
        }
#endif
        fn();
    }
    void layer(LayerW& l, const stage::Token& t) {
#ifdef STRATA_FORK_WARM
        HostScope host_dense(strata_warm_host_partition_v1(),HostPart::dense);
#endif
#ifdef STRATA_FORK_WARM
        dense_trace_.begin(cfg_.warm_kernel_timing,t.request,l.role.il,t.position);
        if(cfg_.warm_measure){if(!warm_dense_)warm_dense_=std::make_unique<WarmDeviceTiming<HipDecodeClock>>(size_t(g_.nl));warm_dense_->begin(s_);}
#endif
        events_.begin(DecodePart::other,s_);
        const auto& g = g_; const auto& role = l.role; const uint32_t il = role.il; const int64_t p0 = int64_t(t.position);
        const int D = g.dim, H = g.hc, HD = g.hd, NHL = g.heads, HQ = NHL * HD, OG = g.o_groups, T = 1;
        auto& d = d_; hipStream_t s = s_;
        const float* freqs = role.ratio ? d.freqs_comp : d.freqs_win;
        if (role.eg >= 0) {  // ds4_model.cu:10010-10045
            const int cols = g.eg_cols(), ehd = g.eg_hd;
            const size_t per = size_t(cols) * size_t(ehd + ehd / 32);
            std::vector<std::byte> host(per);
            timing_.host(timing_.engram_read_ns,[&]{
#ifdef STRATA_FORK_WARM
                if (engram_reads_) { host = engram_reads_->take(role.eg); return; }
#endif
                ws_->engram_rows(role.eg, engram_slice(plan_, t, il), host);
            });
            CK(hipMemcpyAsync(d.eg_rows, host.data(), per, hipMemcpyHostToDevice, s)); sync();
            const uint8_t* sc = d.eg_rows + size_t(cols) * size_t(ehd);
            dense_launch("k4::engram_dequant",[&]{return k4::engram_dequant(d.eg_rows, sc, T * cols, ehd, d.eg_x, s, ws_->engram_format(role.eg));});
            dump("eg_rows", il, p0, T, d.eg_rows, size_t(cols) * size_t(ehd));
            dump("eg_x", il, p0, T, d.eg_x, size_t(cols) * size_t(ehd) * 4);
            dense_launch("mv",[&]{return mv({job(l.eg_wkv, d.eg_x, d.eg_kv, T, size_t(cols) * size_t(ehd), size_t(H + 1) * size_t(D))});});
            dump("eg_kv", il, p0, T, d.eg_kv, size_t(H + 1) * size_t(D) * 4);
            dump("eg_h0", il, p0, T, d.h, size_t(H) * size_t(D) * 4);
            dump("eg_w", il, p0, 1, l.eg_w, size_t(H) * size_t(D) * 4);
            dense_launch("k4::engram_gate",[&]{return k4::engram_gate(d.h, d.eg_kv, l.eg_w, H, D, T, g.eps, s);});
            dump("eg_h", il, p0, T, d.h, size_t(H) * size_t(D) * 4);
        }
        events_.end(s_);
        events_.begin(DecodePart::attention_shared,s_);
        dump("h_in", il, p0, T, d.h, size_t(H) * size_t(D) * 4);
        dump("ring_in", il, p0, T, l.win, size_t(g.win) * g.win_rb());
        // Non-Metal hc_mix_split takes per-token zeroed block counters; T == 1 needs one (allocated with mix).
        dense_group(il,0,[&]{ // static prefix; RoPE/position and KV work stay outside
#ifdef STRATA_FORK_WARM
        if(fuse_small()) {
            k4::hipfuse::Hc p{d.h,l.hca_fn,l.hca_scale,l.hca_base,d.pre,l.attn_norm,
                d.mix,d.a_pre,d.a_post,d.a_comb,d.x,hccnt_,D,H,(2+H)*H,g.sinkhorn,g.eps,g.hc_eps};
            dense_launch("hipfuse::hc_pre",[&]{k4::hipfuse::hc_pre(p,s);});
        } else
#endif
        {
        dense_launch("k4::hc_mix_split",[&]{return k4::hc_mix_split(d.h, l.hca_fn, H * D, (2 + H) * H, T, g.eps, d.mix, hccnt_, l.hca_scale, l.hca_base, H,
                         g.sinkhorn, g.hc_eps, d.a_pre, d.a_post, d.a_comb, s);});
        dump("a_mix", il, p0, T, d.mix, kMixStride * 4); dump("a_pre", il, p0, T, d.a_pre, kPreStride * 4);
        dump("a_post", il, p0, T, d.a_post, kPreStride * 4); dump("a_comb", il, p0, T, d.a_comb, kCombStride * 4);
        dense_launch("k4::hc_pre_norm",[&]{return k4::hc_pre_norm(d.h, d.pre, l.attn_norm, D, H, T, g.eps, d.x, s);});
        }
        dump("pre_in", il, p0, T, d.pre, kPreStride * 4); dump("attn_x", il, p0, T, d.x, size_t(D) * 4);
        {
            x3::MvJob j[4]; int nj = 0;
            j[nj++] = job(l.wq_a, d.x, d.qr, T, D, g.q_lora);
            j[nj++] = job(l.wkv, d.x, d.kvr, T, D, HD);
            if (role.kv_src) j[nj++] = job(l.c_wkv, d.x, d.ckr, T, D, HD);
            if (role.kv_src && role.ratio > 1) j[nj++] = job(l.c_wgate, d.x, d.cgr, T, D, HD);
            dense_launch("mv_n",[&]{return mv_n(j, nj);});
        }
        dump("qr_raw", il, p0, T, d.qr, size_t(g.q_lora) * 4); dump("kvr_raw", il, p0, T, d.kvr, size_t(HD) * 4);
        { k4::Finish f; f.w = l.q_norm; f.eps = g.eps; dense_launch("k4::finish",[&]{return k4::finish(d.qr, T, g.q_lora, g.q_lora, f, s);}); }
        dump("qr", il, p0, T, d.qr, size_t(g.q_lora) * 4);
        dense_launch("mv",[&]{return mv({job(l.wq_b, d.qr, d.q, T, g.q_lora, HQ)});});
        });
        { k4::Finish f; f.rd = g.rd; f.pos = int(p0); f.vpp = NHL; f.freqs = freqs; dense_launch("k4::finish",[&]{return k4::finish(d.q, T * NHL, HD, HD, f, s);}); }
        dump("q", il, p0, T, d.q, size_t(HQ) * 4);
        if(!fuse_small()) {
        dense_launch("k4::ring_to_ext",[&]{return k4::ring_to_ext(l.win, d.ext, p0, g.win, HD, s);});
        }
        {
            k4::Finish f; f.w = l.kv_norm; f.eps = g.eps; f.rd = g.rd; f.pos = int(p0); f.freqs = freqs; f.quant = 1;
            f.dst = d.ext + size_t(g.win) * size_t(HD); f.dst_stride = HD;
            dense_launch("k4::finish",[&]{return k4::finish(d.kvr, T, HD, HD, f, s);});
        }
#ifdef STRATA_FORK_WARM
        if(fuse_small())dense_launch("hipfuse::ring",[&]{k4::hipfuse::ring(l.win,d.ext,p0,g.win,HD,s);});
        else
#endif
        {
        dense_launch("k4::ext_to_ring",[&]{return k4::ext_to_ring(d.ext, l.win, p0, g.win, T, HD, s, nullptr);});
        }
        dump("ext", il, p0, T, d.ext, (size_t(g.win) + T) * size_t(HD) * 2);
        k4::AttnRows ar;
        {
            const int first = int(std::max<int64_t>(int64_t(g.win) - p0, 1));
            ar.win0[0] = first; ar.nwin[0] = g.win - first + 1; ar.nsel[0] = 0;
        }
        const void* ckv = nullptr;
        if (role.ratio > 0) {
            auto& sr = src_[size_t(role.src)];
            const int R = role.ratio;
            const int64_t g_first = p0 / R, g_end = (p0 + T) / R;
            const int ng = int(std::max<int64_t>(0, g_end - g_first));
            if (role.kv_src) {
                if (R == 1) {
                    k4::Finish fc; fc.w = l.c_norm; fc.eps = g.eps; dense_launch("k4::finish",[&]{return k4::finish(d.ckr, T, HD, HD, fc, s);});
                    CK(hipMemcpyAsync(d.latent, d.ckr, size_t(T) * size_t(HD) * 4, hipMemcpyDeviceToDevice, s));
                } else {
                    dense_launch("k4::compress_win",[&]{return k4::compress_win(d.ckr, d.cgr, l.c_kv, l.c_score, R, int(p0), T, HD, l.c_norm, g.eps, d.latent, s);});
                    dump("ckr", il, p0, T, d.ckr, size_t(HD) * 4); dump("cgr", il, p0, T, d.cgr, size_t(HD) * 4);
                }
            }
            k4::Rows lim;
            lim.n[0] = int((p0 + 1) / R); lim.k[0] = std::min(g.ix_topk, lim.n[0]);
            const int clen_max = lim.n[0];
            if (role.ix_src) {
                auto& ks = src_[size_t(role.ik_src)];
                if (role.owns_k && ng > 0) {
                    dense_launch("mv",[&]{return mv({job(l.i_wk, d.latent, d.ikr, ng, HD, g.ix_hd)});});
                    k4::Finish fi; fi.w = l.ik_norm; fi.eps = g.eps; fi.rd = g.rd; fi.pos = int(g_first * R); fi.pos_step = R;
                    fi.freqs = freqs; fi.quant = 3; fi.pdst = ks.ik + size_t(g_first) * g.ik_rb(); fi.pdst_stride = (long long) g.ik_rb();
                    dense_launch("k4::finish",[&]{return k4::finish(d.ikr, ng, g.ix_hd, g.ix_hd, fi, s);});
                }
                if (clen_max > 0) {
                    dense_launch("mv",[&]{return mv({job(l.i_wq_b, d.qr, d.iq, T, g.q_lora, size_t(g.ix_heads) * size_t(g.ix_hd))});});
                    k4::Finish fi; fi.rd = g.rd; fi.pos = int(p0); fi.vpp = g.ix_heads; fi.freqs = freqs; fi.quant = 3;
                    dense_launch("k4::finish",[&]{return k4::finish(d.iq, T * g.ix_heads, g.ix_hd, g.ix_hd, fi, s);});
                    k4::gemv_f16_rows(l.i_wproj, d.x, size_t(D), g.ix_heads, D, T,
                                      1.0f / std::sqrt(float(g.ix_hd)) / std::sqrt(float(g.ix_heads)), d.iw, kIwStride, s);
                    const long long ss = (long long) cfg_.max_context + 64;
                    const bool restrict_ = (lim.n[0] + g.cand_blk - 1) / g.cand_blk > g.cand_topk;
                    if (role.uses_cand && restrict_) {
                        k4::Rows cl; cl.n[0] = lim.n[0]; cl.k[0] = planner_.cand[il];
                        const int nmax = planner_.cand[il] * g.cand_blk;
                        k4::index_score(d.iq, d.iw, ks.ik, g.ix_heads, g.ix_hd, T, nmax, d.cand, int(g.cand_stride()),
                                        g.cand_blk, cl, d.scores, ss, s);
                        k4::Rows tr; tr.n[0] = planner_.cand[il] * g.cand_blk; tr.k[0] = lim.k[0];
                        dense_launch("k4::topk",[&]{return k4::topk(d.scores, ss, T, tr, d.cand, int(g.cand_stride()), g.cand_blk, d.topk, int(g.topk_stride()), s);});
                    } else {
                        k4::index_score(d.iq, d.iw, ks.ik, g.ix_heads, g.ix_hd, T, clen_max, nullptr, 0, 1, lim, d.scores, ss, s);
                        if (role.cand_src) {
                            k4::Rows cr; cr.n[0] = lim.n[0]; cr.k[0] = g.cand_topk;
                            k4::cand_blocks(d.scores, ss, T, cr, g.cand_blk, d.bm, (long long) g.bm_stride(cfg_.max_context),
                                            d.cand, int(g.cand_stride()), s);
                        }
                        dense_launch("k4::topk",[&]{return k4::topk(d.scores, ss, T, lim, nullptr, 0, 1, d.topk, int(g.topk_stride()), s);});
                    }
                }
            }
            if (role.kv_src && ng > 0) {
                k4::Finish fc; fc.rd = g.rd; fc.pos = int(g_first * R); fc.pos_step = R; fc.freqs = freqs; fc.quant = 2;
                fc.pdst = sr.ckv + size_t(g_first) * g.ckv_rb(); fc.pdst_stride = (long long) g.ckv_rb();
                dense_launch("k4::finish",[&]{return k4::finish(d.latent, ng, HD, HD, fc, s);});
            }
            if (ng > 0) dump("latent", il, p0, ng, d.latent, size_t(ng) * size_t(HD) * 4);
            dump("topk", il, p0, T, d.topk, g.topk_stride() * 4);
            ar.nsel[0] = lim.n[0] > 0 ? planner_.topk[il] : 0;
            ckv = sr.ckv;
        }
#ifdef STRATA_FORK_WARM
        k4::Finish af; af.rd=g.rd;af.pos=int(p0);af.vpp=NHL;af.freqs=freqs;af.inverse=1;af.quant=1;
        if(fuse_attention())dense_launch("hipfuse::attention_finish",[&]{k4::hipfuse::attention(d.q,d.ext,ckv,d.topk,
            int(g.topk_stride()),ar,l.sink,NHL,HD,1.0f/std::sqrt(float(HD)),d.o,attn_scr_,
            fuse_scores_,fuse_score_stride_,s,&af);});
        else
#endif
        {
        dense_launch("k4::sparse_attn",[&]{return k4::sparse_attn(d.q, d.ext, ckv, d.topk, int(g.topk_stride()), ar, l.sink, NHL, HD, T, 1.0f / std::sqrt(float(HD)),
                        d.o, s, k4::kKvFp4, 0, &attn_scr_);});
        }
        dump("o", il, p0, T, d.o, size_t(HQ) * 4);
        if(!fuse_attention()) { k4::Finish fo; fo.rd = g.rd; fo.pos = int(p0); fo.vpp = NHL; fo.freqs = freqs; fo.inverse = 1; fo.quant = 1;
          dense_launch("k4::finish",[&]{return k4::finish(d.o, T * NHL, HD, HD, fo, s);}); }
        dump("o_fin", il, p0, T, d.o, size_t(HQ) * 4);
        dense_group(il,1,[&]{
        {
            x3::MvJob j[16];
            const int per = HQ / OG;
            for (int gi = 0; gi < OG; ++gi)
                j[gi] = job(l.wo_a[gi], d.o + size_t(gi) * size_t(per), d.oa + size_t(gi) * size_t(g.o_lora), T, HQ,
                            size_t(OG) * size_t(g.o_lora));
            dense_launch("mv_n",[&]{return mv_n(j, OG);});
            dense_launch("mv",[&]{return mv({job(l.wo_b, d.oa, d.ao, T, size_t(OG) * size_t(g.o_lora), D)});});
        }
        dump("ao", il, p0, T, d.ao, size_t(D) * 4);
        dense_launch("k4::hc_post",[&]{return k4::hc_post(d.ao, d.h, d.a_post, d.a_comb, D, H, T, d.h2, s);});
        });
        std::swap(d.h, d.h2);
        dump("h_attn", il, p0, T, d.h, size_t(H) * size_t(D) * 4);
        dense_group(il,2,[&]{ // static FFN; host route readbacks stay outside
#ifdef STRATA_FORK_WARM
        if(fuse_small()) {
            k4::hipfuse::Hc p{d.h,l.hcf_fn,l.hcf_scale,l.hcf_base,d.a_pre,l.ffn_norm,
                d.mix,d.f_pre,d.f_post,d.f_comb,d.x,hccnt_,D,H,(2+H)*H,g.sinkhorn,g.eps,g.hc_eps};
            dense_launch("hipfuse::hc_pre",[&]{k4::hipfuse::hc_pre(p,s);});
        } else
#endif
        {
        dense_launch("k4::hc_mix_split",[&]{return k4::hc_mix_split(d.h, l.hcf_fn, H * D, (2 + H) * H, T, g.eps, d.mix, hccnt_, l.hcf_scale, l.hcf_base, H,
                         g.sinkhorn, g.hc_eps, d.f_pre, d.f_post, d.f_comb, s);});
        dense_launch("k4::hc_pre_norm",[&]{return k4::hc_pre_norm(d.h, d.a_pre, l.ffn_norm, D, H, T, g.eps, d.x, s);});
        }
        dump("f_pre", il, p0, T, d.f_pre, kPreStride * 4); dump("f_post", il, p0, T, d.f_post, kPreStride * 4);
        dump("f_comb", il, p0, T, d.f_comb, kCombStride * 4); dump("ffn_x", il, p0, T, d.x, size_t(D) * 4);
        // ---- synchronous host-routed MoE (ds4_model.cu:10414-10571)
#ifdef STRATA_FORK_WARM
        if(fuse_small())dense_launch("hipfuse::router",[&]{k4::hipfuse::router(l.gate_w,d.x,l.gate_bias,
            g.n_exp,D,g.k_exp,g.route_scale,d.glog,d.ridx,d.rw,hccnt_,s);});
        else
#endif
        {
        dense_launch("k4::gemv_f32",[&]{return k4::gemv_f32(l.gate_w, d.x, g.n_exp, D, T, d.glog, kGlogStride, s);});
        dense_launch("k4::route",[&]{return k4::route(d.glog, l.gate_bias, g.n_exp, g.k_exp, T, g.route_scale, d.ridx, d.rw, s);});
        }
        });
        int h_idx[kPicks]; float h_w[kPicks];
#ifdef STRATA_FORK_WARM
        const bool early=cfg_.warm_expert_issue=="early-v1";
        const bool trace_issue=uint64_t(p0)<cfg_.warm_expert_timeline_limit;
        HipExpertIssue* issue=nullptr;
        if(early||trace_issue){
            if(!expert_issue_)expert_issue_=std::make_unique<HipExpertIssue>();
            issue=expert_issue_.get();issue->begin(s_,request_,p0,il,trace_issue,early);
        }
        // Declared after host route arrays: drains readbacks/callbacks before
        // these stack owners disappear on every exception path.
        HipIssueGuard issue_guard{issue};
#endif
        CK(hipMemcpyAsync(h_idx, d.ridx, kPicks * 4, hipMemcpyDeviceToHost, s));
        CK(hipMemcpyAsync(h_w, d.rw, kPicks * 4, hipMemcpyDeviceToHost, s));
        CK(hipMemcpyAsync(h_x_.data(), d.x, size_t(D) * 4, hipMemcpyDeviceToHost, s));
#ifdef STRATA_FORK_WARM
        if(issue)issue->route();
#endif
        dense_group(il,3,[&]{
        dense_launch("mv",[&]{return mv({job(l.s1, d.x, d.sg, T, D, g.ff), job(l.s3, d.x, d.su, T, D, g.ff)});});
        dense_launch("k4::swiglu",[&]{return k4::swiglu(d.sg, d.su, T * g.ff, g.lim, d.sh, s);});
        dense_launch("mv",[&]{return mv({job(l.s2, d.sh, d.sh_out, T, g.ff, D)});});
        });
        events_.end(s_);
#ifdef STRATA_FORK_WARM
        const bool queued=cfg_.warm_hip_schedule=="queued-v1";
        if(issue)issue->shared(); // completion of shared compute, before its host copy
        if(queued&&!early){
            CK(hipMemcpyAsync(h_shared_.data(),d.sh_out,size_t(D)*4,hipMemcpyDeviceToHost,s));
            if(cfg_.warm_measure)warm_dense_->mark_end(s_);
        }
#endif
#ifdef STRATA_FORK_WARM
        auto finish_shared=[&]{
            if(queued&&early){
                // A pageable D2H can block the submitting host. Enqueue it only
                // after CPU/B65 have been submitted, even when the API is Async.
                CK(hipMemcpyAsync(h_shared_.data(),d.sh_out,size_t(D)*4,hipMemcpyDeviceToHost,s));
                if(cfg_.warm_measure)warm_dense_->mark_end(s_);
            }
#endif
        sync();
#ifdef STRATA_FORK_WARM
        if(issue)issue->completed();
#endif
#ifdef STRATA_FORK_WARM
        dense_trace_.collect("hip");
        route_capture_.record(uint32_t(p0),uint32_t(il),uint32_t(g.nl),std::span(h_idx,size_t(g.k_exp)),std::span(h_w,size_t(g.k_exp)));
#endif
        dump("glog", il, p0, T, d.glog, kGlogStride * 4);
        if (dump_.on(il, "ridx")) dump_.write("ridx", il, uint64_t(p0), T, h_idx, sizeof(h_idx));
        if (dump_.on(il, "rw")) dump_.write("rw", il, uint64_t(p0), T, h_w, sizeof(h_w));
        dump("sh_out", il, p0, T, d.sh_out, size_t(D) * 4);
#ifdef STRATA_FORK_WARM
        if(cfg_.warm_measure){if(queued)warm_dense_->collect(il);else warm_dense_->end(il,s_);}
#endif
#ifdef STRATA_FORK_WARM
        };
        if(early)issue->wait_route();else finish_shared();
        if(cfg_.warm_experts){
            if(queued){
                h_part_=warm_execute(*cpu_,il,std::span(h_idx,size_t(g.k_exp)),std::span(h_w,size_t(g.k_exp)),h_x_,h_shared_,
                    early?std::function<void()>(finish_shared):std::function<void()>{},issue?&issue->timeline:nullptr);
                if(early)++expert_issue_calls_;
                strata_warm_host_partition_v1()->enter(HostPart::completion);
                {HostScope h2d(strata_warm_host_partition_v1(),HostPart::result_h2d);
                CK(hipMemcpyAsync(d.moe_out,h_part_.data(),h_part_.size()*4,hipMemcpyHostToDevice,s));}
                // Final layer sync retains this host owner through H2D + hc_post.
            }else{
                std::vector<float> shared(static_cast<size_t>(D));CK(hipMemcpyAsync(shared.data(),d.sh_out,size_t(D)*4,hipMemcpyDeviceToHost,s));sync();
                auto combined=warm_execute(*cpu_,il,std::span(h_idx,size_t(g.k_exp)),std::span(h_w,size_t(g.k_exp)),h_x_,shared);
                strata_warm_host_partition_v1()->enter(HostPart::completion);
                {HostScope h2d(strata_warm_host_partition_v1(),HostPart::result_h2d);
                CK(hipMemcpyAsync(d.moe_out,combined.data(),combined.size()*4,hipMemcpyHostToDevice,s));sync();}
            }
            events_.begin(DecodePart::other,s_);
        }else
#endif
        {
        // Placement all-cpu: every pick is a CPU-lane entry; slots in expert-ID order (per-e loop, :10469-10500).
        const auto route_tick=timing_.stamp();
        k4::Combine cm; cm.n[0] = 0; cm.cpu[0] = 0;
        std::vector<std::pair<int, float>> picks;
        for (int i = 0; i < g.k_exp; ++i) {
            const int e = h_idx[i];
            require(e >= 0 && e < g.n_exp, "router picked an out-of-range expert");
            require(std::isfinite(h_w[i]), "router produced a nonfinite weight");
            picks.push_back({e, h_w[i]});
        }
        std::sort(picks.begin(), picks.end(), [](auto a, auto b) { return a.first < b.first; });
        if (residency_.workspace) {
            picks.clear();
            for (const auto& p : resident_picks(residency_, il, std::span(h_idx, size_t(g.k_exp)), std::span(h_w, size_t(g.k_exp))))
                picks.push_back({p.expert, p.weight});
        }
        std::fill(h_part_.begin(), h_part_.end(), 0.0f);
        int cpu_count = 0;
        timing_.routing_combine_host_ns+=timing_.elapsed(route_tick);
        // P0 keeps its original loop; P1 only redirects immutable resident hits.
        // Serial completion keeps this out of P3's overlap experiment.
        std::vector<int> cpu_ids;
        for(const auto& [e,wv]:picks)if(!residency_.workspace||residency_.slot(il,e)<0)cpu_ids.push_back(e);
        cpu_->prepare(il,cpu_ids);
        struct FinishCpu {CpuLane* p;~FinishCpu(){p->finish();}} finish_cpu{cpu_.get()};
        for (const auto& [e, wv] : picks) {
            if (residency_.workspace && residency_.slot(il, e) >= 0) {
                const int row = cm.n[0];
                events_.begin(DecodePart::resident,s_);
                residents_.expert(il, residency_.slot(il, e), d.x, d.ed + size_t(row) * size_t(D), g, s);
                events_.end(s_);
                sync();
                warm_note_hip(il,e);
                const auto combine_tick=timing_.stamp();
                cm.row[0][row] = row; cm.w[0][row] = wv; ++cm.n[0];
                timing_.routing_combine_host_ns+=timing_.elapsed(combine_tick);
                continue;
            }
            const auto y = cpu_->expert(il, e, h_x_);
            const auto combine_tick=timing_.stamp();
            require(y.size() == size_t(D), "CPU lane output shape");
            float* o = h_part_.data();
            for (int k = 0; k < D; ++k) o[k] += wv * y[size_t(k)];  // verbatim :10541 (host contraction: see README)
            ++cpu_count;
            timing_.routing_combine_host_ns+=timing_.elapsed(combine_tick);
        }
        cm.cpu[0] = cpu_count ? 1 : 0;
        events_.begin(DecodePart::other,s_);
        if (dump_.on(il, "cpu_part")) dump_.write("cpu_part", il, uint64_t(p0), T, h_part_.data(), size_t(D) * 4);
        CK(hipMemcpyAsync(d.cpu_part, h_part_.data(), size_t(D) * 4, hipMemcpyHostToDevice, s));
        k4::combine(d.ed, cpu_count ? d.cpu_part : nullptr, d.sh_out, cm, T, D, d.moe_out, s);
        dump("moe_out", il, p0, T, d.moe_out, size_t(D) * 4);
        }
        dense_launch("k4::hc_post",[&]{return k4::hc_post(d.moe_out, d.h, d.f_post, d.f_comb, D, H, T, d.h2, s);});
        std::swap(d.h, d.h2);
        dump("h_out", il, p0, T, d.h, size_t(H) * size_t(D) * 4);
        CK(hipMemcpyAsync(d.pre, d.f_pre, size_t(T) * kPreStride * 4, hipMemcpyDeviceToDevice, s));
        events_.end(s_);
        {
#ifdef STRATA_FORK_WARM
        HostScope wait(strata_warm_host_partition_v1(),HostPart::ffn_wait);
#endif
        sync();  // h_part_/h_x_ are reused by the next layer
        }
    }
    unsigned* hccnt_ = nullptr;
    k4::AttnScratch attn_scr_{};

public:
    HipStage(uint32_t device, Config c)
        : cfg_(std::move(c)), ws_(open_pack(cfg_.pack, {cfg_.q5k_fma, cfg_.cpu_timing, cfg_.warm_engram_priority=="first-v1", cfg_.warm_engram_trace_limit})), g_(ws_->meta()),
          topo_(topology(g_, cfg_.max_context)), dev_(int(device)) {
        CK(hipSetDevice(dev_));
        CK(hipStreamCreateWithFlags(&s_, hipStreamNonBlocking));
        arena_ = std::make_unique<Arena>(dev_, s_);
        size_t free_before = 0, free_after = 0, total = 0;
        if (cfg_.hip_experts != "off") { CK(hipMemGetInfo(&free_before, &total)); free_before_prepare_ = free_before; }
        require(x3::prepare(s_) && gg::prepare(s_), "x3/gg prepare failed");  // ds4_model.cu:2191
        if (cfg_.hip_experts != "off") {
            sync(); CK(hipMemGetInfo(&free_after, &total));
            require(free_after <= free_before, "upstream scratch charge changed during prepare");
            upstream_prepare_charge_ = free_before-free_after;
        }
        fault_.where = cfg_.fault_inject;
        dump_.open(cfg_);
    }
    ~HipStage() override {
        invalidate(); teardown();
        arena_.reset();
        hipSetDevice(dev_); hipStreamDestroy(s_);
    }
    // describe(): name, range, measured device bytes (arena incl. dense matrices, head, caches, scratch)
    // and the Plan footprint it was admitted against.
    const char* name() const override { return report_.c_str(); }
    void begin(const stage::Plan& plan, uint32_t first, uint32_t end, uint64_t request) override {
        valid_ = false;
        timing_.active=false;events_.abandon();
        drain();
        struct BeginGuard {
            HipStage* stage; bool committed = false;
            ~BeginGuard() { if (!committed) stage->teardown(); }
        } transaction{this};
        plan.validate();
        require(plan.layers == uint32_t(g_.nl) && plan.hc == uint32_t(g_.hc) && plan.dim == uint32_t(g_.dim),
                "plan geometry differs from the pack");
        auto selected = check_hip_residency_plan(*ws_, cfg_, plan);
        require(plan.model_sha == ws_->identity(),
                "plan model/schema/context identity mismatch");
        require(first < end && end <= plan.layers, "invalid stage range");
        require((first == 0 && end == plan.cut) || first == plan.cut, "stage range must be the plan's prefix or start at its cut");
        CK(hipSetDevice(dev_));
        if (selected.workspace) require((first == 1 || (first==0 && plan.hip_dense_only())) && end == plan.layers, "P1 requires the complete HIP tail at cut1");
        const bool same = loaded_ && r_.first == first && r_.end == end && plan_.model_sha == plan.model_sha &&
                          plan_.schema_sha == plan.schema_sha && plan_.hip.cap == plan.hip.cap && plan_.hip.fixed == plan.hip.fixed;
        if (!same) {
            teardown();
            require(arena_->release_ok && arena_->bytes == 0, "prior HIP allocations failed to release");
            r_ = {first, end};
            residency_ = std::move(selected);
            try { load_range(first, end); loaded_ = true; }
            catch (...) { teardown(); throw; }
        }
        const BoundaryLayout bl = BoundaryLayout::of(plan);
        check_extents(bl, [&](const stage::Edge& e) { return edge_extent(e); });
        const stage::Budget& budget = first == 0 && !plan.hip_dense_only() ? plan.sycl : plan.hip;
        uint64_t observed_charge = 0;
        if (residency_.workspace) {
            size_t free_now = 0, total = 0; CK(hipMemGetInfo(&free_now, &total));
            observed_charge = resident_observed_charge(residency_.policy,free_before_prepare_,free_now,total);
            if(cfg_.warm_experts)std::fprintf(stderr,"AIOPT_WARM_FREE backend=hip total_bytes=%llu free_bytes=%llu\n",(unsigned long long)total,(unsigned long long)free_now);
        }
        const uint64_t measured = std::max(stage::add(arena_->bytes, upstream_prepare_charge_), observed_charge);
        const uint64_t admitted = plan.footprint(budget, first, end);
        report_ = "strata-dsv41-hip-stage range=[" + std::to_string(first) + "," + std::to_string(end) +
                  ") device_bytes=" + std::to_string(measured) + " plan_footprint=" + std::to_string(admitted) +
                  " cap=" + std::to_string(budget.cap);
        require(measured <= budget.cap, "measured device allocation exceeds the frozen stage cap: " + report_);
        if (residency_.workspace) {
            require(admitted >= kHipDriverSafetyMargin && measured <= admitted-kHipDriverSafetyMargin,
                    "P1 measured allocation consumes frozen driver safety margin");
            std::fprintf(stderr, "AIOPT_HIP_RESIDENT policy=%s slots=%zu profile_tag=%llu slots_tag=%llu payload_bytes=%llu allocated_bytes=%llu workspace_bytes=%llu workspace_reserve_bytes=%llu device_bytes=%llu plan_footprint=%llu cap=%llu weight_h2d_bytes=%llu upstream_prepare_charge=%llu arena_bytes=%llu observed_device_charge=%llu\n",
                residency_.policy.c_str(),residency_.experts.size(), (unsigned long long)residency_.profile_tag, (unsigned long long)residency_.slots_tag,
                (unsigned long long)residency_.payload, (unsigned long long)residency_.allocated,
                (unsigned long long)residents_.workspace(), (unsigned long long)residency_.workspace,
                (unsigned long long)measured, (unsigned long long)admitted, (unsigned long long)budget.cap,
                (unsigned long long)residents_.uploaded(), (unsigned long long)upstream_prepare_charge_,
                (unsigned long long)arena_->bytes, (unsigned long long)observed_charge);
        }
        plan_ = plan; bl_ = bl;
        for (auto& l : L_) {  // Impl::reset + deterministic sources
            CK(hipMemsetAsync(l.win, 0, size_t(g_.win) * g_.win_rb(), s_));
            if (l.c_kv) CK(hipMemsetAsync(l.c_kv, 0, size_t(l.role.ratio) * size_t(g_.hd) * 4, s_));
            if (l.c_score) CK(hipMemsetAsync(l.c_score, 0, size_t(l.role.ratio) * size_t(g_.hd) * 4, s_));
        }
        for (size_t si = 0; si < src_.size(); ++si) {
            if (src_[si].ckv) CK(hipMemsetAsync(src_[si].ckv, 0, size_t(topo_.sources[si].cap) * g_.ckv_rb(), s_));
            if (src_[si].ik) CK(hipMemsetAsync(src_[si].ik, 0, size_t(topo_.sources[si].cap) * g_.ik_rb(), s_));
        }
        CK(hipMemsetAsync(d_.topk, 0, g_.topk_stride() * 4, s_)); CK(hipMemsetAsync(d_.cand, 0, g_.cand_stride() * 4, s_));
        sync();
        request_ = request; executed_step_ = prepared_ = UINT64_MAX; imported_ = false;
        valid_ = true;
        transaction.committed = true;
    }
    stage::HostMemory pinned(size_t n) override {
        CK(hipSetDevice(dev_)); void* p = nullptr; CK(hipHostMalloc(&p, n, hipHostMallocDefault));
        int d = dev_;
        return {p, n, [d](void* x) { hipSetDevice(d); hipHostFree(x); }};
    }
    void execute(const stage::Token& t, const std::atomic_bool& cancel) override {
        require(valid_, "stage invalid; begin() required");
        require(t.request == request_ && t.step == t.position, "token request/position mismatch");
        // Upstream forward(): pos + T <= max_context (ds4_model.cu:9705), before ANY state change or submit.
        require(t.position < uint64_t(cfg_.max_context), "the context is full (position >= max_context)");
        require(t.id >= 0 && t.id < g_.vocab, "token out of range");
#ifdef STRATA_FORK_WARM
        if (cfg_.warm_engram=="read-ahead-v1" && !engram_reads_ && g_.eg_nlayers) {
            engram_reads_=std::make_unique<EngramReadAhead>(size_t(g_.eg_nlayers),size_t(g_.eg_cols()),
                size_t(g_.eg_cols())*size_t(g_.eg_hd+g_.eg_hd/32),
                [this](int li,std::span<const uint64_t> ids,std::span<std::byte> dst,const std::function<void()>& queued){ws_->engram_rows_scheduled(li,ids,dst,queued);});
        }
        EngramReadAhead::TokenGuard engram_guard{engram_reads_.get()};
        require(!cancel.load(std::memory_order_acquire), "request cancelled");
        if (engram_reads_) {
            ws_->engram_token(t.request,t.position);bool first=true;
            for (const auto& l : L_) if (l.role.eg >= 0) {
                // L_ is in model order. First layer owns its complete row queue
                // before later layer workers can compete for the shared pool.
                engram_reads_->issue(l.role.eg,engram_slice(plan_,t,l.role.il),first && cfg_.warm_engram_priority=="first-v1");
                first=false;
            }
        }
#endif
        {
#ifdef STRATA_FORK_WARM
        HostScope planner(strata_warm_host_partition_v1(),HostPart::planner);
#endif
        CK(hipSetDevice(dev_));
        if(r_.prefix())timing_begin(t); // tail already started at its H2D import
        planner_.advance(g_, topo_, t.position);
        }
        if (r_.prefix()) {
#ifdef STRATA_FORK_WARM
            HostScope host_embedding(strata_warm_host_partition_v1(),HostPart::embedding);
#endif
            events_.begin(DecodePart::other,s_);
            std::vector<uint16_t> row(size_t(g_.dim));
            timing_.host(timing_.embedding_read_ns,[&]{ws_->embedding_bf16(t.id, row);});
            CK(hipMemcpyAsync(d_.emb_rows, row.data(), row.size() * 2, hipMemcpyHostToDevice, s_));
            k4::embed(d_.emb_rows, 1, d_.h, g_.dim, g_.hc, s_);
            k4::init_pre(d_.pre, 1, s_);
            events_.end(s_);
            sync();
        } else require(imported_, "tail stage executed without an imported boundary");
        for (auto& l : L_) {
            require(!cancel.load(std::memory_order_acquire), "request cancelled");
            layer(l, t);
        }
        if (tail(r_, g_)) {  // head_rows_dev + head_share for one row on a whole head (nm == 1)
            events_.begin(DecodePart::other,s_);
#ifdef STRATA_FORK_WARM
            strata_warm_host_partition_v1()->enter(HostPart::head);
            dense_trace_.begin(cfg_.warm_kernel_timing,t.request,uint32_t(g_.nl),t.position);
            if(cfg_.warm_measure)warm_dense_->begin(s_);
#endif
            dense_group(uint32_t(g_.nl),0,[&]{
            dense_launch("k4::hc_pre_norm",[&]{return k4::hc_pre_norm(d_.h, d_.pre, norm_w_, g_.dim, g_.hc, 1, g_.eps, d_.x, s_);});
            dense_launch("mv_head",[&]{return mv({job(head_, d_.x, d_.logits, 1, size_t(g_.dim), size_t(g_.vocab))});});
            });
#ifdef STRATA_FORK_WARM
            if(cfg_.warm_measure){if(cfg_.warm_hip_schedule=="queued-v1")warm_dense_->mark_end(s_);else warm_dense_->end(size_t(g_.nl),s_);}
#endif
            events_.end(s_);
        }
        sync();
#ifdef STRATA_FORK_WARM
        if(tail(r_,g_)){
            if(cfg_.warm_measure&&cfg_.warm_hip_schedule=="queued-v1")warm_dense_->collect(size_t(g_.nl));
            dense_trace_.collect("hip");
        }
#endif
        executed_step_ = t.step; imported_ = false;
    }
    std::unique_ptr<stage::Completion> export_boundary(const stage::Token& t, std::span<std::byte> host) override {
        require(valid_ && executed_step_ == t.step && host.size() == bl_.total, "export before execute / extent");
        events_.begin(DecodePart::boundary,s_);
        CK(hipMemcpyAsync(host.data(), d_.h, size_t(bl_.residual), hipMemcpyDeviceToHost, s_));
        for (const auto& sl : bl_.slots)
            CK(hipMemcpyAsync(host.data() + sl.offset, edge_buffer(sl.edge), size_t(sl.edge.bytes), hipMemcpyDeviceToHost, s_));
        events_.end(s_);
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
        events_.begin(DecodePart::boundary,s_);
        CK(hipMemcpyAsync(d_.h, host.data(), size_t(bl_.residual), hipMemcpyHostToDevice, s_));
        for (const auto& sl : bl_.slots)
            CK(hipMemcpyAsync(edge_buffer(sl.edge), host.data() + sl.offset, size_t(sl.edge.bytes), hipMemcpyHostToDevice, s_));
        events_.end(s_);
        sync();
        imported_ = true;
        struct Done final : stage::Completion { size_t n; explicit Done(size_t b) : n(b) {} void wait() override {}
                                                size_t bytes() const override { return n; } };
        return std::make_unique<Done>(host.size());
    }
    std::vector<float> logits(const stage::Token& t) override {
#ifdef STRATA_FORK_WARM
        HostScope readback(strata_warm_host_partition_v1(),HostPart::readback);
#endif
        require(valid_ && tail(r_, g_) && executed_step_ == t.step, "logits only on the executed tail");
        std::vector<float> out(size_t(g_.vocab));
        events_.begin(DecodePart::other,s_);
        CK(hipMemcpyAsync(out.data(), d_.logits, out.size() * 4, hipMemcpyDeviceToHost, s_));
        events_.end(s_); sync();
        return out;
    }
#ifdef STRATA_FORK_WARM
    std::vector<double> warm_dense_costs()const override{return warm_dense_?warm_dense_->means():std::vector<double>{};}
    bool warm_contains(uint32_t l,int e)const override{return residency_.workspace&&l<residency_.slots.size()&&e>=0&&size_t(e)<residency_.slots[l].size()&&residency_.slot(l,e)>=0;}
    std::vector<float> warm_compute(uint32_t l,int e,std::span<const float> x)override{
        require(warm_contains(l,e)&&x.size()==size_t(g_.dim),"warm HIP request extent/residency");CK(hipSetDevice(dev_));
        if(!warm_expert_clock_)warm_expert_clock_=std::make_unique<HipDecodeClock>();warm_expert_clock_->begin(s_);
        CK(hipMemcpyAsync(d_.x,x.data(),x.size_bytes(),hipMemcpyHostToDevice,s_));
        residents_.expert(l,residency_.slot(l,e),d_.x,d_.ed,g_,s_);
        std::vector<float> y(x.size());CK(hipMemcpyAsync(y.data(),d_.ed,y.size()*4,hipMemcpyDeviceToHost,s_));warm_expert_clock_->end(s_);warm_gpu_span(true,l,warm_expert_clock_->elapsed_ns());return y;
    }
    bool warm_grouped()const override{return cfg_.warm_hip_experts!="serial-v1";}
    std::vector<std::vector<float>> warm_compute_many(uint32_t l,std::span<const int> ids,std::span<const float> x)override{
        require(warm_grouped() && !ids.empty() && ids.size()<=size_t(g_.k_exp) && x.size()==size_t(g_.dim),"warm HIP batch request extent");
        std::vector<int> slots;slots.reserve(ids.size());
        for(int id:ids){require(warm_contains(l,id),"warm HIP batch nonresident selection");slots.push_back(residency_.slot(l,id));}
        CK(hipSetDevice(dev_));
        if(!warm_expert_clock_)warm_expert_clock_=std::make_unique<HipDecodeClock>();
        try{
            warm_expert_clock_->begin(s_);
            residents_.enqueue_layer(l,slots,x,g_,s_);
            warm_expert_clock_->end(s_); // one completion wait, after the whole layer's D2H
            warm_gpu_span(true,l,warm_expert_clock_->elapsed_ns());
        }catch(...){hipStreamSynchronize(s_);throw;} // host transfer owners remain in residents_ even if drain fails
        auto flat=residents_.layer_output();std::vector<std::vector<float>> rows;rows.reserve(ids.size());
        for(size_t i=0;i<ids.size();++i)rows.emplace_back(flat.begin()+i*x.size(),flat.begin()+(i+1)*x.size());
        return rows;
    }
    int32_t sample(float temperature,float top_p,int top_k,float min_p,uint64_t seed,uint64_t counter)override{
        require(valid_&&tail(r_,g_)&&cfg_.warm_experts,"warm sample requires valid warm HIP tail");
        strata::dsv41k::SamplerParams params;params.temperature=temperature;params.greedy=temperature<=0;
        params.top_p=top_p;params.top_k=top_k;params.min_p=min_p;params.seed=seed;params.counter=counter;
        strata::dsv41k::sample_tokens(d_.logits,1,g_.vocab,nullptr,0,params,d_.sample_token,s_);
        HostScope readback(strata_warm_host_partition_v1(),HostPart::sample_readback);
        int out=-1;CK(hipMemcpyAsync(&out,d_.sample_token,sizeof(int),hipMemcpyDeviceToHost,s_));sync();
        require(out>=0&&out<g_.vocab,"sampled token outside vocabulary");return out;
    }
#endif
    void prepare_commit(uint64_t step) override {
        require(valid_ && executed_step_ == step, "prepare_commit for a step this stage did not execute");
        sync(); prepared_ = step;
        timing_.emit("hip",r_.first,r_.end,cpu_->decode_phases());
    }
    void commit(uint64_t step) override { require(valid_ && prepared_ == step, "commit without prepare"); }
    void invalidate() noexcept override { valid_ = false; }
    void drain() noexcept override {
        if (cpu_) cpu_->drain();
        hipSetDevice(dev_); hipStreamSynchronize(s_);
    }
};
}  // namespace

std::unique_ptr<stage::Backend> make_model_backend(uint32_t device, const Config& c) {
    return std::make_unique<HipStage>(device, c);
}
}  // namespace strata::wiring
