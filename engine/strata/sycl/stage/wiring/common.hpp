// MIT; crosslane contributors. Host-only pieces shared by both model backends.
#pragma once
#include "model_meta.hpp"
#include "weight_source.hpp"
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>

namespace strata::wiring {

struct Config {
    std::string pack;                // pack dir (ds4.txt, tensors.txt, engram.bin)
    int64_t max_context = 0;         // REQUIRED; bound into Plan.schema_sha (schema_for) and checked in begin()
    std::string dump_s;              // DS4_DUMP_S-format sub-step dump file ("" off)
    std::set<int> dump_layers;       // DS4_DUMP_S_LAYERS equivalent
    std::set<std::string> dump_names;// DS4_DUMP_S_NAMES equivalent (empty: all names this driver emits)
    int cpu_threads = 0;
    std::string cpu_dispatch = "spawn-v1"; // pool-v1 is explicit, a11 speed experiment only
    bool cpu_timing = false;
    int64_t decode_timing_start = -1; // -1 off; coordinator injects exact prompt length, never a lifetime mode
    std::string hip_experts = "off"; // P1: profile-v1 is static, HIP tail at cut1 only
    std::string hip_expert_math = "per32-v1"; // cpu-q8k-v1 is explicit; old P1 unchanged
    std::string expert_profile;     // uint32 LE counts; external SHA256 in freeze
    bool warm_experts = false;
    bool warm_measure = false;
    std::string warm_dense_owner = "split-v1"; // hip-v1: explicit empty SYCL dense range
    std::string warm_hip_experts = "serial-v1"; // layer-batch-v1: exact kernels, one layer transfer
    std::string warm_hip_schedule = "baseline-v1"; // queued-v1: same kernels/order, fewer host waits
    std::string warm_expert_issue = "off"; // early-v1: CPU/B65 overlap same-stream shared FFN
    uint32_t warm_expert_timeline_limit = 0; // first N absolute positions/request, diagnostic only
    std::string warm_engram = "off"; // read-ahead-v1: same-token host rows, HIP cut0 only
    bool warm_kernel_timing = false; // bounded launch spans; read only after dense-end wait
    std::string warm_hip_fuse = "off"; // small-v1, attention-v1, all-v1; native qualification required
    std::string warm_hip_gemv = "off"; // exact Q5 activation sums, 8/4 warps, native gate required
    std::string warm_hip_dense = "off"; // graphs-v1: four static device sequences per layer

    std::string warm_sycl_experts = "serial-v1"; // explicit grouped packed candidate
    bool warm_packed_gu = false; // requires separately accepted gate/up native evidence
    std::string warm_placement; // opt-in complete freeze-time assignment
    std::string warm_sycl_attention = "scalar-v1"; // exact-order candidate, explicit only
    uint64_t warm_hip_fixed=0,warm_sycl_fixed=0;
    uint64_t warm_hip_bytes = 0;
    uint64_t warm_sycl_bytes = 0; // explicit bank payload cap, separately charged in plan
    std::string expert_placement = "all-cpu";  // first gate: every routed expert on the CPU lane (see README)
    bool q5k_fma = false;            // Q5_K embedding decode with fused d*q-m (must match ggml's build; embed_check)
    // TEST ONLY (parity/begin_retry_test.cpp): one-shot throw inside begin() at "load_weights" (after the
    // first layer's weights) or "cpu_lane" (instead of constructing the CPU lane). Empty in every real run.
    std::string fault_inject;
    static Config parse(const char* text) {
        Config c;
        require(text != nullptr, "null stage config");
        std::istringstream in(text);
        std::string line;
        std::set<std::string> seen;
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            const auto eq = line.find('=');
            require(eq != std::string::npos, "stage config line without '='");
            const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
            // Review 2026-11-02 #1: no silent override (e.g. a second max_context after the host plan's).
            require(seen.insert(k).second, "duplicate stage config key: " + k);
            if (k == "pack") c.pack = v;
            else if (k == "max_context") c.max_context = std::stoll(v);
            else if (k == "dump_s") c.dump_s = v;
            else if (k == "dump_layers" || k == "dump_names") {
                std::stringstream ss(v); std::string x;
                while (std::getline(ss, x, ',')) if (!x.empty()) {
                    if (k == "dump_layers") c.dump_layers.insert(std::stoi(x)); else c.dump_names.insert(x);
                }
            } else if (k == "cpu_threads") c.cpu_threads = std::stoi(v);
            else if (k == "cpu_dispatch") c.cpu_dispatch = v;
            else if (k == "decode_timing_start") {
                c.decode_timing_start = std::stoll(v);
                require(std::to_string(c.decode_timing_start) == v && c.decode_timing_start >= 0,
                        "decode_timing_start must be a canonical nonnegative position");
            }
            else if (k == "hip_experts") c.hip_experts = v;
            else if (k == "hip_expert_math") c.hip_expert_math = v;
            else if (k == "expert_profile") c.expert_profile = v;
            else if (k == "warm_measure") { require(v=="0"||v=="1","warm_measure must be 0 or 1"); c.warm_measure=v=="1"; }
            else if (k == "warm_experts") { require(v=="0"||v=="1","warm_experts must be 0 or 1"); c.warm_experts=v=="1"; }
            else if (k == "warm_packed_gu") { require(v=="0"||v=="1","warm_packed_gu must be 0 or 1"); c.warm_packed_gu=v=="1"; }
            else if (k == "warm_dense_owner") c.warm_dense_owner=v;
            else if (k == "warm_engram") c.warm_engram=v;
            else if (k == "warm_hip_experts") c.warm_hip_experts=v;
            else if (k == "warm_expert_issue") c.warm_expert_issue=v;
            else if (k == "warm_expert_timeline_limit") { auto n=std::stoull(v);require(std::to_string(n)==v&&n<=2000,"expert timeline bound");c.warm_expert_timeline_limit=uint32_t(n); }
            else if (k == "warm_hip_schedule") c.warm_hip_schedule=v;
            else if (k == "warm_hip_fuse") c.warm_hip_fuse=v;
            else if (k == "warm_hip_gemv") c.warm_hip_gemv=v;
            else if (k == "warm_hip_dense") c.warm_hip_dense=v;
            else if (k == "warm_kernel_timing") { require(v=="0"||v=="1","warm_kernel_timing must be 0 or 1"); c.warm_kernel_timing=v=="1"; }

            else if (k == "warm_placement") c.warm_placement=v;
            else if (k == "warm_sycl_experts") c.warm_sycl_experts=v;
            else if (k == "warm_sycl_attention") c.warm_sycl_attention=v;
            else if (k == "warm_hip_fixed") c.warm_hip_fixed=std::stoull(v);
            else if (k == "warm_sycl_fixed") c.warm_sycl_fixed=std::stoull(v);
            else if (k == "warm_hip_bytes") { c.warm_hip_bytes=std::stoull(v); require(std::to_string(c.warm_hip_bytes)==v,"invalid warm_hip_bytes"); }
            else if (k == "warm_sycl_bytes") { c.warm_sycl_bytes=std::stoull(v); require(std::to_string(c.warm_sycl_bytes)==v,"invalid warm_sycl_bytes"); }
            else if (k == "cpu_timing") { require(v == "0" || v == "1", "cpu_timing must be 0 or 1"); c.cpu_timing = v == "1"; }
            else if (k == "expert_placement") c.expert_placement = v;
            else if (k == "q5k_fma") { require(v == "0" || v == "1", "q5k_fma must be 0 or 1"); c.q5k_fma = v == "1"; }
            else if (k == "fault_inject") {
                require(v == "load_weights" || v == "cpu_lane", "fault_inject must be load_weights or cpu_lane");
                c.fault_inject = v;
            } else throw std::runtime_error("unknown stage config key: " + k);
        }
        require(!c.pack.empty() && c.max_context > 0, "stage config needs pack and max_context");
        require(c.decode_timing_start < c.max_context, "decode timing start exceeds context");
        require(c.cpu_dispatch == "spawn-v1" || c.cpu_dispatch == "pool-v1", "unknown CPU dispatch version");
        require(c.hip_experts == "off" || c.hip_experts == "profile-v1" || c.hip_experts == "pooled-layer-v1" ||
                c.hip_experts == "pooled-layer-p1-control-v1", "unknown HIP expert policy");
        require(c.hip_expert_math == "per32-v1" || c.hip_expert_math == "cpu-q8k-v1", "unknown HIP expert arithmetic");
        require(c.hip_expert_math == "per32-v1" || c.hip_experts != "off", "Q8_K requires HIP residency");
        require((c.hip_experts == "off") == c.expert_profile.empty(), "profile-v1 requires expert_profile; off forbids it");
        require(c.expert_placement == "all-cpu",
                "only expert_placement=all-cpu is implemented (frozen placement file: TODO(row2))");
        require(c.warm_placement.empty() || c.warm_experts,"placement without warm mode");
        require(!c.warm_experts || (c.hip_experts=="pooled-layer-v1" && c.warm_hip_fixed>0 && c.warm_sycl_fixed>0),"warm experts require pooled HIP and explicit SYCL cap");
        require(c.warm_experts || (c.warm_sycl_bytes==0 && c.warm_hip_bytes==0),"bank cap without warm mode");
        require(!c.warm_experts || c.warm_measure,"warm serving requires event-time reporting");
        require(c.warm_sycl_experts=="serial-v1" || ((c.warm_sycl_experts=="grouped-v1" || c.warm_sycl_experts=="packed-strip-v1" || c.warm_sycl_experts=="mmvq-soa-v1" || c.warm_sycl_experts=="mmvq-soa-v2" || c.warm_sycl_experts=="mmvq-soa-v3") && c.warm_experts && c.warm_packed_gu),"unadmitted warm SYCL expert path");
        require(!c.warm_packed_gu || c.warm_experts,"packed gate/up without warm mode");
        require(c.warm_sycl_attention=="scalar-v1" || (c.warm_sycl_attention=="subgroup32-v1" && c.warm_experts),"unadmitted warm SYCL attention path");
        require(c.warm_dense_owner=="split-v1" || (c.warm_dense_owner=="hip-v1" && c.warm_measure),"unadmitted warm dense owner");
        require(c.warm_hip_schedule=="baseline-v1" || (c.warm_hip_schedule=="queued-v1" && c.warm_experts),"unadmitted HIP schedule");
        require(c.warm_expert_issue=="off" || c.warm_expert_issue=="early-v1","unknown expert issue mode");
        require((c.warm_expert_issue=="off" && !c.warm_expert_timeline_limit) ||
                (c.warm_experts && c.warm_dense_owner=="hip-v1" && c.warm_hip_schedule=="queued-v1" &&
                 c.warm_hip_dense=="off" && c.dump_s.empty()),
                "expert issue/timeline requires warm HIP cut0 queued, graphs off and no dumps");
        require(c.warm_hip_experts=="serial-v1" || ((c.warm_hip_experts=="layer-batch-v1" || c.warm_hip_experts=="soa-v2") && c.warm_experts &&
                c.warm_hip_schedule=="queued-v1" && c.hip_expert_math=="per32-v1" && c.warm_hip_dense=="off" && c.dump_s.empty()),
                "HIP layer batching requires warm experts, queued-v1, per32 math, no graphs or substep dumps");
        require(c.warm_engram=="off" || (c.warm_engram=="read-ahead-v1" && c.warm_experts && c.warm_dense_owner=="hip-v1"),"Engram read-ahead requires warm HIP dense ownership");
        require(!c.warm_kernel_timing || c.warm_measure,"kernel timing requires warm measurement");
        require(c.warm_hip_gemv=="off" || ((c.warm_hip_gemv=="q5-sums-v1" || c.warm_hip_gemv=="q5-sums-w4-v1") &&
                c.warm_experts && c.warm_dense_owner=="hip-v1" && c.warm_hip_schedule=="queued-v1" && c.warm_hip_dense=="off" && c.dump_s.empty()),
                "HIP dense GEMV requires warm HIP dense, queued-v1, no graphs or substep dumps");
        require(c.warm_hip_dense=="off" || (c.warm_hip_dense=="graphs-v1" && c.warm_experts &&
                c.warm_hip_schedule=="queued-v1" && !c.warm_kernel_timing && c.dump_s.empty()),
                "HIP dense graphs require warm experts, queued-v1, no substep dumps or per-kernel events");
        require(c.warm_hip_fuse=="off" || ((c.warm_hip_fuse=="small-v1" || c.warm_hip_fuse=="attention-v1" || c.warm_hip_fuse=="all-v1") &&
                c.warm_experts && c.warm_hip_schedule=="queued-v1" && c.warm_hip_dense=="off" && c.dump_s.empty()),
                "HIP fusion requires warm experts, queued-v1, no graphs or substep dumps");
#ifndef STRATA_FORK_WARM
        require(c.warm_hip_gemv=="off" && c.warm_engram=="off" && !c.warm_experts && c.warm_dense_owner=="split-v1" && !c.warm_kernel_timing && c.warm_hip_dense=="off" && c.warm_hip_fuse=="off","warm serving requires STRATA_FORK_WARM build");
#endif
        return c;
    }
};
// One-shot fault for the begin() recovery test; fires once per backend object.
struct Fault {
    std::string where; bool fired = false;
    void at(const char* point) {
        if (!fired && where == point) { fired = true; throw std::runtime_error(std::string("fault injection at ") + point); }
    }
};

// Writer for ds4_model.cu dump_sub_host's record format: char name[16], int64 il, p0, T, nbytes, bytes.
// Same names, same order as the reference so aiopt.experiments.strata_compare can bind one manifest.
class SubDump {
    std::FILE* f_ = nullptr;
    std::set<int> layers_;
    std::set<std::string> names_;
    std::mutex mu_;
public:
    SubDump() = default;
    SubDump(const SubDump&) = delete;
    SubDump& operator=(const SubDump&) = delete;
    void open(const Config& c) {
        if (c.dump_s.empty()) return;
        f_ = std::fopen(c.dump_s.c_str(), "wb");
        require(f_ != nullptr, "cannot open dump_s");
        layers_ = c.dump_layers.empty() ? std::set<int>{0, 1, 2, 3} : c.dump_layers;
        names_ = c.dump_names;
    }
    ~SubDump() { if (f_) std::fclose(f_); }
    bool on(uint32_t il, const char* name) const {
        return f_ && layers_.count(int(il)) && (names_.empty() || names_.count(name));
    }
    void write(const char* name, uint32_t il, uint64_t p0, int T, const void* host, size_t n) {
        if (!on(il, name)) return;
        std::lock_guard lk(mu_);
        char nm[16] = {};
        std::strncpy(nm, name, 15);
        const int64_t hdr[4] = {int64_t(il), int64_t(p0), T, int64_t(n)};
        require(std::fwrite(nm, 1, 16, f_) == 16 && std::fwrite(hdr, 8, 4, f_) == 4 &&
                    (n == 0 || std::fwrite(host, 1, n, f_) == n) && std::fflush(f_) == 0,
                "dump_s write failed");
    }
};

// Host residual / boundary layout (stage README): full F32 residual [hc,dim], then each Plan::crossing()
// edge's full-capacity snapshot in order. Valid rows INCLUDING this step come from the position only
// (PARITY.md; row34 review finding 6), never from a mutable per-buffer counter.
struct BoundaryLayout {
    struct Slot { stage::Edge edge; uint64_t offset; };
    uint64_t residual = 0, total = 0;
    std::vector<Slot> slots;
    static BoundaryLayout of(const stage::Plan& p) {
        BoundaryLayout b;
        // hip-v1 literal cut0: HIP owns every dense layer, so no stage boundary
        // is exported or imported; the wire packet is empty (Plan::wire_bytes()==0).
        if (p.hip_dense_only()) { require(p.crossing().empty(), "hip-v1 cut0 has no crossing edges"); return b; }
        b.residual = p.activation_bytes;
        uint64_t off = b.residual;
        for (const auto& e : p.crossing()) { b.slots.push_back({e, off}); off = stage::add(off, e.bytes); }
        b.total = off;
        require(b.total == p.wire_bytes(), "boundary layout differs from Plan::wire_bytes");
        return b;
    }
    std::vector<uint64_t> rows(const stage::Plan& p, uint64_t position) const {
        std::vector<uint64_t> out;
        for (const auto& s : slots) out.push_back(p.expected_rows(s.edge, position));
        return out;
    }
};

// Every crossing edge must map to a buffer this stage allocated with exactly the edge's extent
// (review #1: plan capacities vs the backend's own topology). `extent(edge)` returns that allocation's
// bytes, or 0 when the stage holds no buffer for it.
template <class Extent> void check_extents(const BoundaryLayout& bl, Extent extent) {
    for (const auto& s : bl.slots)
        require(extent(s.edge) == s.edge.bytes, "boundary edge extent differs from this stage's allocation (kind " +
                                                     std::to_string(uint32_t(s.edge.kind)) + ", producer " +
                                                     std::to_string(s.edge.producer) + ")");
}

// Stage-local identity of a range for begin(): what this backend owns.
struct Range {
    uint32_t first = 0, end = 0;
    bool owns(uint32_t il) const { return il >= first && il < end; }
    bool prefix() const { return first == 0; }
};
inline bool tail(const Range& r, const ModelMeta& g) { return r.end == uint32_t(g.nl); }

// Engram IDs of one layer for one token: Token.engram_ids sliced by the frozen Plan layout.
inline std::span<const uint64_t> engram_slice(const stage::Plan& p, const stage::Token& t, uint32_t il) {
    for (const auto& s : p.engram_layout)
        if (s.layer == il) {
            require(s.offset + s.count <= t.engram_ids.size(), "Engram IDs shorter than layout");
            return std::span<const uint64_t>(t.engram_ids).subspan(s.offset, s.count);
        }
    throw std::runtime_error("Engram layer missing from frozen layout");
}
}  // namespace strata::wiring
