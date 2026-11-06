// MIT; crosslane contributors. CPU-testable P1 placement/admission.
// No device API and no weight reads here. P0's default plan is unchanged.
#pragma once
#include "common.hpp"
#include "../../moe/format.hpp"
#include <array>
#include <fstream>
#include <numeric>

namespace strata::wiring {
constexpr uint64_t kHipExpertWorkspaceReserve = 256ull << 20;
// smoke-a3: max driver-used minus completed P0 arena = 972378880 B.
// Round UP to 1 GiB, then leave another fixed 256 MiB unused. ROCr's
// fragment BlockAllocator rounds separate matrices to 2 MiB backing blocks.
// Count those blocks as well: a flat context reserve alone misses thousands
// of 2.58--3.41 MB matrices backed by separate 4 MiB allocations.
constexpr uint64_t kHipDriverBaseReserve = 1ull << 30;
constexpr uint64_t kHipDriverSafetyMargin = 256ull << 20;
constexpr uint64_t kHipDriverGranule = 2ull << 20;
inline uint64_t expert_align(uint64_t n) { return stage::add(n, 255) / 256 * 256; }
inline uint64_t expert_driver_charge(uint64_t n) {
    return stage::add(n, kHipDriverGranule-1) / kHipDriverGranule * kHipDriverGranule;
}
struct ResidentExpert {
    uint32_t layer = 0;
    int expert = 0, slot = 0;
    std::array<TensorInfo, 3> matrices;
    std::array<uint64_t, 3> offsets{}; // pooled: nonowning, relative to this layer's slab
    uint64_t payload = 0, allocated = 0, driver_charge = 0;
};
struct HipResidency {
    bool cpu_q8k = false;
    bool all_layers = false;
    uint64_t bank_reservation = 0;
    std::string policy = "off";
    std::vector<std::vector<int>> slots; // [layer][expert], -1 means CpuLane
    std::vector<ResidentExpert> experts; // layer then device-slot order
    std::vector<uint64_t> layer_allocated, layer_driver_charge;
    uint64_t payload = 0, allocated = 0, driver_charge = 0, workspace = 0, profile_tag = 0, slots_tag = 0, layout_tag = 0;
    int slot(uint32_t l, int e) const {
        require(l < slots.size() && e >= 0 && size_t(e) < slots[l].size(), "resident request range");
        return slots[l][size_t(e)];
    }
};
inline bool pooled_policy(const std::string& policy) {
    return policy=="pooled-layer-v1" || policy=="pooled-layer-p1-control-v1";
}
inline uint64_t resident_observed_charge(const std::string& policy,uint64_t baseline_free,uint64_t free,uint64_t total) {
    require(free<=total && baseline_free<=total && free<=baseline_free,"device charge changed during resident startup");
    // P1's archived ledger remains reproducible. Pooled admission uses the full
    // driver-wide usage, including idle; its measured 1GiB base includes idle.
    return pooled_policy(policy)?total-free:baseline_free-free;
}
inline uint64_t slab_extent(uint64_t unit, uint64_t count) {
    require(!count || unit <= UINT64_MAX / count, "resident slab overflow");
    return unit * count;
}
inline uint64_t residency_layout_tag(const HipResidency& r) {
    // Explicit little-endian words; independent of struct padding/ABI.
    std::vector<uint8_t> raw;
    auto word = [&](uint64_t v) { for (int i=0;i<8;++i) raw.push_back(uint8_t(v>>(8*i))); };
    for (size_t l=0;l<r.slots.size();++l) { word(l); word(r.layer_allocated.at(l)); word(r.layer_driver_charge.at(l)); }
    for (const auto& e:r.experts) {
        word(e.layer); word(uint64_t(e.expert)); word(uint64_t(e.slot));
        for (size_t role=0;role<3;++role) { word(e.offsets[role]); word(e.matrices[role].bytes); }
    }
    return stage::checksum(raw.data(),raw.size());
}
inline void validate_pooled_layout(const HipResidency& r) {
    require(pooled_policy(r.policy) && r.layer_allocated.size()==r.slots.size() &&
            r.layer_driver_charge.size()==r.slots.size(), "pooled layout geometry");
    std::vector<uint64_t> ends(r.slots.size(),0);
    std::vector<int> counts(r.slots.size(),0);
    uint64_t payload=0, allocated=0, charge=0;
    uint32_t previous=0;
    for (const auto& e:r.experts) {
        require((e.layer>0 || r.all_layers) && e.layer>=previous && e.layer<ends.size() && e.slot==counts[e.layer]++ && r.slot(e.layer,e.expert)==e.slot,
                "pooled slot identity/contiguity");
        previous=e.layer;
        for (size_t role=0;role<3;++role) {
            const auto offset=e.offsets[role], bytes=e.matrices[role].bytes;
            require(bytes>0 && offset%256==0 && offset==ends[e.layer], "pooled offset alignment/overlap");
            ends[e.layer]=stage::add(offset,expert_align(bytes));
            require(ends[e.layer]<=r.layer_allocated[e.layer], "pooled view exceeds slab");
            payload=stage::add(payload,bytes);
        }
    }
    for(size_t l=0;l<ends.size();++l) {
        require(ends[l]==r.layer_allocated[l] && ends[l]%256==0 &&
                r.layer_driver_charge[l]==expert_driver_charge(ends[l]), "pooled slab extent/charge");
        int mapped=0;for(int s:r.slots[l])if(s>=0){require(s<counts[l],"pooled extra slot");++mapped;}
        require(mapped==counts[l],"pooled map extent");
        allocated=stage::add(allocated,ends[l]); charge=stage::add(charge,r.layer_driver_charge[l]);
    }
    require(payload==r.payload && allocated==r.allocated && charge==r.driver_charge &&
            residency_layout_tag(r)==r.layout_tag, "pooled totals/layout tag");
}
inline ResidentExpert expert_metadata(const WeightSource& ws, uint32_t l, int e) {
    namespace m = strata::ds4::moe;
    const auto& g = ws.meta();
    ResidentExpert x; x.layer = l; x.expert = e;
    const char* roles[] = {"w1", "w3", "w2"};
    for (size_t r = 0; r < 3; ++r) {
        const int n = r < 2 ? g.ff : g.dim, k = r < 2 ? g.dim : g.ff;
        auto t = ws.require_tensor(layer_prefix(l) + "ffn.experts." + std::to_string(e) + "." + roles[r] + ".weight");
        auto ty = m::type(t.dtype); (void)m::ggml_type(ty);
        require(t.shape == std::vector<int64_t>{n, k} &&
                t.bytes == m::product(size_t(n), m::row_bytes(ty, k)), "resident expert shape/extent");
        x.payload = stage::add(x.payload, t.bytes);
        x.allocated = stage::add(x.allocated, expert_align(t.bytes));
        x.driver_charge = stage::add(x.driver_charge, expert_driver_charge(expert_align(t.bytes)));
        x.matrices[r] = std::move(t);
    }
    require(x.matrices[0].dtype == x.matrices[1].dtype, "resident gate/up formats differ");
    return x;
}
inline std::vector<uint32_t> expert_profile(const Config& cfg, const ModelMeta& g, uint64_t& tag) {
    // Retained target-only or target+three-MTP profile. Bounded before allocation.
    require(g.nl > 1 && g.n_exp > 0, "profile geometry");
    const size_t words = strata::ds4::moe::product(size_t(g.nl), size_t(g.n_exp));
    const size_t extra = strata::ds4::moe::product(size_t(3), size_t(g.n_exp));
    std::ifstream f(cfg.expert_profile, std::ios::binary | std::ios::ate);
    require(bool(f), "cannot open expert_profile");
    const auto size = f.tellg();
    require(size >= 0 && (uint64_t(size) == words * 4 || uint64_t(size) == (words + extra) * 4), "expert_profile extent");
    std::vector<uint8_t> raw(size_t(size), uint8_t{});
    f.seekg(0); f.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size()));
    require(bool(f) && f.peek() == std::char_traits<char>::eof(), "expert_profile short/changed read");
    tag = stage::checksum(raw.data(), raw.size()); // internal tag; freeze binds SHA256
    std::vector<uint32_t> out(words);
    for (size_t i = 0; i < words; ++i) out[i] = strata::ds4::moe::u32(raw.data() + 4 * i);
    return out;
}
inline HipResidency select_hip_residency(const WeightSource& ws, const Config& cfg, const stage::Plan& base) {
    HipResidency out;
    if (cfg.hip_experts == "off") return out; // no profile or metadata reads on P0
    require((cfg.hip_experts == "profile-v1" || pooled_policy(cfg.hip_experts)) && !cfg.expert_profile.empty(), "HIP expert config");
    out.policy=cfg.hip_experts;
    const bool pooled=out.policy=="pooled-layer-v1";
    out.cpu_q8k = cfg.hip_expert_math == "cpu-q8k-v1";
    const auto& g = ws.meta();
    require(base.cut == 1 && base.layers == uint32_t(g.nl), "P1 HIP-only requires cut1");
    const auto used = base.footprint(base.hip, base.cut, base.layers);
    constexpr auto reserve = kHipExpertWorkspaceReserve + kHipDriverBaseReserve + kHipDriverSafetyMargin;
    require(used <= base.hip.cap && base.hip.cap - used >= reserve, "no HIP expert driver/workspace headroom");
    const uint64_t room = base.hip.cap - used - reserve;
    const auto counts = expert_profile(cfg, g, out.profile_tag);
    out.workspace = kHipExpertWorkspaceReserve;
    out.slots.assign(size_t(g.nl), std::vector<int>(size_t(g.n_exp), -1));
    out.layer_allocated.assign(size_t(g.nl), 0);
    out.layer_driver_charge.assign(size_t(g.nl), 0);
    std::vector<uint64_t> unit(size_t(g.nl), 0);
    uint64_t total = 0;
    // Validate all metadata before allocating or reading a matrix. Upstream's
    // equal-fraction rule requires uniform aligned slot size within each layer.
    for (uint32_t l = base.cut; l < base.layers; ++l) {
        for (int e = 0; e < g.n_exp; ++e) {
            auto x = expert_metadata(ws, l, e);
            const auto cost=pooled?x.allocated:x.driver_charge;
            if (!e) unit[l] = cost;
            require(unit[l] == cost, "nonuniform resident slot size within layer");
            total = stage::add(total, cost);
        }
    }
    require(total > 0, "empty resident expert inventory");
    auto cost=[&](uint32_t l,int n){auto b=slab_extent(unit[l],uint64_t(n));return pooled?expert_driver_charge(b):b;};
    int floor=0;
    if(pooled) {
        // Greatest common integer floor under nonlinear per-layer rounding.
        for(int candidate=1;candidate<=g.n_exp;++candidate) {
            uint64_t sum=0;for(uint32_t l=base.cut;l<base.layers;++l)sum=stage::add(sum,cost(l,candidate));
            if(sum>room)break;
            floor=candidate;
        }
    } else {
        const uint64_t available=std::min(room,total);
        const uint64_t numerator=strata::ds4::moe::product(size_t(available),size_t(g.n_exp));
        floor=int(numerator/total);
    }
    std::vector<int> n(size_t(g.nl), floor);
    for (uint32_t l = base.cut; l < base.layers; ++l)
        out.driver_charge = stage::add(out.driver_charge, cost(l,n[l]));
    for (uint32_t l = base.cut; l < base.layers; ++l)
        while (n[l] < g.n_exp) {
            const auto delta=cost(l,n[l]+1)-cost(l,n[l]);
            if(delta>room-out.driver_charge)break;
            ++n[l];out.driver_charge=stage::add(out.driver_charge,delta);
        }
    for (uint32_t l = base.cut; l < base.layers; ++l) {
        std::vector<int> rank(size_t(g.n_exp)); std::iota(rank.begin(), rank.end(), 0);
        std::stable_sort(rank.begin(), rank.end(), [&](int a, int b) {
            return counts[size_t(l) * size_t(g.n_exp) + size_t(a)] > counts[size_t(l) * size_t(g.n_exp) + size_t(b)];
        }); // ties keep ascending ID; no spare/adaptation/profile mutation
        for (int s = 0; s < n[l]; ++s) {
            const int e = rank[size_t(s)];
            auto x = expert_metadata(ws, l, e); x.slot = s;
            out.slots[l][size_t(e)] = s;
            uint64_t offset=out.layer_allocated[l];
            for(size_t role=0;role<3;++role){x.offsets[role]=offset;offset=stage::add(offset,expert_align(x.matrices[role].bytes));}
            out.layer_allocated[l] = stage::add(out.layer_allocated[l],x.allocated); out.payload = stage::add(out.payload,x.payload);
            out.allocated = stage::add(out.allocated,x.allocated);
            out.layer_driver_charge[l] = stage::add(out.layer_driver_charge[l],x.driver_charge);
            out.experts.push_back(std::move(x));
        }
        if(pooled_policy(out.policy))out.layer_driver_charge[l]=expert_driver_charge(out.layer_allocated[l]);
    }
    std::vector<int32_t> flat;
    for (const auto& row : out.slots) flat.insert(flat.end(), row.begin(), row.end());
    out.slots_tag = stage::checksum(flat.data(), flat.size() * sizeof(int32_t));
    if(pooled_policy(out.policy)){
        out.driver_charge=0;for(auto b:out.layer_driver_charge)out.driver_charge=stage::add(out.driver_charge,b);
        out.layout_tag=residency_layout_tag(out);validate_pooled_layout(out);
    }
    return out;
}
inline void apply_hip_residency(stage::Plan& p, const HipResidency& r) {
    if (!r.workspace) return;
    p.hip.fixed = stage::add(p.hip.fixed, r.workspace + kHipDriverBaseReserve + kHipDriverSafetyMargin);
    if(r.all_layers)p.hip.fixed=stage::add(p.hip.fixed,r.bank_reservation);
    else for (uint32_t l = p.cut; l < p.layers; ++l) p.hip.layers[l] = stage::add(p.hip.layers[l], r.layer_driver_charge[l]);
    p.schema_sha += ";hip-experts=" + r.policy + ";profile=" + std::to_string(r.profile_tag) + ";slots=" + std::to_string(r.slots_tag);
    if(pooled_policy(r.policy))p.schema_sha += ";layout=" + std::to_string(r.layout_tag);
    p.validate();
}
struct ResidentPick { int expert; float weight; int slot; };
inline std::vector<ResidentPick> resident_picks(const HipResidency& r, uint32_t layer,
                                              std::span<const int> ids, std::span<const float> weights) {
    require(ids.size() == weights.size() && ids.size() <= size_t(kPicks), "resident picks extent");
    std::vector<ResidentPick> out;
    for (size_t i = 0; i < ids.size(); ++i) {
        require(std::isfinite(weights[i]), "nonfinite route weight");
        out.push_back({ids[i], weights[i], r.slot(layer, ids[i])});
    }
    std::sort(out.begin(), out.end(), [](auto a, auto b) { return a.expert < b.expert; });
    for (size_t i = 1; i < out.size(); ++i) require(out[i-1].expert != out[i].expert, "duplicate routed expert");
    return out;
}
} // namespace strata::wiring
