// MIT; crosslane contributors. Plain-g++ host. Teacher-forced single-layer parity (README "Gate 1"):
// replays layer L of ONE stage DSO (HIP upstream or SYCL port) over positions 0..P-1, feeding at every
// position the residual and pre-mix that the AMD-only T=1 reference recorded entering layer L (DS4_DUMP_S),
// and the same token / Engram IDs. The layer's own state (window ring, compressor group, its KV/index
// source rows) evolves inside the backend from those identical inputs. The DSO writes its sub-steps in
// DS4_DUMP_S format; aiopt.experiments.strata_compare compares them with the reference dump.
//   strata-layer-parity --lib <dso> --backend hip|sycl --index 0 --pack <pack> --ctx 4096 --layer L
//     --tokens ids.txt --ref <reference DUMP_S file> --names a,b,c --out <stage DUMP_S file> [--positions P]
#include "../common.hpp"
#include "../host_plan.hpp"
#include "../loader_v2.hpp"
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

using namespace strata;
using namespace strata::wiring;

namespace {
struct Rec { std::string name; int64_t il, p0, T; std::vector<uint8_t> data; };
std::vector<Rec> read_dump_s(const std::string& path) {
    std::vector<Rec> v;
    std::FILE* f = std::fopen(path.c_str(), "rb");
    require(f != nullptr, "cannot open reference dump");
    for (;;) {
        char nm[16]; int64_t h[4];
        if (std::fread(nm, 1, 16, f) != 16) break;
        require(std::fread(h, 8, 4, f) == 4 && h[3] >= 0 && h[3] < (int64_t(1) << 32), "truncated dump header");
        Rec r{std::string(nm, strnlen(nm, 16)), h[0], h[1], h[2], std::vector<uint8_t>(size_t(h[3]))};
        require(std::fread(r.data.data(), 1, r.data.size(), f) == r.data.size(), "truncated dump payload");
        v.push_back(std::move(r));
    }
    std::fclose(f);
    return v;
}
}  // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> a;
    for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
    auto need = [&](const char* k) { auto it = a.find(k); if (it == a.end()) throw std::runtime_error(std::string("missing ") + k); return it->second; };
    try {
        const std::string pack = need("--pack");
        const int64_t ctx = std::stoll(need("--ctx"));
        const uint32_t L = uint32_t(std::stoul(need("--layer")));
        const bool hip = need("--backend") == "hip";
        auto ws = open_pack(pack);
        const auto& g = ws->meta();
        const Topology topo = topology(g, ctx);
        auto ngram = make_ngram(g);
        auto plan = layer_plan(*ws, ctx, L, *ngram);
        // Eligibility: only hc_pre may cross into L. selected_topk / candidates / earlier-source KV are not
        // in the reference dump yet (see README proposal P4: DS4_DUMP_S "topk_in"/"cand_in"/source rows).
        for (const auto& e : plan.crossing())
            require(e.kind == stage::Kind::hc_pre,
                    "layer reads state produced by an earlier layer that the reference dump does not carry; pick a "
                    "layer whose KV/index source is itself, that computes its own top-k and does not use candidates");
        const BoundaryLayout bl = BoundaryLayout::of(plan);
        require(bl.slots.size() == (L == 0 ? 0 : 1), "unexpected parity boundary slots");
        // Inputs entering L: residual before Engram (eg_h0 on Engram layers, else h_in) and pre_in.
        const bool engram = topo.layers[L].eg >= 0;
        const std::string hname = engram ? "eg_h0" : "h_in";
        std::map<int64_t, const Rec*> h_in, pre_in;
        const auto ref = read_dump_s(need("--ref"));
        for (const auto& r : ref) {
            if (r.il != int64_t(L) || r.T != 1) continue;
            if (r.name == hname) h_in[r.p0] = &r;
            if (r.name == "pre_in") pre_in[r.p0] = &r;
        }
        std::vector<int32_t> toks;
        { std::ifstream f(need("--tokens")); int32_t x; while (f >> x) toks.push_back(x); }
        size_t P = a.count("--positions") ? std::stoul(a["--positions"]) : toks.size();
        require(P > 0 && P <= toks.size(), "positions exceed the token stream");
        const std::string names = need("--names");
        const std::string cfg = "pack=" + pack + "\nmax_context=" + std::to_string(ctx) + "\nq5k_fma=0\ndump_s=" + need("--out") +
                                "\ndump_layers=" + std::to_string(L) + "\ndump_names=" + names + "\n";
        LoadedBackend be(need("--lib").c_str(), hip ? STRATA_STAGE_HIP : STRATA_STAGE_SYCL,
                         uint32_t(std::stoul(a.count("--index") ? a["--index"] : "0")), cfg);
        be.begin(plan, L, L + 1, 1);
        auto host = be.pinned(size_t(bl.total));
        std::atomic_bool cancel{false};
        for (size_t p = 0; p < P; ++p) {
            stage::Token t{1, p, p, toks[p], ngram->advance(toks[p])};
            if (L > 0) {
            require(h_in.count(int64_t(p)) && pre_in.count(int64_t(p)), "reference dump lacks layer inputs at a position");
            const auto& hr = h_in[int64_t(p)]->data; const auto& pr = pre_in[int64_t(p)]->data;
            require(hr.size() == bl.residual && pr.size() == bl.slots[0].edge.bytes, "reference input extents");
            auto* b = static_cast<uint8_t*>(host.data);
            std::memcpy(b, hr.data(), hr.size());
            std::memcpy(b + bl.slots[0].offset, pr.data(), pr.size());
            stage::Packet pk{2, plan.model_sha, plan.schema_sha, 1, p, p, 1, bl.total,
                             stage::checksum(host.data, size_t(bl.total)), plan.cut - 1, plan.crossing(), bl.rows(plan, p)};
            auto c = be.import_boundary(t, {static_cast<const std::byte*>(host.data), size_t(bl.total)}, pk);
            c->wait();
            } // L0 derives embedding/init_pre from the same frozen token.
            be.execute(t, cancel);
            be.prepare_commit(p);
            be.commit(p);
        }
        std::fprintf(stderr, "PASS layer=%u positions=%zu backend=%s\n", L, P, be.name());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
