// MIT; crosslane contributors. Plain-g++ host, needs the GPU and the real row-2 / CPU lane (native host,
// guarded window). Fault injection inside begin() followed by a same-range retry (review 2026-11-02 #5):
// for each injection point a fresh backend must (1) fail begin() with the injected error, (2) succeed on a
// retry for the SAME range, (3) run one teacher-forced position. Exit 4 = blocked by a stub (not a crash).
//   strata-begin-retry-test --lib <dso> --backend hip|sycl --pack <pack> --ctx 4096 --layer L [--index 0]
#include "../common.hpp"
#include "../host_plan.hpp"
#include "../loader_v2.hpp"
#include <cstdio>
#include <map>

using namespace strata;
using namespace strata::wiring;

int main(int argc, char** argv) {
    std::map<std::string, std::string> a;
    for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
    auto need = [&](const char* k) { auto it = a.find(k); if (it == a.end()) throw std::runtime_error(std::string("missing ") + k); return it->second; };
    int failures = 0, blocked = 0;
    try {
        const std::string pack = need("--pack");
        const int64_t ctx = std::stoll(need("--ctx"));
        const uint32_t L = uint32_t(std::stoul(need("--layer")));
        const bool hip = need("--backend") == "hip";
        auto ws = open_pack(pack);
        auto ngram = make_ngram(ws->meta());
        const auto plan = layer_plan(*ws, ctx, L, *ngram);
        const BoundaryLayout bl = BoundaryLayout::of(plan);
        for (const char* point : {"load_weights", "cpu_lane"}) {
            const std::string cfg = "pack=" + pack + "\nmax_context=" + std::to_string(ctx) + "\nfault_inject=" + point + "\n";
            LoadedBackend be(need("--lib").c_str(), hip ? STRATA_STAGE_HIP : STRATA_STAGE_SYCL,
                             uint32_t(std::stoul(a.count("--index") ? a["--index"] : "0")), cfg);
            bool injected = false;
            try { be.begin(plan, L, L + 1, 1); }
            catch (const std::exception& e) { injected = std::string(e.what()).find("fault injection") != std::string::npos;
                                              std::printf("[%s] first begin failed: %s\n", point, e.what()); }
            if (!injected) { std::printf("[%s] FAIL: injected fault did not surface from begin()\n", point); ++failures; continue; }
            try { be.begin(plan, L, L + 1, 2); }
            catch (const std::exception& e) {
                const std::string m = e.what();
                std::printf("[%s] retry begin failed: %s\n", point, m.c_str());
                if (m.find("TODO(row2)") != std::string::npos) ++blocked; else ++failures;
                continue;
            }
            std::printf("[%s] retry begin ok: %s\n", point, be.describe().c_str());
            auto host = be.pinned(size_t(bl.total));
            std::memset(host.data, 0, host.bytes);
            const float pre[8] = {1.0f};
            std::memcpy(static_cast<uint8_t*>(host.data) + bl.slots.at(0).offset, pre, sizeof(pre));
            stage::Token t{2, 0, 0, 0, ngram->advance(0)};
            stage::Packet pk{2, plan.model_sha, plan.schema_sha, 2, 0, 0, 1, bl.total,
                             stage::checksum(host.data, size_t(bl.total)), plan.cut - 1, plan.crossing(), bl.rows(plan, 0)};
            std::atomic_bool cancel{false};
            be.import_boundary(t, {static_cast<const std::byte*>(host.data), size_t(bl.total)}, pk)->wait();
            be.execute(t, cancel);
            be.prepare_commit(0); be.commit(0);
            std::printf("[%s] PASS (one position executed after recovery)\n", point);
            ngram->reset();
        }
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("%s failures=%d blocked=%d\n", failures ? "FAIL" : blocked ? "BLOCKED" : "PASS", failures, blocked);
    return failures ? 1 : blocked ? 4 : 0;
}
