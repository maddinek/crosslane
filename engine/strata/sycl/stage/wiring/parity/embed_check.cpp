// MIT; crosslane contributors. Built in the HIP (upstream top-level) configure with the shared row-2
// CPU library: compares WeightSource::embedding_bf16 (our Q5_K decode + upstream rounding) against upstream's
// own gg::gg_row_to_bf16 (ds4_gg_cpu.cpp:266, ggml to_float) for EVERY row of embed.weight, for both
// contraction variants. Exactly the matching variant is frozen as stage config q5k_fma=0|1. CPU only.
//   strata-wiring-embed-check <pack>
#include "../weight_source.hpp"
#include "ds4_gguf_cpu.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <pack>\n", argv[0]); return 2; }
    try {
        auto ws0 = strata::wiring::open_pack(argv[1], {false});
        auto ws1 = strata::wiring::open_pack(argv[1], {true});
        const auto& g = ws0->meta();
        const auto t = ws0->require_tensor("embed.weight");
        if (t.dtype != "Q5_K") { std::printf("embed.weight is %s, not Q5_K: nothing to check\n", t.dtype.c_str()); return 0; }
        const size_t rb = size_t(g.dim) / 256 * 176;
        std::vector<std::byte> row(rb);
        std::vector<uint16_t> ref(size_t(g.dim)), a(size_t(g.dim)), b(size_t(g.dim));
        uint64_t bad0 = 0, bad1 = 0;
        long first0 = -1, first1 = -1;
        for (int r = 0; r < g.vocab; ++r) {
            ws0->read(t.name, uint64_t(r) * rb, row);
            if (!strata::ds4::gguf::gg_row_to_bf16(13 /* GGML_TYPE_Q5_K */, reinterpret_cast<const uint8_t*>(row.data()), g.dim, ref.data()))
                throw std::runtime_error("upstream gg_row_to_bf16 refused Q5_K (STRATA_NATIVE_EXPERTS off?)");
            ws0->embedding_bf16(r, a);
            ws1->embedding_bf16(r, b);
            for (int i = 0; i < g.dim; ++i) {
                if (a[size_t(i)] != ref[size_t(i)]) { ++bad0; if (first0 < 0) first0 = r; }
                if (b[size_t(i)] != ref[size_t(i)]) { ++bad1; if (first1 < 0) first1 = r; }
            }
        }
        std::printf("rows=%d dim=%d mismatches: q5k_fma=0 -> %llu (first row %ld), q5k_fma=1 -> %llu (first row %ld)\n",
                    g.vocab, g.dim, (unsigned long long) bad0, first0, (unsigned long long) bad1, first1);
        if (bad0 == 0 || bad1 == 0) { std::printf("PASS freeze q5k_fma=%d\n", bad0 == 0 ? 0 : 1); return 0; }
        std::printf("FAIL: neither variant reproduces upstream\n");
        return 1;
    } catch (const std::exception& e) {
        std::printf("FAIL: %s\n", e.what());
        return 1;
    }
}
