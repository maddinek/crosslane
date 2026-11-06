// MIT; crosslane contributors. Plain-g++ host, no GPU work: the v2 factory must validate pointers,
// version and caller struct size BEFORE writing *out (review 2026-11-02 #8).
//   strata-abi-v2-canary <libstrata-stage-{sycl,hip}-model.so>
#include "../abi_v2.h"
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) { std::fprintf(stderr, "usage: %s <stage DSO>\n", argv[0]); return 2; }
    void* lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "FAIL dlopen: %s\n", dlerror()); return 1; }
    auto f = reinterpret_cast<strata_stage_backend_factory_v2>(dlsym(lib, STRATA_STAGE_BACKEND_FACTORY));
    if (!f) { std::fprintf(stderr, "FAIL: no %s\n", STRATA_STAGE_BACKEND_FACTORY); return 1; }
    const size_t n = sizeof(strata_stage_backend_api_v2);
    int bad = 0;
    auto canary = [&](uint32_t version, uint32_t bytes, const char* what) {
        std::vector<unsigned char> buf(n + 64, 0xA5);
        strata_stage_error_v1 e{};
        const int32_t rc = f(version, bytes, 0, "pack=/nonexistent\nmax_context=16\n",
                             reinterpret_cast<strata_stage_backend_api_v2*>(buf.data()), &e);
        bool intact = true;
        for (unsigned char c : buf) intact = intact && c == 0xA5;
        std::printf("%-28s rc=%d intact=%d msg=%s\n", what, rc, int(intact), e.message);
        if (rc == 0 || !intact) ++bad;
    };
    canary(STRATA_STAGE_ABI_V2, uint32_t(n - 8), "short struct");
    canary(STRATA_STAGE_ABI_V2, uint32_t(n + 8), "long struct");
    canary(STRATA_STAGE_ABI_V2 + 1, uint32_t(n), "future version");
    {   // null output: must fail, error filled
        strata_stage_error_v1 e{};
        const int32_t rc = f(STRATA_STAGE_ABI_V2, uint32_t(n), 0, "", nullptr, &e);
        std::printf("%-28s rc=%d msg=%s\n", "null out", rc, e.message);
        if (rc == 0 || e.code == 0) ++bad;
    }
    {   // valid version/size, invalid config: fails AFTER the checks, *out must then be zeroed (no stale pointers)
        std::vector<unsigned char> buf(n, 0xA5);
        strata_stage_error_v1 e{};
        const int32_t rc = f(STRATA_STAGE_ABI_V2, uint32_t(n), 0, "bogus_key=1\n",
                             reinterpret_cast<strata_stage_backend_api_v2*>(buf.data()), &e);
        bool zero = true;
        for (unsigned char c : buf) zero = zero && c == 0;
        std::printf("%-28s rc=%d zeroed=%d msg=%s\n", "bad config", rc, int(zero), e.message);
        if (rc == 0 || !zero) ++bad;
    }
    dlclose(lib);
    std::printf(bad ? "FAIL %d\n" : "PASS\n", bad);
    return bad ? 1 : 0;
}
