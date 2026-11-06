// MIT; crosslane contributors. Compiled independently in each toolchain next to the UNCHANGED
// ../abi_plugin.cpp (v1 transport). Exposes the runtime's stage::Backend as strata_stage_open_backend_v2.
#include "abi_v2.h"
#include "codec.hpp"
#include "common.hpp"
#include <atomic>
#include <cstdio>
#include <map>

namespace strata::wiring {
// Implemented by exactly one of sycl_backend.cpp / hip_backend.cpp in each DSO.
std::unique_ptr<stage::Backend> make_model_backend(uint32_t device_index, const Config&);
}

using namespace strata;
using namespace strata::wiring;
namespace {
static_assert(sizeof(std::atomic_bool) == 1 && std::atomic_bool::is_always_lock_free,
              "cancel flag crosses the ABI as a 1-byte lock-free atomic");
struct Context {
    std::unique_ptr<stage::Backend> backend;
    std::map<void*, stage::HostMemory> hosts;
    stage::Plan plan;
    bool begun = false, failed = false;
    void live() const { stage::require(begun && !failed, "stage backend not begun or poisoned"); }
};
int32_t fail(strata_stage_error_v1* e, const char* s) noexcept {
    if (e) { e->code = 1; std::snprintf(e->message, sizeof(e->message), "%s", s); }
    return 1;
}
template <class F> int32_t guard(void* p, strata_stage_error_v1* e, F f) noexcept {
    if (!e) return 1;
    *e = {};
    if (!p) return fail(e, "null context");
    auto& c = *static_cast<Context*>(p);
    try { f(c); return 0; }
    catch (const std::exception& x) { c.failed = true; c.backend->invalidate(); c.backend->drain(); return fail(e, x.what()); }
    catch (...) { c.failed = true; c.backend->invalidate(); c.backend->drain(); return fail(e, "unknown native exception"); }
}
stage::Token token_of(const uint8_t* b, uint64_t n) { return codec::decode_token(b, size_t(n)); }

int32_t begin(void* p, const uint8_t* plan, uint64_t n, uint32_t first, uint32_t end, uint64_t request,
              strata_stage_error_v1* e) noexcept {
    return guard(p, e, [&](Context& c) {
        c.failed = false; c.begun = false;
        c.plan = codec::decode_plan(plan, size_t(n));
        c.backend->begin(c.plan, first, end, request);
        c.begun = true;
    });
}
int32_t pinned(void* p, uint64_t n, void** out, strata_stage_error_v1* e) noexcept {
    if (out) *out = nullptr;
    return guard(p, e, [&](Context& c) {
        stage::require(out && n && n <= SIZE_MAX, "invalid pinned request");
        auto h = c.backend->pinned(size_t(n));
        stage::require(h.data && h.bytes == n, "backend returned a short pinned allocation");
        void* raw = h.data; c.hosts.emplace(raw, std::move(h)); *out = raw;
    });
}
int32_t free_pinned(void* p, void* h, strata_stage_error_v1* e) noexcept {
    return guard(p, e, [&](Context& c) {
        stage::require(c.hosts.count(h), "foreign pinned allocation");
        c.backend->drain(); c.hosts.erase(h);
    });
}
int32_t execute(void* p, const uint8_t* t, uint64_t tn, const void* cancel, strata_stage_error_v1* e) noexcept {
    return guard(p, e, [&](Context& c) {
        c.live(); stage::require(cancel != nullptr, "null cancel flag");
        c.backend->execute(token_of(t, tn), *static_cast<const std::atomic_bool*>(cancel));
    });
}
int32_t export_boundary(void* p, const uint8_t* t, uint64_t tn, void* h, uint64_t hn, uint64_t* written,
                        strata_stage_error_v1* e) noexcept {
    if (written) *written = 0;
    return guard(p, e, [&](Context& c) {
        c.live();
        auto it = c.hosts.find(h);
        stage::require(written && it != c.hosts.end() && hn <= it->second.bytes, "export needs this context's pinned base");
        auto d = c.backend->export_boundary(token_of(t, tn), {static_cast<std::byte*>(h), size_t(hn)});
        stage::require(bool(d), "missing completion"); d->wait(); *written = d->bytes();
    });
}
int32_t committed_rows(void* p, const uint8_t* t, uint64_t tn, uint64_t* rows, uint64_t cap, uint64_t* count,
                       strata_stage_error_v1* e) noexcept {
    if (count) *count = 0;
    return guard(p, e, [&](Context& c) {
        c.live();
        auto v = c.backend->boundary_committed_rows(token_of(t, tn));
        stage::require(count && (v.empty() || rows) && v.size() <= cap, "committed row buffer too small");
        std::copy(v.begin(), v.end(), rows); *count = v.size();
    });
}
int32_t import_boundary(void* p, const uint8_t* t, uint64_t tn, const void* h, uint64_t hn, const uint8_t* pk,
                        uint64_t pn, uint64_t* read, strata_stage_error_v1* e) noexcept {
    if (read) *read = 0;
    return guard(p, e, [&](Context& c) {
        c.live();
        auto it = c.hosts.find(const_cast<void*>(h));
        stage::require(read && it != c.hosts.end() && hn <= it->second.bytes, "import needs this context's pinned base");
        auto d = c.backend->import_boundary(token_of(t, tn), {static_cast<const std::byte*>(h), size_t(hn)},
                                            codec::decode_packet(pk, size_t(pn)));
        stage::require(bool(d), "missing completion"); d->wait(); *read = d->bytes();
    });
}
int32_t logits(void* p, const uint8_t* t, uint64_t tn, float* out, uint64_t cap, uint64_t* count,
               strata_stage_error_v1* e) noexcept {
    if (count) *count = 0;
    return guard(p, e, [&](Context& c) {
        c.live();
        auto v = c.backend->logits(token_of(t, tn));
        stage::require(count && out && v.size() <= cap, "logit buffer too small");
        std::copy(v.begin(), v.end(), out); *count = v.size();
    });
}
int32_t prepare_commit(void* p, uint64_t step, strata_stage_error_v1* e) noexcept {
    return guard(p, e, [&](Context& c) { c.live(); c.backend->prepare_commit(step); });
}
int32_t commit(void* p, uint64_t step, strata_stage_error_v1* e) noexcept {
    return guard(p, e, [&](Context& c) { c.live(); c.backend->commit(step); });
}
void invalidate(void* p) { if (p) { auto& c = *static_cast<Context*>(p); c.failed = true; c.backend->invalidate(); } }
void drain(void* p) { if (p) static_cast<Context*>(p)->backend->drain(); }
int32_t describe(void* p, char* out, uint64_t n, strata_stage_error_v1* e) noexcept {
    return guard(p, e, [&](Context& c) {
        std::string s = c.backend->name();
        stage::require(out && n > s.size(), "description buffer too small");
        std::memcpy(out, s.c_str(), s.size() + 1);
    });
}
void destroy(void* p) {
    try {
        auto* c = static_cast<Context*>(p);
        if (c) { c->backend->invalidate(); c->backend->drain(); c->hosts.clear(); delete c; }
    } catch (...) {}
}
}  // namespace

extern "C" int32_t strata_stage_open_backend_v2(uint32_t version, uint32_t bytes, uint32_t index, const char* config,
                                                strata_stage_backend_api_v2* out, strata_stage_error_v1* e) {
    // Review 2026-11-02 #8: validate pointers, version and the caller's struct size BEFORE writing *out, so an
    // older/smaller caller struct is never overrun. *e is the fixed v1 error struct and may be written.
    if (!e) return 1;
    *e = {};
    if (!out) return fail(e, "null factory output");
    if (version != STRATA_STAGE_ABI_V2 || bytes != sizeof(*out)) return fail(e, "unsupported backend ABI version/size");
    *out = {};
    try {
        auto c = std::make_unique<Context>();
        c->backend = make_model_backend(index, Config::parse(config));
#if defined(STRATA_STAGE_PLUGIN_HIP)
        constexpr auto backend = STRATA_STAGE_HIP;
#else
        constexpr auto backend = STRATA_STAGE_SYCL;
#endif
        *out = {STRATA_STAGE_ABI_V2, sizeof(*out), backend, 0, c.get(), begin, pinned, free_pinned, execute,
                export_boundary, committed_rows, import_boundary, logits, prepare_commit, commit, invalidate,
                drain, describe, destroy};
        c.release();
        return 0;
    } catch (const std::exception& x) { return fail(e, x.what()); }
    catch (...) { return fail(e, "unknown factory exception"); }
}

#ifdef STRATA_FORK_WARM
// Additive symbol; the frozen v2 vtable and factory signatures remain intact.
extern "C" __attribute__((visibility("default"))) int32_t strata_stage_sample_warm_v1(
 void* p,float temperature,float top_p,int32_t top_k,float min_p,uint64_t seed,uint64_t counter,
 int32_t* token,strata_stage_error_v1* e) noexcept {
 return guard(p,e,[&](Context& c){c.live();stage::require(token!=nullptr,"null sample output");
  *token=c.backend->sample(temperature,top_p,top_k,min_p,seed,counter);});
}
#endif

#ifdef STRATA_FORK_WARM
extern "C" __attribute__((visibility("default"))) int32_t strata_stage_warm_costs_v1(void* p,double* costs,uint64_t capacity,uint64_t* count,strata_stage_error_v1* e)noexcept{
 return guard(p,e,[&](Context& c){c.live();auto data=c.backend->warm_dense_costs();stage::require(costs&&count&&data.size()<=capacity,"warm cost buffer");std::copy(data.begin(),data.end(),costs);*count=data.size();});
}
#endif
