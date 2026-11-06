// MIT; crosslane contributors. Plain-g++ host side: a stage::Backend proxy over a dlopened v2
// DSO, so the unchanged stage::Runtime coordinates the real model stages. No GPU headers here.
#pragma once
#include "../loader.hpp"
#include "abi_v2.h"
#include "codec.hpp"
#include <atomic>
#include <memory>

namespace strata::wiring {
// Context + library, shared by the backend proxy and every pinned allocation it handed out
// (review 2026-11-02 #7): destroy() and dlclose() run when the last owner goes.
struct BackendHandle {
    void* library = nullptr;
    strata_stage_backend_api_v2 api{};
    BackendHandle() = default;
    BackendHandle(const BackendHandle&) = delete;
    BackendHandle& operator=(const BackendHandle&) = delete;
    ~BackendHandle() {
        if (api.context && api.destroy) api.destroy(api.context);
        if (library) dlclose(library);
    }
};
class LoadedBackend final : public stage::Backend {
    std::shared_ptr<BackendHandle> h_;
    strata_stage_backend_api_v2& api_;
    std::string name_;
    struct Ready final : stage::Completion {
        size_t n; explicit Ready(size_t b) : n(b) {}
        void wait() override {}
        size_t bytes() const override { return n; }
    };
    static void check(int32_t rc, const strata_stage_error_v1& e) { stage::abi_check(rc, e); }
public:
    LoadedBackend(const char* path, uint32_t backend, uint32_t device, const std::string& config)
        : h_(std::make_shared<BackendHandle>()), api_(h_->api) {
        // RTLD_LOCAL keeps the two runtimes' symbols apart; v1 may be opened from the same file separately.
        h_->library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (!h_->library) throw std::runtime_error(dlerror());
        dlerror();
        auto f = reinterpret_cast<strata_stage_backend_factory_v2>(dlsym(h_->library, STRATA_STAGE_BACKEND_FACTORY));
        if (const char* why = dlerror()) throw std::runtime_error(why);
        strata_stage_error_v1 e{};
        check(f(STRATA_STAGE_ABI_V2, sizeof(api_), device, config.c_str(), &api_, &e), e);
        if (api_.abi_version != STRATA_STAGE_ABI_V2 || api_.struct_bytes != sizeof(api_) || api_.backend != backend ||
            api_.reserved || !api_.context || !api_.begin || !api_.pinned || !api_.free_pinned || !api_.execute ||
            !api_.export_boundary || !api_.committed_rows || !api_.import_boundary || !api_.logits ||
            !api_.prepare_commit || !api_.commit || !api_.invalidate || !api_.drain || !api_.describe || !api_.destroy)
            throw std::runtime_error("invalid v2 backend vtable/backend");
        name_ = describe();
    }
    LoadedBackend(const LoadedBackend&) = delete;
    LoadedBackend& operator=(const LoadedBackend&) = delete;
    ~LoadedBackend() override = default;  // the handle outlives us while pinned allocations remain
    std::string describe() {
        std::vector<char> text(1 << 16); strata_stage_error_v1 e{};
        check(api_.describe(api_.context, text.data(), text.size(), &e), e);
        return text.data();
    }
    const char* name() const override { return name_.c_str(); }
#ifdef STRATA_FORK_WARM
    template<class T> T* warm_host_partition()const {
        dlerror();auto fn=reinterpret_cast<T*(*)()>(dlsym(h_->library,"strata_warm_host_partition_v1"));
        if(const char* why=dlerror())throw std::runtime_error(why);return fn();
    }
    auto warm_stop_function()const -> void(*)(){
        dlerror();auto fn=reinterpret_cast<void(*)()>(dlsym(h_->library,"strata_warm_stop_workers_v1"));
        if(const char* why=dlerror())throw std::runtime_error(why);return fn;
    }
    std::vector<double> warm_dense_costs()const override{
        using Fn=int32_t(*)(void*,double*,uint64_t,uint64_t*,strata_stage_error_v1*);
        dlerror();auto fn=reinterpret_cast<Fn>(dlsym(h_->library,"strata_stage_warm_costs_v1"));if(const char* why=dlerror())throw std::runtime_error(why);
        std::vector<double> costs(4096);uint64_t count=0;strata_stage_error_v1 e{};
        check(fn(api_.context,costs.data(),costs.size(),&count,&e),e);if(count>costs.size())throw std::runtime_error("warm cost count");costs.resize(size_t(count));return costs;
    }
    int32_t sample(float temperature,float top_p,int top_k,float min_p,uint64_t seed,uint64_t counter)override{
        using Fn=int32_t(*)(void*,float,float,int32_t,float,uint64_t,uint64_t,int32_t*,strata_stage_error_v1*);
        dlerror();auto fn=reinterpret_cast<Fn>(dlsym(h_->library,"strata_stage_sample_warm_v1"));
        if(const char* why=dlerror())throw std::runtime_error(why);
        int32_t token=-1;strata_stage_error_v1 e{};
        check(fn(api_.context,temperature,top_p,top_k,min_p,seed,counter,&token,&e),e);return token;
    }
#endif
    void begin(const stage::Plan& p, uint32_t first, uint32_t end, uint64_t request) override {
        auto b = codec::encode(p); strata_stage_error_v1 e{};
        check(api_.begin(api_.context, b.data(), b.size(), first, end, request, &e), e);
        name_ = describe();  // measured allocation report (review #3)
    }
    stage::HostMemory pinned(size_t n) override {
        void* p = nullptr; strata_stage_error_v1 e{};
        check(api_.pinned(api_.context, n, &p, &e), e);
        auto h = h_;  // shared ownership: context and DSO stay alive until this allocation is freed
        return {p, n, [h](void* x) { strata_stage_error_v1 e2{}; h->api.free_pinned(h->api.context, x, &e2); }};
    }
    void execute(const stage::Token& t, const std::atomic_bool& cancel) override {
        auto b = codec::encode(t); strata_stage_error_v1 e{};
        check(api_.execute(api_.context, b.data(), b.size(), &cancel, &e), e);
    }
    std::unique_ptr<stage::Completion> export_boundary(const stage::Token& t, std::span<std::byte> h) override {
        auto b = codec::encode(t); uint64_t n = 0; strata_stage_error_v1 e{};
        check(api_.export_boundary(api_.context, b.data(), b.size(), h.data(), h.size(), &n, &e), e);
        return std::make_unique<Ready>(size_t(n));
    }
    std::vector<uint64_t> boundary_committed_rows(const stage::Token& t) override {
        auto b = codec::encode(t); std::vector<uint64_t> rows(4096); uint64_t n = 0; strata_stage_error_v1 e{};
        check(api_.committed_rows(api_.context, b.data(), b.size(), rows.data(), rows.size(), &n, &e), e);
        rows.resize(size_t(n)); return rows;
    }
    std::unique_ptr<stage::Completion> import_boundary(const stage::Token& t, std::span<const std::byte> h,
                                                       const stage::Packet& p) override {
        auto b = codec::encode(t); auto pk = codec::encode(p); uint64_t n = 0; strata_stage_error_v1 e{};
        check(api_.import_boundary(api_.context, b.data(), b.size(), h.data(), h.size(), pk.data(), pk.size(), &n, &e), e);
        return std::make_unique<Ready>(size_t(n));
    }
    std::vector<float> logits(const stage::Token& t) override {
        auto b = codec::encode(t); std::vector<float> out(size_t(1) << 20); uint64_t n = 0; strata_stage_error_v1 e{};
        check(api_.logits(api_.context, b.data(), b.size(), out.data(), out.size(), &n, &e), e);
        out.resize(size_t(n)); return out;
    }
    void prepare_commit(uint64_t s) override { strata_stage_error_v1 e{}; check(api_.prepare_commit(api_.context, s, &e), e); }
    void commit(uint64_t s) override { strata_stage_error_v1 e{}; check(api_.commit(api_.context, s, &e), e); }
    void invalidate() noexcept override { api_.invalidate(api_.context); }
    void drain() noexcept override { api_.drain(api_.context); }
};
}  // namespace strata::wiring
