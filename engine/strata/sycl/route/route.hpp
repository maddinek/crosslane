// Strata c62d5e3 router contract; Project Maya contributors, MIT.
// Keep third_party/project-maya/{LICENSE,NOTICE.md} with distributions.
#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
namespace strata::ds4::route_port {
struct Routing {
    size_t tokens{}, experts{}, picks{};
    std::vector<int> ids;
    std::vector<float> weights; // compact [tokens,picks]; never renormalized by service
};
Routing route_cpu(std::span<const float> logits, size_t logit_stride,
                  std::span<const float> bias, size_t tokens, size_t full_picks, float route_scale);
enum class CombinePath { upstream_default, synchronous, x3 };
struct Req {
    uint64_t request{}, epoch{}, seq{};
    size_t layer{}, dim{};
    Routing route;
    std::vector<float> x, shared; // compact [tokens,dim], both required
    CombinePath combine_path = CombinePath::upstream_default; // default DS4_ASYNC_T=8
    bool stage_store = false; // pinned request policy; upstream zc_stage is decode-only
};
struct Resp {
    uint64_t request{}, epoch{}, seq{};
    std::vector<int> cpu;
    std::vector<float> part, combined;
    size_t resident_hits{}, store_hits{}, cpu_misses{};
};
// Every method returns only AFTER explicit copies/compute complete. Native
// adapters must wait_and_throw / synchronize their private runtime queue.
// No mapped-host mailboxes, system fences, device waits or foreign pointers.
class ExpertBackend {
public:
    virtual ~ExpertBackend() = default;
    virtual bool resident(size_t layer, int expert) const = 0;
    virtual bool store_contains(size_t layer, int expert) const = 0;
    virtual void stage(size_t layer, int expert) = 0;
    virtual std::vector<float> device(size_t layer, int expert, std::span<const float> x) = 0;
    virtual std::vector<float> cpu(size_t layer, int expert, std::span<const float> x) = 0;
    virtual void publish(const Resp&) = 0; // explicit response/output H2D before host ack
    virtual void drain() noexcept = 0;
};
class HostStore {
    std::map<std::pair<size_t,int>,std::vector<std::byte>> data_;
public:
    void insert(size_t layer, int expert, std::vector<std::byte> bytes, size_t expected);
    bool contains(size_t layer, int expert) const;
    std::span<const std::byte> get(size_t layer, int expert, size_t expected) const;
};
class ExpertService {
    size_t experts_, picks_;
    uint64_t request_{}, epoch_{}, completed_{};
    bool initialized_{}, poisoned_{}, store_policy_set_{}, store_policy_{};
    std::atomic<bool> cancelled_{false};
    std::mutex mutex_;
public:
    ExpertService(size_t experts, size_t full_picks);
    void reset(uint64_t request, uint64_t epoch);
    void cancel() noexcept { cancelled_.store(true, std::memory_order_release); }
    Resp execute(const Req&, ExpertBackend&);
    uint64_t completed() const noexcept { return completed_; } // serial host owner only
};
}
