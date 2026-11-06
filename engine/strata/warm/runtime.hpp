// MIT. Shared CPU DSO is the single owner of the warm store and B65 redirect.
#pragma once
#include "../sycl/stage/wiring/weight_source.hpp"
#include "../sycl/stage/wiring/cpu_lane.hpp"
#include <array>
#include <functional>
#include "expert_timeline.hpp"
#include <memory>
namespace strata::wiring {
struct WarmRedirect {
    virtual ~WarmRedirect() = default;
    virtual bool warm_grouped() const { return false; }
    virtual std::vector<std::vector<float>> warm_compute_many(uint32_t,std::span<const int>,std::span<const float>) { throw std::runtime_error("grouped expert path unavailable"); }
    virtual bool warm_contains(uint32_t, int) const = 0;
    virtual std::vector<float> warm_compute(uint32_t, int, std::span<const float>) = 0;
};
#ifdef STRATA_FORK_WARM
void warm_register(WarmRedirect*,bool hip=false);
extern "C" void strata_warm_stop_workers_v1();
void warm_set_residency(const std::vector<std::vector<int>>&,const std::vector<std::vector<int>>&);
void warm_unregister(WarmRedirect*) noexcept;
WarmRedirect* warm_redirect(bool hip=false);
std::vector<float> warm_execute(CpuLane&,uint32_t,std::span<const int>,std::span<const float>,std::span<const float>,std::span<const float>,const std::function<void()>& before_hip={},ExpertTimeline* timeline=nullptr);
class WarmStore {
public:
    virtual ~WarmStore() = default;
    virtual void prepare(uint32_t,std::span<const int>) = 0;
    virtual std::array<const uint8_t*,3> matrices(uint32_t,int) = 0;
    virtual void finish() noexcept = 0;
};
std::shared_ptr<WarmStore> warm_store(const WeightSource&);
void warm_note_cpu(uint32_t,int,uint64_t ns);
void warm_note_sycl(uint32_t,int,uint64_t ns);
void warm_note_hip(uint32_t,int);
void warm_gpu_span(bool hip,uint32_t,uint64_t);
#else
inline void warm_note_hip(uint32_t,int) {}
#endif
}
