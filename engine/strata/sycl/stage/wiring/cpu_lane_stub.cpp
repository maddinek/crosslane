// MIT; crosslane contributors. TODO(row2): replaced by the upstream-CPU-lane binding (cpu_lane.hpp).
#include "cpu_lane.hpp"
namespace strata::wiring {
std::unique_ptr<CpuLane> make_cpu_lane(const WeightSource&, uint32_t, uint32_t, int, const std::string&, bool) {
    throw std::runtime_error("TODO(row2): CPU expert lane (upstream ds4_gg_cpu/ds4_iq_cpu binding) not yet available");
}
}  // namespace strata::wiring
