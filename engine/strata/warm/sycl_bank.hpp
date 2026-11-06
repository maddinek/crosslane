#pragma once
#include "residency.hpp"
#include "../sycl/stage/wiring/sycl_row2.hpp"
namespace strata::wiring {
std::unique_ptr<strata::ds4::row2::ExpertCompute> make_warm_sycl_experts(
 sycl::queue&,const WeightSource&,uint32_t first,uint32_t end,const Config&,const WarmResidency&);
}
