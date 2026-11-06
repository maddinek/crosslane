// Never a native selector. Only used by the explicit CPU native-branch API test.
#pragma once
#ifndef STRATA_IQ2_HOST_TEST
#error "CPU API fake only"
#endif
#include <sycl/sycl.hpp>
namespace strata{inline sycl::gpu_selector intel_level_zero_device(){return {};}}
