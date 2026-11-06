// The SYCL stage targets Intel through Level Zero, even with a HIP plugin installed.
#pragma once
#include <sycl/sycl.hpp>
#include <cstddef>
#include <stdexcept>
#include <vector>
namespace strata {
inline bool intel_level_zero(const sycl::device& d) {
    return d.is_gpu() && d.get_backend()==sycl::backend::ext_oneapi_level_zero &&
        d.get_info<sycl::info::device::vendor_id>()==0x8086;
}
inline sycl::device intel_level_zero_device(size_t index=0) {
    std::vector<sycl::device> matches;
    for(const auto& d:sycl::device::get_devices(sycl::info::device_type::gpu))
        if(intel_level_zero(d)) matches.push_back(d);
    if(index>=matches.size()) throw std::runtime_error("Intel Level Zero GPU index absent (index is within filtered devices)");
    return matches[index];
}
}
