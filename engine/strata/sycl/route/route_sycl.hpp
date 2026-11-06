#pragma once
#include "route.hpp"
#include <sycl/sycl.hpp>
namespace strata::ds4::route_port {
// Caller owns device allocations and 2*T*n scratch floats. Text only.
// IDs/weights compact [T,k]; logits have explicit stride. No mapped mailbox.
sycl::event route_device(sycl::queue&, const float* logits, size_t stride, const float* bias,
                         size_t n, size_t k, size_t T, float scale,
                         int* ids, float* weights, float* scratch);
Req read_request(sycl::queue&, uint64_t request, uint64_t epoch, uint64_t seq,
                 size_t layer, size_t D, size_t n, size_t k, size_t T,
                 const int* ids, const float* weights, const float* x, const float* shared);
void publish_response(sycl::queue&, const Resp&, int* cpu, float* part, float* combined);
void stage_copy(sycl::queue&, void* device_dst, std::span<const std::byte> host_src, size_t expected);
}
