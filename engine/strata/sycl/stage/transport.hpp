#pragma once
#include "stage.hpp"
namespace strata::stage {
// Native adapter copies always use the adapter's own stream/queue and events.
// Native device allocations never cross this interface between adapters.
struct Transport {
    virtual ~Transport()=default;
    virtual HostMemory pinned(size_t)=0;
    virtual void* allocate(size_t)=0;
    virtual void release(void*) noexcept=0;
    virtual void release_checked(void*)=0;
    virtual void free_pinned_checked(void*)=0;
    virtual std::unique_ptr<Completion> to_host(void* host,const void* device,size_t)=0;
    virtual std::unique_ptr<Completion> to_device(void* device,const void* host,size_t)=0;
    virtual std::string describe() const=0;
    virtual void drain_checked()=0;
    virtual void drain() noexcept=0;
};
std::unique_ptr<Transport> make_hip_transport(int device);
std::unique_ptr<Transport> make_sycl_transport(size_t gpu_index);
}
