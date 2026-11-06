#include "transport.hpp"
#include <hip/hip_runtime.h>
namespace strata::stage {
namespace {
void hip_check(hipError_t e){if(e!=hipSuccess)throw std::runtime_error(hipGetErrorString(e));}
struct HipCompletion final:Completion {
    hipEvent_t event{};int device;size_t extent;
    explicit HipCompletion(int d,hipStream_t stream,size_t n):device(d),extent(n){
        hip_check(hipSetDevice(device));hip_check(hipEventCreateWithFlags(&event,hipEventDisableTiming));
        try{hip_check(hipEventRecord(event,stream));}catch(...){hipEventDestroy(event);throw;}
    }
    void wait() override{hip_check(hipSetDevice(device));hip_check(hipEventSynchronize(event));}
    size_t bytes() const override{return extent;}
    ~HipCompletion(){hipSetDevice(device);hipEventDestroy(event);}
};
class HipTransport final:public Transport {
    int device_;hipStream_t stream_{};
public:
    explicit HipTransport(int d):device_(d){hip_check(hipSetDevice(d));hip_check(hipStreamCreate(&stream_));}
    ~HipTransport(){drain();hipSetDevice(device_);hipStreamDestroy(stream_);}
    HostMemory pinned(size_t n) override {
        hip_check(hipSetDevice(device_));void* p=nullptr;hip_check(hipHostMalloc(&p,n,hipHostMallocDefault));
        int d=device_;return {p,n,[d](void* x){hipSetDevice(d);hipHostFree(x);}};
    }
    void* allocate(size_t n) override{hip_check(hipSetDevice(device_));void* p=nullptr;hip_check(hipMalloc(&p,n));return p;}
    void release(void* p) noexcept override{hipSetDevice(device_);hipFree(p);}
    void release_checked(void* p) override{hip_check(hipSetDevice(device_));hip_check(hipFree(p));}
    void free_pinned_checked(void* p) override{hip_check(hipSetDevice(device_));hip_check(hipHostFree(p));}
    std::unique_ptr<Completion> to_host(void* h,const void* d,size_t n) override {
        hip_check(hipSetDevice(device_));hip_check(hipMemcpyAsync(h,d,n,hipMemcpyDeviceToHost,stream_));
        return std::make_unique<HipCompletion>(device_,stream_,n);
    }
    std::unique_ptr<Completion> to_device(void* d,const void* h,size_t n) override {
        hip_check(hipSetDevice(device_));hip_check(hipMemcpyAsync(d,h,n,hipMemcpyHostToDevice,stream_));
        return std::make_unique<HipCompletion>(device_,stream_,n);
    }
    std::string describe() const override {
        hipDeviceProp_t props{};hip_check(hipGetDeviceProperties(&props,device_));return props.name;
    }
    void drain_checked() override{hip_check(hipSetDevice(device_));hip_check(hipStreamSynchronize(stream_));}
    void drain() noexcept override{try{drain_checked();}catch(...) {}}
};
}
std::unique_ptr<Transport> make_hip_transport(int device){return std::make_unique<HipTransport>(device);}
}
