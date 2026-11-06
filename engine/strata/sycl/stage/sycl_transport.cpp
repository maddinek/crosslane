#include "transport.hpp"
#include "../device.hpp"
namespace strata::stage {
namespace {
struct SyclCompletion final:Completion {
    sycl::event event;size_t extent;
    explicit SyclCompletion(sycl::event e,size_t n):event(std::move(e)),extent(n){}
    void wait() override{event.wait_and_throw();}
    size_t bytes()const override{return extent;}
};
class SyclTransport final:public Transport {
    sycl::queue queue_;

public:
    explicit SyclTransport(size_t index):queue_(strata::intel_level_zero_device(index),
        [](sycl::exception_list errors){for(const auto& e:errors)std::rethrow_exception(e);},
        sycl::property::queue::in_order{}){}
    ~SyclTransport(){drain();}
    HostMemory pinned(size_t n) override {
        void* p=sycl::malloc_host(n,queue_);if(!p)throw std::bad_alloc();
        auto context=queue_.get_context();return {p,n,[context](void* x){try{sycl::free(x,context);}catch(...) {}}};
    }
    void* allocate(size_t n) override{auto p=sycl::malloc_device(n,queue_);if(!p)throw std::bad_alloc();return p;}
    void release(void* p) noexcept override{try{sycl::free(p,queue_);}catch(...) {}}
    void release_checked(void* p) override{sycl::free(p,queue_);}
    void free_pinned_checked(void* p) override{sycl::free(p,queue_);}
    std::unique_ptr<Completion> to_host(void* h,const void* d,size_t n) override {
        return std::make_unique<SyclCompletion>(queue_.memcpy(h,d,n),n);
    }
    std::unique_ptr<Completion> to_device(void* d,const void* h,size_t n) override {
        return std::make_unique<SyclCompletion>(queue_.memcpy(d,h,n),n);
    }
    std::string describe() const override{return queue_.get_device().get_info<sycl::info::device::name>();}
    void drain_checked() override{queue_.wait_and_throw();}
    void drain() noexcept override{try{drain_checked();}catch(...) {}}
};
}
std::unique_ptr<Transport> make_sycl_transport(size_t index){return std::make_unique<SyclTransport>(index);}
}
