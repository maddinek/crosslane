// Compiled independently in each toolchain. Only abi.h crosses the DSO boundary.
#include "abi.h"
#include "transport.hpp"
#include <cstdio>
#include <map>
using namespace strata::stage;
namespace {
struct Context {
    std::unique_ptr<Transport> transport;
    std::map<void*, HostMemory> hosts;
    std::map<void*, size_t> devices;
    bool failed=false;
    explicit Context(uint32_t index) {
#if defined(STRATA_STAGE_PLUGIN_HIP)
        require(index<=uint32_t(INT32_MAX),"HIP index overflow");
        transport=make_hip_transport(static_cast<int>(index));
#else
        transport=make_sycl_transport(index);
#endif
    }
    ~Context(){transport->drain();for(auto [p,n]:devices){(void)n;transport->release(p);}}
    void live(){require(!failed,"transport invalidated by prior failure");}
    size_t extent(uint64_t n){require(n && n<=SIZE_MAX,"invalid allocation/copy extent");return size_t(n);}
    void bounds(const void* h,const void* d,uint64_t n){
        live();extent(n);
        auto a=hosts.find(const_cast<void*>(h));auto b=devices.find(const_cast<void*>(d));
        require(a!=hosts.end() && b!=devices.end(),"foreign allocation or non-base pointer");
        require(n<=a->second.bytes && n<=b->second,"copy extent exceeds allocation");
    }
};
int32_t error(strata_stage_error_v1* e,const char* s) noexcept {
    if(e){e->code=1;std::snprintf(e->message,sizeof(e->message),"%s",s);}return 1;
}
template<class F> int32_t guard(void* p,strata_stage_error_v1* e,F f) noexcept {
    if(!e)return 1;
    *e={};
    if(!p)return error(e,"null context");
    auto& c=*static_cast<Context*>(p);
    try{f(c);return 0;}
    catch(const std::exception& x){c.failed=true;c.transport->drain();return error(e,x.what());}
    catch(...){c.failed=true;c.transport->drain();return error(e,"unknown native exception");}
}
int32_t pinned(void* p,uint64_t n,void** out,strata_stage_error_v1* e) noexcept {
    if(out)*out=nullptr;
    return guard(p,e,[&](Context& c){c.live();require(out,"null output");auto h=c.transport->pinned(c.extent(n));
        void* raw=h.data;c.hosts.emplace(raw,std::move(h));*out=raw;});
}
int32_t allocate(void* p,uint64_t n,void** out,strata_stage_error_v1* e) noexcept {
    if(out)*out=nullptr;
    return guard(p,e,[&](Context& c){c.live();require(out,"null output");auto bytes=c.extent(n);
        void* d=c.transport->allocate(bytes);
        try{c.devices.emplace(d,bytes);}catch(...){c.transport->release(d);throw;}*out=d;});
}
int32_t free_host(void* p,void* h,strata_stage_error_v1* e) noexcept {
    return guard(p,e,[&](Context& c){require(c.hosts.count(h),"foreign host allocation");c.transport->drain_checked();c.transport->free_pinned_checked(h);c.hosts.at(h).data=nullptr;c.hosts.erase(h);});
}
int32_t free_device(void* p,void* d,strata_stage_error_v1* e) noexcept {
    return guard(p,e,[&](Context& c){require(c.devices.count(d),"foreign device allocation");c.transport->drain_checked();c.transport->release_checked(d);c.devices.erase(d);});
}
int32_t to_host(void* p,void* h,const void* d,uint64_t n,strata_stage_error_v1* e) noexcept {
    return guard(p,e,[&](Context& c){c.bounds(h,d,n);c.transport->to_host(h,d,size_t(n))->wait();});
}
int32_t to_device(void* p,void* d,const void* h,uint64_t n,strata_stage_error_v1* e) noexcept {
    return guard(p,e,[&](Context& c){c.bounds(h,d,n);c.transport->to_device(d,h,size_t(n))->wait();});
}
int32_t drain(void* p,strata_stage_error_v1* e) noexcept {
    return guard(p,e,[](Context& c){c.transport->drain_checked();});
}
int32_t describe(void* p,char* out,uint64_t n,strata_stage_error_v1* e) noexcept {
    return guard(p,e,[&](Context& c){require(out && n,"invalid description buffer");auto s=c.transport->describe();
        require(s.size()<n,"description buffer too small");std::memcpy(out,s.c_str(),s.size()+1);});
}
void destroy(void* p) noexcept {try{delete static_cast<Context*>(p);}catch(...) {}}
}
extern "C" int32_t strata_stage_open_v1(uint32_t version,uint32_t bytes,uint32_t index,
    strata_stage_api_v1* out,strata_stage_error_v1* e) {
    if(out)*out={};
    if(!out || !e)return error(e,"null factory output/error");
    *e={};
    if(version!=STRATA_STAGE_ABI_V1 || bytes!=sizeof(*out))return error(e,"unsupported stage ABI version/size");
    try{
        auto c=std::make_unique<Context>(index);
#if defined(STRATA_STAGE_PLUGIN_HIP)
        constexpr auto backend=STRATA_STAGE_HIP;
#else
        constexpr auto backend=STRATA_STAGE_SYCL;
#endif
        *out={STRATA_STAGE_ABI_V1,sizeof(*out),backend,0,c.get(),pinned,allocate,free_host,free_device,
              to_host,to_device,drain,describe,destroy};
        c.release();return 0;
    }catch(const std::exception& x){return error(e,x.what());}
    catch(...){return error(e,"unknown factory exception");}
}
