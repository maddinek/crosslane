// CPU-only queue/subgroup API fake. No driver/runtime and no native qualification.
#pragma once
#ifndef STRATA_IQ2_HOST_TEST
#error "Explicit CPU test only"
#endif
#include "../../../row34/host_api/sycl/cooperative.hpp"
#include <sycl/ext/intel/esimd.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <span>
namespace sycl {
using std::fma;using std::fabs;using std::rint;using std::fmin;using std::fmax;using std::copysign;using std::isnan;using std::exp;
template<class To,class From>To bit_cast(From v){return std::bit_cast<To>(v);}
namespace info {namespace device {struct name{};}namespace event_profiling {struct command_start{};struct command_end{};}}
// Match oneAPI 2026.1: event waits are non-const; profiling reads are const.
struct event{void wait(){}void wait_and_throw(){}template<class T>uint64_t get_profiling_info()const{return 1;}};
namespace property::queue{struct in_order{};struct enable_profiling{};}
struct property_list{template<class... T>property_list(T...){}};
struct gpu_selector{};inline constexpr gpu_selector gpu_selector_v{};
template<int N>struct id{size_t n;size_t operator[](int)const{return n;}operator size_t()const{return n;}};
template<int N>struct range{size_t n;range(size_t v):n(v){}};
template<int N>struct nd_range{range<N> global,local;nd_range(range<N> a,range<N> b):global(a),local(b){}};
struct sub_group{size_t lane;};
template<int N>struct nd_item{size_t group,local;size_t get_group_linear_id()const{return group;}
    size_t get_local_linear_id()const{return local;}sub_group get_sub_group()const{return {local%32};}};
inline float permute_group_by_xor(sub_group sg,float v,int offset){
    static thread_local std::array<float,32> exchange;
    exchange.at(sg.lane)=v;host_test::active->barrier();float r=exchange.at(sg.lane^offset);host_test::active->barrier();return r;
}
struct fake_device{template<class T>std::string get_info()const{return "CPU fake, not a SYCL device";}};
inline int fake_grf256_launches=0,fake_grf128_launches=0;
struct handler {
    // ESIMD local4 API fake: one independent vector row per invocation.
    template<class Properties,class Fn>void parallel_for(nd_range<1> r,Properties,Fn fn){
        static_assert(Properties::grf_size==256||Properties::grf_size==128);
        if constexpr(Properties::grf_size==256)++fake_grf256_launches;else ++fake_grf128_launches;
        if((r.local.n!=4&&r.local.n!=16)||r.global.n%r.local.n)throw std::invalid_argument("fake ESIMD local4");
        if(r.local.n==4){for(size_t group=0;group<r.global.n/4;++group)for(size_t local=0;local<4;++local)fn(nd_item<1>{group,local});}
        else {host_test::group wg(r.local.n);
            for(size_t group=0;group<r.global.n/r.local.n;++group){ext::intel::esimd::fake::slm_reset();wg.run([&](size_t local){fn(nd_item<1>{group,local});});}}

    }
    template<class Properties,class Fn>void parallel_for(range<1> r,Properties,Fn fn){
        static_assert(Properties::grf_size==256);++fake_grf256_launches;
        for(size_t i=0;i<r.n;++i)fn(id<1>{i});
    }
};
struct queue {
    template<class F>event single_task(F f){f();return {};}
    queue()=default;queue(gpu_selector,property_list){}
    fake_device get_device()const{return {};}
    template<class P>bool has_property()const{return true;}
    template<class Fn>event submit(Fn fn){handler h;fn(h);return {};}
    template<class Fn>event parallel_for(range<1> r,Fn fn){for(size_t i=0;i<r.n;++i)fn(id<1>{i});return {};}
    template<class Kernel,class Fn>event parallel_for(range<1> r,Fn fn){for(size_t i=0;i<r.n;++i)fn(id<1>{i});return {};}
    template<class Fn>event parallel_for(nd_range<1> r,Fn fn){
        if(!r.local.n||r.local.n%32||r.global.n%r.local.n)throw std::invalid_argument("fake ND geometry");
        // There are no group-wide barriers in these kernels. Simulate each
        // independent subgroup, retaining the actual group/local IDs.
        host_test::group sg(32);
        for(size_t group=0;group<r.global.n/r.local.n;++group)for(size_t sub=0;sub<r.local.n/32;++sub)
            sg.run([&](size_t lane){fn(nd_item<1>{group,sub*32+lane});});
        return {};
    }
    event memcpy(void* d,const void* s,size_t n){std::memcpy(d,s,n);return {};}
    void wait_and_throw()const{}
};
inline void* malloc_device(size_t n,queue&){auto p=std::malloc(n);if(p)ext::intel::esimd::fake::allow(p,n);return p;}
template<class T>T* malloc_device(size_t n,queue& q){return static_cast<T*>(malloc_device(n*sizeof(T),q));}
inline void free(void* p,queue&){std::free(p);}
}
