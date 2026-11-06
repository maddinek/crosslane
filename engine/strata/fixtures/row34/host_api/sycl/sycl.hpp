// Host-only API diagnostic shim, NOT SYCL and never a native backend.
// Test include path is opt-in; production CMake does not reference this directory.
#pragma once
#ifndef STRATA_ROW34_HOST_API_TEST
#error "This shim may only be included by the explicit offline host API test"
#endif
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include "cooperative.hpp"
namespace sycl {
using std::fma;using std::fabs;using std::rint;using std::fmin;using std::fmax;using std::copysign;using std::isnan;using std::sin;using std::cos;using std::max;
template<class To,class From>To bit_cast(From x){return std::bit_cast<To>(x);}
inline float rsqrt(float x){return 1/std::sqrt(x);}
struct event {void wait(){}void wait_and_throw(){}};
namespace property::queue {struct in_order {};}
struct gpu_selector {};inline constexpr gpu_selector gpu_selector_v{};
template<int N>struct id {size_t n;size_t operator[](int)const{return n;}};
template<int N>struct range {size_t n;explicit range(size_t count):n(count){}};
template<int N>struct nd_range {range<N> global,local;nd_range(range<N> g,range<N> l):global(g),local(l){}};
namespace access {enum class fence_space {local_space};}
struct sub_group {size_t lane;};
inline float select_from_group(sub_group sg,float value,id<1> target){
    auto& g=*host_test::active;g.subgroup_exchange.at(sg.lane)=value;g.barrier();
    const auto result=g.subgroup_exchange.at(sg.lane/32*32+target[0]);g.barrier();return result;
}
inline float permute_group_by_xor(sub_group sg,float value,int mask){return select_from_group(sg,value,id<1>{(sg.lane%32)^size_t(mask)});}
template<int N>struct nd_item {
    size_t local,group;
    size_t get_local_linear_id()const{return local;}
    size_t get_group_linear_id()const{return group;}
    sub_group get_sub_group()const{return {local};}
    void barrier(access::fence_space)const{host_test::active->barrier();}
};
class handler {
public:
    template<class Function>void parallel_for(nd_range<1> r,Function fn){
        if(!r.local.n||r.global.n%r.local.n)throw std::invalid_argument("host ND geometry");
        host_test::group work(r.local.n);
        for(size_t g=0;g<r.global.n/r.local.n;++g)work.run([&](size_t lane){fn(nd_item<1>{lane,g});});
    }
};
template<class T,int N>class local_accessor {
    std::shared_ptr<std::vector<T>> data;
public:
    local_accessor(range<N> r,handler&):data(std::make_shared<std::vector<T>>(r.n)){}
    T& operator[](size_t i)const{return data->at(i);}
};
class queue {
    bool ordered_=false;
public:
    queue()=default;queue(gpu_selector,property::queue::in_order):ordered_(true){}
    template<class Property>bool has_property()const{return ordered_;}
    template<class Function>event parallel_for(range<1> r,Function fn){for(size_t i=0;i<r.n;++i)fn(id<1>{i});return {};}
    template<class Function>event parallel_for(nd_range<1> r,Function fn){handler command;command.parallel_for(r,fn);return {};}
    template<class Function>event single_task(Function fn){fn();return {};}
    template<class Function>event submit(Function fn){handler command;fn(command);return {};}
    event memcpy(void* dst,const void* src,size_t n){std::memcpy(dst,src,n);return {};}
    void wait_and_throw()const{}
};
inline void* malloc_shared(size_t n,queue&){return std::malloc(n);}
inline void free(void* p,queue&){std::free(p);}
}
