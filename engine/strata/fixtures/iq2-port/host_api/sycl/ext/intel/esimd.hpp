// CPU-only ESIMD arithmetic/API fake; NEVER a SYCL compiler or GPU qualification.
#pragma once
#ifndef STRATA_IQ2_HOST_TEST
#error "Opt-in CPU tests only"
#endif
#define ESIMD_INLINE inline
#define SYCL_ESIMD_KERNEL
#include "../../../../../row34/host_api/sycl/cooperative.hpp"
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <vector>
namespace sycl::ext::intel::esimd {
template<class T,int N>struct simd;
template<class T,int N,int S>struct view;
namespace fake {
struct region{uintptr_t begin;size_t size;};
inline std::vector<region> regions;
inline size_t reads=0,masked=0,blocks=0,gathers=0,dots=0,dword_tiles=0;
inline void allow(const void* p,size_t n){regions.push_back({uintptr_t(p),n});}
inline void reset(){regions.clear();reads=masked=blocks=gathers=dots=dword_tiles=0;}
inline void check(uintptr_t p,size_t n){
    for(auto r:regions)if(p>=r.begin&&p-r.begin<=r.size&&n<=r.size-(p-r.begin)){++reads;return;}
    throw std::out_of_range("ESIMD fake: unbound/overread memory");
}
}
template<class T,int N>struct simd {
    std::array<T,N> a{};
    simd()=default;
    template<class U>requires std::is_arithmetic_v<U> simd(U v){a.fill(T(v));}
    template<class U,class V>simd(U base,V step){for(int i=0;i<N;++i)a[i]=T(base+i*step);}
    template<class U>simd(const simd<U,N>& b){for(int i=0;i<N;++i)a[i]=T(b[i]);}
    template<class U,int S>simd(const view<U,N,S>& b):simd(b.read()){}
    T& operator[](int i){return a.at(i);}T operator[](int i)const{return a.at(i);}
    template<int M,int S>view<T,M,S> select(int i){return {a.data(),N,i};}
    template<int M>simd<T,M> iselect(const simd<uint16_t,M>& ix){
        simd<T,M> r;for(int i=0;i<M;++i)r[i]=a.at(ix[i]);return r;
    }
    template<class U>void merge(const simd<U,N>& b,const simd<uint16_t,N>& m){for(int i=0;i<N;++i)if(m[i])a[i]=T(b[i]);}
    template<class U>struct cast_view {
        const simd& v;
        simd<U,N*sizeof(T)/sizeof(U)> read()const {
            static_assert((N*sizeof(T))%sizeof(U)==0);simd<U,N*sizeof(T)/sizeof(U)> r;
            if constexpr(std::is_same_v<T,int8_t>&&N==1024&&std::is_same_v<U,int32_t>)
                ++fake::dword_tiles;
            std::memcpy(r.a.data(),v.a.data(),sizeof(v.a));return r;
        }
        // Native simd_view has no iselect: read into a simd before selecting.
    };
    template<class U>cast_view<U> bit_cast_view(){return {*this};}
};
template<class T,int N,int S>struct view {
    T* data;int extent,offset;
    simd<T,N> read()const {simd<T,N> r;for(int i=0;i<N;++i){if(offset+i*S>=extent)throw std::out_of_range("view");r[i]=data[offset+i*S];}return r;}
    template<class U>view& operator=(const simd<U,N>& b){for(int i=0;i<N;++i){if(offset+i*S>=extent)throw std::out_of_range("view");data[offset+i*S]=T(b[i]);}return *this;}
};
#define FAKE_BINARY(OP) \
template<class T,class U,int N>auto operator OP(const simd<T,N>& a,const simd<U,N>& b){ \
using R=decltype(T{} OP U{});simd<R,N> r;for(int i=0;i<N;++i)r[i]=a[i] OP b[i];return r;} \
template<class T,class U,int N>requires std::is_arithmetic_v<U> auto operator OP(const simd<T,N>& a,U b){return a OP simd<U,N>(b);} \
template<class T,class U,int N>requires std::is_arithmetic_v<T> auto operator OP(T a,const simd<U,N>& b){return simd<T,N>(a) OP b;}
FAKE_BINARY(+) FAKE_BINARY(-) FAKE_BINARY(*) FAKE_BINARY(/) FAKE_BINARY(%)
FAKE_BINARY(&) FAKE_BINARY(|) FAKE_BINARY(^) FAKE_BINARY(<<) FAKE_BINARY(>>)
#undef FAKE_BINARY
#define FAKE_COMPARE(OP) \
template<class T,class U,int N>auto operator OP(const simd<T,N>& a,const simd<U,N>& b){ \
simd<uint16_t,N> r;for(int i=0;i<N;++i)r[i]=a[i] OP b[i];return r;} \
template<class T,class U,int N>requires std::is_arithmetic_v<U> auto operator OP(const simd<T,N>& a,U b){return a OP simd<U,N>(b);}
FAKE_COMPARE(==) FAKE_COMPARE(!=) FAKE_COMPARE(<) FAKE_COMPARE(>=)
#undef FAKE_COMPARE
template<class T,int N>auto operator-(const simd<T,N>& a){simd<T,N> r;for(int i=0;i<N;++i)r[i]=-a[i];return r;}
template<class T,int N,class U>simd<T,N> gather(const T* p,const simd<U,N>& off,
 const simd<uint16_t,N>& mask,const simd<T,N>& pass){
    ++fake::gathers;
    simd<T,N> r=pass;
    for(int i=0;i<N;++i)if(mask[i]){uintptr_t at=uintptr_t(p)+off[i];fake::check(at,sizeof(T));std::memcpy(&r[i],reinterpret_cast<const void*>(at),sizeof(T));}else ++fake::masked;
    return r;
}
template<int Alignment>struct overaligned_tag {};
template<class T,int N,int Alignment>simd<T,N> block_load(const T* p,overaligned_tag<Alignment>){
    if(uintptr_t(p)%Alignment)throw std::invalid_argument("ESIMD fake: false alignment assertion");
    fake::check(uintptr_t(p),N*sizeof(T));++fake::blocks;
    simd<T,N> r;std::memcpy(r.a.data(),p,N*sizeof(T));return r;
}
template<class T,int N,int Alignment>void block_store(T* p,simd<T,N> v,overaligned_tag<Alignment>){
    if(uintptr_t(p)%Alignment)throw std::invalid_argument("ESIMD fake: false store alignment");
    fake::check(uintptr_t(p),N*sizeof(T));std::memcpy(p,v.a.data(),N*sizeof(T));
}
// Match Intel's accumulator-FIRST API, NOT CUDA's. A signed dword type means
// signed 8-bit multiplicands. Restrict this fake to the adopted s32/s32 mode.
template<class R,class A,class W,class X,int N>
simd<R,N> dp4a(simd<A,N> acc,simd<W,N> w,simd<X,N> x){
    static_assert(std::is_same_v<R,int32_t>&&std::is_same_v<A,int32_t>&&
                  std::is_same_v<W,int32_t>&&std::is_same_v<X,int32_t>);
    ++fake::dots;simd<R,N> result;
    for(int lane=0;lane<N;++lane){
        uint32_t sum=std::bit_cast<uint32_t>(acc[lane]);
        for(int j=0;j<4;++j){
            uint32_t wb=(std::bit_cast<uint32_t>(w[lane])>>(8*j))&255u;
            uint32_t xb=(std::bit_cast<uint32_t>(x[lane])>>(8*j))&255u;
            int wi=wb<128?int(wb):int(wb)-256,xi=xb<128?int(xb):int(xb)-256;
            sum+=uint32_t(wi*xi);
        }
        result[lane]=std::bit_cast<int32_t>(sum);
    }return result;
}
namespace fake {
inline std::vector<uint8_t> slm;
inline std::vector<bool> written;
inline size_t slm_reads=0,slm_writes=0,barriers=0;
inline void slm_reset(){slm.clear();written.clear();}
}
template<uint32_t N>void slm_init(){
 if(fake::slm.empty()){fake::slm.resize(N);fake::written.resize(N);}
 if(fake::slm.size()!=N)throw std::invalid_argument("inconsistent group SLM");
}
template<class T,int N>void slm_block_store(uint32_t off,simd<T,N> v){
 if(off%16||off>fake::slm.size()||N*sizeof(T)>fake::slm.size()-off)throw std::out_of_range("SLM block store");
 for(size_t i=off;i<off+N*sizeof(T);++i){if(fake::written[i])throw std::runtime_error("overlapping SLM initialization");fake::written[i]=true;}
 std::memcpy(fake::slm.data()+off,v.a.data(),N*sizeof(T));++fake::slm_writes;
}
template<class T,int N>simd<T,N> slm_gather(simd<uint32_t,N> offsets){
 simd<T,N> out;
 for(int i=0;i<N;++i){size_t off=offsets[i];if(off%sizeof(T)||off>fake::slm.size()||sizeof(T)>fake::slm.size()-off)throw std::out_of_range("SLM gather");
  for(size_t j=off;j<off+sizeof(T);++j)if(!fake::written[j])throw std::runtime_error("uninitialized SLM");
  std::memcpy(&out[i],fake::slm.data()+off,sizeof(T));}
 ++fake::slm_reads;return out;
}
inline void barrier(){if(!sycl::host_test::active)throw std::runtime_error("SLM barrier outside group");++fake::barriers;sycl::host_test::active->barrier();}

}
