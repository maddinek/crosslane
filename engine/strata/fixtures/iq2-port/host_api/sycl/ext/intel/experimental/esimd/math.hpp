#pragma once
#include <sycl/ext/intel/esimd.hpp>
namespace sycl::ext::intel::experimental::esimd {
template<class T,int N>sycl::ext::intel::esimd::simd<T,N> fma(
 sycl::ext::intel::esimd::simd<T,N> a,sycl::ext::intel::esimd::simd<T,N> b,sycl::ext::intel::esimd::simd<T,N> c){
    sycl::ext::intel::esimd::simd<T,N> r;for(int i=0;i<N;++i)r[i]=std::fma(a[i],b[i],c[i]);return r;
}
template<class R,class T,int N>sycl::ext::intel::esimd::simd<R,N> lzd(sycl::ext::intel::esimd::simd<T,N> v){
 sycl::ext::intel::esimd::simd<R,N> out;for(int i=0;i<N;++i)out[i]=std::countl_zero(uint32_t(v[i]));return out;
}

}
