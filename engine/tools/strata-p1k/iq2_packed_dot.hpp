// MIT. Explicit micro only: down-cached parent, signed integer-dot change only.
// No engine policy/automatic dispatch; raw 66B blocks and original lane tree.
#pragma once
#include "down_geometry.hpp"
namespace p1k_packed {
using namespace strata::ds4::row2::iq2;

ESIMD_INLINE I iq2_weight_word(U grid,U signs,int half4) {
    U word=0;
#pragma unroll
    for(int j=0;j<4;++j){
        const uint32_t offset=uint32_t(4*half4+j);
        U c=(grid>>(2u*offset))&3u;
        I weight=8u+17u*c+(c>>1);
        weight.merge(-weight,((signs>>offset)&1u)!=0u);
        word=word|((U(weight)&255u)<<uint32_t(8*j));
    }
    return word.template bit_cast_view<int32_t>().read();
}
ESIMD_INLINE I activation_word(e::simd<int32_t,256>& xw,int group,int half4) {
    U lane(0,1);
    // Select from the materialized tile: native simd_view has no iselect.
    // One dword selects the same four bytes, preserving -128 verbatim.
    return xw.template iselect<32>(
        Index(lane*8u+uint32_t(2*group+half4)));
}
ESIMD_INLINE F iq2_dot(e::simd<uint8_t,288> bytes,e::simd<int8_t,1024> x,
                      const uint16_t* table) {
    // Once per base tile, before the group loop; no new memory load.
    e::simd<int32_t,256> xw=x.template bit_cast_view<int32_t>().read();
    U lane(0,1);Index p=(lane/8u)*66u+2u+(lane%8u)*8u;
    U a0=bytes.template iselect<32>(Index(p+4u)),a1=bytes.template iselect<32>(Index(p+5u));
    U a2=bytes.template iselect<32>(Index(p+6u)),a3=bytes.template iselect<32>(Index(p+7u));
    U aux=a0|(a1<<8)|(a2<<16)|(a3<<24);
    I lo=0,hi=0;
#pragma unroll
    for(int group=0;group<4;++group){
        Index index=bytes.template iselect<32>(Index(p+uint16_t(group)));
#ifdef STRATA_IQ2_HOST_TEST
        auto grid=e::gather<uint16_t,32>(table,U(index)*2u,e::simd<uint16_t,32>(1),e::simd<uint16_t,32>(0));
#else
        auto grid=e::gather<uint16_t,32>(table,U(index)*2u,
            e::properties{e::cache_hint_L1<e::cache_hint::cached>,e::cache_hint_L2<e::cache_hint::cached>});
#endif
        U signs=parity_sign(aux>>uint32_t(7*group));
        I sum=group<2?lo:hi;
#pragma unroll
        for(int half4=0;half4<2;++half4)
            sum=e::dp4a<int32_t>(sum,iq2_weight_word(U(grid),signs,half4),activation_word(xw,group,half4));
        if(group<2)lo=sum;else hi=sum;
    }
    // |scale*lo+scale*hi| <= 5,459,968 < 2^24: exact integer regrouping.
    I scale=2u*(aux>>28)+1u;
    Index b=(lane/8u)*66u;
    U h0=bytes.template iselect<32>(b),h1=bytes.template iselect<32>(Index(b+1u));
    F d=half(h0|(h1<<8))*0.125f;
    F integer=scale*lo+scale*hi;
    return d*integer; // original rounded subdot BEFORE explicit lane FMA
}
ESIMD_INLINE void iq2_strip(const uint8_t* w,size_t ld,const int8_t* xq,
        const float* xd,int k,int row,float* y,const uint16_t* table) {
    F acc=0.0f;U lane(0,1);
    for(int base=0;base<k/32;base+=32){
        auto x=input_load(xq,base,k);auto active=lane<uint32_t(k/32-base);
        F scales=e::gather<float,32>(xd+base,lane*uint32_t(sizeof(float)),active,F(0));
        auto bytes=packed(w+size_t(row)*ld,base,k);
        F dot=iq2_dot(bytes,x,table);
        F next=ex::fma<float,32>(scales,dot,acc);acc.merge(next,active);
    }
    acc=xor_tree(acc);float value=1.0f*float(acc[0]);y[row]=value+0.0f;
}
#ifndef P1K_DOWN_HOST_TEST
inline sycl::event iq2_local4(sycl::queue& q,const uint8_t* w,size_t ld,int n,int k,
        float* y,const int8_t* xq,const float* xd,const uint16_t* table) {
    return q.submit([&](sycl::handler& cgh){
        cgh.parallel_for(sycl::nd_range<1>((size_t(n)+3)/4*4,4),
            sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                int row=int(it.get_group_linear_id())*4+int(it.get_local_linear_id());if(row>=n)return;
                iq2_strip(w,ld,xq,xd,k,row,y,table);
            });
    });
}
#endif
}
