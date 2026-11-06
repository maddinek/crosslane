// MIT. Dedicated IQ1_M ESIMD decoder, explicit down micro only.
// Its new ESIMD row mapping is a separate rewrite, not a geometry-only claim.
#pragma once
#include "iq2_packed_dot.hpp"
namespace p1k_packed {
ESIMD_INLINE I iq1m_weight_word(U grid,I delta,int half4) {
    U word=0;
#pragma unroll
    for(int j=0;j<4;++j){
        I symbol=(grid>>uint32_t(2*(4*half4+j)))&3u;
        I weight=8*(symbol-1)+delta;
        word=word|((U(weight)&255u)<<uint32_t(8*j));
    }
    return word.template bit_cast_view<int32_t>().read();
}
ESIMD_INLINE e::simd<uint8_t,224> iq1m_packed(const uint8_t* row,int base,int k) {
    e::simd<uint8_t,224> result=0;U byte(0,1);
    const uint32_t start=uint32_t(base/8)*56u,remaining=uint32_t(k/256)*56u-start;
#pragma unroll
    for(int part=0;part<7;++part){
        U off=byte+uint32_t(part*32);
        auto data=e::gather<uint8_t,32>(row+start,off,off<remaining,e::simd<uint8_t,32>(0));
        result.template select<32,1>(part*32)=data;
    }
    return result;
}
ESIMD_INLINE U iq1m_u16(e::simd<uint8_t,224>& bytes,Index offset) {
    U a=bytes.template iselect<32>(offset),b=bytes.template iselect<32>(Index(offset+1u));
    return a|(b<<8);
}
ESIMD_INLINE F iq1m_dot(e::simd<uint8_t,224> bytes,e::simd<int8_t,1024> x,
                       const uint16_t* table) {
    // Once per base tile, shared by all eight dword selections.
    e::simd<int32_t,256> xw=x.template bit_cast_view<int32_t>().read();
    U lane(0,1),ib=lane%8u;Index b=(lane/8u)*56u;
    U h0=iq1m_u16(bytes,Index(b+48u)),h1=iq1m_u16(bytes,Index(b+50u));
    U h2=iq1m_u16(bytes,Index(b+52u)),h3=iq1m_u16(bytes,Index(b+54u));
    U encoded=(h0>>12)|((h1>>8)&0xf0u)|((h2>>4)&0xf00u)|(h3&0xf000u);
    U sc=iq1m_u16(bytes,Index(b+48u+2u*(ib>>1)));
    I s0=2u*((sc>>(6u*(ib&1u)))&7u)+1u;
    I s1=2u*((sc>>(6u*(ib&1u)+3u))&7u)+1u;
    I lo=0,hi=0;
#pragma unroll
    for(int group=0;group<4;++group){
        U h=bytes.template iselect<32>(Index(b+32u+2u*ib+uint32_t(group/2)));
        U low=bytes.template iselect<32>(Index(b+4u*ib+uint32_t(group)));
        U index=low|((h<<uint32_t(group&1?4:8))&0x700u);
        I delta=1;delta.merge(I(-1),(h&uint32_t(group&1?128:8))!=0u);
        // Default readonly gather policy, like IQ1_M down-cached. The 4096B
        // immutable table is the same exact 2048x16-bit compact ternary grid.
        auto grid=e::gather<uint16_t,32>(table,index*2u,e::simd<uint16_t,32>(1),e::simd<uint16_t,32>(0));
        I sum=group<2?lo:hi;
#pragma unroll
        for(int half4=0;half4<2;++half4)
            sum=e::dp4a<int32_t>(sum,iq1m_weight_word(U(grid),delta,half4),activation_word(xw,group,half4));
        if(group<2)lo=sum;else hi=sum;
    }
    F d=half(encoded)*0.125f;
    F integer=s0*lo+s1*hi; // <=552960 <2^24; no FP reassociation across kb.
    return d*integer;
}
ESIMD_INLINE void iq1m_strip(const uint8_t* w,size_t ld,const int8_t* xq,
        const float* xd,int k,int row,float* y,const uint16_t* table) {
    F acc=0.0f;U lane(0,1);
    for(int base=0;base<k/32;base+=32){
        auto x=input_load(xq,base,k);auto active=lane<uint32_t(k/32-base);
        F scales=e::gather<float,32>(xd+base,lane*uint32_t(sizeof(float)),active,F(0));
        auto bytes=iq1m_packed(w+size_t(row)*ld,base,k);
        F dot=iq1m_dot(bytes,x,table);
        F next=ex::fma<float,32>(scales,dot,acc);acc.merge(next,active);
    }
    acc=xor_tree(acc);float value=1.0f*float(acc[0]);y[row]=value+0.0f;
}
#ifndef P1K_DOWN_HOST_TEST
inline sycl::event iq1m_local4(sycl::queue& q,const uint8_t* w,size_t ld,int n,int k,
        float* y,const int8_t* xq,const float* xd,const uint16_t* table) {
    return q.submit([&](sycl::handler& cgh){
        cgh.parallel_for(sycl::nd_range<1>((size_t(n)+3)/4*4,4),
            sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<128>},
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                int row=int(it.get_group_linear_id())*4+int(it.get_local_linear_id());if(row>=n)return;
                iq1m_strip(w,ld,xq,xd,k,row,y,table);
            });
    });
}
#endif
}
