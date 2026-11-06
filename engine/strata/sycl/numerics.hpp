// Derived from Strata c62d5e3 src/dsv41/ds4_{numerics,kernels}.cu[h].
// Copyright (c) 2026 Peasant Smith and Project Maya contributors. MIT.
// Preserve third_party/project-maya/{LICENSE,NOTICE.md} with distributions.
#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <stdexcept>
namespace strata::ds4::sycl_port {
inline uint32_t bits(float x) { return sycl::bit_cast<uint32_t>(x); }
inline float value(uint32_t x) { return sycl::bit_cast<float>(x); }
inline uint16_t bf16_rn(float x) {
    auto b=bits(x);
    if ((b&0x7fffffff)>0x7f800000) return uint16_t((b>>16)|0x40);
    return uint16_t((b+0x7fff+((b>>16)&1))>>16);
}
// finish uses upstream f2bf16 (distinct NaN payload policy from bf16_rn).
inline uint16_t finish_bf16(float x) {
    auto b=bits(x);
    return (b&0x7f800000)==0x7f800000 ? uint16_t(b>>16) : bf16_rn(x);
}
inline float bf16_value(uint16_t x) { return value(uint32_t(x)<<16); }
inline float e4m3_value(uint8_t q) {
    unsigned e=(q>>3)&15, m=q&7;
    float v=e==15&&m==7 ? value(0x7fc00000) : e==0 ? float(m)*0.001953125f : value(((e+120)<<23)|(m<<20));
    return q&128 ? -v : v;
}
inline float e4m3_round(float v) {
    float a=sycl::fabs(v), r;
    if (sycl::isnan(a)) return v;
    if (a>=448) r=448;
    else {
        int e=int((bits(a)>>23)&255)-127;
        float step=e< -6 ? 0.001953125f : value(uint32_t(e+124)<<23);
        r=a==0 ? 0 : sycl::rint(a/step)*step;
        r=sycl::fmin(r,448.0f);
    }
    return v<0 ? -r : r; // upstream round maps -0 to +0
}
// Input MUST already be on the E4M3 grid after scaling; not a general encoder.
inline uint8_t e4m3_code(float v, int k=0) {
    auto b=bits(v), a=b&0x7fffffff, s=(b>>24)&128;
    if (!a) return uint8_t(s);
    int e=int(a>>23)-127-k;
    return uint8_t(s|(e>=-6 ? (uint32_t(e+7)<<3)|((a>>20)&7) : ((a&0x7fffff)|0x800000)>>(14-e)));
}
inline float e2m1_round(float v) {
    float a=sycl::fabs(v);
    float r=a<=.25f?0:a<.75f?.5f:a<=1.25f?1:a<1.75f?1.5f:a<=2.5f?2:a<3.5f?3:a<=5?4:6;
    return sycl::copysign(r,v);
}
inline uint8_t e2m1_code(float q) {
    float a=sycl::fabs(q);
    unsigned m=a<=1.5f?unsigned(a*2):a<=2?4:a<=3?5:a<=4?6:7;
    return uint8_t(m|((bits(q)>>31)<<3));
}
inline float e2m1_value(unsigned c) {
    constexpr float t[8]={0,.5f,1,1.5f,2,3,4,6};
    return sycl::copysign(t[c&7],c&8?-1.0f:1.0f);
}
inline float pow2_ceil(float x) {
    auto b=bits(x); return value((((b>>23)&255)+((b&0x7fffff)!=0))<<23);
}
inline size_t row_bytes(int fmt,int d) {
    return fmt==0?size_t(d)*2:((fmt==1?size_t(d)+d/32:size_t(d)/2+d/(fmt==2?16:32))+15)/16*16;
}
struct Finish {
    const float* w=nullptr; float eps=1e-20f;
    int rd=0,pos=0,inverse=0,vpp=1,pos_step=1;
    const float* freqs=nullptr; int quant=0;
    uint16_t* dst=nullptr; long long dst_stride=0;
    uint8_t* pdst=nullptr; long long pdst_stride=0;
};
sycl::event finish(sycl::queue&,float*,int,int,long long,Finish);
sycl::event embed(sycl::queue&,const uint16_t*,int,float*,int,int);
sycl::event ring_to_ext(sycl::queue&,const uint8_t*,uint16_t*,int64_t,int,int);
sycl::event ext_to_ring(sycl::queue&,const uint16_t*,uint8_t*,int64_t,int,int,int,uint8_t* = nullptr);
sycl::event ring_restore(sycl::queue&,uint8_t*,const uint8_t*,int64_t,int,int,int,int);
// Caller owns USM bounds/lifetime and uses an in-order queue. No implicit fallback.
inline void ordered(sycl::queue& q) {
    if (!q.has_property<sycl::property::queue::in_order>()) throw std::invalid_argument("in-order queue required");
}
}
