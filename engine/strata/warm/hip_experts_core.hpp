// Adapted from mmvq_project.hpp and pinned Strata ds4_iq.cu (MIT,
// Copyright 2026 Peasant Smith, Project Maya contributors and ggml authors).
// Pure per-subblock integer/scale decoding, shared by CPU proof and HIP.
#pragma once
#include <cstddef>
#include <cstdint>
#ifndef HIPEX_FN
#define HIPEX_FN inline
#endif
namespace strata::wiring::hipex {
HIPEX_FN uint32_t word(const uint8_t* p){
    // SoA word planes are aligned; CPU proof permits unaligned input too.
    uint32_t v;__builtin_memcpy(&v,p,4);return v;
}
HIPEX_FN uint16_t halfword(const uint8_t* p){uint16_t v;__builtin_memcpy(&v,p,2);return v;}
HIPEX_FN uint32_t signs(uint32_t v){v&=127;return v|((__builtin_popcount(v)&1u)<<7);}
HIPEX_FN uint32_t signed_word(uint32_t v,unsigned b){
    unsigned lo=((b&1)|((b&2)<<7)|((b&4)<<14)|((b&8)<<21));
    return (v^(lo*255u))+lo;
}
struct Dot {int integer;uint16_t half_scale;};
// M::dot4 is a signed integer dot. No floating operation before the subdot's
// explicit scale rounding. Type numbers below are internal layout IDs 0..2.
template<int Type,class M> HIPEX_FN Dot subdot(const uint8_t* w,size_t sub,size_t s,const int8_t* x,const uint32_t* table){
    const size_t b=s/8,blocks=sub/8;unsigned lo=0,hi=0,sc=0,dh=0;
    if constexpr(Type!=2){lo=word(w+4*s);hi=word(w+4*sub+4*s);sc=Type==0?hi>>28:w[8*sub+s];dh=halfword(w+(Type==0?8:9)*sub+2*b);}
    else {unsigned h[4];for(int j=0;j<4;++j)h[j]=halfword(w+6*sub+2*(size_t(j)*blocks+b));
        dh=(h[0]>>12)|((h[1]>>8)&0xf0)|((h[2]>>4)&0xf00)|(h[3]&0xf000);sc=(h[(s%8)/2]>>(6*(s&1)))&63;}
    int sums[2]={0,0};
#pragma unroll
    for(int j=0;j<4;++j){unsigned index,sign=0;
        if constexpr(Type==0){index=(lo>>(8*j))&255;sign=signs(hi>>(7*j));}
        else if constexpr(Type==1){unsigned c=((j<2?lo:hi)>>(16*(j&1)))&65535;index=256+(c&511);sign=signs(c>>9);}
        else {unsigned h=w[size_t(4+j/2)*sub+s]>>(4*(j&1));index=256+512+2*(unsigned(w[size_t(j)*sub+s])|((h&7)<<8))+((h>>3)&1);}
        uint32_t a=table[2*index],c=table[2*index+1];
        if constexpr(Type!=2){a=signed_word(a,sign);c=signed_word(c,sign>>4);}
        sums[j/2]=M::dot4(a,word(reinterpret_cast<const uint8_t*>(x)+8*j),sums[j/2]);
        sums[j/2]=M::dot4(c,word(reinterpret_cast<const uint8_t*>(x)+8*j+4),sums[j/2]);
    }
    const int s0=Type==0?2*int(sc)+1:Type==1?2*int(sc&15)+1:2*int(sc&7)+1;
    const int s1=Type==0?s0:Type==1?2*int(sc>>4)+1:2*int(sc>>3)+1;
    return {s0*sums[0]+s1*sums[1],uint16_t(dh)};
}
}
