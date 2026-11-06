// Derived from Strata c62d5e3, Project Maya contributors (MIT).
// Preserve third_party/project-maya/{LICENSE,NOTICE.md} with distributions.
#pragma once
#include <bit>
#include <cstdint>
namespace strata::ds4::engram {
inline float fbits(uint32_t u) { return std::bit_cast<float>(u); }
inline float half(uint16_t h) {
    unsigned e=(h>>10)&31, m=h&1023;
    if(e==31) return fbits((uint32_t(h&0x8000)<<16)|0x7f800000u|(m<<13));
    if(e) return fbits((uint32_t(h&0x8000)<<16)|((e+112)<<23)|(m<<13));
    // All half subnormals are exactly representable in F32.
    return (h&0x8000 ? -1.0f:1.0f)*float(m)*0x1p-24f;
}
inline float fp8(uint8_t q) {
    unsigned e=(q>>3)&15, m=q&7;
    float v=e==15&&m==7 ? fbits(0x7fc00000u) : e ? fbits(((e+120)<<23)|(m<<20)):float(m)*0x1p-9f;
    return q&128?-v:v;
}
inline float scale8(uint8_t b) { return b==255?fbits(0x7fc00000u):b==0?0x1p-127f:fbits(uint32_t(b)<<23); }
inline float row_value(const uint8_t* row,const uint8_t* scales,int i,int fmt) {
    if(fmt==0) return fp8(row[i])*scale8(scales[i/32]);
    if(fmt==1) {
        const auto* b=row+(i/256)*210; int e=i%256;
        int h=e/128, q=(e%128)/32, l=e%32;
        int lo=b[h*64+l+(q&1?32:0)];
        int v=((q&2?lo>>4:lo&15)|(((b[128+h*32+l]>>(2*q))&3)<<4))-32;
        int sc=int(b[192+h*8+l/16+2*q]); if(sc>=128) sc-=256;
        // Two F32 multiplications, -ffp-contract=off. Never fuse scale with input.
        float d=half(uint16_t(b[208])|(uint16_t(b[209])<<8));
        return (d*float(sc))*float(v);
    }
    constexpr int cb[16]={-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
    const auto* b=row+(i/32)*18; int e=i%32;
    int q=b[2+(e%16)]; q=e&16?q>>4:q&15;
    return half(uint16_t(b[0])|(uint16_t(b[1])<<8))*float(cb[q]);
}
}
