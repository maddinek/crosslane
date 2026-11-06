// MIT; CPU-testable arithmetic port of ggml6b9ba4f2 x86 AVX2 Q8_K IQ dots.
// Tables and integer groups match the pinned CPU lane. No HIP API here.
#pragma once
#include <cstdint>
#include <cstddef>
#include <cmath>
#ifndef Q8K_FN
#define Q8K_FN inline
#define Q8K_TABLE static constexpr
#endif
namespace strata::wiring::q8k {
#include "q8k_tables.inc"
struct Block { float d; int8_t qs[256]; int16_t bsums[16]; };
static_assert(sizeof(Block)==292 && offsetof(Block,qs)==4 && offsetof(Block,bsums)==260);
Q8K_FN uint16_t u16(const uint8_t* p){return uint16_t(p[0])|(uint16_t(p[1])<<8);}
Q8K_FN uint32_t u32(const uint8_t* p){return uint32_t(u16(p))|(uint32_t(u16(p+2))<<16);}
Q8K_FN float half(uint16_t h){
    // All finite half values are exact binary32, including subnormals.
    int e=(h>>10)&31; float v;
    if(!e)v=ldexpf(float(h&1023),-24);
    else if(e==31)v=(h&1023)?NAN:INFINITY;
    else v=ldexpf(float(1024+(h&1023)),e-25);
    return h&32768?-v:v;
}
Q8K_FN int nearest(float x){
    // ggml nearest_int; union bit transfer has the same representation on HIP/x86.
    union {float f; uint32_t u;} v; v.f=x+12582912.0f;
    return int(v.u&0x007fffff)-0x00400000;
}
Q8K_FN void quant_block(const float* x,Block& y){
    float mx=0,am=0;
    for(int j=0;j<256;++j){const float a=fabsf(x[j]);if(a>am){am=a;mx=x[j];}}
    if(am==0){y.d=0;for(int j=0;j<256;++j)y.qs[j]=0;for(int j=0;j<16;++j)y.bsums[j]=0;return;}
    const float is=-127.0f/mx;
    for(int j=0;j<256;++j){int v=nearest(is*x[j]);y.qs[j]=int8_t(v<127?v:127);}
    for(int j=0;j<16;++j){int s=0;for(int z=0;z<16;++z)s+=y.qs[16*j+z];y.bsums[j]=int16_t(s);}
    y.d=1.0f/is;
}
Q8K_FN bool supported(int t){return t==16||t==17||t==29;}
Q8K_FN int block_bytes(int t){return t==29?56:t==17?74:t==16?66:0;}
struct Ints {int main=0,delta=0;};
// Lane L of AVX2's madd_epi16: positions 4L..4L+3 in every32 chunk.
Q8K_FN Ints integer_lane(int t,const uint8_t* w,const Block& a,int lane){
    Ints out;
    for(int c=0;c<8;++c){
        const int local=4*lane,group=local/8;
        int grid=0,sign=0,scale=0,delta=0;uint64_t tab=0;
        if(t==29){
            const uint8_t hi=w[32+2*c+group/2];
            grid=w[4*c+group]|((int(hi)<<(8-4*(group%2)))&0x700);
            tab=iq1s_grid[grid];delta=(hi&(group%2?128:8))?-1:1;
            const auto sc=u16(w+48+2*(c/2));
            scale=2*((sc>>(6*(c%2)+3*(local/16)))&7)+1;
        }else if(t==17){
            const auto q=u16(w+2+8*c+2*group);tab=iq2xs_grid[q&511];sign=ksigns_iq2xs[q>>9];
            scale=2*((w[66+c]>>(4*(local/16)))&15)+1;
        }else{
            const uint8_t* chunk=w+2+8*c;const auto hi=u32(chunk+4);
            tab=iq2xxs_grid[chunk[group]];sign=ksigns_iq2xs[(hi>>(7*group))&127];scale=2*(hi>>28)+1;
        }
        for(int j=local%8;j<local%8+4;++j){
            const int q=a.qs[32*c+8*group+j];
            const int code=t==29?int(int8_t(uint8_t(tab>>(8*j)))):int(uint8_t(tab>>(8*j)))*((sign&(1<<j))?-1:1);
            out.main+=scale*code*q;
            if(t==29)out.delta+=scale*delta*q;
        }
    }
    return out;
}
Q8K_FN float scale(int t,const uint8_t* w){
    if(t!=29)return half(u16(w));
    const uint16_t h=(u16(w+48)>>12)|((u16(w+50)>>8)&0xf0)|((u16(w+52)>>4)&0xf00)|(u16(w+54)&0xf000);
    return half(h);
}
struct Sums {float main=0,delta=0;};
Q8K_FN Sums dot_lane(int t,const uint8_t* w,const Block* a,int nb,int lane){
    Sums s;
    for(int b=0;b<nb;++b){
        const auto* p=w+size_t(b)*block_bytes(t);const auto v=integer_lane(t,p,a[b],lane);
        const float d=scale(t,p)*a[b].d;
        s.main=fmaf(d,float(v.main),s.main);
        if(t==29)s.delta=fmaf(d,float(v.delta),s.delta);
    }
    return s;
}
Q8K_FN float hsum(const float* v){return ((v[0]+v[4])+(v[2]+v[6]))+((v[1]+v[5])+(v[3]+v[7]));}
Q8K_FN float dot(int t,const uint8_t* w,const Block* a,int nb){
    float m[8],d[8];for(int l=0;l<8;++l){const auto s=dot_lane(t,w,a,nb,l);m[l]=s.main;d[l]=s.delta;}
    const float main=hsum(m);
    return t==29?main+0.125f*hsum(d):0.125f*main;
}
Q8K_FN float swiglu(float g,float u,float limit){
    if(limit>0){g=fminf(g,limit);u=fminf(fmaxf(u,-limit),limit);}
    return (g/(1.0f+expf(-g)))*u;
}
} // namespace strata::wiring::q8k
