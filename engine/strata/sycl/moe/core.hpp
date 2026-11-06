// Adapted from Strata c62d5e3 ds4_iq.cu decode_b/sub_dot and ds4_gguf.cu.
// Copyright (c) 2026 Peasant Smith, Project Maya contributors and ggml authors, MIT.
#pragma once
#include "format.hpp"
#include "tables.hpp"
namespace strata::ds4::moe {
struct Sub {int q[32]{}; float d=0,m[2]{};int s[2]{1,1};};
inline unsigned signs(unsigned v) {v&=127;return v|((std::popcount(v)&1u)<<7);}
inline void scale_min(const uint8_t* q,int j,int& d,int& m) {
    if(j<4){d=q[j]&63;m=q[j+4]&63;}
    else {d=(q[j+4]&15)|((q[j-4]>>6)<<4);m=(q[j+4]>>4)|((q[j]>>6)<<4);}
}
inline Sub decode(Type t,const uint8_t* row,int kb) {
    const int ib=kb&7;
    const auto* b=row+size_t(block_values(t)==32?kb:kb/8)*block_bytes(t);
    Sub s;
    if(t==Type::q8_0){s.d=half(u16(b));for(int j=0;j<32;++j)s.q[j]=int8_t(b[2+j]);}
    else if(t==Type::iq4_nl){s.d=half(u16(b));for(int j=0;j<32;++j)s.q[j]=tables::kvalues_iq4nl[j<16?b[2+j]&15:b[2+j-16]>>4];}
    else if(t==Type::iq2_xxs || t==Type::iq2_xs) {
        const auto* p=b+2+8*ib;const unsigned aux=u32(p+4);
        for(int l=0;l<4;++l) {
            const unsigned v=u16(p+2*l);
            const uint64_t grid=t==Type::iq2_xxs?tables::iq2xxs_grid[p[l]]:tables::iq2xs_grid[v&511];
            const unsigned sg=signs(t==Type::iq2_xxs?aux>>(7*l):v>>9);
            for(int j=0;j<8;++j){int q=int((grid>>(8*j))&255);s.q[8*l+j]=(sg>>j)&1?-q:q;}
        }
        s.d=half(u16(b))*0.125f;
        if(t==Type::iq2_xxs)s.s[0]=s.s[1]=2*int(aux>>28)+1;
        else {s.s[0]=2*int(b[66+ib]&15)+1;s.s[1]=2*int(b[66+ib]>>4)+1;}
    } else if(t==Type::iq2_s) {
        s.d=half(u16(b))*0.125f;s.s[0]=2*int(b[74+ib]&15)+1;s.s[1]=2*int(b[74+ib]>>4)+1;
        for(int l=0;l<4;++l){unsigned idx=b[2+4*ib+l]|((unsigned(b[66+ib])<<(8-2*l))&0x300);
            auto grid=tables::iq2s_grid[idx];unsigned sg=b[34+4*ib+l];
            for(int j=0;j<8;++j){int q=int((grid>>(8*j))&255);s.q[8*l+j]=(sg>>j)&1?-q:q;}}
    } else if(t==Type::iq1_s) {
        unsigned h=u16(b+34+2*ib);s.d=half(u16(b))*0.125f;s.s[0]=s.s[1]=2*int((h>>12)&7)+1;
        for(int l=0;l<4;++l){unsigned idx=b[2+4*ib+l]|(((h>>(3*l))&7)<<8);unsigned grid=tables::iq1s_grid_gpu[idx];
            for(int j=0;j<8;++j){int q=int((grid>>(8*(j&3)+(j>=4?4:0)))&15);s.q[8*l+j]=8*(q-1)+((h&0x8000)?-1:1);}}
    } else if(t==Type::iq1_m) {
        const unsigned sc=u16(b+48+2*(ib>>1));
        const uint16_t d=uint16_t((u16(b+48)>>12)|((u16(b+50)>>8)&0xf0)|((u16(b+52)>>4)&0xf00)|(u16(b+54)&0xf000));
        s.d=half(d)*0.125f;s.s[0]=2*int((sc>>(6*(ib&1)))&7)+1;s.s[1]=2*int((sc>>(6*(ib&1)+3))&7)+1;
        for(int l=0;l<4;++l){unsigned h=b[32+2*ib+l/2];unsigned idx=b[4*ib+l]|((h<<(l&1?4:8))&0x700);
            unsigned grid=tables::iq1s_grid_gpu[idx];int delta=(h&(l&1?128:8))?-1:1;
            for(int j=0;j<8;++j){int q=int((grid>>(8*(j&3)+(j>=4?4:0)))&15);s.q[8*l+j]=8*(q-1)+delta;}
        }
    } else if(t==Type::q4_k || t==Type::q5_k) {
        int d,m;scale_min(b+4,ib,d,m);s.s[0]=s.s[1]=d;s.d=half(u16(b));s.m[0]=s.m[1]=half(u16(b+2))*float(m);
        const auto* lo=b+(t==Type::q5_k?48:16)+32*(ib>>1);
        for(int j=0;j<32;++j){s.q[j]=(lo[j]>>(4*(ib&1)))&15;if(t==Type::q5_k)s.q[j]|=((b[16+j]>>ib)&1)<<4;}
    } else if(t==Type::q6_k) {
        int h=ib>>2,quarter=ib&3;const auto* lo=b+64*h+(quarter&1?32:0);
        for(int j=0;j<32;++j)s.q[j]=int(((lo[j]>>(4*(quarter>>1)))&15)|(((b[128+32*h+j]>>(2*quarter))&3)<<4))-32;
        s.s[0]=int8_t(b[192+8*h+2*quarter]);s.s[1]=int8_t(b[193+8*h+2*quarter]);s.d=half(u16(b+208));
    }
    return s;
}
// HIP contracts the MIN expression through its inline __f{mul,add,sub}_rn
// wrappers: first m0*sum_lo + rounded(m1*sum_hi), then d*dot - correction.
// Preserve those two FMAs explicitly with global contraction still disabled.
// Own-input proof: parity-g01/a9-engram (all 64 L14 eg_kv vectors).
template<class M> inline float sub_dot(const Sub& s,const int8_t* x,bool minimum) {
    int a=0,b=0,sa=0,sb=0;for(int j=0;j<16;++j){a+=s.q[j]*x[j];b+=s.q[16+j]*x[16+j];sa+=x[j];sb+=x[16+j];}
    float v=s.d*float(s.s[0]*a+s.s[1]*b);
    if(minimum){float mb=s.m[1]*float(sb);float correction=M::fma(s.m[0],float(sa),mb);
        v=M::fma(s.d,float(s.s[0]*a+s.s[1]*b),-correction);}return v;
}
// M supplies explicit FMA and native scalar transcendentals. No implicit contraction.
template<class M> inline float lane(Type t,const uint8_t* w,int k,const float* x,const int8_t* xq,const float* xd,int l,bool vec) {
    float a=0;
    if(t==Type::bf16) {
        if(vec)for(int i=8*l;i<k;i+=256)for(int j=0;j<8;++j)a=M::fma(bf16(u16(w+2*(i+j))),x[i+j],a);
        else for(int i=l;i<k;i+=32)a=M::fma(bf16(u16(w+2*i)),x[i],a);
    } else if(t==Type::f32) {
        for(int i=4*l;i<k;i+=128)for(int j=0;j<4;++j)a=M::fma(std::bit_cast<float>(u32(w+4*(i+j))),x[i+j],a);
    } else if(t==Type::f16) {
        // HIP's `acc += f.x*x0 + f.y*x1`: contracts the inner expression then the outer add.
        for(int i=8*l;i<k;i+=256)for(int j=0;j<8;j+=2){float hi=half(u16(w+2*(i+j+1)))*x[i+j+1];a=a+M::fma(half(u16(w+2*(i+j))),x[i+j],hi);}
    } else for(int kb=l;kb<k/32;kb+=32){Sub s=decode(t,w,kb);
        if(t==Type::q8_0){int dot=0;for(int j=0;j<32;++j)dot+=s.q[j]*xq[32*kb+j];a=M::fma(s.d*xd[kb],float(dot),a);}
        else {float dot=sub_dot<M>(s,xq+32*kb,t==Type::q4_k || t==Type::q5_k);a=M::fma(xd[kb],dot,a);}
    }
    return a;
}
template<class M> inline float swiglu(float g,float u,float lim) {
    if(lim>0){g=M::min(g,lim);u=M::min(M::max(u,-lim),lim);}return (g/(1.0f+M::exp(-g)))*u;
}
}
