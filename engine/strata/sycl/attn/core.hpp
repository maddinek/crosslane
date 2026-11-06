// Derived from Strata c62d5e3 ds4_kernels.cu and ds4_attn_split.cuh.
// Copyright (c) 2026 Peasant Smith and Project Maya contributors (MIT).
// Preserve third_party/project-maya/{LICENSE,NOTICE.md} with distributions.
#pragma once
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#if defined(SYCL_LANGUAGE_VERSION) || defined(STRATA_ROW34_HOST_API_TEST)
#include "../division.hpp"
#endif
namespace strata::ds4::attn_scalar {
constexpr int kWin=16;
struct Rows {int n[kWin]{},k[kWin]{};int gen=0,R=1,kcap=0,ccap=0,blk=1;long long p0=0;};
struct AttnRows {int win0[kWin]{},nwin[kWin]{},nsel[kWin]{};int gen=0,t0=0,win=0,R=0,kcap=0;long long p0=0;};
// Explicit contraction at upstream indexer/attention multiply-adds. precise/off stays
// in force everywhere else; native accepted HIP ISA must be checked at handoff.
inline float index_fma(float a,float b,float c){
#if defined(SYCL_LANGUAGE_VERSION) || defined(STRATA_ROW34_HOST_API_TEST)
    return sycl::fma(a,b,c);
#else
    return ::fmaf(a,b,c);
#endif
}
inline float attention_div(float a,float b){
#if defined(SYCL_LANGUAGE_VERSION) || defined(STRATA_ROW34_HOST_API_TEST)
    return sycl_port::div_rn(a,b);
#else
    return a/b;
#endif
}
inline int min_(int a,int b){return a<b?a:b;}
inline size_t row_bytes(int fmt,int d){return fmt==0?size_t(d)*2:((fmt==1?size_t(d)+d/32:size_t(d)/2+d/(fmt==2?16:32))+15)/16*16;}
inline float bf(uint16_t b){return std::bit_cast<float>(uint32_t(b)<<16);}
inline float fp8(uint8_t b){if((b&127)==127)return NAN;int e=(b>>3)&15,m=b&7;float x=e==0?float(m)/512:float(8+m)*::ldexpf(1.0f,e-10);return b&128?-x:x;}
inline float decode(const uint8_t* r,int fmt,int d,int e){
    if(fmt==0)return bf(uint16_t(r[e*2])|(uint16_t(r[e*2+1])<<8));
    if(fmt==1)return fp8(r[e])*std::bit_cast<float>(uint32_t(r[d+e/32])<<23);
    constexpr float tab[8]={0,.5f,1,1.5f,2,3,4,6};int c=(r[e/2]>>(4*(e%2)))&15;
    float s=fmt==2?fp8(r[d/2+e/16]):std::bit_cast<float>(uint32_t(r[d/2+e/32])<<23);
    return (c&8?-tab[c&7]:tab[c&7])*s;
}
inline void rows_at(const Rows& r,int t,int& n,int& k){
    if(!r.gen){n=r.n[t];k=r.k[t];return;}
    int64_t nn=(r.p0+t+1)/r.R,cb=min_(r.ccap,int((nn+r.blk-1)/r.blk));
    if(r.gen==1){n=nn;k=min_(r.kcap,nn);}else if(r.gen==2){n=nn;k=cb;}
    else if(r.gen==3){n=int(cb*r.blk);k=min_(r.kcap,nn);}else{n=(nn+r.blk-1)/r.blk;k=r.kcap;}
}
inline void attn_at(const AttnRows& a,int t,int& w0,int& nw,int& ns){
    if(!a.gen){w0=a.win0[t];nw=a.nwin[t];ns=a.nsel[t];return;}
    long long ta=a.t0+t, first=a.win-a.p0>ta+1?a.win-a.p0:ta+1;
    w0=int(first);nw=int(a.win+ta-first+1);ns=a.R>0?int((a.p0+ta+1)/a.R):0;ns=min_(a.kcap,ns);
}
inline void compress(const float* kv,const float* score,float* ks,float* ss,int ratio,int p0,int T,int D,const float* w,float eps,float* latent){
    int g=0;float v[1024];
    for(int t=0;t<T;++t){int slot=(p0+t)%ratio;
        for(int e=0;e<D;++e){ks[size_t(slot)*D+e]=kv[size_t(t)*D+e];ss[size_t(slot)*D+e]=score[size_t(t)*D+e];}
        if((p0+t+1)%ratio)continue;
        float norm=0;
        for(int e=0;e<D;++e){float m=-INFINITY;for(int s=0;s<ratio;++s)m=::fmaxf(m,ss[size_t(s)*D+e]);
            float den=0;for(int s=0;s<ratio;++s)den+=::expf(ss[size_t(s)*D+e]-m);
            float x=0;for(int s=0;s<ratio;++s)x+=ks[size_t(s)*D+e]*(::expf(ss[size_t(s)*D+e]-m)/den);
            v[e]=x;norm+=x*x;}
        float inv=1/::sqrtf(norm/float(D)+eps);for(int e=0;e<D;++e)latent[size_t(g)*D+e]=v[e]*inv*w[e];++g;
    }
}
inline void index(const float* q,const float* w,const uint8_t* keys,int H,int D,int t,int n,const int* cand,int cs,int blk,Rows lim,float* out,long long stride,int fmt){
    int ln,lk;rows_at(lim,t,ln,lk);out+=int64_t(t)*stride;
    for(int i=0;i<n;++i){if(cand&&int64_t(i)>=int64_t(lk)*blk){out[i]=-INFINITY;continue;}int64_t p=cand?int64_t(cand[size_t(t)*cs+i/blk])*blk+i%blk:i;
        if(p<0||p>=ln){out[i]=-INFINITY;continue;}
        float s=0;const auto r=keys+size_t(p)*row_bytes(fmt,D);
        for(int h=0;h<H;++h){float dot=0;for(int e=0;e<D;++e)dot=index_fma(q[(size_t(t)*H+h)*D+e],decode(r,fmt,D,e),dot);
            s=index_fma(::fmaxf(dot,0),w[size_t(t)*64+h],s);}out[i]=s;}
}
inline uint32_t key(float x){uint32_t b=std::bit_cast<uint32_t>(x);return b&0x80000000u?~b:b|0x80000000u;}
// Invalid device candidate IDs never overflow a signed product or become a
// usable index. Index scoring masks them; a standalone mapping returns -1.
inline int mapped_id(const int* cand,int blk,int i){
    int64_t p=cand?int64_t(cand[i/blk])*blk+i%blk:i;
    return p<0||p>INT32_MAX?-1:int(p);
}
inline void select(const float* v,int n,int k,const int* cand,int blk,int* out){
    if(n<=0||k<=0)return;
    if(k>=n){for(int i=0;i<n;++i)out[i]=mapped_id(cand,blk,i);return;}
    // Same radix threshold and two stable output partitions as upstream; O(32*n), no private n-sized array.
    uint32_t prefix=0,mask=0;int remaining=k;
    for(int bit=31;bit>=0;--bit){uint32_t b=uint32_t(1)<<bit;int count=0;
        for(int i=0;i<n;++i){auto u=key(v[i]);count+=((u&mask)==prefix)&&(u&b);}
        if(count>=remaining)prefix|=b;else remaining-=count;mask|=b;
    }
    int a=0,b=k-remaining;
    for(int i=0;i<n;++i){auto u=key(v[i]);int id=mapped_id(cand,blk,i);
        if(u>prefix)out[a++]=id;else if(u==prefix&&b<k)out[b++]=id;}
}
inline void blocks(const float* score,long long stride,int t,Rows r,int blk,float* bm,long long bs,int* out,int os){
    int n,k;rows_at(r,t,n,k);if(r.gen)k=r.kcap;int nb=(n+blk-1)/blk;
    for(int b=0;b<nb;++b){float m=-INFINITY;for(int j=b*blk;j<min_(n,(b+1)*blk);++j)m=::fmaxf(m,score[t*stride+j]);
        bm[t*bs+b]=b==(n-1)/blk?INFINITY:m;}
    select(bm+t*bs,nb,k,nullptr,1,out+size_t(t)*os);
}
inline float sum32(float* a){for(int s=16;s;s>>=1){float old[32];for(int i=0;i<32;++i)old[i]=a[i];for(int i=0;i<32;++i)a[i]=old[i]+old[i^s];}return a[0];}
inline float row_value(const uint16_t* win,const uint8_t* kv,const int* sel,int w0,int nw,int j,int e,int fmt){
    if(j<nw)return bf(win[(size_t(w0)+j)*512+e]);
    int64_t p=sel[j-nw];if(p<0||p>=1048576)return NAN;
    return decode(kv+size_t(p)*row_bytes(fmt,512),fmt,512,e);
}
// A partial is {max,den,numerator[512]}; no learned sink until the final merge.
inline void partial(const float* q,const uint16_t* win,const uint8_t* kv,const int* sel,int w0,int nw,int begin,int end,int fmt,float scale,float* part){
    float m=-1e30f,den=0,acc[512]={};
    for(int first=begin;first<end;first+=64){int nt=min_(64,end-first);float sc[64],mx=-INFINITY;
        for(int j=0;j<nt;++j){float lanes[32]={};for(int l=0;l<32;++l)for(int e=l*16;e<(l+1)*16;++e)lanes[l]=index_fma(q[e],row_value(win,kv,sel,w0,nw,first+j,e,fmt),lanes[l]);
            sc[j]=sum32(lanes)*scale;mx=::fmaxf(mx,sc[j]);}
        float nm=::fmaxf(m,mx),cs=::expf(m-nm),lanes[32]={};
        for(int j=0;j<nt;++j){sc[j]=::expf(sc[j]-nm);lanes[j%32]+=sc[j];}
        den=index_fma(den,cs,sum32(lanes));
        for(int e=0;e<512;++e){acc[e]*=cs;for(int j=0;j<nt;++j)acc[e]=index_fma(sc[j],row_value(win,kv,sel,w0,nw,first+j,e,fmt),acc[e]);}
        m=nm;
    }
    part[0]=m;part[1]=den;for(int e=0;e<512;++e)part[e+2]=acc[e];
}
// Upstream attn_part<G,R>: one R-row softmax, no running-max rescaling.
// __expf's backend approximation is tolerance-tested, not claimed bit-identical.
inline void split_partial(const float* q,const uint16_t* win,const uint8_t* kv,const int* sel,int w0,int nw,int begin,int end,int fmt,float scale,float* part){
    float sc[1024],m=-INFINITY,lanes[32]={};int nr=end-begin;
    if(nr<=0){part[0]=-INFINITY;part[1]=0;for(int e=0;e<512;++e)part[e+2]=0;return;}
    for(int j=0;j<nr;++j){float dot[32]={};for(int l=0;l<32;++l)for(int e=l*16;e<(l+1)*16;++e)dot[l]=index_fma(q[e],row_value(win,kv,sel,w0,nw,begin+j,e,fmt),dot[l]);
        sc[j]=sum32(dot)*scale;m=::fmaxf(m,sc[j]);}
    for(int j=0;j<nr;++j){sc[j]=::expf(sc[j]-m);lanes[j%32]+=sc[j];}
    part[0]=m;part[1]=sum32(lanes);
    for(int e=0;e<512;++e){float acc=0;for(int j=0;j<nr;++j)acc=index_fma(sc[j],row_value(win,kv,sel,w0,nw,begin+j,e,fmt),acc);part[e+2]=acc;}
}
inline void attention(const float* q,const uint16_t* win,const uint8_t* kv,const int* sel,int stride,AttnRows a,const float* sink,int H,int t,int h,float scale,int fmt,float* out){
    int w0,nw,ns;attn_at(a,t,w0,nw,ns);float p[514];
    partial(q+(size_t(t)*H+h)*512,win,kv,sel?sel+size_t(t)*stride:nullptr,w0,nw,0,nw+ns,fmt,scale,p);
    if(nw+ns==0){for(int e=0;e<512;++e)out[(size_t(t)*H+h)*512+e]=0;return;}
    float den=p[1]+::expf(sink[h]-p[0]);for(int e=0;e<512;++e)out[(size_t(t)*H+h)*512+e]=attention_div(p[e+2],den);
}
}
