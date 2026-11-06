// Derived from Strata c62d5e3 ds4_kernels.cu; Copyright (c) 2026 Peasant
// Smith and Project Maya contributors, MIT. Preserve Project Maya LICENSE/NOTICE.
#pragma once
#include <cmath>
#include <cstddef>
namespace strata::ds4::hc_scalar {
// Pin's block_sum: lane-zero result of XOR 16,8,4,2,1, then the
// same 32-lane tree over zero-padded warp totals. No arbitrary reduce order.
inline float warp_sum(const float* p) {
    float v[32];for(int i=0;i<32;++i)v[i]=p[i];
    for(int o=16;o;o>>=1)for(int i=0;i<32;++i)if(!(i&o))v[i]+=v[i^o];
    return v[0];
}
inline float block_sum(const float* p,int threads) {
    float red[32]={};for(int i=0;i<threads/32;++i)red[i]=warp_sum(p+i*32);
    return warp_sum(red);
}
// HIP contracts the float4 expression before the separate outer lane add:
// fma(x0,w0,x1*w1), then fma(x2,w2,...), then fma(x3,w3,...).
// The first pair's direction matters. A1 L0 own-input CPU reconstruction
// reproduces both p0 reference vectors with these three FMAs; the separate
// expression reproduces SYCL. Global implicit contraction remains disabled.
inline float quad_dot(const float* x,const float* w,int k,int D) {
    if(k+3<D) {
        float a=::fmaf(x[k],w[k],x[k+1]*w[k+1]);
        a=::fmaf(x[k+2],w[k+2],a);
        return ::fmaf(x[k+3],w[k+3],a);
    }
    float s=0;
    for(int j=0;j<4;++j){float v=k+j<D?x[k+j]*w[k+j]:0.f;if(j==0)s=v;else s+=v;}
    return s;
}
inline float mix_lane(const float* x,const float* w,int D,int lane) {
    float a=0;for(int k=lane*4;k<D;k+=256*4)a+=quad_dot(x,w,k,D);return a;
}
inline float pre_value(const float* h,const float* mix,int D,int H,int e) {
    float a=0;for(int j=0;j<H;++j)a+=mix[j]*h[size_t(j)*D+e];return a;
}
inline void mix(const float* h,const float* fn,int D,int N,int t,float eps,float* mx,bool scale_only=false) {
    h+=size_t(t)*D; mx+=size_t(t)*32;
    float lanes[256];for(int i=0;i<256;++i)lanes[i]=mix_lane(h,h,D,i);
    float inv=1.0f/::sqrtf(block_sum(lanes,256)/float(D)+eps);
    for(int m=0;m<N;++m) {
        float a=0; if(scale_only) a=mx[m];
        else {for(int i=0;i<256;++i)lanes[i]=mix_lane(h,fn+size_t(m)*D,D,i);a=block_sum(lanes,256);}
        mx[m]=a*inv;
    }
}
// Match the hc==4 shuffle tree. Other supported stream counts use lane-0 order.
inline float sum_h(const float* p,int stride,int H) {
    if(H==4)return (p[0]+p[stride])+(p[2*stride]+p[3*stride]);
    float s=0;for(int i=0;i<H;++i)s+=p[i*stride];return s;
}
inline void split(const float* mx,const float* scale,const float* base,int H,int iters,float eps,int t,
                  float* pre,float* post,float* comb) {
    mx+=size_t(t)*32; pre+=size_t(t)*8;post+=size_t(t)*8;comb+=size_t(t)*64;
    float c[16];
    for(int j=0;j<H;++j) {
        pre[j]=1.0f/(1.0f+::expf(-(mx[j]*scale[0]+base[j])))+eps;
        post[j]=2.0f/(1.0f+::expf(-(mx[H+j]*scale[1]+base[H+j])));
        float m=-INFINITY;
        for(int k=0;k<H;++k) {int i=j*H+k;c[i]=mx[2*H+i]*scale[2]+base[2*H+i];m=::fmaxf(m,c[i]);}
        for(int k=0;k<H;++k)c[j*H+k]=::expf(c[j*H+k]-m);
        float s=sum_h(c+j*H,1,H);
        for(int k=0;k<H;++k)c[j*H+k]=c[j*H+k]/s+eps;
    }
    for(int it=0;it<iters;++it) {
        if(it)for(int j=0;j<H;++j){float s=sum_h(c+j*H,1,H);
            for(int k=0;k<H;++k)c[j*H+k]/=(s+eps);}
        for(int k=0;k<H;++k){float s=sum_h(c+k,H,H);
            for(int j=0;j<H;++j)c[j*H+k]/=(s+eps);}
    }
    for(int i=0;i<H*H;++i)comb[i]=c[i];
}
inline void pre_norm(const float* h,const float* mix,const float* w,int D,int H,int t,float eps,float* x) {
    h+=size_t(t)*H*D;mix+=size_t(t)*8;x+=size_t(t)*D;
    float lanes[1024]={};
    for(int i=0;i<1024;++i)for(int e=i;e<D;e+=1024){float a=pre_value(h,mix,D,H,e);x[e]=a;lanes[i]+=a*a;}
    if(w){float inv=1.0f/::sqrtf(block_sum(lanes,1024)/float(D)+eps);for(int e=0;e<D;++e)x[e]=x[e]*inv*w[e];}
}
inline void post(const float* y,const float* res,const float* post,const float* comb,int D,int H,int t,int e,float* out) {
    res+=size_t(t)*H*D;out+=size_t(t)*H*D;post+=size_t(t)*8;comb+=size_t(t)*64;
    float r[4];for(int j=0;j<H;++j)r[j]=res[size_t(j)*D+e];
    float v=y[size_t(t)*D+e];
    for(int k=0;k<H;++k){float a=post[k]*v;for(int j=0;j<H;++j)a+=comb[j*H+k]*r[j];out[size_t(k)*D+e]=a;}
}
}
