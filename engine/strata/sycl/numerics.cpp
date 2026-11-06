// MIT, Project Maya attribution and source contract in numerics.hpp.
#include "numerics.hpp"
#include "division.hpp"
namespace strata::ds4::sycl_port {
sycl::event finish(sycl::queue& q,float* x,int nv,int len,long long stride,Finish f) {
    ordered(q);
    if(nv<=0||len<=0||len>8192||stride<len||f.quant<0||f.quant>3||f.rd<0||f.rd>len||f.rd%2||(len-f.rd)%2||f.vpp<=0||
       (f.rd&&!f.freqs)||(f.quant&&len%(f.quant==2?16:32))||
       (f.dst&&f.dst_stride<len)||(f.pdst&&(!f.quant||f.pdst_stride<(long long)row_bytes(f.quant,len))))
        throw std::invalid_argument("unsupported finish geometry");
    return q.parallel_for(sycl::range<1>(nv),[=](sycl::id<1> id) {
        int v=int(id[0]); float r[8192];
        for(int e=0;e<len;++e) r[e]=x[v*stride+e];
        if(f.w) {
            // Reproduce HIP's 256-thread block_sum XOR tree without subgroup assumptions.
            float lanes[256]={};
            for(int t=0;t<256;++t) for(int e=t;e<len;e+=256) lanes[t]+=r[e]*r[e];
            for(int o=16;o;o>>=1) for(int t=0;t<256;t+=32) {
                float old[32]; for(int i=0;i<32;++i) old[i]=lanes[t+i];
                for(int i=0;i<32;++i) lanes[t+i]=old[i]+old[i^o];
            }
            float sums[32]={}; for(int i=0;i<8;++i) sums[i]=lanes[i*32];
            for(int o=16;o;o>>=1) { float old[32]; for(int i=0;i<32;++i) old[i]=sums[i];
                for(int i=0;i<32;++i) sums[i]=old[i]+old[i^o]; }
            float inv=sycl::rsqrt(sums[0]/float(len)+f.eps);
            for(int e=0;e<len;++e) r[e]=r[e]*inv*f.w[e];
        }
        for(int e=len-f.rd;e<len;e+=2) {
            float a=float(f.pos+(v/f.vpp)*f.pos_step)*f.freqs[(e-len+f.rd)/2];
            float c=sycl::cos(a),s=sycl::sin(a)*(f.inverse?-1.0f:1.0f),re=r[e],im=r[e+1];
            r[e]=re*c-im*s; r[e+1]=re*s+im*c;
        }
        if(f.quant) {
            int g=f.quant==2?16:32;
            for(int b=0;b<len;b+=g) {
                float a=0; for(int i=0;i<g;++i) a=sycl::fmax(a,sycl::fabs(r[b+i]));
                float s=f.quant==1?pow2_ceil(sycl::fmax(a,1e-4f)*(1.0f/448.0f)):
                    f.quant==2?e4m3_round(div_rn(sycl::fmax(a,6.0f*.001953125f),6.0f)):
                    pow2_ceil(sycl::fmax(a,6.0f*1.1754943508222875e-38f)*(1.0f/6.0f));
                auto p=f.pdst?f.pdst+v*f.pdst_stride:nullptr;
                for(int i=0;i<g;++i) {
                    float lim=f.quant==1?448.0f:6.0f;
                    // Only quant==2 has a non-power-of-two scale.
                    float scaled=f.quant==2?div_rn(r[b+i],s):r[b+i]/s;
                    float z=sycl::fmin(sycl::fmax(scaled,-lim),lim);
                    float zq=f.quant==1?e4m3_round(z):e2m1_round(z);
                    r[b+i]=zq*s;
                    if(p) {
                        if(f.quant==1) p[b+i]=e4m3_code(zq);
                        else if(i%2==0) p[(b+i)/2]=e2m1_code(zq);
                        else p[(b+i)/2]|=uint8_t(e2m1_code(zq)<<4);
                    }
                }
                if(p) p[(f.quant==1?len:len/2)+b/g]=f.quant==2?e4m3_code(s):uint8_t(bits(s)>>23);
            }
        }
        for(int e=0;e<len;++e) {
            if(f.dst) f.dst[v*f.dst_stride+e]=finish_bf16(r[e]);
            else if(!f.pdst) x[v*stride+e]=r[e];
        }
    });
}
sycl::event embed(sycl::queue& q,const uint16_t* rows,int T,float* h,int d,int hc) {
    ordered(q); if(T<=0||d<=0||hc<=0) throw std::invalid_argument("embed geometry");
    return q.parallel_for(sycl::range<1>(size_t(T)*hc*d),[=](sycl::id<1> i) {
        size_t j=i[0]; h[j]=bf16_value(rows[(j/(hc*d))*d+j%d]);
    });
}
sycl::event ring_to_ext(sycl::queue& q,const uint8_t* ring,uint16_t* ext,int64_t p0,int n,int d) {
    ordered(q); if(p0<0||n<=0||d<=0||d%32) throw std::invalid_argument("ring geometry");
    return q.parallel_for(sycl::range<1>(size_t(n)*d),[=](sycl::id<1> i) {
        int r=i[0]/d,e=i[0]%d; int64_t p=p0-n+r;
        auto row=ring+size_t(p>=0?p%n:0)*row_bytes(1,d);
        ext[i[0]]=p<0?0:finish_bf16(e4m3_value(row[e])*value(uint32_t(row[d+e/32])<<23));
    });
}
sycl::event ext_to_ring(sycl::queue& q,const uint16_t* ext,uint8_t* ring,int64_t p0,int n,int T,int d,uint8_t* backup) {
    ordered(q); if(p0<0||n<=0||T<=0||d<=0||d%32) throw std::invalid_argument("ring geometry");
    return q.parallel_for(sycl::range<1>(T),[=](sycl::id<1> id) {
        int t=id[0]; if(T-t>n) return;
        auto rb=row_bytes(1,d); auto row=ring+size_t((p0+t)%n)*rb;
        if(backup) for(size_t i=0;i<rb;++i) backup[size_t(t)*rb+i]=row[i];
        auto src=ext+size_t(n+t)*d;
        for(int b=0;b<d;b+=32) {
            uint32_t a=0; for(int i=0;i<32;++i) a=sycl::max(a,bits(bf16_value(src[b+i]))&0x7fffffff);
            int k=a==0?0:int(a>>23)-135+((a&0x7fffff)>0x600000);
            for(int i=0;i<32;++i) row[b+i]=e4m3_code(bf16_value(src[b+i]),k);
            row[d+b/32]=uint8_t(k+127);
        }
    });
}
sycl::event ring_restore(sycl::queue& q,uint8_t* ring,const uint8_t* bak,int64_t p0,int n,int keep,int T,int d) {
    ordered(q); if(p0<0||n<=0||keep<0||keep>=T||T>n||d<=0||d%32) throw std::invalid_argument("restore geometry");
    return q.parallel_for(sycl::range<1>(T-keep),[=](sycl::id<1> id) {
        int t=keep+id[0]; auto rb=row_bytes(1,d);
        for(size_t i=0;i<rb;++i) ring[size_t((p0+t)%n)*rb+i]=bak[size_t(t)*rb+i];
    });
}
}
