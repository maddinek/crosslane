// MIT; derived from pinned Strata Engram kernels; see decode.hpp.
#include "engram.hpp"
#include "decode.hpp"
#include <stdexcept>
#include <limits>
#include <cmath>
#include "../device.hpp"
namespace strata::ds4::engram {
static void ordered(sycl::queue& q) {
    if(!strata::intel_level_zero(q.get_device())) throw std::invalid_argument("Intel Level Zero queue required");
    if(!q.has_property<sycl::property::queue::in_order>()) throw std::invalid_argument("Engram requires in-order queue");
}
static size_t extent(size_t a,size_t b) {
    if(b && a>std::numeric_limits<size_t>::max()/b) throw std::invalid_argument("Engram extent overflow");
    return a*b;
}
static void geometry(int n,int hd,int fmt) {
    if(n<=0||hd<=0||hd%32||fmt<0||fmt>2||(fmt==1&&hd%256)) throw std::invalid_argument("Engram geometry/format");
}
sycl::event dequant(sycl::queue& q,const uint8_t* rows,const uint8_t* sc,int n,int hd,float* out,int fmt) {
    ordered(q); geometry(n,hd,fmt); extent(extent(size_t(n),size_t(hd)),sizeof(float));
    if(!rows||!out||(fmt==0&&!sc)) throw std::invalid_argument("Engram null");
    return q.parallel_for(sycl::range<1>(size_t(n)*hd),[=](sycl::id<1> id) {
        size_t j=id[0],r=j/hd; out[j]=row_value(rows+r*hd,fmt==0?sc+r*(hd/32):nullptr,j%hd,fmt);
    });
}
sycl::event dequant_chunk(sycl::queue& q,const uint8_t* src,size_t per,int cols,int T,int hd,float* out,int fmt) {
    ordered(q); geometry(T,hd,fmt);
    if(cols<=0||!src||!out||per<extent(size_t(cols),size_t(hd)+hd/32)) throw std::invalid_argument("Engram chunk stride");
    extent(extent(extent(size_t(T),size_t(cols)),size_t(hd)),sizeof(float));
    extent(size_t(T),per);
    return q.parallel_for(sycl::range<1>(size_t(T)*cols*hd),[=](sycl::id<1> id) {
        size_t j=id[0],r=j/hd,t=r/cols,c=r%cols;
        out[j]=row_value(src+t*per+c*hd,fmt==0?src+t*per+size_t(cols)*hd+c*(hd/32):nullptr,j%hd,fmt);
    });
}
sycl::event gate(sycl::queue& q,float* h,const float* kv,const float* w,int hc,int dim,int T,float eps) {
    ordered(q);
    if(!h||!kv||!w||hc<=0||dim<=0||T<=0||!std::isfinite(eps)||eps<0) throw std::invalid_argument("Engram gate geometry");
    extent(extent(extent(size_t(T),size_t(hc)+1),size_t(dim)),sizeof(float));
    // Scalar baseline: arithmetic gate under unchanged primitive tolerance, not bitwise reduction parity.
    return q.parallel_for(sycl::range<1>(size_t(T)*hc),[=](sycl::id<1> id) {
        size_t j=id[0],t=j/hc,k=j%hc; auto hk=h+j*dim; auto key=kv+(t*(hc+1)+k)*dim;
        auto val=kv+(t*(hc+1)+hc)*dim; auto wk=w+k*dim;
        float sh=0,sk=0,sd=0;
        for(int e=0;e<dim;++e) {float hv=hk[e],v=key[e];sh+=hv*hv;sk+=v*v;sd+=(hv*wk[e])*v;}
        float dot=sd*(sycl::rsqrt(sh/float(dim)+eps)*sycl::rsqrt(sk/float(dim)+eps))*sycl::rsqrt(float(dim));
        float sq=sycl::copysign(sycl::sqrt(sycl::fmax(sycl::fabs(dot),1e-6f)),dot);
        float g=1.0f/(1.0f+sycl::exp(-sq));
        for(int e=0;e<dim;++e) hk[e]=sycl::fma(g,val[e],hk[e]);
    });
}
template<class W> static sycl::event projection(sycl::queue& q,const float* x,const W* w,float* y,int T,int in,int out) {
    ordered(q); if(!x||!w||!y||T<=0||in<=0||out<=0) throw std::invalid_argument("Engram projection geometry");
    extent(extent(size_t(T),size_t(in)),sizeof(float)); extent(extent(size_t(T),size_t(out)),sizeof(float));
    extent(extent(size_t(in),size_t(out)),sizeof(W));
    return q.parallel_for(sycl::range<1>(size_t(T)*out),[=](sycl::id<1> id) {
        size_t j=id[0],t=j/out,r=j%out;float a=0;
        for(int i=0;i<in;++i) {
            float v;if constexpr(sizeof(W)==2) v=fbits(uint32_t(w[r*in+i])<<16);else v=w[r*in+i];
            a+=x[t*in+i]*v;
        }
        y[j]=a;
    });
}
sycl::event project_f32(sycl::queue& q,const float* x,const float* w,float* y,int T,int in,int out) { return projection(q,x,w,y,T,in,out); }
sycl::event project_bf16(sycl::queue& q,const float* x,const uint16_t* w,float* y,int T,int in,int out) { return projection(q,x,w,y,T,in,out); }
}
