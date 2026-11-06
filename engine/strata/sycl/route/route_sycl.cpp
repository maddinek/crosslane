// Scalar correctness path; no subgroup-size assumption or device spin waits.
#include "route_sycl.hpp"
#include <cmath>
#include "../device.hpp"
#include <limits>
namespace strata::ds4::route_port {
namespace {
void ordered(sycl::queue& q) {
    if(!strata::intel_level_zero(q.get_device())) throw std::invalid_argument("Intel Level Zero queue required");
    if(!q.has_property<sycl::property::queue::in_order>()) throw std::invalid_argument("in-order queue required");
}
size_t product(size_t a,size_t b) {
    if(b && a>std::numeric_limits<size_t>::max()/b) throw std::invalid_argument("shape overflow");
    return a*b;
}
}
sycl::event route_device(sycl::queue& q,const float* logits,size_t stride,const float* bias,
                         size_t n,size_t k,size_t T,float scale,int* ids,float* weights,float* scratch) {
    ordered(q);
    if(!n || !k || !T || k>n || stride<n || n>size_t(std::numeric_limits<int>::max()) ||
       !logits || !bias || !ids || !weights || !scratch || !std::isfinite(scale) || scale<0)
        throw std::invalid_argument("route geometry/pointer/scale");
    product(product(T,n),2); product(T,stride); product(T,k);
    return q.parallel_for(sycl::range<1>(T),[=](sycl::id<1> tid) {
        size_t t=tid[0]; float* sc=scratch+t*n*2; float* bs=sc+n;
        for(size_t e=0;e<n;++e) {
            float l=logits[t*stride+e]; sc[e]=sycl::sqrt(l>20.0f?l:sycl::log1p(sycl::exp(l))); bs[e]=sc[e]+bias[e];
        }
        float sum=0;
        for(size_t i=0;i<k;++i) {
            size_t best=0; for(size_t e=1;e<n;++e) if(bs[e]>bs[best]) best=e;
            ids[t*k+i]=int(best); sum+=sc[best]; bs[best]=-std::numeric_limits<float>::max();
        }
        for(size_t i=0;i<k;++i) weights[t*k+i]=sc[ids[t*k+i]]/(sum+1e-20f)*scale;
    });
}
Req read_request(sycl::queue& q,uint64_t request,uint64_t epoch,uint64_t seq,size_t layer,
                 size_t D,size_t n,size_t k,size_t T,const int* ids,const float* weights,
                 const float* x,const float* shared) {
    ordered(q);
    if(!D || !T || !n || !k || k>n || !ids || !weights || !x || !shared) throw std::invalid_argument("request geometry/pointer");
    Req r; r.request=request; r.epoch=epoch; r.seq=seq; r.layer=layer; r.dim=D;
    r.route.tokens=T; r.route.experts=n; r.route.picks=k;
    size_t slots=product(T,k),elts=product(T,D);
    r.route.ids.resize(slots); r.route.weights.resize(slots); r.x.resize(elts); r.shared.resize(elts);
    q.memcpy(r.route.ids.data(),ids,product(slots,sizeof(int))).wait_and_throw();
    q.memcpy(r.route.weights.data(),weights,product(slots,sizeof(float))).wait_and_throw();
    q.memcpy(r.x.data(),x,product(elts,sizeof(float))).wait_and_throw();
    q.memcpy(r.shared.data(),shared,product(elts,sizeof(float))).wait_and_throw(); return r;
}
void publish_response(sycl::queue& q,const Resp& r,int* cpu,float* part,float* combined) {
    ordered(q);
    if(!cpu || !part || !combined) throw std::invalid_argument("response pointer");
    q.memcpy(cpu,r.cpu.data(),product(r.cpu.size(),sizeof(int))).wait_and_throw();
    q.memcpy(part,r.part.data(),product(r.part.size(),sizeof(float))).wait_and_throw();
    q.memcpy(combined,r.combined.data(),product(r.combined.size(),sizeof(float))).wait_and_throw();
}
void stage_copy(sycl::queue& q,void* dst,std::span<const std::byte> src,size_t expected) {
    ordered(q);
    if(!dst || !expected || src.size()!=expected) throw std::invalid_argument("short store staging transfer");
    q.memcpy(dst,src.data(),expected).wait_and_throw();
}
}
