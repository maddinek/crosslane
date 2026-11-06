// MIT, Project Maya attribution in core.hpp.
#include "hc.hpp"
#include "core.hpp"
namespace strata::ds4::sycl_port {
static void geometry(sycl::queue& q,int D,int H,int T,float eps=1) {
    ordered(q);if(D<=0||D>32768||H<1||H>4||T<1||T>8192||!std::isfinite(eps)||eps<=0)
        throw std::invalid_argument("HC geometry/epsilon");
}
static void disjoint(const void* a,size_t an,const void* b,size_t bn) {
    if(!a||!b)return;
    auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    if((x<=y?y-x<an:x-y<bn))throw std::invalid_argument("HC overlapping output/input");
}
// Fixed 256 physical work items, with four virtual lanes each for pre-norm's
// upstream 1024-lane launch. Local-memory shuffles deliberately preserve the
// pin's 32-lane XOR tree regardless of the device's actual sub-group width.
// All work items participate in every barrier, including unused virtual lanes.
template<int Threads>
static float block_sum(sycl::nd_item<1> item,sycl::local_accessor<float,1> scratch) {
    const int lane=int(item.get_local_linear_id());
    item.barrier(sycl::access::fence_space::local_space);
    for(int offset=16;offset;offset>>=1) {
        float next[Threads/256];
        for(int j=0;j<Threads/256;++j){int i=lane+j*256;next[j]=scratch[i]+scratch[i^offset];}
        item.barrier(sycl::access::fence_space::local_space);
        for(int j=0;j<Threads/256;++j)scratch[lane+j*256]=next[j];
        item.barrier(sycl::access::fence_space::local_space);
    }
    float v=lane<Threads/32?scratch[lane*32]:0.f;
    item.barrier(sycl::access::fence_space::local_space);
    if(lane<32)scratch[lane]=v;
    item.barrier(sycl::access::fence_space::local_space);
    for(int offset=16;offset;offset>>=1) {
        if(lane<32)v+=scratch[lane^offset];
        item.barrier(sycl::access::fence_space::local_space);
        if(lane<32)scratch[lane]=v;
        item.barrier(sycl::access::fence_space::local_space);
    }
    float total=scratch[0];
    item.barrier(sycl::access::fence_space::local_space);
    return total;
}
sycl::event hc_mix(sycl::queue& q,const float* h,const float* fn,int D,int N,int T,float eps,float* m) {
    geometry(q,D,1,T,eps);if(N<1||N>32||!h||!fn||!m)throw std::invalid_argument("HC mix");
    disjoint(m,size_t(T)*32*4,h,size_t(T)*D*4);disjoint(m,size_t(T)*32*4,fn,size_t(N)*D*4);
    return q.submit([&](sycl::handler& handler){
        sycl::local_accessor<float,1> scratch(sycl::range<1>(256),handler);
        handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(T)*N*256),sycl::range<1>(256)),[=](sycl::nd_item<1> item){
            const int lane=int(item.get_local_linear_id()),group=int(item.get_group_linear_id()),t=group/N,j=group%N;
            const float* x=h+size_t(t)*D;const float* w=fn+size_t(j)*D;
            scratch[lane]=hc_scalar::mix_lane(x,w,D,lane);
            float acc=block_sum<256>(item,scratch);
            scratch[lane]=hc_scalar::mix_lane(x,x,D,lane);
            float ss=block_sum<256>(item,scratch);
            if(lane==0)m[size_t(t)*32+j]=acc*(1.0f/::sqrtf(ss/float(D)+eps));
        });
    });
}
sycl::event hc_mix_scale(sycl::queue& q,const float* h,int D,int N,int T,float eps,float* m) {
    geometry(q,D,1,T,eps);if(N<1||N>32||!h||!m)throw std::invalid_argument("HC scale");
    disjoint(m,size_t(T)*32*4,h,size_t(T)*D*4);
    return q.submit([&](sycl::handler& handler){
        sycl::local_accessor<float,1> scratch(sycl::range<1>(256),handler);
        handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(T)*256),sycl::range<1>(256)),[=](sycl::nd_item<1> item){
            const int lane=int(item.get_local_linear_id()),t=int(item.get_group_linear_id());
            const float* x=h+size_t(t)*D;scratch[lane]=hc_scalar::mix_lane(x,x,D,lane);
            float ss=block_sum<256>(item,scratch);
            if(lane<N)m[size_t(t)*32+lane]*=1.0f/::sqrtf(ss/float(D)+eps);
        });
    });
}
sycl::event hc_split(sycl::queue& q,const float* m,const float* s,const float* b,int H,int it,float eps,int T,float* pre,float* post,float* c) {
    geometry(q,1,H,T,eps);if(it<1||it>1024||!m||!s||!b||!pre||!post||!c)throw std::invalid_argument("HC split");
    const void* outputs[3]={pre,post,c};size_t sizes[3]={size_t(T)*8*4,size_t(T)*8*4,size_t(T)*64*4};
    for(int j=0;j<3;++j){disjoint(outputs[j],sizes[j],m,size_t(T)*32*4);disjoint(outputs[j],sizes[j],s,12);disjoint(outputs[j],sizes[j],b,size_t(2*H+H*H)*4);
        for(int k=0;k<j;++k)disjoint(outputs[j],sizes[j],outputs[k],sizes[k]);}
    return q.parallel_for(sycl::range<1>(T),[=](sycl::id<1> i){hc_scalar::split(m,s,b,H,it,eps,i[0],pre,post,c);});
}
sycl::event hc_mix_split(sycl::queue& q,const float* h,const float* f,int D,int N,int T,float eps,float* m,const float* s,const float* b,int H,int it,float he,float* pre,float* post,float* c) {
    geometry(q,D,H,T,eps);if(N!=2*H+H*H||it<1||it>1024||!std::isfinite(he)||he<=0||!h||!f||!m||!s||!b||!pre||!post||!c)
        throw std::invalid_argument("HC mix/split");
    hc_mix(q,h,f,D,N,T,eps,m);return hc_split(q,m,s,b,H,it,he,T,pre,post,c);
}
sycl::event hc_mix_split(sycl::queue& q,const float* h,const float* f,int D,int N,int T,float eps,float* m,unsigned* cnt,const float* s,const float* b,int H,int it,float he,float* pre,float* post,float* c) {
    if(!cnt)throw std::invalid_argument("HC counters");
    hc_mix_split(q,h,f,D,N,T,eps,m,s,b,H,it,he,pre,post,c);
    return q.parallel_for(sycl::range<1>(T),[=](sycl::id<1> i){cnt[i[0]]=0;});
}
sycl::event hc_pre_norm(sycl::queue& q,const float* h,const float* m,const float* w,int D,int H,int T,float eps,float* x) {
    geometry(q,D,H,T,eps);if(!h||!m||!x||x==h)throw std::invalid_argument("HC pre alias/input");
    disjoint(x,size_t(T)*D*4,h,size_t(T)*H*D*4);disjoint(x,size_t(T)*D*4,m,size_t(T)*8*4);disjoint(x,size_t(T)*D*4,w,size_t(D)*4);
    return q.submit([&](sycl::handler& handler){
        sycl::local_accessor<float,1> scratch(sycl::range<1>(1024),handler);
        handler.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(T)*256),sycl::range<1>(256)),[=](sycl::nd_item<1> item){
            const int lane=int(item.get_local_linear_id()),t=int(item.get_group_linear_id());
            const float* hh=h+size_t(t)*H*D;const float* mm=m+size_t(t)*8;float* xx=x+size_t(t)*D;
            for(int j=0;j<4;++j){float ss=0;
                for(int e=lane+j*256;e<D;e+=1024){float a=hc_scalar::pre_value(hh,mm,D,H,e);xx[e]=a;ss+=a*a;}
                scratch[lane+j*256]=ss;
            }
            float ss=block_sum<1024>(item,scratch);
            if(w){float inv=1.0f/::sqrtf(ss/float(D)+eps);
                for(int e=lane;e<D;e+=256)xx[e]=xx[e]*inv*w[e];}
        });
    });
}
sycl::event hc_post(sycl::queue& q,const float* y,const float* h,const float* p,const float* c,int D,int H,int T,float* out) {
    geometry(q,D,H,T);if(!y||!h||!p||!c||!out||y==out||h==out)throw std::invalid_argument("HC post alias/input");
    disjoint(out,size_t(T)*H*D*4,h,size_t(T)*H*D*4);disjoint(out,size_t(T)*H*D*4,y,size_t(T)*D*4);disjoint(out,size_t(T)*H*D*4,p,size_t(T)*8*4);disjoint(out,size_t(T)*H*D*4,c,size_t(T)*64*4);
    return q.parallel_for(sycl::range<1>(size_t(T)*D),[=](sycl::id<1> i){hc_scalar::post(y,h,p,c,D,H,i[0]/D,i[0]%D,out);});
}
sycl::event hc_post_inplace(sycl::queue& q,const float* y,float* h,const float* p,const float* c,int D,int H,int T) {
    geometry(q,D,H,T);if(!y||!h||!p||!c||y==h)throw std::invalid_argument("HC inplace input");
    disjoint(h,size_t(T)*H*D*4,y,size_t(T)*D*4);disjoint(h,size_t(T)*H*D*4,p,size_t(T)*8*4);disjoint(h,size_t(T)*H*D*4,c,size_t(T)*64*4);
    return q.parallel_for(sycl::range<1>(size_t(T)*D),[=](sycl::id<1> i){hc_scalar::post(y,h,p,c,D,H,i[0]/D,i[0]%D,h);});
}
}
