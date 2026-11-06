// MIT; Project Maya attribution in core.hpp. Scalar correctness path, no subgroup assumptions.
#include "attn.hpp"
#include <algorithm>
#include <limits>
namespace strata::ds4::sycl_port {
static void count(sycl::queue& q,int T,bool generated=false){ordered(q);if(T<=0||T>(generated?8192:16))throw std::invalid_argument("attention query count");}
static void rows_check(Rows r,int T,long long stride,int out=0){
    if(r.gen<0||r.gen>4||r.R<=0||r.p0<0||r.p0>INT32_MAX-T||r.blk<=0||r.blk>1048576||r.kcap<0||r.kcap>1048576||r.ccap<0||r.ccap>1048576)throw std::invalid_argument("index rows descriptor");
    // Bound generated arithmetic before rows_at's int products/additions.
    if(r.gen){long long nn=(r.p0+T)/r.R,cb=std::min<long long>(r.ccap,(nn+r.blk-1)/r.blk);
        if(nn>1048576||(r.gen==3&&cb*r.blk>1048576))throw std::invalid_argument("generated index extent");}
    for(int t=0;t<T;++t){int n,k;attn_scalar::rows_at(r,t,n,k);if(n<0||k<0||k>1048576||n>1048576||stride<n||stride>INT64_MAX/T||out<0)throw std::invalid_argument("index rows/stride");}
}
sycl::event compress_win(sycl::queue& q,const float* kv,const float* score,float* ks,float* ss,int R,int p0,int T,int D,const float* w,float eps,float* latent){
    count(q,T,true);if(!kv||!score||!ks||!ss||!w||!latent||R<1||R>1048576||D<1||D>1024||p0<0||p0>INT32_MAX-T||!std::isfinite(eps)||eps<=0)
        throw std::invalid_argument("compressor geometry/input");
    return q.single_task([=]{attn_scalar::compress(kv,score,ks,ss,R,p0,T,D,w,eps,latent);});
}
sycl::event index_score(sycl::queue& q,const float* x,const float* w,const void* keys,int H,int D,int T,int n,const int* c,int cs,int blk,const Rows& r,float* out,long long stride,int fmt){
    count(q,T,r.gen!=0);rows_check(r,T,1048576);if(!x||!w||!keys||!out||H<1||H>32||D!=128||n<0||n>1048576||stride<n||stride>INT64_MAX/T||blk<=0||blk>1048576||(c&&cs<1)||(fmt!=0&&fmt!=3))throw std::invalid_argument("index geometry/format");
    for(int t=0;t<T;++t){int rn,k;attn_scalar::rows_at(r,t,rn,k);if(c&&cs<k)throw std::invalid_argument("candidate stride");}
    return q.parallel_for(sycl::range<1>(T),[=](sycl::id<1> i){attn_scalar::index(x,w,(const uint8_t*)keys,H,D,i[0],n,c,cs,blk,r,out,stride,fmt);});
}
sycl::event topk(sycl::queue& q,const float* v,long long stride,int T,const Rows& r,const int* c,int cs,int blk,int* out,int os){
    count(q,T,r.gen!=0);rows_check(r,T,stride);if(!v||!out||os<1||blk<1||blk>1048576||(c&&cs<1))throw std::invalid_argument("topk input");
    for(int t=0;t<T;++t){int n,k;attn_scalar::rows_at(r,t,n,k);if(os<attn_scalar::min_(n,k)||(c&&cs<(n+blk-1)/blk))throw std::invalid_argument("topk output/candidate stride");}
    return q.parallel_for(sycl::range<1>(T),[=](sycl::id<1> i){int t=i[0],n,k;attn_scalar::rows_at(r,t,n,k);attn_scalar::select(v+t*stride,n,k,c?c+size_t(t)*cs:nullptr,blk,out+size_t(t)*os);});
}
sycl::event cand_blocks(sycl::queue& q,const float* v,long long stride,int T,const Rows& r,int blk,float* bm,long long bs,int* out,int os){
    count(q,T,r.gen!=0);rows_check(r,T,stride);if(r.gen==3||r.gen==4)throw std::invalid_argument("cand_blocks does not accept candidate/block Rows gen 3/4");
    if(!v||!bm||!out||blk<1||blk>1048576||bs<1||bs>INT64_MAX/T||os<1)throw std::invalid_argument("candidate input");
    for(int t=0;t<T;++t){int n,k;attn_scalar::rows_at(r,t,n,k);if(r.gen)k=r.kcap;int nb=(n+blk-1)/blk;if(bs<nb||os<attn_scalar::min_(nb,k))throw std::invalid_argument("candidate strides");}
    return q.parallel_for(sycl::range<1>(T),[=](sycl::id<1> i){attn_scalar::blocks(v,stride,i[0],r,blk,bm,bs,out,os);});
}
static void attn_check(sycl::queue& q,const float* x,const uint16_t* win,const void* kv,const int* sel,int ss,AttnRows a,const float* sink,int H,int D,int T,float scale,float* out,int fmt){
    count(q,T,a.gen!=0);if(!x||!win||!sink||!out||D!=512||H<1||H>128||(fmt!=0&&fmt!=2)||!std::isfinite(scale)||a.gen<0||a.gen>1||a.t0<0||a.t0>INT32_MAX-T||a.p0<0||a.p0>INT32_MAX-a.t0-T||a.win<0||a.R<0||a.kcap<0)throw std::invalid_argument("attention geometry/format");
    for(int t=0;t<T;++t){int w0,nw,ns;attn_scalar::attn_at(a,t,w0,nw,ns);if(w0<0||w0>1048576||nw<0||ns<0||nw>1048576||ns>1048576||int64_t(nw)+ns>1048576||ss<ns||(ns&&(!kv||!sel)))throw std::invalid_argument("attention visibility/selection");}
}
sycl::event sparse_attn(sycl::queue& q,const float* x,const uint16_t* win,const void* kv,const int* sel,int ss,const AttnRows& a,const float* sink,int H,int D,int T,float scale,float* out,int fmt){
    attn_check(q,x,win,kv,sel,ss,a,sink,H,D,T,scale,out,fmt);
    return q.parallel_for(sycl::range<1>(size_t(T)*H),[=](sycl::id<1> i){attn_scalar::attention(x,win,(const uint8_t*)kv,sel,ss,a,sink,H,i[0]/H,i[0]%H,scale,fmt,out);});
}
sycl::event sparse_attn_subgroup32(sycl::queue& q,const float* x,const uint16_t* win,const void* kv,const int* sel,int ss,const AttnRows& a,const float* sink,int H,int D,int T,float scale,float* out,int fmt){
    attn_check(q,x,win,kv,sel,ss,a,sink,H,D,T,scale,out,fmt);
    return q.parallel_for(sycl::nd_range<1>(sycl::range<1>(size_t(T)*H*32),sycl::range<1>(32)),[=](sycl::nd_item<1> it)
#ifndef STRATA_ROW34_HOST_API_TEST
        [[sycl::reqd_sub_group_size(32)]]
#endif
    {
        const size_t head=it.get_group_linear_id();const int lane=int(it.get_local_linear_id()),t=int(head/H),h=int(head%H);
        const auto sg=it.get_sub_group();int w0,nw,ns;attn_scalar::attn_at(a,t,w0,nw,ns);
        const auto* query=x+head*512;const auto* selected=sel?sel+size_t(t)*ss:nullptr;
        float acc[16]={},m=-1e30f,den=0;
        // This is core.hpp::partial, with one work-item per original logical
        // lane. Keep both the per-lane FMA order and the XOR tree unchanged.
        for(int first=0;first<nw+ns;first+=64){
            const int nt=attn_scalar::min_(64,nw+ns-first);float sc[64],mx=-INFINITY;
            for(int j=0;j<nt;++j){
                float dot=0;
                for(int e=lane*16;e<(lane+1)*16;++e)
                    dot=attn_scalar::index_fma(query[e],attn_scalar::row_value(win,(const uint8_t*)kv,selected,w0,nw,first+j,e,fmt),dot);
                for(int offset=16;offset;offset>>=1)dot+=sycl::permute_group_by_xor(sg,dot,offset);
                // Make the scalar oracle's lane-zero return explicit.
                sc[j]=sycl::select_from_group(sg,dot,sycl::id<1>(0))*scale;mx=::fmaxf(mx,sc[j]);
            }
            const float nm=::fmaxf(m,mx),cs=::expf(m-nm);float d=0;
            for(int j=0;j<nt;++j){sc[j]=::expf(sc[j]-nm);if(j%32==lane)d+=sc[j];}
            for(int offset=16;offset;offset>>=1)d+=sycl::permute_group_by_xor(sg,d,offset);
            den=attn_scalar::index_fma(den,cs,sycl::select_from_group(sg,d,sycl::id<1>(0)));
            for(int e=0;e<16;++e){
                acc[e]*=cs;
                for(int j=0;j<nt;++j)acc[e]=attn_scalar::index_fma(sc[j],attn_scalar::row_value(win,(const uint8_t*)kv,selected,w0,nw,first+j,lane*16+e,fmt),acc[e]);
            }
            m=nm;
        }
        den+=::expf(sink[h]-m);
        for(int e=0;e<16;++e)out[head*512+lane*16+e]=nw+ns?attn_scalar::attention_div(acc[e],den):0.0f;
    });
}
sycl::event sparse_attn_split(sycl::queue& q,const float* x,const uint16_t* win,const void* kv,const int* sel,int ss,const AttnRows& a,const float* sink,int H,int D,int T,float scale,float* out,int fmt,int R,float* scratch,size_t floats){
    attn_check(q,x,win,kv,sel,ss,a,sink,H,D,T,scale,out,fmt);
    if(R<1||R>1024)throw std::invalid_argument("split attention rows per chunk");
    int total=0;for(int t=0;t<T;++t){int w0,nw,ns;attn_scalar::attn_at(a,t,w0,nw,ns);total=std::max(total,nw+ns);}
    int splits=std::max(1,(total+R-1)/R);
    if(splits>1024||!scratch||floats<size_t(T)*H*splits*514)throw std::invalid_argument("split attention scratch");
    q.parallel_for(sycl::range<1>(size_t(T)*H*splits),[=](sycl::id<1> i){size_t j=i[0];int s=j%splits,h=(j/splits)%H,t=j/(size_t(splits)*H);int w0,nw,ns;attn_scalar::attn_at(a,t,w0,nw,ns);int chunk=R;
        attn_scalar::split_partial(x+(size_t(t)*H+h)*512,win,(const uint8_t*)kv,sel?sel+size_t(t)*ss:nullptr,w0,nw,s*chunk,attn_scalar::min_(nw+ns,(s+1)*chunk),fmt,scale,scratch+j*514);});
    return q.parallel_for(sycl::range<1>(size_t(T)*H),[=](sycl::id<1> i){size_t j=i[0];int h=j%H;auto p=scratch+j*splits*514;
        float m=sink[h];for(int s=0;s<splits;++s)if(p[s*514+1]>0)m=::fmaxf(m,p[s*514]);
        float den=::expf(sink[h]-m);for(int s=0;s<splits;++s)if(p[s*514+1]>0)den=sycl::fma(p[s*514+1],::expf(p[s*514]-m),den);
        for(int e=0;e<512;++e){float acc=0;for(int s=0;s<splits;++s)if(p[s*514+1]>0)acc=sycl::fma(p[s*514+e+2],::expf(p[s*514]-m),acc);out[j*512+e]=attn_scalar::attention_div(acc,den);}});
}
}
