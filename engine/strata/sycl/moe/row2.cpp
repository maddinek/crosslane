// Strata c62d5e3 row-2 SYCL port; MIT, see licenses/ and source provenance.
#include "../stage/wiring/sycl_row2.hpp"
#include "../stage/wiring/cpu_lane.hpp"
#include "core.hpp"
#include "iq2_esimd.hpp"
#include "../division.hpp"
#include <array>
#include <cmath>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
namespace strata::ds4::row2 {
namespace {
namespace m=moe;
struct Math {
    static float fma(float a,float b,float c){return sycl::fma(a,b,c);}
    static float min(float a,float b){return sycl::fmin(a,b);}
    static float max(float a,float b){return sycl::fmax(a,b);}
    static float exp(float x){return sycl::exp(x);}
};
void need(bool ok,const char* msg){if(!ok)throw std::invalid_argument(msg);}
// FP8-grid o_fin hits int8 near-ties: a 1-ulp '/' error changes the code.
// See results/strata-cross-vendor/parity-g01/step18-sycl-div/DIAGNOSIS.md.
using sycl_port::div_rn;
struct Temp {
    sycl::queue& q;std::vector<void*> ptrs;
    explicit Temp(sycl::queue& queue):q(queue){}
    ~Temp(){try{q.wait_and_throw();}catch(...){}for(auto p:ptrs)sycl::free(p,q);}
    template<class T>T* get(size_t n){auto p=sycl::malloc_device<T>(n,q);if(!p)throw std::bad_alloc();ptrs.push_back(p);return p;}
};
// SPMD geometry arm: independent subgroup32 rows, no cross-row operations.
template<int Rows>
sycl::event iq_spmd(sycl::queue& q,const uint8_t* W,size_t ld,int n,int k,int cols,
                    float alpha,float* y,size_t ys,const int8_t* xq,const float* xd) {
    const size_t strips=(size_t(n)+Rows-1)/Rows;
    return q.parallel_for(sycl::nd_range<1>(strips*cols*Rows*32,Rows*32),[=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        size_t group=it.get_group_linear_id(),c=group/strips;
        int r=int(group%strips)*Rows+int(it.get_local_linear_id()/32),l=int(it.get_local_linear_id()%32);
        if(r>=n)return; // the entire subgroup is absent, never a partial subgroup
        float a=m::lane<Math>(m::Type::iq2_xxs,W+size_t(r)*ld,k,nullptr,xq+c*k,xd+c*(k/32),l,true);
        auto sg=it.get_sub_group();for(int o=16;o;o>>=1)a+=sycl::permute_group_by_xor(sg,a,o);
        if(!l){float v=alpha*a;y[c*ys+r]=v+0.0f;}
    });
}
template<int Rows,bool Load=false,bool Compact=false>
sycl::event iq_esimd(sycl::queue& q,const uint8_t* W,size_t ld,int n,int k,int cols,
                     float alpha,float* y,size_t ys,const int8_t* xq,const float* xd) {
    const size_t strips=(size_t(n)+Rows-1)/Rows;
    // One ESIMD work-item owns its own 32*Rows logical lane registers.
    return q.parallel_for(sycl::range<1>(strips*cols),[=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
        size_t c=id[0]/strips;int first=int(id[0]%strips)*Rows;
        iq2::strip<Rows,Load,Compact>(W,ld,xq+c*k,xd+c*(k/32),n,k,first,alpha,y+c*ys);
    });
}
// Single conditional arm, pinned llama.cpp esimd.hpp property form. The
// existing compact128 instantiation above and all control launches stay intact.
sycl::event iq_esimd_grf256(sycl::queue& q,const uint8_t* W,size_t ld,int n,int k,int cols,
                           float alpha,float* y,size_t ys,const int8_t* xq,const float* xd) {
    const size_t strips=size_t(n);
    return q.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(sycl::range<1>(strips*cols),
                         sycl::ext::oneapi::experimental::properties{
                             sycl::ext::intel::experimental::grf_size<256> },
                         [=](sycl::id<1> id) SYCL_ESIMD_KERNEL {
            size_t c=id[0]/strips;int first=int(id[0]%strips);
            iq2::strip<1,true,true>(W,ld,xq+c*k,xd+c*(k/32),n,k,first,alpha,y+c*ys);
        });
    });
}
// One logical CUDA warp per row; subgroup width is an explicit contract.
sycl::event launch(sycl::queue& q,m::Type t,const uint8_t* W,size_t ld,const float* x,size_t xs,
                   int n,int k,int cols,float alpha,float* y,size_t ys,const int8_t* xq,const float* xd,bool vec,IqPath path=IqPath::control) {
    check_iq_path(path);
    if(t==m::Type::iq2_xxs) {
        switch(path) {
        case IqPath::esimd1:return iq_esimd<1>(q,W,ld,n,k,cols,alpha,y,ys,xq,xd);
        case IqPath::esimd1_load:return iq_esimd<1,true>(q,W,ld,n,k,cols,alpha,y,ys,xq,xd);
        case IqPath::esimd1_cb:return iq_esimd<1,false,true>(q,W,ld,n,k,cols,alpha,y,ys,xq,xd);
        case IqPath::esimd1_load_cb:return iq_esimd<1,true,true>(q,W,ld,n,k,cols,alpha,y,ys,xq,xd);
        case IqPath::esimd1_load_cb_grf256:return iq_esimd_grf256(q,W,ld,n,k,cols,alpha,y,ys,xq,xd);
        case IqPath::esimd2:return iq_esimd<2>(q,W,ld,n,k,cols,alpha,y,ys,xq,xd);
        case IqPath::esimd4:return iq_esimd<4>(q,W,ld,n,k,cols,alpha,y,ys,xq,xd);
        case IqPath::spmd4:return iq_spmd<4>(q,W,ld,n,k,cols,alpha,y,ys,xq,xd);
        case IqPath::spmd8:return iq_spmd<8>(q,W,ld,n,k,cols,alpha,y,ys,xq,xd);
        case IqPath::control:break;
        }
    }
    return q.parallel_for(sycl::nd_range<1>(size_t(n)*cols*32,32),[=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        size_t rc=it.get_group_linear_id();int r=int(rc%size_t(n)),c=int(rc/size_t(n)),l=int(it.get_local_linear_id());
        float a=m::lane<Math>(t,W+size_t(r)*ld,k,x+size_t(c)*xs,xq?xq+size_t(c)*k:nullptr,xd?xd+size_t(c)*(k/32):nullptr,l,vec);
        auto sg=it.get_sub_group();for(int o=16;o;o>>=1)a+=sycl::permute_group_by_xor(sg,a,o);
        if(!l){float v=alpha*a;y[size_t(c)*ys+r]=(t==m::Type::bf16||m::block_values(t)>1)?v+0.0f:v;}
    });
}
sycl::event quant(sycl::queue& q,const float* x,size_t xs,int k,int cols,int8_t* a,float* d) {
    return q.parallel_for(sycl::nd_range<1>(size_t(k)*cols,32),[=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        size_t b=it.get_group_linear_id(),c=b/size_t(k/32),kb=b%size_t(k/32),l=it.get_local_linear_id();
        float v=x[c*xs+kb*32+l],am=sycl::fabs(v);auto sg=it.get_sub_group();
        for(int o=16;o;o>>=1)am=sycl::fmax(am,sycl::permute_group_by_xor(sg,am,o));
        float sc=div_rn(am,127.0f);a[c*size_t(k)+kb*32+l]=int8_t(am>0?int(sycl::rint(div_rn(v,sc))):0);if(!l)d[b]=sc;
    });
}
std::string tensor_name(const wiring::WeightSource& ws,const std::string& n){return ws.find(n)?n:n+".weight";}
void geometry(int n,int k,int cols,size_t xs,size_t ys){need(n>0&&k>0&&cols>0&&cols<=16,"row2 invalid matrix geometry");
    need(xs>=size_t(k)&&ys>=size_t(n),"row2 short column stride");m::product(size_t(cols),xs);m::product(size_t(cols),ys);}
}
DenseMat upload(sycl::queue& q,const wiring::WeightSource& ws,const std::string& name) {
    sycl_port::ordered(q);auto n=tensor_name(ws,name);auto t=ws.require_tensor(n);auto ty=m::type(t.dtype);
    need(t.shape.size()==2&&t.shape[0]>0&&t.shape[1]>0&&t.shape[0]<=INT32_MAX&&t.shape[1]<=INT32_MAX,"row2 matrix shape");
    DenseMat a{n,t.dtype,int(t.shape[0]),int(t.shape[1]),0,nullptr};a.ld=m::row_bytes(ty,a.k);
    need(t.bytes==m::product(size_t(a.n),a.ld),"row2 tensor byte extent");
    need(ty!=m::Type::f16 || a.k%8==0,"row2 F16 projection requires whole eight-value vectors");
    need(ty!=m::Type::f32 || a.k%4==0,"row2 F32 projection requires whole four-value vectors");
    auto bytes=ws.read_all(n);a.raw=sycl::malloc_device(bytes.size(),q);if(!a.raw)throw std::bad_alloc();
    try{q.memcpy(a.raw,bytes.data(),bytes.size()).wait_and_throw();}catch(...){sycl::free(a.raw,q);throw;}return a;
}
void release(sycl::queue& q,DenseMat& a) noexcept{if(a.raw){try{q.wait_and_throw();}catch(...){}try{sycl::free(a.raw,q);}catch(...){}a.raw=nullptr;}}
sycl::event mv(sycl::queue& q,std::span<const MvJob> jobs,IqPath path) {
    check_iq_path(path);
    sycl_port::ordered(q);need(!jobs.empty()&&jobs.size()<=8,"row2 expects one to eight mv jobs");
    bool vec=true; // Upstream groups BF16 jobs and selects VEC for the entire group.
    for(const auto& b:jobs){need(b.m&&b.m->raw&&b.x&&b.y,"row2 null mv operand");auto& a=*b.m;auto t=m::type(a.dtype);
        geometry(a.n,a.k,b.ncol,b.x_cs?b.x_cs:size_t(a.k),b.y_cs?b.y_cs:size_t(a.n));
        need(a.ld>=m::row_bytes(t,a.k),"row2 short weight stride");need(std::isfinite(b.alpha),"row2 nonfinite alpha");
        need(t!=m::Type::f16 || a.k%8==0,"row2 F16 vector tail");need(t!=m::Type::f32 || a.k%4==0,"row2 F32 vector tail");
        if(t==m::Type::bf16)vec=vec&&!(uintptr_t(b.x)&15)&&!(uintptr_t(a.raw)&15)&&a.k%8==0&&a.ld%16==0&&(b.ncol==1||b.x_cs%4==0);
    }
    Temp temp(q);sycl::event last;
    for(const auto& b:jobs){auto& a=*b.m;auto t=m::type(a.dtype);size_t xs=b.x_cs?b.x_cs:size_t(a.k),ys=b.y_cs?b.y_cs:size_t(a.n);
        int8_t* xq=nullptr;float* xd=nullptr;
        if(m::block_values(t)>1){xq=temp.get<int8_t>(size_t(a.k)*b.ncol);xd=temp.get<float>(size_t(a.k/32)*b.ncol);quant(q,b.x,xs,a.k,b.ncol,xq,xd);}
        last=launch(q,t,static_cast<const uint8_t*>(a.raw),a.ld,b.x,xs,a.n,a.k,b.ncol,b.alpha,b.y,ys,xq,xd,vec,path);
    }
    // Scratch belongs to this call: explicit completion before release, no host_task lifetime/thread assumptions.
    last.wait_and_throw();return last;
}
sycl::event gemv_f32(sycl::queue& q,const float* W,const float* x,int n,int k,int T,float* y,int ldy) {
    sycl_port::ordered(q);need(W&&x&&y&&k%4==0&&ldy>=n,"row2 F32 operands/vector tail");geometry(n,k,T,size_t(k),size_t(ldy));
    return launch(q,m::Type::f32,reinterpret_cast<const uint8_t*>(W),size_t(k)*4,x,size_t(k),n,k,T,1,y,size_t(ldy),nullptr,nullptr,true);
}
sycl::event gemv_f16_rows(sycl::queue& q,const uint16_t* W,const float* x,size_t ldx,int n,int k,int T,float alpha,float* y,size_t ldy) {
    sycl_port::ordered(q);need(W&&x&&y&&k%8==0&&std::isfinite(alpha),"row2 F16 operands/vector tail");geometry(n,k,T,ldx,ldy);
    return launch(q,m::Type::f16,reinterpret_cast<const uint8_t*>(W),size_t(k)*2,x,ldx,n,k,T,alpha,y,ldy,nullptr,nullptr,true);
}
sycl::event swiglu(sycl::queue& q,const float* g,const float* u,int n,float lim,float* h) {
    sycl_port::ordered(q);need(g&&u&&h&&n>0&&std::isfinite(lim),"row2 swiglu geometry/limit");
    return q.parallel_for(sycl::range<1>(size_t(n)),[=](sycl::id<1> i){h[i]=m::swiglu<Math>(g[i],u[i],lim);});
}
namespace {
class Experts final:public ExpertCompute {
    sycl::queue& q_;const wiring::WeightSource& ws_;uint32_t first_,end_;std::unique_ptr<wiring::CpuLane> cpu_;IqPath path_;
public:
    Experts(sycl::queue& q,const wiring::WeightSource& ws,uint32_t first,uint32_t end,const std::string& placement,int threads,const std::string& dispatch,bool timing,IqPath path)
    :q_(q),ws_(ws),first_(first),end_(end),path_(path){check_iq_path(path);need(placement=="all-cpu","row2 only frozen all-cpu placement is admitted");cpu_=wiring::make_cpu_lane(ws,first,end,threads,dispatch,timing);}
    void check(size_t l,int e)const{need(l>=first_&&l<end_&&e>=0&&e<ws_.meta().n_exp,"row2 expert index");}
    bool resident(size_t l,int e)const override{check(l,e);return false;}
    bool store_contains(size_t l,int e)const override{check(l,e);return false;}
    void stage(size_t,int)override{throw std::invalid_argument("row2 all-cpu placement has no device cache");}
    std::vector<float> cpu(size_t l,int e,std::span<const float> x)override{check(l,e);return cpu_->expert(uint32_t(l),e,x);}
    std::vector<float> device(size_t l,int e,std::span<const float> x)override{
        check(l,e);need(x.size()==size_t(ws_.meta().dim),"row2 expert input extent");
        std::string p=wiring::layer_prefix(uint32_t(l))+"ffn.experts."+std::to_string(e)+".";
        std::array<DenseMat,3>a{};Temp tmp(q_);
        try{a[0]=upload(q_,ws_,p+"w1.weight");a[1]=upload(q_,ws_,p+"w3.weight");a[2]=upload(q_,ws_,p+"w2.weight");
            int d=int(x.size()),ff=ws_.meta().ff;need(a[0].k==d&&a[0].n==ff&&a[1].k==d&&a[1].n==ff&&a[2].k==ff&&a[2].n==d,"row2 expert shape");
            float* dx=tmp.get<float>(x.size()),*g=tmp.get<float>(ff),*u=tmp.get<float>(ff),*h=tmp.get<float>(ff),*y=tmp.get<float>(d);
            q_.memcpy(dx,x.data(),x.size_bytes());std::array<MvJob,2> gu{{{&a[0],dx,g,1,size_t(d),size_t(ff),1},{&a[1],dx,u,1,size_t(d),size_t(ff),1}}};
            mv(q_,gu,path_);swiglu(q_,g,u,ff,ws_.meta().lim,h);MvJob down{&a[2],h,y,1,size_t(ff),size_t(d),1};mv(q_,std::span(&down,1),path_);
            std::vector<float> out(x.size());q_.memcpy(out.data(),y,out.size()*4).wait_and_throw();for(auto& z:a)release(q_,z);return out;
        }catch(...){for(auto& z:a)release(q_,z);throw;}
    }
    uint64_t device_bytes(uint32_t l)const override{need(l>=first_&&l<end_,"row2 layer index");return 0;}
    void begin_decode_step(bool active)override{cpu_->begin_decode_step(active);}
    wiring::CpuPhases decode_phases()const override{return cpu_->decode_phases();}
    wiring::CpuLane* host_lane()override{return cpu_.get();}
    void drain()noexcept override{try{q_.wait_and_throw();}catch(...){}cpu_->drain();}
};
}
std::unique_ptr<ExpertCompute> make_experts(sycl::queue& q,const wiring::WeightSource& ws,uint32_t first,uint32_t end,const std::string& placement,int threads,const std::string& dispatch,bool timing,IqPath path){sycl_port::ordered(q);return std::make_unique<Experts>(q,ws,first,end,placement,threads,dispatch,timing,path);}
}
namespace strata::ds4::moe {
sycl::event quantize(sycl::queue& q,const float* x,size_t stride,int k,int T,int8_t* a,float* d){
    sycl_port::ordered(q);if(!x||!a||!d||k<=0||k%32||T<=0||stride<size_t(k))throw std::invalid_argument("row2 quant geometry");
    return row2::quant(q,x,stride,k,T,a,d);
}
}
