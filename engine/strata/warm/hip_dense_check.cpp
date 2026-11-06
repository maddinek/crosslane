// MIT. Native HIP exact-byte and paired timing discriminator, no model needed.
// Same upstream GGUF kernels, nonzero Q5_K/Q8_0/BF16 matrices and mutable inputs.
#include "hip_dense_graph.hpp"
#include "ds4_gguf.cuh"
#include "ds4_kernels.cuh"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace gg=strata::ds4::gguf;
namespace k4=strata::ds4;
namespace x3=strata::dsv41k::exl3;
using strata::wiring::HipDenseGraph;
using strata::wiring::HipDenseGraphApi;
static void ck(hipError_t e){HipDenseGraphApi::check(e);}
struct Device {
    void* p{};
    explicit Device(size_t n){ck(hipMalloc(&p,n));}
    ~Device(){if(p)hipFree(p);}
    template<class T>T* as(){return static_cast<T*>(p);}
};
static void matrix(std::vector<unsigned char>& w,int type,int k,int n){
    const size_t block=type==13?176:34,values=type==13?256:32;
    uint32_t state=19;
    for(auto& c:w){state=1664525u*state+1013904223u;c=static_cast<unsigned char>(state>>24);}
    if(type==0){
        for(size_t at=0;at<w.size();at+=2){uint16_t v=uint16_t(0x3a00+(at/2)%127);std::memcpy(w.data()+at,&v,2);}
    }else for(size_t at=0;at<w.size();at+=block){
        const uint16_t scale=0x2400;std::memcpy(w.data()+at,&scale,2);
        if(type==13)std::memcpy(w.data()+at+2,&scale,2);
    }
    (void)k;(void)n;(void)values;
}
static void hc_test(hipStream_t s){
    constexpr int D=5120,H=4,M=24,E=384;
    Device h(size_t(H)*D*4),fn(size_t(M)*H*D*4),weights(size_t(E)*D*4),nw(D*4),scale(3*4),base(M*4),bias(E*4),count(4);
    // One zero-initialized result allocation also compares all padding bytes.
    constexpr size_t words=32+8+8+64+D+H*D+1024+8+8;
    Device outputs(words*4);auto* mix=outputs.as<float>();auto* pre=mix+32;auto* post=pre+8;
    auto* comb=post+8;auto* x=comb+64;auto* h2=x+D;auto* glog=h2+H*D;
    auto* ids=reinterpret_cast<int*>(glog+1024);auto* rw=reinterpret_cast<float*>(ids+8);
    auto fill=[&](Device& d,size_t n,float value){std::vector<float>v(n,value);ck(hipMemcpyAsync(d.p,v.data(),n*4,hipMemcpyHostToDevice,s));ck(hipStreamSynchronize(s));};
    fill(fn,size_t(M)*H*D,0.0001f);fill(weights,size_t(E)*D,0.0003f);fill(nw,D,1.0f);
    fill(scale,3,0.5f);fill(base,M,0.01f);fill(bias,E,0);ck(hipMemsetAsync(count.p,0,4,s));
    auto submit=[&]{
        k4::hc_mix_split(h.as<float>(),fn.as<float>(),H*D,M,1,1e-6f,mix,count.as<unsigned>(),scale.as<float>(),base.as<float>(),H,20,1e-6f,pre,post,comb,s);
        k4::hc_pre_norm(h.as<float>(),pre,nw.as<float>(),D,H,1,1e-6f,x,s);
        k4::hc_post(x,h.as<float>(),post,comb,D,H,1,h2,s);
        k4::gemv_f32(weights.as<float>(),x,E,D,1,glog,1024,s);
        k4::route(glog,bias.as<float>(),E,6,1,1.5f,ids,rw,s);
    };
    HipDenseGraph graph;std::vector<unsigned char> ref(words*4),got(words*4);std::vector<float>input(size_t(H)*D,0);
    for(int round=0;round<8;++round){
        for(size_t i=0;i<input.size();++i)input[i]=float(int((i*17+round*19)%97)-48)*0.03125f;
        ck(hipMemcpyAsync(h.p,input.data(),input.size()*4,hipMemcpyHostToDevice,s));
        ck(hipMemsetAsync(outputs.p,0,words*4,s));submit();
        ck(hipMemcpyAsync(ref.data(),outputs.p,ref.size(),hipMemcpyDeviceToHost,s));ck(hipStreamSynchronize(s));
        ck(hipMemsetAsync(outputs.p,0,words*4,s));graph.run(true,s,{h.p,h2},submit);
        ck(hipMemcpyAsync(got.data(),outputs.p,got.size(),hipMemcpyDeviceToHost,s));
        unsigned cnt=99;ck(hipMemcpyAsync(&cnt,count.p,4,hipMemcpyDeviceToHost,s));ck(hipStreamSynchronize(s));
        if(cnt||got!=ref)throw std::runtime_error("HC/Sinkhorn/router graph identity or counter reset failed");
    }
    graph.reset();std::printf("IDENTITY hc-sinkhorn-prenorm-post-router rounds=8 bytes=%zu PASS\n",ref.size()*8);
}
static void test(hipStream_t s,int type,int k,int n,bool timing){
    const size_t rb=type==0?size_t(k)*2:type==8?gg::q8_row_bytes(k):gg::type_row_bytes(type,k);
    if(!rb)throw std::runtime_error("zero fixture row bytes");
    std::vector<unsigned char>w(size_t(n)*rb);matrix(w,type,k,n);
    std::vector<float>x(size_t(k),0),norm(size_t(n),1),ref(size_t(n),0),got(size_t(n),0);
    Device dw(w.size()),dx(size_t(k)*4),dy(size_t(n)*4),dn(size_t(n)*4);
    ck(hipMemcpyAsync(dw.p,w.data(),w.size(),hipMemcpyHostToDevice,s));
    ck(hipMemcpyAsync(dn.p,norm.data(),norm.size()*4,hipMemcpyHostToDevice,s));
    ck(hipStreamSynchronize(s));
    x3::Mat m{};m.raw=dw.p;m.k=k;m.n=n;m.ld=int(type==0?k:rb);
    m.fmt=type==0?gg::kFmtBF16:type==8?gg::kFmtQ8_0:gg::kFmtGG;m.K=type;
    auto submit=[&]{
        x3::MvJob j{};j.m=&m;j.xf=dx.as<float>();j.y=dy.as<float>();j.ncol=1;j.alpha=1.0f;
        if(!gg::mv(&j,1,s))throw std::runtime_error("native dense mv rejected");
        k4::Finish f;f.w=dn.as<float>();f.eps=1e-6f;
        const int len=n<=8192?n:512; // scalar norms plus head-sized vectors, within upstream finish bound
        if(n%len)throw std::runtime_error("fixture finish geometry");
        k4::finish(dy.as<float>(),n/len,len,len,f,s);
    };
    HipDenseGraph graph;
    for(int round=0;round<8;++round){
        for(int i=0;i<k;++i)x[size_t(i)]=float(((i*37+round*11)%251)-125)*0.0078125f;
        ck(hipMemcpyAsync(dx.p,x.data(),x.size()*4,hipMemcpyHostToDevice,s));
        submit();ck(hipMemcpyAsync(ref.data(),dy.p,ref.size()*4,hipMemcpyDeviceToHost,s));ck(hipStreamSynchronize(s));
        // Poison output: capture itself must not execute and replay must overwrite it.
        ck(hipMemsetAsync(dy.p,0xff,got.size()*4,s));
        const auto capture_tick=std::chrono::steady_clock::now();
        graph.run(true,s,{dx.p,dy.p},submit);
        if(round==0)std::printf("SETUP type=%d k=%d n=%d first_graph_enqueue_us=%.6f scope=capture-instantiate-and-first-enqueue-no-wait\n",type,k,n,std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-capture_tick).count());
        ck(hipMemcpyAsync(got.data(),dy.p,got.size()*4,hipMemcpyDeviceToHost,s));ck(hipStreamSynchronize(s));
        for(size_t i=0;i<got.size();++i)if(!std::isfinite(got[i])||std::memcmp(&got[i],&ref[i],4))
            throw std::runtime_error("native graph differs at word "+std::to_string(i));
    }
    std::printf("IDENTITY type=%d k=%d n=%d rounds=8 words=%zu PASS\n",type,k,n,got.size()*8);
    if(timing){
        hipEvent_t first{},last{};ck(hipEventCreate(&first));ck(hipEventCreate(&last));
        // Host wall cost + event span. Alternate order, 5 paired repeats, no capture in timing.
        for(int rep=0;rep<5;++rep)for(int a=0;a<2;++a){
            const bool graphs=((rep+a)&1)!=0;
            const auto begin=std::chrono::steady_clock::now();ck(hipEventRecord(first,s));
            for(int i=0;i<64;++i)graph.run(graphs,s,{dx.p,dy.p},submit);
            ck(hipEventRecord(last,s));ck(hipEventSynchronize(last));
            const double wall=std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-begin).count()/64;
            float ms=0;ck(hipEventElapsedTime(&ms,first,last));
            std::printf("TIMING type=%d k=%d n=%d rep=%d graphs=%d iterations=64 event_us=%.6f wall_us=%.6f\n",type,k,n,rep,int(graphs),double(ms)*1000/64,wall);
        }
        hipEventDestroy(last);hipEventDestroy(first);
    }
    ck(hipStreamSynchronize(s));graph.reset(); // destruction always precedes scratch ownership release
}
int main(int argc,char**argv){try{
    const bool timing=argc==2&&std::strcmp(argv[1],"--timing")==0;
    if(argc!=2||(!timing&&std::strcmp(argv[1],"--identity")!=0))throw std::runtime_error("use --identity or --timing");
    ck(hipSetDevice(0));hipDeviceProp_t props{};ck(hipGetDeviceProperties(&props,0));
    std::printf("DEVICE name=%s arch=%s\n",props.name,props.gcnArchName);
    hipStream_t s{};ck(hipStreamCreateWithFlags(&s,hipStreamNonBlocking));
    if(!gg::prepare(s))throw std::runtime_error("prepare failed");
    hc_test(s);
    for(int type:{8,13,0})for(auto shape:{std::pair{5120,1280},std::pair{1280,32768},std::pair{4096,1024},std::pair{8192,5120},std::pair{5120,2304},std::pair{2304,5120}})
        test(s,type,shape.first,shape.second,timing);
    if(gg::launch_errors())throw std::runtime_error("upstream launch error counter nonzero");
    ck(hipStreamDestroy(s));std::puts("HIP_DENSE_NATIVE PASS");return 0;
}catch(const std::exception&e){std::fprintf(stderr,"HIP_DENSE_NATIVE FAIL %s\n",e.what());return 1;}}
