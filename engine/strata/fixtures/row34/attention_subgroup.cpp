// MIT. Execute the production scalar and subgroup kernels on the cooperative
// host API shim, or (STRATA_ATTN_NATIVE_CHECK) real device USM. Same inputs.
#include <sycl/sycl.hpp>
#include "../../sycl/attn/attn.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
namespace s=strata::ds4::sycl_port;
#ifdef STRATA_ATTN_NATIVE_CHECK
template<class T> struct Device {
    sycl::queue& q; T* p;
    Device(sycl::queue& queue,const std::vector<T>& host):q(queue),p(sycl::malloc_device<T>(host.size(),q)) {
        if(!p)throw std::bad_alloc();
        try{q.memcpy(p,host.data(),host.size()*sizeof(T)).wait_and_throw();}
        catch(...){sycl::free(p,q);throw;}
    }
    Device(const Device&)=delete;
    ~Device(){sycl::free(p,q);}
};
#endif
void run(sycl::queue& q,const std::vector<float>& query,const std::vector<uint16_t>& window,
         const std::vector<uint8_t>& kv,const std::vector<int>& selected,int stride,
         const s::AttnRows& rows,const std::vector<float>& sink,int heads,int tokens,float scale,
         std::vector<float>& control,std::vector<float>& candidate,int fmt) {
#ifdef STRATA_ATTN_NATIVE_CHECK
    Device<float> dq(q,query),ds(q,sink),dc(q,control),dn(q,candidate);
    Device<uint16_t> dw(q,window);Device<uint8_t> dk(q,kv);Device<int> di(q,selected);
    // Never pass ordinary host pointers to a native GPU, and never compare
    // before both kernels and both device-to-host copies have completed.
    s::sparse_attn(q,dq.p,dw.p,dk.p,di.p,stride,rows,ds.p,heads,512,tokens,scale,dc.p,fmt).wait_and_throw();
    s::sparse_attn_subgroup32(q,dq.p,dw.p,dk.p,di.p,stride,rows,ds.p,heads,512,tokens,scale,dn.p,fmt).wait_and_throw();
    q.memcpy(control.data(),dc.p,control.size()*sizeof(float)).wait_and_throw();
    q.memcpy(candidate.data(),dn.p,candidate.size()*sizeof(float)).wait_and_throw();
#else
    s::sparse_attn(q,query.data(),window.data(),kv.data(),selected.data(),stride,rows,sink.data(),heads,512,tokens,scale,control.data(),fmt).wait_and_throw();
    s::sparse_attn_subgroup32(q,query.data(),window.data(),kv.data(),selected.data(),stride,rows,sink.data(),heads,512,tokens,scale,candidate.data(),fmt).wait_and_throw();
#endif
    for(size_t i=0;i<control.size();++i)
        if(!std::isfinite(control[i])||!std::isfinite(candidate[i]))
            throw std::runtime_error("nonfinite/unwritten attention output");
}
}
int main() try {
    namespace a=strata::ds4::attn_scalar;namespace s=strata::ds4::sycl_port;
    sycl::queue queue(sycl::gpu_selector_v,sycl::property::queue::in_order{});
#ifdef STRATA_ATTN_NATIVE_CHECK
    const auto device=queue.get_device();
    const auto widths=device.get_info<sycl::info::device::sub_group_sizes>();
    if(!device.is_gpu()||device.get_info<sycl::info::device::vendor_id>()!=0x8086||
       std::find(widths.begin(),widths.end(),size_t(32))==widths.end())
        throw std::runtime_error("requires selected Intel GPU with subgroup32");
    std::printf("NATIVE device=%s driver=%s\n",device.get_info<sycl::info::device::name>().c_str(),device.get_info<sycl::info::device::driver_version>().c_str());
#endif
    std::mt19937 rng(17435);auto real=[&]{return float(int(rng()%2049)-1024)/512;};
    constexpr int W=160,K=600,SS=512;
    std::vector<uint16_t> window(W*512);
    for(auto& x:window)x=uint16_t(std::bit_cast<uint32_t>(real())>>16);
    size_t cases=0,words=0;
    for(int fmt:{0,2}){
        const auto row=a::row_bytes(fmt,512);std::vector<uint8_t> kv(K*row);
        if(fmt==0){for(size_t i=0;i<kv.size();i+=2){auto b=uint16_t(std::bit_cast<uint32_t>(real())>>16);kv[i]=uint8_t(b);kv[i+1]=uint8_t(b>>8);}}
        else for(int k=0;k<K;++k){for(int e=0;e<256;++e)kv[k*row+e]=uint8_t(rng());for(int e=0;e<32;++e)kv[k*row+256+e]=uint8_t(0x30+rng()%8);}
        for(int visible:{0,1,16,31,32,63,64,65,127,128,256,640}){
            const int T=visible==65?2:1,H=visible==128?3:1;
            std::vector<float> query(size_t(T)*H*512),sink(H),control(query.size()),candidate(query.size());
            std::vector<int> selected(size_t(T)*SS);for(auto& x:selected)x=int(rng()%K);
            for(auto& x:query)x=real();for(auto& x:sink)x=real();
            // Cancellation exercises the prescribed tree association and
            // per-lane16-element FMA sequence.
            if(visible==31){
                query[0]=1e8f;query[16]=1;query[256]=-1e8f;query[272]=1;
                for(int w=0;w<W;++w)window[w*512]=window[w*512+256]=0x3f80;
            }
            s::AttnRows ar{};
            for(int t=0;t<T;++t){ar.win0[t]=visible?5:0;ar.nwin[t]=std::min(128,visible);ar.nsel[t]=visible-ar.nwin[t];}
            const float scale=1/std::sqrt(512.f);
            std::fill(control.begin(),control.end(),NAN);std::fill(candidate.begin(),candidate.end(),NAN);
            run(queue,query,window,kv,selected,SS,ar,sink,H,T,scale,control,candidate,fmt);
            if(std::memcmp(control.data(),candidate.data(),control.size()*4)){
                for(size_t i=0;i<control.size();++i)if(std::bit_cast<uint32_t>(control[i])!=std::bit_cast<uint32_t>(candidate[i])){
                    std::fprintf(stderr,"fmt=%d visible=%d index=%zu scalar=%a subgroup=%a\n",fmt,visible,i,control[i],candidate[i]);break;}
                throw std::runtime_error("attention subgroup changed bits");
            }
            ++cases;words+=control.size();
        }
        // Generated visibility uses the same descriptor as production prefill.
        std::vector<float> query(2*512,0.25f),sink(1,0.5f),control(2*512),candidate(2*512);
        std::vector<int> selected(2*SS,7);s::AttnRows ar{};ar.gen=1;ar.p0=130;ar.t0=0;ar.win=128;ar.R=2;ar.kcap=512;
        std::fill(control.begin(),control.end(),NAN);std::fill(candidate.begin(),candidate.end(),NAN);
        run(queue,query,window,kv,selected,SS,ar,sink,1,2,0.04f,control,candidate,fmt);
        if(std::memcmp(control.data(),candidate.data(),control.size()*4))throw std::runtime_error("generated visibility changed bits");
        ++cases;words+=control.size();
    }
    std::printf("PASS: %zu production-kernel cases, %zu bit-identical F32 words; BF16/FP4, rows0..640, chunk boundaries, multiple heads/tokens, generated visibility, cancellation. %s\n",cases,words,
#ifdef STRATA_ATTN_NATIVE_CHECK
                "Native Intel GPU USM."
#else
                "CPU shim only."
#endif
    );
    return 0;
}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
