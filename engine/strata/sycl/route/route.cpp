#include "route.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
namespace strata::ds4::route_port {
namespace {
void require(bool b,const char* s) { if(!b) throw std::invalid_argument(s); }
size_t product(size_t a,size_t b) {
    require(!b || a<=std::numeric_limits<size_t>::max()/b,"shape overflow"); return a*b;
}
void finite(std::span<const float> x) {
    for(float v:x) require(std::isfinite(v),"nonfinite arithmetic input/output");
}
}
Routing route_cpu(std::span<const float> logits,size_t stride,std::span<const float> bias,
                  size_t T,size_t k,float scale) {
    const size_t n=bias.size();
    require(T && n && k && k<=n && stride>=n && n<=size_t(std::numeric_limits<int>::max()),"route geometry");
    require(logits.size()==product(T,stride),"route logits size"); finite(logits); finite(bias);
    require(std::isfinite(scale) && scale>=0,"route scale");
    Routing out{T,n,k,{}, {}}; out.ids.resize(product(T,k)); out.weights.resize(T*k);
    std::vector<float> sc(n),bs(n);
    for(size_t t=0;t<T;++t) {
        for(size_t e=0;e<n;++e) {
            float l=logits[t*stride+e]; float sp=l>20.0f?l:std::log1p(std::exp(l));
            sc[e]=std::sqrt(sp); bs[e]=sc[e]+bias[e];
            require(std::isfinite(bs[e]),"nonfinite route score");
        }
        float sum=0;
        for(size_t i=0;i<k;++i) {
            size_t best=0; for(size_t e=1;e<n;++e) if(bs[e]>bs[best]) best=e;
            out.ids[t*k+i]=int(best); sum+=sc[best]; bs[best]=-std::numeric_limits<float>::max();
        }
        for(size_t i=0;i<k;++i) out.weights[t*k+i]=sc[out.ids[t*k+i]]/(sum+1e-20f)*scale;
    }
    return out;
}
void HostStore::insert(size_t l,int e,std::vector<std::byte> b,size_t expected) {
    require(e>=0 && expected && b.size()==expected,"short/invalid expert blob");
    require(data_.emplace(std::make_pair(l,e),std::move(b)).second,"duplicate expert blob");
}
bool HostStore::contains(size_t l,int e) const { return data_.contains({l,e}); }
std::span<const std::byte> HostStore::get(size_t l,int e,size_t expected) const {
    auto& b=data_.at({l,e}); require(expected && b.size()==expected,"expert blob extent mismatch"); return b;
}
ExpertService::ExpertService(size_t n,size_t k):experts_(n),picks_(k) {
    require(n && k && k<=n && n<=size_t(std::numeric_limits<int>::max()),"trained route geometry");
}
void ExpertService::reset(uint64_t req,uint64_t epoch) {
    std::unique_lock lock(mutex_,std::try_to_lock); require(lock.owns_lock(),"request active");
    require(!initialized_ || req!=request_ || epoch>epoch_,"reset must change request or advance epoch");
    request_=req; epoch_=epoch; completed_=0; initialized_=true; poisoned_=false; cancelled_=false; store_policy_set_=false;
}
Resp ExpertService::execute(const Req& r,ExpertBackend& b) {
    std::unique_lock lock(mutex_,std::try_to_lock); require(lock.owns_lock(),"request active");
    require(initialized_ && !poisoned_,"request uninitialized/invalid; reset required");
    require(r.request==request_ && r.epoch==epoch_ && completed_!=UINT64_MAX && r.seq==completed_+1,"stale request/sequence");
    auto check_cancel=[&] { if(cancelled_.load(std::memory_order_acquire)) throw std::runtime_error("request cancelled"); };
    const auto& route=r.route; size_t T=route.tokens,k=route.picks,D=r.dim;
    require(T && D && route.experts==experts_ && k==picks_,"full trained routing required");
    require(r.combine_path==CombinePath::upstream_default || r.combine_path==CombinePath::synchronous ||
            r.combine_path==CombinePath::x3,"unknown combine path");
    const bool x3=r.combine_path==CombinePath::x3 || (r.combine_path==CombinePath::upstream_default && T<=8);
    require(!x3 || (T<=8 && k<=8),"x3 window exceeds retained eight-slot combine");
    require(!r.stage_store || (T==1 && x3),"store-to-device staging is decode-only");
    require(!store_policy_set_ || r.stage_store==store_policy_,"request store policy changed");
    size_t slots=product(T,k),numel=product(T,D);
    require(route.ids.size()==slots && route.weights.size()==slots && r.x.size()==numel && r.shared.size()==numel,"request shape");
    finite(route.weights); finite(r.x); finite(r.shared);
    for(size_t t=0;t<T;++t) for(size_t i=0;i<k;++i) {
        int id=route.ids[t*k+i]; require(id>=0 && size_t(id)<experts_ && route.weights[t*k+i]>=0,"route ID/weight");
        for(size_t j=0;j<i;++j) require(route.ids[t*k+j]!=id,"duplicate expert pick");
    }
    Resp o{r.request,r.epoch,r.seq,std::vector<int>(T),std::vector<float>(numel),std::vector<float>(numel),0,0,0};
    try {
        check_cancel();
        store_policy_=r.stage_store; store_policy_set_=true;
        std::vector<std::vector<float>> ys(slots); std::vector<bool> gpu(slots);
        for(size_t slot=0;slot<slots;++slot) {
            check_cancel(); int id=route.ids[slot]; bool hit=b.resident(r.layer,id);
            if(hit) ++o.resident_hits;
            else if(b.store_contains(r.layer,id)) {
                ++o.store_hits;
                if(r.stage_store) { b.stage(r.layer,id); check_cancel(); hit=true; }
            }
            else ++o.cpu_misses;
            gpu[slot]=hit;
            auto x=std::span<const float>(r.x).subspan((slot/k)*D,D);
            ys[slot]=hit?b.device(r.layer,id,x):b.cpu(r.layer,id,x);
            require(ys[slot].size()==D,"expert output shape"); finite(ys[slot]);
        }
        for(size_t t=0;t<T;++t) {
            // Upstream CPU lane sorts (expert,token,pick); preserve its sum order.
            std::vector<size_t> cpu_slots,gpu_slots;
            for(size_t i=0;i<k;++i) (gpu[t*k+i]?gpu_slots:cpu_slots).push_back(t*k+i);
            if(!x3) std::sort(gpu_slots.begin(),gpu_slots.end(),[&](size_t a,size_t z){ return route.ids[a]<route.ids[z]; });
            std::sort(cpu_slots.begin(),cpu_slots.end(),[&](size_t a,size_t z){ return route.ids[a]<route.ids[z]; });
            o.cpu[t]=!cpu_slots.empty();
            for(size_t e=0;e<D;++e) {
                float cpu=0; for(size_t s:cpu_slots) cpu+=route.weights[s]*ys[s][e];
                o.part[t*D+e]=cpu;
                // x3 exl3.cu:597: every pick, zero for CPU, explicit RN FMA.
                // Synchronous ds4_model.cu:10490 / combine_kernel: expert-ID order.
                float acc=0;
                if(x3) for(size_t i=0;i<k;++i) {
                    size_t s=t*k+i; acc=std::fma(route.weights[s],gpu[s]?ys[s][e]:0.0f,acc);
                }
                else for(size_t s:gpu_slots) acc=std::fma(route.weights[s],ys[s][e],acc);
                if(o.cpu[t]) acc+=cpu;
                acc+=r.shared[t*D+e]; o.combined[t*D+e]=acc;
            }
        }
        finite(o.part); finite(o.combined); check_cancel(); b.publish(o);
        // Publication is the completion point. A later cancel affects the next
        // dispatch, never retracts a response already delivered to the device.
        completed_=r.seq; return o;
    } catch(...) { poisoned_=true; b.drain(); throw; }
}
}
