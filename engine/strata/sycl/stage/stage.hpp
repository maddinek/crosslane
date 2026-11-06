// MIT; experimental Strata stage coordinator, crosslane contributors.
#pragma once
#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace strata::stage {
inline void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
inline uint64_t add(uint64_t a, uint64_t b) {
    require(b <= std::numeric_limits<uint64_t>::max()-a, "byte count overflow"); return a+b;
}
enum class Kind : uint32_t { compressed_kv, index_rows, visible_ring, candidates, hc_pre, selected_topk };
enum class RowMode : uint32_t { accumulated, ring, current_step };
struct Edge {
    uint32_t producer=0, consumer=0;
    Kind kind=Kind::compressed_kv;
    uint64_t bytes=0; // Exact frozen snapshot extent; no variable/truncated transfer.
    uint64_t row_bytes=0;
    uint64_t ratio=0; // Required adapter metadata; zero rejects admission.
    RowMode row_mode=RowMode::accumulated;
    friend bool operator==(const Edge&, const Edge&) = default;
};
struct Budget {
    uint64_t cap=0, fixed=0, mirrors=0;
    std::vector<uint64_t> layers;
};
struct EngramSlice {
    uint32_t layer; size_t offset,count;
    friend bool operator==(const EngramSlice&,const EngramSlice&)=default;
};
struct Plan {
    std::string model_sha, schema_sha;
    uint32_t layers=0, cut=0;
    uint32_t hc=0,dim=0;
    uint64_t activation_bytes=0;
    Budget sycl, hip;
    // Complete graph supplied by the architecture adapter, not inferred here.
    bool dependencies_complete=false;
    std::vector<Edge> dependencies;
    std::vector<EngramSlice> engram_layout; // Frozen per-layer offsets, delivered by begin().
    std::vector<Edge> crossing() const {
        std::vector<Edge> out;
        for (auto e:dependencies) if (e.producer<cut && e.consumer>=cut &&
            std::none_of(out.begin(),out.end(),[&](auto a){return a.producer==e.producer&&a.kind==e.kind;})) out.push_back(e);
        return out;
    }
    uint64_t crossing_bytes() const {
        uint64_t n=0; for (auto e:crossing()) n=add(n,e.bytes); return n;
    }
    bool hip_dense_only() const { return cut==0 && schema_sha.find(";warm-dense=hip-v1;")!=std::string::npos; }
    uint64_t wire_bytes() const { return hip_dense_only()?0:add(activation_bytes,crossing_bytes()); }
    uint64_t expected_rows(const Edge& e,uint64_t position) const {
        require(position<UINT64_MAX && e.ratio && e.row_bytes,"invalid row clock");
        uint64_t rows=(position+1)/e.ratio, capacity=e.bytes/e.row_bytes;
        if(e.row_mode==RowMode::current_step) rows=1;
        if(e.row_mode==RowMode::ring) rows=std::min(rows,capacity);
        require(rows<=capacity,"valid rows exceed source capacity");
        return rows;
    }
    uint64_t footprint(const Budget& b, uint32_t first, uint32_t end) const {
        uint64_t n=add(b.fixed,b.mirrors);
        for (uint32_t l=first;l<end;++l) n=add(n,b.layers.at(l));
        return n;
    }
    void validate() const {
        require(!model_sha.empty()&&!schema_sha.empty(), "missing model/schema identity");
        require(layers>1 && (cut>0 || hip_dense_only()) && cut<layers, "invalid contiguous cut");
        require(activation_bytes && activation_bytes%sizeof(float)==0, "invalid residual extent");
        require(hc&&dim&&uint64_t(hc)<=std::numeric_limits<uint64_t>::max()/(uint64_t(dim)*sizeof(float))&&
                activation_bytes==uint64_t(hc)*uint64_t(dim)*sizeof(float),"residual HC/dimension mismatch");
        require(dependencies_complete, "dependency inventory incomplete");
        require(sycl.layers.size()==layers && hip.layers.size()==layers, "missing layer accounting");
        require(footprint(sycl,0,cut)<=sycl.cap && footprint(hip,cut,layers)<=hip.cap,
                "stage exceeds frozen device cap");
        size_t engram_count=0;std::vector<uint32_t> engram_layers;
        for(auto slice:engram_layout) {
            require(slice.layer<layers && slice.count && slice.offset==engram_count &&
                    slice.count<=std::numeric_limits<size_t>::max()-engram_count,"invalid Engram layout");
            require(std::find(engram_layers.begin(),engram_layers.end(),slice.layer)==engram_layers.end(),"duplicate Engram layer");
            engram_layers.push_back(slice.layer);engram_count+=slice.count;
        }
        std::vector<Edge> seen;
        for (auto e:dependencies) {
            require(e.producer<layers && e.consumer<layers && e.producer<e.consumer && e.bytes&&e.row_bytes&&e.bytes%e.row_bytes==0 && e.ratio,
                    "invalid or backwards source dependency");
            require(uint32_t(e.kind)<=uint32_t(Kind::selected_topk), "unknown dependency kind");
            require(uint32_t(e.row_mode)<=uint32_t(RowMode::current_step),"unknown row mode");
            require(e.row_mode!=RowMode::current_step || e.ratio==1,"current-step ratio must be one");
            require(std::find_if(seen.begin(),seen.end(),[&](auto a){
                return a.producer==e.producer&&a.consumer==e.consumer&&a.kind==e.kind;
            })==seen.end(), "duplicate dependency");
            for(auto a:seen)if(a.producer==e.producer&&a.kind==e.kind)
                require(a.bytes==e.bytes&&a.row_bytes==e.row_bytes&&a.ratio==e.ratio&&a.row_mode==e.row_mode,"inconsistent source snapshot geometry");
            seen.push_back(e);
        }
        // A later consumer's entire read-only source snapshot counts on HIP.
        require(hip.mirrors==crossing_bytes(), "cross-cut mirror budget must equal unique source bytes");
        require(wire_bytes()<=std::numeric_limits<size_t>::max(), "host packet too large");
    }
};
// SYCL owns the prefix; HIP owns the tail, final norm and output head. The
// caller supplies header-derived footprints, frozen runtime headroom and caps.
// Choose the smallest fitting SYCL prefix, leaving the larger tail on HIP;
// this is a deterministic capacity rule, not a measured performance optimum.
// No sum-of-VRAM admission.
inline Plan choose_cut(Plan p) {
    require(p.layers>1,"not enough layers");
    std::string reasons;
    for(uint32_t k=1;k<p.layers;++k) {
        p.cut=k;
        try { p.hip.mirrors=p.crossing_bytes(); p.validate(); return p; }
        catch(const std::runtime_error& e) { reasons+="cut "+std::to_string(k)+": "+e.what()+"; "; }
    }
    throw std::runtime_error("no cut admitted: "+reasons);
}
struct Token {
    uint64_t request=0, step=0, position=0;
    int32_t id=0;
    // Flattened in Ngram::layout() order, then each table's (ngram-2,head).
    // UINT64_MAX is the image/zero-row marker. Immutable to both stages.
    std::vector<uint64_t> engram_ids;
};
struct Completion {
    virtual ~Completion()=default;
    virtual void wait()=0;
    virtual size_t bytes() const=0;
};
struct HostMemory {
    void* data=nullptr; size_t bytes=0;
    std::function<void(void*)> release;
    HostMemory()=default;
    HostMemory(void* p,size_t n,std::function<void(void*)> f):data(p),bytes(n),release(std::move(f)){}
    HostMemory(const HostMemory&)=delete;
    HostMemory& operator=(const HostMemory&)=delete;
    HostMemory(HostMemory&& x) noexcept:data(std::exchange(x.data,nullptr)),bytes(x.bytes),release(std::move(x.release)){}
    HostMemory& operator=(HostMemory&& x) noexcept {
        if(this!=&x){if(data&&release)release(data);data=std::exchange(x.data,nullptr);bytes=x.bytes;release=std::move(x.release);}
        return *this;
    }
    ~HostMemory(){if(data&&release)release(data);}
};
inline uint64_t checksum(const void* p,size_t n) {
    auto b=static_cast<const uint8_t*>(p); uint64_t h=14695981039346656037ULL;
    // Host-local word hash; memcpy avoids alignment/aliasing assumptions. The
    // packet never leaves this process or host endian domain. Not byte FNV-1a.
    size_t i=0;
    for(;n-i>=sizeof(uint64_t);i+=sizeof(uint64_t)) {
        uint64_t word;std::memcpy(&word,b+i,sizeof(word));h^=word;h*=1099511628211ULL;
    }
    for(;i<n;++i){h^=b[i];h*=1099511628211ULL;} return h;
}
struct Packet {
    uint32_t version=2;
    std::string model_sha,schema_sha;
    uint64_t request=0,step=0,position=0,epoch=0,bytes=0,digest=0;
    uint32_t producer=0;
    std::vector<Edge> edges;
    // Valid rows INCLUDING this step, even though transactional commit is later.
    // Exact counts are derived from position+1, ratio and row mode in the plan.
    // Full capacity snapshots start at zero; readers use only these valid rows.
    // Ring/window schemas encode their logical origin inside the bound format.
    std::vector<uint64_t> committed_rows;
};
// Each implementation owns its runtime, stream/queue, weights, local KV/rings,
// expert service and scratch. No pointers or native events cross this seam.
// export/import layout: full F32 residual, then exact edge snapshots in order.
// Stage0 owns producing state; Stage1 mirrors are read-only, tagged with epoch.
struct Backend {
    virtual ~Backend()=default;
    virtual const char* name() const=0;
    virtual void begin(const Plan&,uint32_t first,uint32_t end,uint64_t request)=0;
    virtual HostMemory pinned(size_t)=0;
    virtual void execute(const Token&,const std::atomic_bool& cancel)=0;
    virtual std::unique_ptr<Completion> export_boundary(const Token&,std::span<std::byte>)=0;
    virtual std::vector<uint64_t> boundary_committed_rows(const Token&)=0;
    virtual std::unique_ptr<Completion> import_boundary(const Token&,std::span<const std::byte>,const Packet&)=0;
    virtual std::vector<float> logits(const Token&)=0; // Tail only; completes its D2H internally.
#ifdef STRATA_FORK_WARM
    virtual std::vector<double> warm_dense_costs()const { return {}; }
    virtual int32_t sample(float,float,int,float,uint64_t,uint64_t) { throw std::runtime_error("sampling requires HIP tail"); }
#endif
    virtual void prepare_commit(uint64_t step)=0;
    virtual void commit(uint64_t step)=0;
    virtual void invalidate() noexcept=0;
    virtual void drain() noexcept=0;
};
struct Ngram {
    virtual ~Ngram()=default;
    virtual void reset()=0;
    virtual std::vector<uint64_t> advance(int32_t id)=0;
    virtual std::span<const EngramSlice> layout() const {return {};}
    // Text-only implementations must reject image tokens explicitly.
    virtual std::vector<uint64_t> advance_image(int32_t) {throw std::runtime_error("image ngram not supported");}
};
struct TokenTiming { // CPU-observed completion intervals, never kernel-only durations.
    uint64_t prefix_ns=0,d2h_ns=0,host_boundary_ns=0,h2d_ns=0,tail_logits_ns=0,total_ns=0;
};
class Runtime {
    const Plan plan_;
    Backend& prefix_; Backend& tail_; Ngram& ngram_;
    HostMemory source_,target_;
    uint64_t request_=0,step_=0,epoch_=0;
    bool valid_=false;
    std::atomic_bool cancel_{false};
    std::mutex dispatch_;
    void check_cancel() const {require(!cancel_.load(),"request cancelled");}
    void poison() noexcept {
        valid_=false; prefix_.invalidate();tail_.invalidate();prefix_.drain();tail_.drain();
    }
    void validate_packet(const Packet& p) const {
        require(p.version==2&&p.model_sha==plan_.model_sha&&p.schema_sha==plan_.schema_sha,
                "boundary schema/model mismatch");
        require(p.request==request_&&p.step==step_&&p.epoch==epoch_&&p.position==step_,
                "stale boundary epoch/step");
        require(p.producer==plan_.cut-1&&p.edges==plan_.crossing()&&p.bytes==plan_.wire_bytes(),
                "boundary geometry/ownership mismatch");
        require(p.committed_rows.size()==p.edges.size(),"missing committed row inventory");
        for(size_t i=0;i<p.edges.size();++i)
            require(p.committed_rows[i]==plan_.expected_rows(p.edges[i],p.position),"valid rows differ from position/ratio");
        require(p.digest==checksum(target_.data,size_t(p.bytes)),"boundary checksum mismatch");
    }
public:
    Runtime(Plan p,Backend& a,Backend& b,Ngram& n):plan_(std::move(p)),prefix_(a),tail_(b),ngram_(n) {
        plan_.validate();
        const auto layout=ngram_.layout();
        require(std::equal(layout.begin(),layout.end(),plan_.engram_layout.begin(),plan_.engram_layout.end()),"coordinator/plan Engram layout mismatch");
        if(plan_.hip_dense_only())return; // no transport buffers for an expert-only prefix
        auto bytes=size_t(plan_.wire_bytes());
        source_=prefix_.pinned(bytes);target_=tail_.pinned(bytes);
        require(source_.data&&target_.data&&source_.data!=target_.data&&source_.bytes==bytes&&target_.bytes==bytes,
                "invalid private pinned allocations");
    }
    ~Runtime(){poison();}
    Runtime(const Runtime&)=delete;Runtime& operator=(const Runtime&)=delete;
    const Plan& plan() const{return plan_;}
    bool valid() const{return valid_;}
    uint64_t committed_steps() const{return step_;}
    void cancel() noexcept{cancel_.store(true);}
    void begin(uint64_t request) {
        std::unique_lock lock(dispatch_,std::try_to_lock);require(lock.owns_lock(),"concurrent stage dispatch");
        poison();cancel_.store(false);request_=request;step_=0;++epoch_;ngram_.reset();
        try {
            prefix_.begin(plan_,0,plan_.cut,request);tail_.begin(plan_,plan_.cut,plan_.layers,request);
            valid_=true;
        } catch(...) {poison();throw;}
    }
    std::vector<float> token(int32_t id,TokenTiming* timing=nullptr) {
        std::unique_lock lock(dispatch_,std::try_to_lock);require(lock.owns_lock(),"concurrent stage dispatch");
        require(valid_,"request must begin/reset after failure");
        auto stamp=[&]()->uint64_t{return timing?uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count()):0;};
        if(timing)*timing={};
        const auto start=stamp();auto tick=start;
        try {
            check_cancel();
            Token t{request_,step_,step_,id,ngram_.advance(id)};
            const auto& layout=plan_.engram_layout;
            const size_t expected_ids=layout.empty()?0:layout.back().offset+layout.back().count;
            require(t.engram_ids.size()==expected_ids,"Engram ID count differs from frozen layout");
            if(!plan_.hip_dense_only()){
            prefix_.execute(t,cancel_);check_cancel();
            if(timing){timing->prefix_ns=stamp()-tick;tick=stamp();}
            auto d2h=prefix_.export_boundary(t,{static_cast<std::byte*>(source_.data),source_.bytes});
            require(bool(d2h),"missing D2H completion");d2h->wait();
            require(d2h->bytes()==source_.bytes,"short D2H transfer");check_cancel();
            if(timing){timing->d2h_ns=stamp()-tick;tick=stamp();}
            Packet p{2,plan_.model_sha,plan_.schema_sha,request_,step_,step_,epoch_,
                     plan_.wire_bytes(),checksum(source_.data,source_.bytes),plan_.cut-1,plan_.crossing(),
                     prefix_.boundary_committed_rows(t)};
            // Each runtime pins its own buffer; no cross-runtime host registration
            // or event interoperability is assumed. Reuse only after both waits.
            std::memcpy(target_.data,source_.data,source_.bytes);validate_packet(p);
            if(timing){timing->host_boundary_ns=stamp()-tick;tick=stamp();}
            auto h2d=tail_.import_boundary(t,{static_cast<const std::byte*>(target_.data),target_.bytes},p);
            require(bool(h2d),"missing H2D completion");h2d->wait();
            require(h2d->bytes()==target_.bytes,"short H2D transfer");check_cancel();
            if(timing){timing->h2d_ns=stamp()-tick;tick=stamp();}
            } // cut0 goes straight from ngram ownership to HIP embedding/layer0
            tail_.execute(t,cancel_);check_cancel();auto out=tail_.logits(t);check_cancel();
            if(timing)timing->tail_logits_ns=stamp()-tick;
            require(!out.empty(),"empty logits");
            for(auto x:out) require(std::isfinite(x),"nonfinite logits");
            if(!plan_.hip_dense_only())prefix_.prepare_commit(step_);
            tail_.prepare_commit(step_);check_cancel();
            if(!plan_.hip_dense_only())prefix_.commit(step_);
            tail_.commit(step_);++step_;
            if(timing)timing->total_ns=stamp()-start;
            return out;
        } catch(...) {poison();throw;}
    }
};
} // namespace strata::stage
