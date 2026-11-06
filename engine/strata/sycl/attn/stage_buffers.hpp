// MIT; connects row3/4 buffers to aa28c52e's stage::Backend boundary methods.
// Requires sibling row56/M2 stage headers at integration; not a full layer driver.
#pragma once
#include "numerics.hpp"
#include "stage/wiring/common.hpp"
namespace strata::ds4::sycl_port {
struct StageBuffer {
    uint32_t producer=0;
    stage::Kind kind=stage::Kind::compressed_kv;
    void* usm=nullptr;
    uint64_t bytes=0,row_bytes=0;
    bool mirror=false;
};
class RowBoundary {
    sycl::queue& queue_;
    float* residual_;
    size_t residual_bytes_;
    std::vector<StageBuffer>& buffers_;
    struct Done final:stage::Completion {
        size_t n;explicit Done(size_t bytes):n(bytes){}
        void wait() override {} // All local DMA already joined before return.
        size_t bytes() const override{return n;}
    };
    StageBuffer& find(stage::Edge e,bool mirror) {
        StageBuffer* found=nullptr;
        for(auto& b:buffers_)if(b.producer==e.producer&&b.kind==e.kind){
            stage::require(!found,"duplicate row buffer binding");found=&b;
        }
        stage::require(found&&found->usm&&found->bytes==e.bytes&&found->row_bytes==e.row_bytes&&found->mirror==mirror,"row buffer format/ownership mismatch");
        stage::require(found->row_bytes&&found->bytes%found->row_bytes==0,"invalid row capacity");
        return *found;
    }
    void check(const stage::Plan& plan,size_t bytes,bool mirror) {
        stage::require(!plan.hip_dense_only(),"RowBoundary has no cut0 transfer");
        plan.validate();stage::require(residual_&&residual_bytes_==plan.activation_bytes&&bytes==plan.wire_bytes(),"row boundary extent mismatch");
        for(auto e:plan.crossing())find(e,mirror);
    }
public:
    RowBoundary(sycl::queue& q,float* residual,size_t bytes,std::vector<StageBuffer>& buffers):queue_(q),residual_(residual),residual_bytes_(bytes),buffers_(buffers){ordered(q);}
    // Delegate to the wiring's canonical position clock, just as its backends do.
    std::vector<uint64_t> boundary_committed_rows(const stage::Plan& plan,const stage::Token& token) {
        check(plan,plan.wire_bytes(),false);queue_.wait_and_throw();
        return strata::wiring::BoundaryLayout::of(plan).rows(plan,token.position);
    }
    std::unique_ptr<stage::Completion> export_boundary(const stage::Plan& plan,const stage::Token& token,std::span<std::byte> host) {
        check(plan,host.size(),false);(void)boundary_committed_rows(plan,token);queue_.wait_and_throw();
        queue_.memcpy(host.data(),residual_,residual_bytes_).wait_and_throw();size_t offset=residual_bytes_;
        for(auto e:plan.crossing()){auto& b=find(e,false);queue_.memcpy(host.data()+offset,b.usm,b.bytes).wait_and_throw();offset+=b.bytes;}
        return std::make_unique<Done>(offset);
    }
    std::unique_ptr<stage::Completion> import_boundary(const stage::Plan& plan,const stage::Token& token,std::span<const std::byte> host,const stage::Packet& packet) {
        check(plan,host.size(),true);
        stage::require(packet.edges==plan.crossing()&&packet.bytes==host.size()&&packet.committed_rows.size()==packet.edges.size(),"row packet schema mismatch");
        // Coordinator validates request/step/epoch/model/schema/checksum before this call.
        stage::require(packet.position==token.position&&packet.committed_rows==strata::wiring::BoundaryLayout::of(plan).rows(plan,token.position),"row packet committed rows differ from position");
        queue_.wait_and_throw();queue_.memcpy(residual_,host.data(),residual_bytes_).wait_and_throw();size_t offset=residual_bytes_;
        for(auto e:packet.edges){auto& b=find(e,true);queue_.memcpy(b.usm,host.data()+offset,b.bytes).wait_and_throw();offset+=b.bytes;}
        return std::make_unique<Done>(offset);
    }
};
}
