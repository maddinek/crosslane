// MIT. One complementary B65 slab, immutable after upload; shared CPU DSO seam.
#include "sycl_bank.hpp"
#include "runtime.hpp"
#include "activation_capture.hpp"
#include "../sycl/stage/wiring/sycl_decode_events.hpp"
#include "../sycl/moe/core.hpp"
#include "../sycl/moe/moe.hpp"
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#define STRATA_WARM_PACKED
#include "packed_project.hpp"
#include "grouped_project.hpp"
#include "mmvq_v3.hpp"
#include <chrono>
namespace strata::wiring {
namespace {
namespace r=strata::ds4::row2;namespace m=strata::ds4::moe;
class Bank final:public r::ExpertCompute,public WarmRedirect {
 sycl::queue& q;const WeightSource& ws;WarmResidency plan;uint32_t first,end;bool packed_gu;
 std::unique_ptr<r::ExpertCompute> fallback;
 uint8_t* slab=nullptr;float* work=nullptr;int8_t* quant=nullptr;float* scales=nullptr;
 uint16_t* table2=nullptr;uint16_t* table1=nullptr;uint16_t* tablexs=nullptr;
 std::vector<std::vector<std::array<r::DenseMat,3>>> matrices;
 std::unique_ptr<WarmGroupedWorkspace> grouped;
 bool soa=false,exact_order=false,low_n=false;uint32_t* mmvq_table=nullptr;
 uint64_t workspace_bytes=0;SyclDecodeClock clock;
 ActivationCapture capture;
 void clear()noexcept{warm_unregister(this);try{q.wait_and_throw();}catch(...){}
  for(void* p:{static_cast<void*>(slab),static_cast<void*>(work),static_cast<void*>(quant),static_cast<void*>(scales),static_cast<void*>(table2),static_cast<void*>(table1),static_cast<void*>(tablexs)})if(p)try{sycl::free(p,q);}catch(...){}
  if(mmvq_table){try{sycl::free(mmvq_table,q);}catch(...){}mmvq_table=nullptr;}
  grouped.reset();
  slab=nullptr;work=nullptr;quant=nullptr;scales=nullptr;table2=table1=tablexs=nullptr;
 }
 void project(const r::DenseMat& a,float* y,bool packed,bool gu=false){
  if(!packed){r::MvJob job{&a,work,y,1,size_t(a.k),size_t(a.n),1};r::mv(q,std::span(&job,1));return;}
  const auto* w=static_cast<const uint8_t*>(a.raw);
  warm_packed_project(q,a.dtype,gu,w,a.ld,a.n,a.k,y,quant,scales,table2,tablexs,table1);
 }
public:
 Bank(sycl::queue& queue,const WeightSource& source,uint32_t f,uint32_t e,const Config& cfg,const WarmResidency& p)
  :q(queue),ws(source),plan(p),first(f),end(e),packed_gu(cfg.warm_packed_gu),capture(uint32_t(source.meta().nl),uint32_t(source.meta().dim)){
  require(cfg.warm_experts&&first==0&&(end>=1||cfg.warm_dense_owner=="hip-v1"),"warm B65 bank requires cut1 prefix");
  if(first<end)fallback=r::make_experts(q,ws,first,end,cfg.expert_placement,cfg.cpu_threads,cfg.cpu_dispatch,cfg.cpu_timing);
  try{
   low_n=cfg.warm_sycl_experts=="mmvq-soa-v3";
   exact_order=low_n||cfg.warm_sycl_experts=="mmvq-soa-v2";
   soa=exact_order||cfg.warm_sycl_experts=="mmvq-soa-v1";
   const auto& g=ws.meta();const size_t maxk=size_t(std::max(g.dim,g.ff));
   slab=sycl::malloc_device<uint8_t>(size_t(plan.allocated),q);
   work=sycl::malloc_device<float>(size_t(g.dim)*2+size_t(g.ff)*3,q);
   quant=sycl::malloc_device<int8_t>(maxk,q);scales=sycl::malloc_device<float>(maxk/32,q);
   table2=sycl::malloc_device<uint16_t>(r::iq2::compact_codebook.size(),q);
   table1=sycl::malloc_device<uint16_t>(p1k_down::iq1_compact.size(),q);
   tablexs=sycl::malloc_device<uint16_t>(p1k_iq2xs::compact.size(),q);
   require(slab&&work&&quant&&scales&&table2&&table1&&tablexs,"warm SYCL allocation failed");
   workspace_bytes=(size_t(g.dim)*2+size_t(g.ff)*3)*4+maxk+(maxk/32)*4+sizeof(r::iq2::compact_codebook)+sizeof(p1k_down::iq1_compact)+sizeof(p1k_iq2xs::compact);
   if(cfg.warm_sycl_experts!="serial-v1"){
    grouped=std::make_unique<WarmGroupedWorkspace>(q,g.dim,g.ff,g.k_exp);workspace_bytes+=grouped->bytes();grouped->fast=cfg.warm_sycl_experts=="packed-strip-v1";
   }
   if(soa){const auto table=mmvq::codebooks();mmvq_table=sycl::malloc_device<uint32_t>(table.size(),q);require(mmvq_table,"MMVQ table allocation");q.memcpy(mmvq_table,table.data(),sizeof(table)).wait_and_throw();workspace_bytes+=sizeof(table);}
   require(workspace_bytes<=kWarmSyclWorkspace,"warm workspace reservation exceeded");
   q.memcpy(tablexs,p1k_iq2xs::compact.data(),sizeof(p1k_iq2xs::compact));
   q.memcpy(table2,r::iq2::compact_codebook.data(),sizeof(r::iq2::compact_codebook));
   q.memcpy(table1,p1k_down::iq1_compact.data(),sizeof(p1k_down::iq1_compact)).wait_and_throw();
   matrices.resize(size_t(g.nl));
   for(const auto& x:plan.experts){auto& row=matrices[x.layer];row.resize(size_t(x.slot)+1);
    for(size_t i=0;i<3;++i){const auto& t=x.matrices[i];auto raw=ws.read_all(t.name);
     require(x.offsets[i]<=plan.allocated&&t.bytes<=plan.allocated-x.offsets[i],"warm bank view extent");
     if(soa){require(t.shape.size()==2&&t.shape[0]>0&&t.shape[0]<=INT32_MAX&&t.shape[1]>0&&t.shape[1]<=INT32_MAX,"MMVQ bank shape");
      auto encoded=mmvq::reorder(mmvq::kind(t.dtype),int(t.shape[0]),int(t.shape[1]),std::span(reinterpret_cast<const uint8_t*>(raw.data()),raw.size()));
      q.memcpy(slab+x.offsets[i],encoded.data(),encoded.size()).wait_and_throw();
     }else q.memcpy(slab+x.offsets[i],raw.data(),raw.size()).wait_and_throw();
     row[size_t(x.slot)][i]={t.name,t.dtype,int(t.shape[0]),int(t.shape[1]),m::row_bytes(m::type(t.dtype),int(t.shape[1])),slab+x.offsets[i]};
    }}
#if !defined(STRATA_IQ2_HOST_TEST) && !defined(STRATA_CPU_SYNTAX_AUDIT)
   std::fprintf(stderr,"AIOPT_WARM_FREE backend=sycl total_bytes=%llu scope=global-memory-capacity-accounted-free-in-DRM-snapshot\n",(unsigned long long)q.get_device().get_info<sycl::info::device::global_mem_size>());
#endif
   if(grouped)for(const auto& layer:matrices)for(const auto& a:layer){
    require(a[0].dtype==a[1].dtype&&a[0].dtype==layer[0][0].dtype&&a[2].dtype==layer[0][2].dtype,"grouped bank requires uniform layer roles");
    require(a[0].n==g.ff&&a[1].n==g.ff&&a[0].k==g.dim&&a[1].k==g.dim&&a[2].n==g.dim&&a[2].k==g.ff,"grouped bank shape");
   }
   if(soa)std::fprintf(stderr,"AIOPT_WARM_MMVQ layout=soa-v1 same_payload=1 reduction=%s numerical_gate=unchanged\n",low_n?"exact-stride32-v3":exact_order?"exact-stride32-v2":"paired-subblock-sg32-v1");
   if(grouped)std::fprintf(stderr,"AIOPT_WARM_GROUPED capacity=%d workspace=%zu completion=one-per-layer-lane\n",grouped->capacity,grouped->bytes());
   warm_register(this);std::fprintf(stderr,"AIOPT_WARM_SYCL experts=%zu payload=%llu allocated=%llu workspace=%llu packed_gu=%d\n",plan.experts.size(),(unsigned long long)plan.payload,(unsigned long long)plan.allocated,(unsigned long long)workspace_bytes,int(packed_gu));
  }catch(...){clear();throw;}
 }
 ~Bank()override{clear();}
 bool warm_contains(uint32_t l,int e)const override{return plan.slot(l,e)>=0;}
 std::vector<float> warm_compute(uint32_t l,int e,std::span<const float> x)override{
  if(soa){const int id=e;return warm_compute_many(l,std::span(&id,1),x).at(0);}
  const auto& g=ws.meta();require(x.size()==size_t(g.dim)&&warm_contains(l,e),"warm SYCL request extent/residency");
  require(std::all_of(x.begin(),x.end(),[](float z){return std::isfinite(z);}),"nonfinite warm input");
  capture.record(l,x);
  const auto& a=matrices.at(l).at(size_t(plan.slot(l,e)));
  float* gate=work+g.dim;float* up=gate+g.ff;float* hidden=up+g.ff;float* y=hidden+g.ff;
  clock.begin(q);q.memcpy(work,x.data(),x.size_bytes());m::quantize(q,work,size_t(g.dim),g.dim,1,quant,scales);
  project(a[0],gate,packed_gu,true);project(a[1],up,packed_gu,true);
  r::swiglu(q,gate,up,g.ff,g.lim,hidden);m::quantize(q,hidden,size_t(g.ff),g.ff,1,quant,scales);
  project(a[2],y,true);std::vector<float> out(size_t(g.dim));q.memcpy(out.data(),y,out.size()*4);clock.end(q);warm_gpu_span(false,l,clock.elapsed_ns());
  require(std::all_of(out.begin(),out.end(),[](float z){return std::isfinite(z);}),"nonfinite warm output");return out;
 }
 bool warm_grouped()const override{return bool(grouped);}
 std::vector<std::vector<float>> warm_compute_many(uint32_t l,std::span<const int> ids,std::span<const float> x)override{
  require(grouped&&x.size()==size_t(grouped->dim)&&!ids.empty()&&ids.size()<=size_t(grouped->capacity),"grouped bank request");
  require(std::all_of(x.begin(),x.end(),[](float v){return std::isfinite(v);}),"nonfinite grouped input");
  capture.record(l,x);
  WarmPackedBatch a;std::string gu,dn;
  for(size_t i=0;i<ids.size();++i){require(warm_contains(l,ids[i]),"grouped selection not resident");
   for(size_t j=0;j<i;++j)require(ids[i]!=ids[j],"duplicate grouped selection");
   const auto& m=matrices.at(l).at(size_t(plan.slot(l,ids[i])));
   gu=m[0].dtype;dn=m[2].dtype;
   a.gate[i]=static_cast<const uint8_t*>(m[0].raw);a.up[i]=static_cast<const uint8_t*>(m[1].raw);a.down[i]=static_cast<const uint8_t*>(m[2].raw);
   a.gate_ld[i]=m[0].ld;a.up_ld[i]=m[1].ld;a.down_ld[i]=m[2].ld;
  }
  std::vector<float> flat(ids.size()*size_t(grouped->dim));
  try{
   clock.begin(q);q.memcpy(grouped->x,x.data(),x.size_bytes());
   if(low_n)mmvq::v3::enqueue(q,*grouped,a,int(ids.size()),gu,dn,ws.meta().lim,mmvq_table);
   else if(soa)mmvq::enqueue(q,*grouped,a,int(ids.size()),gu,dn,ws.meta().lim,mmvq_table,exact_order);
   else grouped->enqueue(a,int(ids.size()),gu,dn,ws.meta().lim,table2,tablexs,table1);
   q.memcpy(flat.data(),grouped->y,flat.size()*sizeof(float));clock.end(q);
   const auto elapsed=clock.elapsed_ns();
   for(size_t i=0;i<ids.size();++i)warm_gpu_span(false,l,i==0?elapsed:0); // keep selection counts for placement-cost reducers
  }catch(...){try{q.wait_and_throw();}catch(...){}throw;} // host buffers survive a partial submit failure
  require(std::all_of(flat.begin(),flat.end(),[](float v){return std::isfinite(v);}),"nonfinite grouped output");
  std::vector<std::vector<float>> out;out.reserve(ids.size());
  for(size_t i=0;i<ids.size();++i)out.emplace_back(flat.begin()+i*size_t(grouped->dim),flat.begin()+(i+1)*size_t(grouped->dim));
  return out;
 }
 bool resident(size_t l,int e)const override{return warm_contains(uint32_t(l),e);}
 bool store_contains(size_t,int)const override{return false;}
 void stage(size_t,int)override{throw std::runtime_error("immutable warm bank cannot stage");}
 std::vector<float> device(size_t l,int e,std::span<const float> x)override{
  const auto tick=std::chrono::steady_clock::now();auto y=warm_compute(uint32_t(l),e,x);
  warm_note_sycl(uint32_t(l),e,uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-tick).count()));return y;
 }
 std::vector<float> cpu(size_t l,int e,std::span<const float> x)override{require(bool(fallback),"expert-only B65 has no CPU lane");return fallback->cpu(l,e,x);}
 uint64_t device_bytes(uint32_t l)const override{return l==first?plan.allocated+workspace_bytes:0;}
 void begin_decode_step(bool a)override{if(fallback)fallback->begin_decode_step(a);}
 wiring::CpuPhases decode_phases()const override{return fallback?fallback->decode_phases():CpuPhases{};}
 CpuLane* host_lane()override{return fallback?fallback->host_lane():nullptr;}
 void drain()noexcept override{try{q.wait_and_throw();}catch(...){}if(fallback)fallback->drain();}
};
}
std::unique_ptr<r::ExpertCompute> make_warm_sycl_experts(sycl::queue& q,const WeightSource& ws,uint32_t first,uint32_t end,const Config& cfg,const WarmResidency& p){return std::make_unique<Bank>(q,ws,first,end,cfg,p);}
}
