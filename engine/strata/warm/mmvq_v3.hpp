// MIT. Opt-in exact-v2 arithmetic with low-N geometry and fused hidden Q8.
#pragma once
#include "mmvq_project.hpp"
#include "../sycl/division.hpp"
namespace strata::wiring::mmvq::v3 {
// Scheduling only: each row still owns the same 32 logical lanes. No shared
// memory, cross-row reduction, numerical reassociation or new weight layout.
template<bool GateUp>sycl::event dispatch(sycl::queue& q,std::string_view type,WarmPackedBatch a,int count,int n,int k,float* y,float* up,const int8_t* x,const float* d,const uint32_t* table){
 const int t=kind(type);extent(t,n,k);
 require(count>0&&count<=kPicks&&y&&x&&d&&table&&(!GateUp||up),"MMVQ v3 launch extent");
 for(int i=0;i<count;++i)require(GateUp?(a.gate[i]&&a.up[i]):bool(a.down[i]),"MMVQ v3 null weight");
 if(count>2)return mmvq::dispatch<GateUp>(q,type,a,count,n,k,y,up,x,d,table,true);
 if(t==0)return project<0,GateUp,true,1>(q,a,count,n,k,y,up,x,d,table);
 if(t==1)return project<1,GateUp,true,1>(q,a,count,n,k,y,up,x,d,table);
 return project<2,GateUp,true,1>(q,a,count,n,k,y,up,x,d,table);
}
struct Math {
 static float min(float a,float b){return sycl::fmin(a,b);}
 static float max(float a,float b){return sycl::fmax(a,b);}
 static float exp(float x){return sycl::exp(x);}
};
inline sycl::event hidden_quant(sycl::queue& q,const float* gate,const float* up,int n,float lim,float* hidden,int8_t* quant,float* scale){
 require(gate&&up&&hidden&&quant&&scale&&n>0&&n%32==0&&std::isfinite(lim),"MMVQ v3 hidden geometry/limit");
 // The existing SwiGLU expression followed by the existing subgroup Q8 max,
 // fdiv_rn and rint sequence. Keeping hidden also preserves the strict oracle.
 return q.parallel_for(sycl::nd_range<1>(size_t(n),32),[=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]]{
  const size_t block=it.get_group_linear_id(),lane=it.get_local_linear_id(),i=block*32+lane;
  const float v=ds4::moe::swiglu<Math>(gate[i],up[i],lim);hidden[i]=v;
  float am=sycl::fabs(v);auto sg=it.get_sub_group();
  for(int offset=16;offset;offset>>=1)am=sycl::fmax(am,sycl::permute_group_by_xor(sg,am,offset));
  const float sc=ds4::sycl_port::div_rn(am,127.0f);
  quant[i]=int8_t(am>0?int(sycl::rint(ds4::sycl_port::div_rn(v,sc))):0);
  if(!lane)scale[block]=sc;
 });
}
inline WarmGroupedWorkspace::Events enqueue(sycl::queue& q,WarmGroupedWorkspace& w,WarmPackedBatch a,int count,std::string_view gu,std::string_view dn,float lim,const uint32_t* table){
 require(count>0&&count<=w.capacity&&std::isfinite(lim),"MMVQ v3 count/limit");
 WarmGroupedWorkspace::Events e;
 e.input=ds4::moe::quantize(q,w.x,size_t(w.dim),w.dim,1,w.quant,w.scale);
 e.gate_up=dispatch<true>(q,gu,a,count,w.ff,w.dim,w.gate,w.up,w.quant,w.scale,table);
 e.hidden_quant=hidden_quant(q,w.gate,w.up,w.ff*count,lim,w.hidden,w.quant,w.scale);
 e.down=dispatch<false>(q,dn,a,count,w.dim,w.ff,w.y,nullptr,w.quant,w.scale,table);
 return e; // queue is in-order; no intermediate host wait or readback
}
}
