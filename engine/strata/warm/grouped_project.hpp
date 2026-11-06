// MIT. Group submissions, never arithmetic: each row calls the accepted strip.
#pragma once
#include "packed_project.hpp"
#include "packed_strip.hpp"
#include "../sycl/moe/moe.hpp"
#include "../sycl/stage/wiring/sycl_row2.hpp"
#include <array>
namespace strata::wiring {
// The existing routed-expert ABI bounds one token to kPicks selections.
// Captured by value in the kernel: no transient descriptor upload/lifetime.
struct WarmPackedBatch {
 std::array<const uint8_t*,kPicks> gate{},up{},down{};
 std::array<size_t,kPicks> gate_ld{},up_ld{},down_ld{};
};
namespace warm_grouped_detail {
template<int Type,bool GateUp,bool Fast=false>
sycl::event project(sycl::queue& q,WarmPackedBatch a,int count,int n,int k,
 float* y,float* up,const int8_t* x,const float* d,const uint16_t* table){
 constexpr int local=Fast&&Type!=2?16:4,rows=Fast&&Type!=2?2:1;
 const size_t strips=(size_t(n)+local*rows-1)/(local*rows);
 if constexpr(Fast&&Type!=2)for(int i=0;i<count;++i){
  auto bounded=[&](size_t ld){return ld<=UINT32_MAX/size_t(n);};
  require(n>0&&k>0&&k%256==0&&(GateUp?(bounded(a.gate_ld[i])&&bounded(a.up_ld[i])):bounded(a.down_ld[i])),"packed strip matrix exceeds 32-bit byte offsets");
 }
 constexpr int grf=Type==2?128:256;
 return q.submit([&](sycl::handler& cgh){
  cgh.parallel_for(sycl::nd_range<1>(strips*local*size_t(count)*(GateUp?2:1),local),
   sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<grf>},
   [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
    if constexpr(Fast&&Type!=2)warm_strip::table_init<Type==1?512:256>(table,int(it.get_local_linear_id()));
    const size_t part=it.get_group_linear_id()/strips;
    const int row=int(it.get_group_linear_id()%strips)*(local*rows)+int(it.get_local_linear_id())*rows;
    if(row>=n)return;
    const size_t expert=GateUp?part/2:part;
    const bool is_up=GateUp&&(part%2);
    const uint8_t* w=GateUp?(is_up?a.up[expert]:a.gate[expert]):a.down[expert];
    const size_t ld=GateUp?(is_up?a.up_ld[expert]:a.gate_ld[expert]):a.down_ld[expert];
    float* dst=(is_up?up:y)+expert*size_t(n);
    const int8_t* qx=x+(GateUp?0:expert*size_t(k));
    const float* dx=d+(GateUp?0:expert*size_t(k/32));
    if constexpr(Fast&&Type!=2)warm_strip::strip2<Type==1>(w,ld,qx,dx,n,k,row,dst);
    else if constexpr(Type==0)p1k_packed::iq2_strip(w,ld,qx,dx,k,row,dst,table);
    else if constexpr(Type==1)p1k_iq2xs::strip(w,ld,qx,dx,k,row,dst,table);
    else p1k_packed::iq1m_strip(w,ld,qx,dx,k,row,dst,table);
   });
 });
}
template<bool GateUp,bool Fast=false>
sycl::event dispatch(sycl::queue& q,std::string_view type,WarmPackedBatch a,int count,int n,int k,
 float* y,float* up,const int8_t* x,const float* d,const uint16_t* xx,const uint16_t* xs,const uint16_t* im){
 if(type=="IQ2_XXS")return project<0,GateUp,Fast>(q,a,count,n,k,y,up,x,d,xx);
 if(type=="IQ2_XS")return project<1,GateUp,Fast>(q,a,count,n,k,y,up,x,d,xs);
 if(type=="IQ1_M")return project<2,GateUp,Fast>(q,a,count,n,k,y,up,x,d,im);
 throw std::invalid_argument("unadmitted grouped packed type");
}
}
// One owner and one in-order queue. Capacity is metadata k_exp, not bank size.
// Never shared concurrently: warm_execute gives each backend one worker.
class WarmGroupedWorkspace {
 sycl::queue& q;
 std::vector<void*> allocations;
 template<class T>T* alloc(size_t n){auto p=sycl::malloc_device<T>(n,q);if(!p)throw std::bad_alloc();allocations.push_back(p);return p;}
 void clear()noexcept{try{q.wait_and_throw();}catch(...){}for(auto p:allocations)try{sycl::free(p,q);}catch(...){}allocations.clear();}
public:
 const int dim,ff,capacity;
 bool fast=false; // explicit packed-strip-v1; no automatic adoption
 float *x=nullptr,*gate=nullptr,*up=nullptr,*hidden=nullptr,*y=nullptr,*scale=nullptr;
 int8_t* quant=nullptr;
 const size_t quant_count;
 struct Events {sycl::event input,gate_up,hidden_quant,down;};
 WarmGroupedWorkspace(sycl::queue& queue,int d,int f,int cap):q(queue),dim(d),ff(f),capacity(cap),
 quant_count(std::max(size_t(d),size_t(f)*size_t(cap))){
  ds4::sycl_port::ordered(q);
  require(d>0&&f>0&&d%256==0&&f%256==0&&cap>0&&cap<=kPicks&&f<=INT32_MAX/cap,"grouped workspace geometry");
  try{x=alloc<float>(size_t(d));gate=alloc<float>(size_t(f)*cap);up=alloc<float>(size_t(f)*cap);
   hidden=alloc<float>(size_t(f)*cap);y=alloc<float>(size_t(d)*cap);
   quant=alloc<int8_t>(quant_count);scale=alloc<float>(quant_count/32);
  }catch(...){clear();throw;}
 }
 ~WarmGroupedWorkspace(){clear();}
 WarmGroupedWorkspace(const WarmGroupedWorkspace&)=delete;
 WarmGroupedWorkspace& operator=(const WarmGroupedWorkspace&)=delete;
 size_t bytes()const{return (size_t(dim)+size_t(capacity)*(3*size_t(ff)+size_t(dim)))*sizeof(float)+quant_count+quant_count/32*sizeof(float);}
 Events enqueue(WarmPackedBatch a,int count,std::string_view gu,std::string_view dn,float lim,
                const uint16_t* xx,const uint16_t* xs,const uint16_t* im){
  require(count>0&&count<=capacity&&std::isfinite(lim),"grouped count/limit");
  require(warm_packed_type(gu)&&warm_packed_type(dn)&&xx&&xs&&im,"grouped type/table");
  for(int i=0;i<count;++i)require(a.gate[i]&&a.up[i]&&a.down[i]&&
   a.gate_ld[i]>=ds4::moe::row_bytes(ds4::moe::type(gu),dim)&&
   a.up_ld[i]>=ds4::moe::row_bytes(ds4::moe::type(gu),dim)&&
   a.down_ld[i]>=ds4::moe::row_bytes(ds4::moe::type(dn),ff),"grouped matrix extent");
  Events e;
  e.input=ds4::moe::quantize(q,x,size_t(dim),dim,1,quant,scale);
  e.gate_up=fast?warm_grouped_detail::dispatch<true,true>(q,gu,a,count,ff,dim,gate,up,quant,scale,xx,xs,im):
   warm_grouped_detail::dispatch<true>(q,gu,a,count,ff,dim,gate,up,quant,scale,xx,xs,im);
  ds4::row2::swiglu(q,gate,up,ff*count,lim,hidden);
  e.hidden_quant=ds4::moe::quantize(q,hidden,size_t(ff),ff,count,quant,scale);
  e.down=fast?warm_grouped_detail::dispatch<false,true>(q,dn,a,count,dim,ff,y,nullptr,quant,scale,xx,xs,im):
   warm_grouped_detail::dispatch<false>(q,dn,a,count,dim,ff,y,nullptr,quant,scale,xx,xs,im);
  return e; // caller retains inputs and scratch until the final readback completes
 }
};
}
