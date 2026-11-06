// MIT. CPU-only complementary bank selection. HIP's existing selector stays intact.
#pragma once
#include "../sycl/stage/wiring/hip_resident.hpp"
namespace strata::wiring {
constexpr uint64_t kWarmSyclWorkspace=256ull<<20;
struct WarmResidency {
 std::vector<std::vector<int>> slots;
 std::vector<ResidentExpert> experts;
 uint64_t reservation=0,allocated=0,payload=0,profile_tag=0,slots_tag=0;
 int slot(uint32_t l,int e)const{return l<slots.size()&&e>=0&&size_t(e)<slots[l].size()?slots[l][size_t(e)]:-1;}
};
inline WarmResidency select_warm_sycl(const WeightSource& ws,const Config& cfg,const HipResidency& hip,uint64_t automatic_room=0){
 WarmResidency r;if(!cfg.warm_experts)return r;r.reservation=cfg.warm_sycl_bytes?cfg.warm_sycl_bytes:automatic_room;require(r.reservation>0,"no warm B65 bank room");
 const auto& g=ws.meta();const auto counts=expert_profile(cfg,g,r.profile_tag);
 r.slots.assign(size_t(g.nl),std::vector<int>(size_t(g.n_exp),-1));
 std::vector<ResidentExpert> candidates;
 for(int l=0;l<g.nl;++l)for(int e=0;e<g.n_exp;++e){
  if(!hip.slots.empty()&&hip.slot(uint32_t(l),e)>=0)continue;
  auto x=expert_metadata(ws,uint32_t(l),e);bool supported=true;
  for(const auto& t:x.matrices)supported&=t.dtype=="IQ2_XXS"||t.dtype=="IQ2_XS"||t.dtype=="IQ1_M";
  if(supported)candidates.push_back(std::move(x));
 }
 std::stable_sort(candidates.begin(),candidates.end(),[&](const auto& a,const auto& b){
  const auto ca=counts[size_t(a.layer)*size_t(g.n_exp)+size_t(a.expert)],cb=counts[size_t(b.layer)*size_t(g.n_exp)+size_t(b.expert)];
  return __uint128_t(ca)*b.allocated>__uint128_t(cb)*a.allocated;
 });
 for(auto& x:candidates)if(x.allocated<=r.reservation-r.allocated){r.allocated+=x.allocated;r.payload+=x.payload;r.experts.push_back(std::move(x));}
 require(!r.experts.empty(),"warm SYCL selection is empty");
 std::sort(r.experts.begin(),r.experts.end(),[](const auto& a,const auto& b){return std::pair(a.layer,a.expert)<std::pair(b.layer,b.expert);});
 uint64_t offset=0;std::vector<int> next(size_t(g.nl),0);
 for(auto& x:r.experts){x.slot=next[x.layer]++;r.slots[x.layer][size_t(x.expert)]=x.slot;
  for(size_t role=0;role<3;++role){x.offsets[role]=offset;offset+=expert_align(x.matrices[role].bytes);}}
 require(offset==r.allocated,"warm slab accounting");
 std::vector<int32_t> flat;for(const auto& row:r.slots)flat.insert(flat.end(),row.begin(),row.end());
 r.slots_tag=stage::checksum(flat.data(),flat.size()*sizeof(int32_t));return r;
}
inline uint64_t warm_sycl_reservation(const Config& cfg){return cfg.warm_experts?stage::add(cfg.warm_sycl_bytes,kWarmSyclWorkspace):0;}
inline void apply_warm_sycl(stage::Plan& p,const Config& cfg,const WarmResidency& r){
 if(!cfg.warm_experts)return;
 p.sycl.fixed=stage::add(p.sycl.fixed,stage::add(r.reservation,kWarmSyclWorkspace));
 p.schema_sha+=";warm-sycl="+std::to_string(r.slots_tag)+";warm-profile="+std::to_string(r.profile_tag)+";warm-bytes="+std::to_string(r.allocated)+";warm-packed-gu="+std::to_string(cfg.warm_packed_gu);
 if(cfg.warm_sycl_experts!="serial-v1")p.schema_sha+=";warm-experts="+cfg.warm_sycl_experts;
 if(cfg.warm_sycl_attention!="scalar-v1")p.schema_sha+=";warm-attn="+cfg.warm_sycl_attention;
 if(cfg.warm_hip_schedule!="baseline-v1")p.schema_sha+=";warm-hip-schedule="+cfg.warm_hip_schedule;
 if(cfg.warm_kernel_timing)p.schema_sha+=";warm-kernel-timing=1";
 if(cfg.warm_hip_dense!="off")p.schema_sha+=";warm-hip-dense="+cfg.warm_hip_dense;
 p.validate();
}
}

namespace strata::wiring {
inline HipResidency select_warm_hip(const WeightSource& ws,const Config& cfg,const stage::Plan& base,const WarmResidency& sycl){
 HipResidency r;r.policy="pooled-layer-v1";r.workspace=kHipExpertWorkspaceReserve;r.all_layers=true;r.bank_reservation=cfg.warm_hip_bytes;
 const auto& g=ws.meta();const auto counts=expert_profile(cfg,g,r.profile_tag);
 r.slots.assign(size_t(g.nl),std::vector<int>(size_t(g.n_exp),-1));
 r.layer_allocated.assign(size_t(g.nl),0);r.layer_driver_charge.assign(size_t(g.nl),0);
 constexpr uint64_t reserve=kHipExpertWorkspaceReserve+kHipDriverBaseReserve+kHipDriverSafetyMargin;
 const auto used=base.footprint(base.hip,base.cut,base.layers);
 require(used<=base.hip.cap&&reserve<=base.hip.cap-used,"no warm HIP headroom");require(cfg.warm_hip_bytes<=base.hip.cap-used-reserve,"warm HIP bank reservation exceeds physical room");const auto room=cfg.warm_hip_bytes?cfg.warm_hip_bytes:base.hip.cap-used-reserve;r.bank_reservation=room;
 std::vector<ResidentExpert> candidates;
 for(uint32_t l=0;l<base.layers;++l)for(int e=0;e<g.n_exp;++e)if(sycl.slot(l,e)<0)candidates.push_back(expert_metadata(ws,l,e));
 std::stable_sort(candidates.begin(),candidates.end(),[&](const auto& a,const auto& b){
  const auto ca=counts[size_t(a.layer)*size_t(g.n_exp)+size_t(a.expert)],cb=counts[size_t(b.layer)*size_t(g.n_exp)+size_t(b.expert)];
  return __uint128_t(ca)*b.allocated>__uint128_t(cb)*a.allocated;
 });
 for(auto& x:candidates){const auto old=r.layer_allocated[x.layer];const auto bytes=stage::add(old,x.allocated);
  const auto delta=expert_driver_charge(bytes)-expert_driver_charge(old);
  if(delta>room-r.driver_charge)continue;r.driver_charge+=delta;r.layer_allocated[x.layer]=bytes;r.experts.push_back(std::move(x));}
 std::sort(r.experts.begin(),r.experts.end(),[](const auto& a,const auto& b){return std::pair(a.layer,a.expert)<std::pair(b.layer,b.expert);});
 std::vector<uint64_t> offsets(size_t(g.nl),0);std::vector<int> next(size_t(g.nl),0);
 for(auto& x:r.experts){x.slot=next[x.layer]++;r.slots[x.layer][size_t(x.expert)]=x.slot;
  for(size_t role=0;role<3;++role){x.offsets[role]=offsets[x.layer];offsets[x.layer]+=expert_align(x.matrices[role].bytes);}
  r.payload+=x.payload;r.allocated+=x.allocated;}
 for(size_t l=0;l<r.slots.size();++l)r.layer_driver_charge[l]=expert_driver_charge(r.layer_allocated[l]);
 std::vector<int32_t> flat;for(const auto& row:r.slots)flat.insert(flat.end(),row.begin(),row.end());
 r.slots_tag=stage::checksum(flat.data(),flat.size()*sizeof(int32_t));r.layout_tag=residency_layout_tag(r);validate_pooled_layout(r);
 require(r.payload>=20000000000ull,"warm HIP bank below 20GB");
 return r;
}
}
