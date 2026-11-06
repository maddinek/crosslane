// MIT. Reconstruct both banks from an opt-in, complete, immutable assignment.
#pragma once
#include "residency.hpp"
namespace strata::wiring {
inline void apply_frozen_warm_placement(const WeightSource& ws,const Config& cfg,HipResidency& hip,WarmResidency& sycl){
 if(cfg.warm_placement.empty())return;
 const auto& g=ws.meta();std::ifstream f(cfg.warm_placement);std::string version;
 uint64_t nl=0,ne=0,tag=0;require(bool(f>>version>>nl>>ne>>tag)&&version=="STRATA_WARM_PLACEMENT_V1"&&nl==uint64_t(g.nl)&&ne==uint64_t(g.n_exp)&&tag==hip.profile_tag&&tag==sycl.profile_tag,"warm placement header/profile mismatch");
 const auto hip_room=hip.bank_reservation,sycl_room=sycl.reservation;
 hip.experts.clear();hip.payload=hip.allocated=hip.driver_charge=0;
 hip.slots.assign(size_t(g.nl),std::vector<int>(size_t(g.n_exp),-1));hip.layer_allocated.assign(size_t(g.nl),0);hip.layer_driver_charge.assign(size_t(g.nl),0);
 sycl.experts.clear();sycl.payload=sycl.allocated=0;sycl.slots.assign(size_t(g.nl),std::vector<int>(size_t(g.n_exp),-1));
 std::vector<int> hn(size_t(g.nl),0),sn(size_t(g.nl),0);uint64_t offset=0;
 for(int l=0;l<g.nl;++l)for(int e=0;e<g.n_exp;++e){
  int layer=-1,expert=-1,lane=-1;require(bool(f>>layer>>expert>>lane)&&layer==l&&expert==e&&lane>=0&&lane<=2,"warm placement missing/duplicate/unsorted row");
  if(!lane)continue;auto x=expert_metadata(ws,uint32_t(l),e);
  if(lane==1){x.slot=hn[size_t(l)]++;hip.slots[size_t(l)][size_t(e)]=x.slot;
   for(size_t r=0;r<3;++r){x.offsets[r]=hip.layer_allocated[size_t(l)];hip.layer_allocated[size_t(l)]=stage::add(hip.layer_allocated[size_t(l)],expert_align(x.matrices[r].bytes));}
   hip.payload=stage::add(hip.payload,x.payload);hip.allocated=stage::add(hip.allocated,x.allocated);hip.experts.push_back(std::move(x));
  }else{
   for(const auto& t:x.matrices)require(t.dtype=="IQ2_XXS"||t.dtype=="IQ2_XS"||t.dtype=="IQ1_M","unadmitted frozen warm B65 dtype");
   x.slot=sn[size_t(l)]++;sycl.slots[size_t(l)][size_t(e)]=x.slot;
   for(size_t r=0;r<3;++r){x.offsets[r]=offset;offset=stage::add(offset,expert_align(x.matrices[r].bytes));}
   sycl.payload=stage::add(sycl.payload,x.payload);sycl.allocated=stage::add(sycl.allocated,x.allocated);sycl.experts.push_back(std::move(x));
  }
 }
 std::string extra;require(!(f>>extra)&&f.eof(),"extra/invalid warm placement data");
 for(size_t l=0;l<hip.slots.size();++l){hip.layer_driver_charge[l]=expert_driver_charge(hip.layer_allocated[l]);hip.driver_charge=stage::add(hip.driver_charge,hip.layer_driver_charge[l]);}
 require(hip.driver_charge<=hip_room&&sycl.allocated<=sycl_room,"frozen warm bank exceeds reserved room");
 require(hip.payload>=20000000000ull&&sycl.payload>=20000000000ull,"frozen warm bank below 20GB");
 auto slot_tag=[](const auto& slots){std::vector<int32_t> v;for(const auto& row:slots)v.insert(v.end(),row.begin(),row.end());return stage::checksum(v.data(),v.size()*sizeof(int32_t));};
 hip.slots_tag=slot_tag(hip.slots);hip.layout_tag=residency_layout_tag(hip);validate_pooled_layout(hip);sycl.slots_tag=slot_tag(sycl.slots);
 // Reject a partially filled result if any complementary expert still fits.
 for(int l=0;l<g.nl;++l)for(int e=0;e<g.n_exp;++e)if(hip.slot(uint32_t(l),e)<0&&sycl.slot(uint32_t(l),e)<0){auto x=expert_metadata(ws,uint32_t(l),e);
  const auto delta=expert_driver_charge(stage::add(hip.layer_allocated[size_t(l)],x.allocated))-hip.layer_driver_charge[size_t(l)];
  require(delta>hip_room-hip.driver_charge,"frozen HIP bank has refillable slack");
  bool supported=true;for(const auto& t:x.matrices)supported&=t.dtype=="IQ2_XXS"||t.dtype=="IQ2_XS"||t.dtype=="IQ1_M";
  require(!supported||x.allocated>sycl_room-sycl.allocated,"frozen B65 bank has refillable slack");
 }
}
}
