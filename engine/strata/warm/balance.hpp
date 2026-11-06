// MIT. Single-stream dense stages execute sequentially, including the hand-off.
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#include <stdexcept>
namespace strata::wiring {
struct WarmBalance {uint32_t cut=0;double sycl_ns=0,hip_ns=0,handoff_ns=0,total_ns=0;uint32_t balanced_cut=0;};
inline WarmBalance balance_dense(const std::vector<double>& sycl,const std::vector<double>& hip,const std::vector<uint32_t>& allowed,const std::vector<double>& handoff){
 if(sycl.size()!=hip.size()||sycl.size()<3||allowed.empty())throw std::runtime_error("warm dense cost shape");
 if(handoff.size()!=sycl.size())throw std::runtime_error("warm hand-off cost shape");
 if(!std::isfinite(hip.back())||hip.back()<=0)throw std::runtime_error("missing measured HIP head cost");
 WarmBalance best;double fastest=std::numeric_limits<double>::infinity(),error=fastest;uint32_t balanced=0;const size_t layers=sycl.size()-1;
 for(auto cut:allowed){if(cut>=layers)throw std::runtime_error("warm dense cut range");double a=0,b=hip[layers];
  for(size_t l=0;l<cut;++l){if(!std::isfinite(sycl[l])||sycl[l]<=0)throw std::runtime_error("missing measured SYCL dense cost");a+=sycl[l];}
  for(size_t l=cut;l<layers;++l){if(!std::isfinite(hip[l])||hip[l]<=0)throw std::runtime_error("missing measured HIP dense cost");b+=hip[l];}
  if(!std::isfinite(b)||b<=0)throw std::runtime_error("invalid HIP dense/head cost");
  if(!std::isfinite(handoff[cut])||handoff[cut]<0)throw std::runtime_error("missing measured hand-off cost");
  const double total=a+b+handoff[cut];
  if(!std::isfinite(total))throw std::runtime_error("invalid total dense cost");
  if(total<fastest){fastest=total;best={cut,a,b,handoff[cut],total,0};}
  if(std::abs(a-b)<error){error=std::abs(a-b);balanced=cut;}
 }best.balanced_cut=balanced;return best;
}
}
