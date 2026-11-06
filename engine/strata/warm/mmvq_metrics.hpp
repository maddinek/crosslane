// MIT. Prospective screening gates, separate from frozen full-model quality.
#pragma once
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <span>
#include <stdexcept>
namespace strata::wiring::mmvq {
struct Metrics {
 double squared=0,reference_squared=0,max_abs=0,peak=0,max_relative=0;uint64_t max_ulp=0,count=0;
 static uint32_t ordered(float v){auto u=std::bit_cast<uint32_t>(v);return u&0x80000000u?~u+1u:u|0x80000000u;}
 void add(float a,float b){if(!std::isfinite(a)||!std::isfinite(b))throw std::runtime_error("MMVQ nonfinite accuracy value");double e=std::abs(double(a)-b);squared+=e*e;reference_squared+=double(b)*b;max_abs=std::max(max_abs,e);peak=std::max(peak,std::abs(double(b)));auto aa=ordered(a),bb=ordered(b);max_ulp=std::max(max_ulp,uint64_t(aa>bb?aa-bb:bb-aa));++count;}
 double nrmse()const{return reference_squared?std::sqrt(squared/reference_squared):squared?INFINITY:0;}
 bool expert_ok()const{return count&&nrmse()<=0.005&&max_abs<=0.02*peak+1e-6;}
};
inline Metrics metrics(std::span<const float> a,std::span<const float> b){if(a.size()!=b.size()||a.empty())throw std::invalid_argument("MMVQ accuracy extent");Metrics r;for(size_t i=0;i<a.size();++i)r.add(a[i],b[i]);for(size_t i=0;i<a.size();++i)if(std::abs(double(b[i]))>1e-3*r.peak)r.max_relative=std::max(r.max_relative,std::abs(double(a[i])-b[i])/std::abs(double(b[i])));return r;}
}
