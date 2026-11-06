// MIT. Closed per-matrix dispatch shared by the bank and native checks.
#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "../sycl/moe/core.hpp"
#ifndef STRATA_WARM_PACKED
#define STRATA_WARM_PACKED
#endif
#include "../../tools/strata-p1k/expert_packed_dot.hpp"
#include "../../tools/strata-p1k/iq2xs_packed_dot.hpp"
#include <string_view>
namespace strata::wiring {
inline bool warm_packed_type(std::string_view type){
 return type=="IQ2_XXS"||type=="IQ2_XS"||type=="IQ1_M";
}
inline sycl::event warm_packed_project(sycl::queue& q,std::string_view type,bool gu,
 const uint8_t* w,size_t ld,int n,int k,float* y,const int8_t* x,const float* d,
 const uint16_t* xx,const uint16_t* xs,const uint16_t* im){
 if(type=="IQ2_XXS")return gu?p1k_expert::iq2_gate_up(q,w,ld,n,k,y,x,d,xx):p1k_packed::iq2_local4(q,w,ld,n,k,y,x,d,xx);
 if(type=="IQ2_XS")return gu?p1k_iq2xs::gate_up(q,w,ld,n,k,y,x,d,xs):p1k_iq2xs::local4(q,w,ld,n,k,y,x,d,xs);
 if(type=="IQ1_M")return gu?p1k_expert::iq1m_gate_up(q,w,ld,n,k,y,x,d,im):p1k_packed::iq1m_local4(q,w,ld,n,k,y,x,d,im);
 throw std::invalid_argument("unadmitted warm packed matrix format");
}
}
