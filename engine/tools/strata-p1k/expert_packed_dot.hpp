// MIT. Explicit whole-expert micro; reuse accepted strip math without fusion.
#pragma once
#include "iq1m_packed_dot.hpp"
namespace p1k_expert {
// Distinct launch symbols identify gate/up in final JIT. Both invocations use
// this same signature; pointers bind w1/dg and w3/du in resident.cpp.
inline sycl::event iq2_gate_up(sycl::queue& q,const uint8_t* w,size_t ld,int n,int k,
        float* y,const int8_t* xq,const float* xd,const uint16_t* table) {
    return q.submit([&](sycl::handler& cgh){
        cgh.parallel_for(sycl::nd_range<1>((size_t(n)+3)/4*4,4),
            sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>},
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                int row=int(it.get_group_linear_id())*4+int(it.get_local_linear_id());if(row>=n)return;
                p1k_packed::iq2_strip(w,ld,xq,xd,k,row,y,table);
            });
    });
}
inline sycl::event iq1m_gate_up(sycl::queue& q,const uint8_t* w,size_t ld,int n,int k,
        float* y,const int8_t* xq,const float* xd,const uint16_t* table) {
    return q.submit([&](sycl::handler& cgh){
        cgh.parallel_for(sycl::nd_range<1>((size_t(n)+3)/4*4,4),
            sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<128>},
            [=](sycl::nd_item<1> it) SYCL_ESIMD_KERNEL {
                int row=int(it.get_group_linear_id())*4+int(it.get_local_linear_id());if(row>=n)return;
                p1k_packed::iq1m_strip(w,ld,xq,xd,k,row,y,table);
            });
    });
}
}
