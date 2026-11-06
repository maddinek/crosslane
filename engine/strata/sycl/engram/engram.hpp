// MIT; Project Maya attribution in decode.hpp.
#pragma once
#include <sycl/sycl.hpp>
#include <cstddef>
#include <cstdint>
namespace strata::ds4::engram {
// Format 0 is host-unpacked MXFP8/legacy FP8; 1 Q6_K; 2 IQ4_NL.
// All rows retain upstream hd-byte slots. Pointers are USM in q's context;
// in-order queue and disjoint input/output allocations required. Caller owns bounds.
sycl::event dequant(sycl::queue&,const uint8_t*,const uint8_t*,int,int,float*,int);
sycl::event dequant_chunk(sycl::queue&,const uint8_t*,size_t,int,int,int,float*,int);
sycl::event gate(sycl::queue&,float*,const float*,const float*,int,int,int,float);
// Dense fixture/reference seam only. Real quantized eg_wkv uses row2's matrix API:
// input [T,cols*hd], weights [out,in], output [T,(hc+1)*dim].
sycl::event project_f32(sycl::queue&,const float*,const float*,float*,int,int,int);
sycl::event project_bf16(sycl::queue&,const float*,const uint16_t*,float*,int,int,int);
}
