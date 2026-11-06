// MIT; attribution in core.hpp. Runtime owns all buffers, committed lengths and epochs.
#pragma once
#include "numerics.hpp"
#include "core.hpp"
namespace strata::ds4::sycl_port {
using Rows=attn_scalar::Rows;using AttnRows=attn_scalar::AttnRows;
sycl::event compress_win(sycl::queue&,const float*,const float*,float*,float*,int,int,int,int,const float*,float,float*);
sycl::event index_score(sycl::queue&,const float*,const float*,const void*,int,int,int,int,const int*,int,int,const Rows&,float*,long long,int=3);
sycl::event topk(sycl::queue&,const float*,long long,int,const Rows&,const int*,int,int,int*,int);
sycl::event cand_blocks(sycl::queue&,const float*,long long,int,const Rows&,int,float*,long long,int*,int);
sycl::event sparse_attn(sycl::queue&,const float*,const uint16_t*,const void*,const int*,int,const AttnRows&,const float*,int,int,int,float,float*,int=2);
// Same 64-row online-softmax arithmetic as sparse_attn, distributed over its
// existing 32 logical lanes. Explicit candidate; the scalar default stays intact.
sycl::event sparse_attn_subgroup32(sycl::queue&,const float*,const uint16_t*,const void*,const int*,int,const AttnRows&,const float*,int,int,int,float,float*,int=2);
// Explicit two-phase split-KV path. Scratch: T*heads*nsplit*514 F32, nsplit>=1.
// Parameter R = fixed rows per chunk (1..1024); nsplit=max(1,ceil(max_visible/R)).
// No in-chunk rescale; empty partitions carry den=0.
sycl::event sparse_attn_split(sycl::queue&,const float*,const uint16_t*,const void*,const int*,int,const AttnRows&,const float*,int,int,int,float,float*,int,int,float*,size_t);
}
