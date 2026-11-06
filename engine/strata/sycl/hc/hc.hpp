// MIT, Project Maya attribution in core.hpp. In-order queue, caller-owned USM.
#pragma once
#include "numerics.hpp"
namespace strata::ds4::sycl_port {
sycl::event hc_mix(sycl::queue&,const float*,const float*,int,int,int,float,float*);
sycl::event hc_mix_scale(sycl::queue&,const float*,int,int,int,float,float*);
sycl::event hc_split(sycl::queue&,const float*,const float*,const float*,int,int,float,int,float*,float*,float*);
sycl::event hc_mix_split(sycl::queue&,const float*,const float*,int,int,int,float,float*,const float*,const float*,int,int,float,float*,float*,float*);
sycl::event hc_mix_split(sycl::queue&,const float*,const float*,int,int,int,float,float*,unsigned*,const float*,const float*,int,int,float,float*,float*,float*);
sycl::event hc_pre_norm(sycl::queue&,const float*,const float*,const float*,int,int,int,float,float*);
sycl::event hc_post(sycl::queue&,const float*,const float*,const float*,const float*,int,int,int,float*);
sycl::event hc_post_inplace(sycl::queue&,const float*,float*,const float*,const float*,int,int,int);
}
