#pragma once
#ifndef STRATA_IQ2_HOST_TEST
#error "Explicit CPU test only"
#endif
namespace sycl::ext::intel::math{inline float fdiv_rn(float a,float b){return a/b;}}
