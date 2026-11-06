// Host-only API shim. This does not test the Intel device library.
#pragma once
#ifndef STRATA_ROW34_HOST_API_TEST
#error "This shim may only be included by the explicit offline host API test"
#endif
namespace sycl::ext::intel::math {
inline float fdiv_rn(float a,float b) {return a/b;}
}
