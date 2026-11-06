// Exact division at quantization boundaries; ordinary SYCL '/' permits 2.5 ulp.
#pragma once
#include <sycl/ext/intel/math.hpp>
namespace strata::ds4::sycl_port {
// Intel IMF specifies IEEE binary32 round-to-nearest-even, including gradual
// underflow and signed zero. Use the library's sign/overflow/underflow handling
// rather than comparing rounded FMA residuals (which may underflow to zero).
// https://www.intel.com/content/www/us/en/docs/dpcpp-cpp-compiler/developer-guide-reference/2026-0/simple-arithmetic-operations-with-rounding-mode.html
inline float div_rn(float a,float b) {
    return sycl::ext::intel::math::fdiv_rn(a,b);
}
}
