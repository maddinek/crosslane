// Host oracle only; never include in, or substitute for, an inference kernel.
// expf algorithm/table: musl v1.2.5 src/math/{expf,exp2f_data}.c.
// Copyright (c) 2017-2018, Arm Limited. SPDX-License-Identifier: MIT
// Full license + exact original sources/hashes: docs/experiments/p1k-evidence/host-exp/.
// Adaptations: fixed binary64 intermediates, explicit rounding barriers, canonical
// exceptional results (no errno/fenv exception contract). No libc exp/SVML calls.
#pragma once
#include <bit>
#include <cfenv>
#include <cstdint>
#include <limits>
#include <stdexcept>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "P1-K host oracle requires precise arithmetic, no fast math"
#endif

namespace p1k_host {
static_assert(sizeof(float)==4 && sizeof(double)==8 &&
              std::numeric_limits<float>::is_iec559 && std::numeric_limits<float>::digits==24 &&
              std::numeric_limits<double>::is_iec559 && std::numeric_limits<double>::digits==53);

// Each operation rounds separately, even if a caller enables FMA contraction.
// Volatile stores also defeat vector-libm substitution and excess precision.
inline double rn64(double x) { volatile double y=x; return y; }
inline float rn32(double x) { volatile float y=static_cast<float>(x); return y; }

inline void check_environment() {
    if (std::fegetround()!=FE_TONEAREST)
        throw std::runtime_error("P1-K host oracle requires round-to-nearest-even");
    volatile float tiny=std::bit_cast<float>(uint32_t{0x00800000});
    volatile float half=tiny*0.5f;
    volatile float restored=half*2.0f;
    if (std::bit_cast<uint32_t>(float(half))!=0x00400000 || restored!=tiny)
        throw std::runtime_error("P1-K host oracle requires gradual underflow (no FTZ/DAZ)");
}

inline float exp(float x) {
    constexpr uint64_t tab[32]={
        0x3ff0000000000000,0x3fefd9b0d3158574,0x3fefb5586cf9890f,0x3fef9301d0125b51,
        0x3fef72b83c7d517b,0x3fef54873168b9aa,0x3fef387a6e756238,0x3fef1e9df51fdee1,
        0x3fef06fe0a31b715,0x3feef1a7373aa9cb,0x3feedea64c123422,0x3feece086061892d,
        0x3feebfdad5362a27,0x3feeb42b569d4f82,0x3feeab07dd485429,0x3feea47eb03a5585,
        0x3feea09e667f3bcd,0x3fee9f75e8ec5f74,0x3feea11473eb0187,0x3feea589994cce13,
        0x3feeace5422aa0db,0x3feeb737b0cdc5e5,0x3feec49182a3f090,0x3feed503b23e255d,
        0x3feee89f995ad3ad,0x3feeff76f2fb5e47,0x3fef199bdd85529c,0x3fef3720dcef9069,
        0x3fef5818dcfba487,0x3fef7c97337b9b5f,0x3fefa4afa2a490da,0x3fefd0765b6e4540
    };
    const uint32_t bits=std::bit_cast<uint32_t>(x), abits=bits&0x7fffffff;
    if (abits>0x7f800000) return std::bit_cast<float>(uint32_t{0x7fc00000});
    if (bits==0xff800000) return 0.0f;
    if (bits==0x7f800000 || x>0x1.62e42ep6f)
        return std::bit_cast<float>(uint32_t{0x7f800000});
    if (x< -0x1.9fe368p6f) return 0.0f;
    double z=rn64((0x1.71547652b82fep+0*32)*double(x));
    double kd=rn64(z+0x1.8p+52);
    const uint64_t ki=std::bit_cast<uint64_t>(kd);
    kd=rn64(kd-0x1.8p+52);
    const double r=rn64(z-kd);
    const double s=std::bit_cast<double>(tab[ki%32]+(ki<<(52-5)));
    z=rn64(rn64((0x1.c6af84b912394p-5/32/32/32)*r)+(0x1.ebfce50fac4f3p-3/32/32));
    const double r2=rn64(r*r);
    double y=rn64(rn64((0x1.62e42ff0c52d6p-1/32)*r)+1.0);
    y=rn64(rn64(z*r2)+y);
    return rn32(rn64(y*s));
}
} // namespace p1k_host
