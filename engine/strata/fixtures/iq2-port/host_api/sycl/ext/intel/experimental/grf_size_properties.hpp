// CPU API fake only. Does not prove native property support or GRF allocation.
#pragma once
#ifndef STRATA_IQ2_HOST_TEST
#error "Explicit CPU test only"
#endif
namespace sycl::ext::intel::experimental {
template<int N>struct grf_size_t{static constexpr int value=N;};
template<int N>inline constexpr grf_size_t<N> grf_size{};
}
namespace sycl::ext::oneapi::experimental {
template<class T>struct properties {
    static constexpr int grf_size=T::value;
    explicit properties(T){}
};
}
