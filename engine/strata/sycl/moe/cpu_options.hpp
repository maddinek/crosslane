// MIT. Strict, bounded experimental CPU options. All defaults preserve a1.
#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <charconv>
#include <atomic>
namespace strata::wiring {
inline unsigned cpu_option(const char* name,unsigned maximum,unsigned fallback=0) {
    const char* s=std::getenv(name);if(!s)return fallback;
    const std::string v(s);unsigned n=0;
    const auto [end,ec]=std::from_chars(v.data(),v.data()+v.size(),n);
    if(ec!=std::errc{}||end!=v.data()+v.size()||n>maximum)
        throw std::invalid_argument(std::string("invalid ")+name);
    return n;
}
inline void cpu_pause() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}
}
