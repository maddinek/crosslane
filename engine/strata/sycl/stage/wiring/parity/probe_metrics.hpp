// MIT. Binary64 diagnostic math.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>
#include <stdexcept>
namespace strata::wiring {
struct ProbeMetrics {
    double error2=0, cpu2=0, hip2=0, max_abs=0, peak_cpu=0;
    bool exact=false;
    double nmse_cpu() const { return error2/std::max(cpu2,1e-30); }
    double nmse_hip() const { return error2/std::max(hip2,1e-30); }
    bool passed(bool zero) const { return zero ? exact : max_abs<=1e-5+0.10*peak_cpu; }
};
inline ProbeMetrics probe_metrics(std::span<const float> hip, std::span<const float> cpu) {
    if(hip.empty() || hip.size()!=cpu.size()) throw std::runtime_error("expert output extent");
    ProbeMetrics m;
    m.exact=std::memcmp(hip.data(),cpu.data(),hip.size_bytes())==0;
    for(size_t i=0;i<cpu.size();++i) {
        if(!std::isfinite(cpu[i]) || !std::isfinite(hip[i])) throw std::runtime_error("nonfinite expert output");
        const double a=hip[i], b=cpu[i], d=a-b;
        m.error2+=d*d; m.cpu2+=b*b; m.hip2+=a*a;
        m.max_abs=std::max(m.max_abs,std::abs(d)); m.peak_cpu=std::max(m.peak_cpu,std::abs(b));
    }
    return m;
}
}
