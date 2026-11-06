// MIT. Opt-in host-only combine candidates; never change the expert order.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string_view>
#include <vector>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define STRATA_OTHER_X86 1
#endif
namespace strata::wiring {
struct OtherOptions {
    bool simd=false;
    static bool flag(const char* name) {
        const char* s=std::getenv(name);
        if(!s)return false;
        if(std::string_view(s)!="0"&&std::string_view(s)!="1")throw std::runtime_error("invalid other candidate flag");
        return s[0]=='1';
    }
    static bool supported() {
#ifdef STRATA_OTHER_X86
        return __builtin_cpu_supports("avx2")&&__builtin_cpu_supports("fma");
#else
        return false;
#endif
    }
    static OtherOptions environment() {
        OtherOptions o{flag("DS4_OTHER_COMBINE_SIMD")};
        if(o.simd&&!supported())throw std::runtime_error("other combine needs host AVX2/FMA");
        if(o.simd)std::fprintf(stderr,"AIOPT_OTHER_COMBINE simd=1\n");
        return o;
    }
};
// Every vector lane is a different output coordinate. No horizontal reduction,
// no reassociation, no fast math and no process-wide compiler/ISA change.
#ifdef STRATA_OTHER_X86
__attribute__((target("avx2,fma"),noinline))
inline void other_fma(float w,const float* y,float* out,size_t n) {
    const auto weight=_mm256_set1_ps(w);size_t k=0;
    for(;k+8<=n;k+=8)
        _mm256_storeu_ps(out+k,_mm256_fmadd_ps(weight,_mm256_loadu_ps(y+k),_mm256_loadu_ps(out+k)));
    for(;k<n;++k)out[k]=std::fma(w,y[k],out[k]);
}
#endif

template<class Results>
std::vector<float> other_combine(std::span<const int> ids,std::span<const float> weights,
                                std::span<const float> shared,const Results& results,
                                OtherOptions options) {
    std::vector<const float*> rows(ids.size(),nullptr);
    std::vector<size_t> order(ids.size());
    for(const auto& tier:results)for(const auto& item:tier){
        if(item.first>=ids.size()||item.second.size()!=shared.size()||rows[item.first])
            throw std::runtime_error("other combine row census");
        rows[item.first]=item.second.data();
    }
    if(weights.size()!=ids.size()||std::any_of(rows.begin(),rows.end(),[](auto p){return !p;}))
        throw std::runtime_error("other combine missing row");
    std::iota(order.begin(),order.end(),0);
    std::sort(order.begin(),order.end(),[&](auto i,auto j){return ids[i]<ids[j];});
    std::vector<float> out(shared.size(),0);
    for(auto i:order){
#ifdef STRATA_OTHER_X86
        if(options.simd)other_fma(weights[i],rows[i],out.data(),out.size());else
#endif
        for(size_t k=0;k<out.size();++k)out[k]=std::fma(weights[i],rows[i][k],out[k]);
    }
    for(size_t k=0;k<out.size();++k)out[k]+=shared[k];
    return out;
}
}
