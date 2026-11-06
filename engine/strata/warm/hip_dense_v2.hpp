// MIT. Exact Q5_K input-sum cache. Own scratch, no implicit stream/global state.
#pragma once
#include "ds4_gguf.cuh"
#include "ds4_iq.cuh"
#include <cstring>
#include <stdexcept>
namespace strata::ds4::hipdense {
namespace x3 = strata::dsv41k::exl3;
namespace iq = strata::ds4::gguf::iq;
enum class Mode { off, sums8, sums4 };
struct Sums { int lo, hi; };
struct Workspace {
    int8_t* q = nullptr;
    float* d = nullptr;
    Sums* sums = nullptr;
    int max_k = 0; // each of kMaxMvJobs slots owns max_k values
};
inline Mode mode(const char* name) {
    if (!std::strcmp(name,"off")) return Mode::off;
    if (!std::strcmp(name,"q5-sums-v1")) return Mode::sums8;
    if (!std::strcmp(name,"q5-sums-w4-v1")) return Mode::sums4;
    throw std::invalid_argument("unknown HIP dense GEMV mode");
}
// Mixed/unsupported batches remain one unchanged upstream call, including its
// cross-format quantizer deduplication. No format or matrix descriptor changes.
inline bool eligible(const x3::MvJob* jobs,int n) {
    if(n<1 || n>x3::kMaxMvJobs) return false;
    for(int i=0;i<n;++i) {
        const auto& j=jobs[i]; if(!j.m) return false; const auto& m=*j.m;
        if(m.fmt!=gguf::kFmtGG || m.K!=iq::kQ5_K || !m.raw || !j.xf || !j.y ||
           j.ncol!=1 || m.k<=0 || m.k%256 || m.n<=0 ||
           (m.ld!=0 && size_t(m.ld)<gguf::type_row_bytes(m.K,m.k))) return false;
    }
    return true;
}
bool mv(const x3::MvJob*,int,Workspace&,Mode,cudaStream_t);
// Native checker only: copied pinned quantization without cached sums. Never
// used by production; permits direct byte checks of quantizer q/d output.
void reference_quant(const float*,int,int8_t*,float*,cudaStream_t);
}
