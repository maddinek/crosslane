// MIT. Opt-in T=1 HIP scheduling experiments. Every arm requires native bitcheck.
#pragma once
#include "ds4_kernels.cuh"
namespace strata::ds4::hipfuse {
struct Hc {
    const float *h, *fn, *scale, *base, *previous_pre, *norm;
    float *mixes, *pre, *post, *comb, *x;
    unsigned* count;
    int dim, hc, n_mix, iters;
    float eps, hc_eps;
};
void hc_pre(const Hc&, cudaStream_t);
void router(const float* weights, const float* x, const float* bias,
            int n, int dim, int k, float scale, float* logits,
            int* idx, float* weights_out, unsigned* count, cudaStream_t);
// KV finish must already have written ext[n]. Each block owns a disjoint old
// ring row. Block zero reads the overwritten row before appending the new one.
void ring(uint8_t* ring, uint16_t* ext, int64_t position, int n, int dim, cudaStream_t);
// Score scratch is heads * score_stride floats; score_stride >= nwin+nsel.
// This splits QK rows only. Softmax/PV keep their original 64-row recurrence.
void attention(const float* q, const uint16_t* win, const void* ckv, const int* sel,
               int sel_stride, const AttnRows&, const float* sink, int heads,
               int dim, float scale, float* out, const AttnScratch&,
               float* scores, int score_stride, cudaStream_t, const Finish* epilogue=nullptr);
}
