// MIT. Warm overlay replaces ONLY the IQ translation unit. Original upstream
// entrypoints remain verbatim; every expert still calls its original kernels.
#include "ds4_iq.cu"
#include "hip_dense_v2.hpp"
#include "hip_dense_v2_generated.inc"
