// MIT. Build this IN PLACE OF the upstream ds4_kernels.cu object in the warm
// stage only. Original entrypoints are compiled verbatim, preserving defaults.
// The include gives the extensions access to the pinned private numerics and
// fallback flag. Never compile both objects into one target.
#include "ds4_kernels.cu"
#include "hip_fuse.hpp"
#include <stdexcept>
#include "hip_fuse_generated.inc"
