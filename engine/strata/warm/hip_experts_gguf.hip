// MIT. Original pinned TU, original launch helper and original compiler flags.
// Only the checker installs a thread-local callback; serving leaves it null.
#include "hip_experts.hpp"
#include "ds4_launch.h"
#include <cstring>
template<class... KA,class... CA>
inline void hipex_launch(const char* spelling,const char* file,int line,void(*kernel)(KA...),
                        dim3 grid,dim3 block,size_t shared,cudaStream_t stream,CA&&... args){
    const auto trace=strata::wiring::hipex::primitive_trace;
    const bool quant=trace.mark && std::strncmp(spelling,"q8_quant",8)==0;
    if(quant)trace("quant_start",stream);
    strata_ds4_launch_at(file,line,kernel,grid,block,shared,stream,static_cast<CA&&>(args)...);
    if(quant)trace("quant_end",stream);
}
#undef strata_ds4_launch
#define strata_ds4_launch(...) hipex_launch(#__VA_ARGS__,__FILE__,__LINE__,__VA_ARGS__)
#include "ds4_gguf.cu"
