// MIT. Test-only, single host thread cooperative SIMT executor. Not a HIP shim
// for production. Executes actual HC/router bodies with 32-lane shuffles and barriers;
// does not model GPU codegen, weak memory or device transcendental intrinsics.
#pragma once
#include <ucontext.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <vector>
struct dim3 {unsigned x,y,z;dim3(unsigned a=1,unsigned b=1,unsigned c=1):x(a),y(b),z(c){}};
inline dim3 threadIdx,blockIdx,blockDim,gridDim;
struct float4 {float x,y,z,w;};
namespace simt {
enum class Wait {ready,done,block,warp,shuffle};
struct Fiber {ucontext_t context{};std::array<char,16384> stack;Wait wait=Wait::ready;float input{},result{};int offset{};};
inline struct Block* active;
struct Block {
    ucontext_t scheduler{};std::vector<Fiber> fibers;std::function<void()> kernel;unsigned current{};
    static void entry(){auto* b=active;b->kernel();b->fibers[b->current].wait=Wait::done;}
    explicit Block(unsigned n,std::function<void()> fn):fibers(n),kernel(std::move(fn)){
        for(auto& f:fibers){getcontext(&f.context);f.context.uc_stack.ss_sp=f.stack.data();
            f.context.uc_stack.ss_size=f.stack.size();f.context.uc_link=&scheduler;
            makecontext(&f.context,&entry,0);}
    }
    float yield(Wait wait,float input=0,int offset=0){
        auto& f=fibers[current];f.wait=wait;f.input=input;f.offset=offset;
        swapcontext(&f.context,&scheduler);return f.result;
    }
    void run(){
        active=this;blockDim=dim3(fibers.size());
        for(;;){
            for(current=0;current<fibers.size();++current)if(fibers[current].wait==Wait::ready){
                threadIdx=dim3(current,0,0);swapcontext(&scheduler,&fibers[current].context);
            }
            bool done=true,barrier=true,progress=false;
            for(auto& f:fibers){done&=f.wait==Wait::done;barrier&=f.wait==Wait::block;}
            if(done)return;
            if(barrier){for(auto& f:fibers)f.wait=Wait::ready;continue;}
            for(size_t w=0;w<fibers.size();w+=32){
                const auto kind=fibers[w].wait;const auto offset=fibers[w].offset;
                if(kind!=Wait::warp&&kind!=Wait::shuffle)continue;
                bool full=true;
                for(size_t l=0;l<32;++l)full&=fibers[w+l].wait==kind&&fibers[w+l].offset==offset;
                if(!full)continue;
                for(size_t l=0;l<32;++l){auto& f=fibers[w+l];f.result=fibers[w+(l^offset)].input;f.wait=Wait::ready;}
                progress=true;
            }
            if(!progress)throw std::runtime_error("SIMT divergent collective/deadlock");
        }
    }
};
inline void launch(unsigned grid,unsigned threads,const std::function<void()>& fn,bool reverse=false){
    gridDim=dim3(grid);for(unsigned b=0;b<grid;++b){blockIdx=dim3(reverse?grid-1-b:b,0,0);Block(threads,fn).run();}
}
}
inline void __syncthreads(){simt::active->yield(simt::Wait::block);}
inline void __syncwarp(){simt::active->yield(simt::Wait::warp);}
inline void __threadfence(){}  // Serial blocks; no weak-memory claim.
inline float __shfl_xor_sync(unsigned,float x,int o){return simt::active->yield(simt::Wait::shuffle,x,o);}
inline int __shfl_xor_sync(unsigned mask,int x,int o){
    // Preserve INT_MAX exactly; converting the router sentinel to float rounds it.
    float bits;std::memcpy(&bits,&x,4);bits=__shfl_xor_sync(mask,bits,o);
    int result;std::memcpy(&result,&bits,4);return result;
}
inline unsigned atomicAdd(unsigned* p,unsigned n){unsigned old=*p;*p+=n;return old;}
inline float rsqrtf(float x){return 1.0f/::sqrtf(x);}
#define __shared__ static
#define __global__
#define __forceinline__ inline
#define __launch_bounds__(...)
