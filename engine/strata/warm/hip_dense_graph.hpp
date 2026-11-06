// MIT. Native HIP adapter; the scheduler itself also runs with a CPU fake API.
#pragma once
#include "dense_graph.hpp"
#include <hip/hip_runtime.h>
#include <string>
namespace strata::wiring {
struct HipDenseGraphApi {
    using Stream=hipStream_t;using Graph=hipGraph_t;using Exec=hipGraphExec_t;
    static void check(hipError_t e) {
        if(e!=hipSuccess)throw std::runtime_error(std::string("HIP dense graph: ")+hipGetErrorString(e));
    }
    static void begin(Stream s){check(hipStreamBeginCapture(s,hipStreamCaptureModeThreadLocal));}
    static Graph end(Stream s){Graph g{};auto e=hipStreamEndCapture(s,&g);if(e!=hipSuccess&&g)hipGraphDestroy(g);check(e);return g;}
    static void cancel(Stream s)noexcept{Graph g{};hipStreamEndCapture(s,&g);if(g)hipGraphDestroy(g);}
    static Exec instantiate(Graph g){Exec e{};auto rc=hipGraphInstantiate(&e,g,nullptr,nullptr,0);if(rc!=hipSuccess&&e)hipGraphExecDestroy(e);check(rc);return e;}
    static void launch(Exec e,Stream s){check(hipGraphLaunch(e,s));}
    static void destroy_exec(Exec e)noexcept{hipGraphExecDestroy(e);}
    static void destroy_graph(Graph g)noexcept{hipGraphDestroy(g);}
};
using HipDenseGraph=DenseGraph<HipDenseGraphApi>;
}
