// MIT. Capture only immutable-argument device sequences. No position, host copy,
// event timing, allocation, pointer swap or host callback may be inside submit().
#pragma once
#include <array>
#include <cstddef>
#include <stdexcept>

namespace strata::wiring {
template<class Api> class DenseGraph {
    using Graph = typename Api::Graph;
    using Exec = typename Api::Exec;
    Graph graph_{};
    Exec exec_{};
    std::array<const void*,2> key_{};
    typename Api::Stream stream_{};
public:
    DenseGraph() = default;
    DenseGraph(const DenseGraph&) = delete;
    DenseGraph& operator=(const DenseGraph&) = delete;
    ~DenseGraph() { reset(); }
    // Caller must drain the owning stream before resetting or freeing inputs.
    void reset() noexcept {
        if (exec_) Api::destroy_exec(exec_);
        if (graph_) Api::destroy_graph(graph_);
        graph_={};exec_={};key_={};stream_={};
    }
    template<class F> void run(bool enabled, typename Api::Stream stream,
                              std::array<const void*,2> key, F submit) {
        if (!enabled) { submit(); return; }
        if (exec_ && (key != key_ || stream != stream_))
            throw std::runtime_error("HIP dense graph input/stream changed before reset");
        if (!exec_) {
            Api::begin(stream);
            try { submit(); }
            catch (...) { Api::cancel(stream); throw; }
            try {
                graph_=Api::end(stream);
                exec_=Api::instantiate(graph_);
                key_=key;stream_=stream;
            } catch (...) { reset(); throw; }
        }
        // Capturing submits no work. First use also launches exactly once.
        Api::launch(exec_,stream);
    }
};
}
