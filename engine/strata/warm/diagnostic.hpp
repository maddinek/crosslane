// Boundary-only, default-off store observation; no expert-loop hooks or device waits.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include "stats.hpp"

namespace strata::forkwarm {
class StoreDiagnostic {
    StoreDiagnosticStats start_, previous_, decode_;
    int64_t request_ = 0, prompt_ = 0;
    bool decoded_ = false;
    static void counters(const StoreDiagnosticStats& x, const StoreDiagnosticStats* base) {
        const Stats zero{};
        const Stats& a = x.model;
        const Stats& b = base ? base->model : zero;
        const StoreDiagnosticStats z{};
        const auto& y = base ? *base : z;
        bool first = true;
        const auto integer = [&](const char* key, int64_t v) {
            std::fprintf(stderr, "%s\"%s\":%lld", first ? "" : ",", key, (long long) v);
            first = false;
        };
        const auto real = [&](const char* key, double v) {
            std::fprintf(stderr, ",\"%s\":%.9f", key, v);
        };
#define SD_I(key, field) integer(key, a.field - b.field)
        SD_I("model_fed_tokens", tokens);
        SD_I("expert_selections", experts);
        SD_I("cpu_expert_selections", cpu_experts);
        SD_I("stream_expert_selections", stream_experts);
        SD_I("legacy_ram_expert_selections", ram_experts);
        SD_I("store_hits", store_hits);
        SD_I("store_misses", store_misses);
        SD_I("store_prefetched", store_prefetched);
        SD_I("store_prefetch_used", store_pf_used);
        SD_I("store_prefetch_wasted", store_pf_wasted);
        SD_I("store_evictions", store_evictions);
        SD_I("store_bytes_read", store_bytes);
        SD_I("store_fails", store_fails);
        SD_I("store_zero_copy", store_zc);
#undef SD_I
        integer("store_no_slot", x.no_slot - y.no_slot);
        integer("store_cancelled", x.cancelled - y.cancelled);
        integer("store_put", x.put - y.put);
        real("store_wait_ms", a.store_wait_ms - b.store_wait_ms);
        real("store_busy_ms", a.store_busy_ms - b.store_busy_ms);
        real("cpu_ms", a.cpu_ms - b.cpu_ms);
        real("gpu_host_wait_ms", a.gpu_wait_ms - b.gpu_wait_ms);
        real("engram_ms", a.engram_ms - b.engram_ms);
    }
    void emit(const char* event, const StoreDiagnosticStats& x, int64_t outputs, bool ok) {
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        std::fprintf(stderr, "store_diag_v1 {\"request\":%lld,\"event\":\"%s\",\"monotonic_ns\":%lld,"
            "\"prompt_tokens\":%lld,\"generated_outputs\":%lld,\"fed_decode_tokens\":%lld,\"ok\":%s,"
            "\"snapshot_atomic\":false,\"background_completion_origin_available\":false,\"totals\":{",
            (long long) request_, event, (long long) ns, (long long) prompt_, (long long) outputs,
            (long long) (decoded_ ? x.model.tokens - decode_.model.tokens : 0), ok ? "true" : "false");
        counters(x, nullptr);
        std::fprintf(stderr, "},\"interval_delta\":{"); counters(x, &previous_);
        std::fprintf(stderr, "},\"request_delta\":{"); counters(x, &start_);
        std::fprintf(stderr, "},\"gauges\":{\"store_slot_bytes\":%lld,\"store_slots_allocated\":%lld,"
            "\"store_slots_max\":%lld,\"store_visible_slots\":%lld}}\n",
            (long long) x.slot_bytes, (long long) x.model.store_slots,
            (long long) x.model.store_slots_max, (long long) x.model.store_zc_visible);
        previous_ = x;
    }
public:
    void begin(const StoreDiagnosticStats& x, int64_t prompt) {
        ++request_; prompt_ = prompt; decoded_ = false;
        start_ = previous_ = x;
        emit("request_begin", x, 0, true);  // also prefill_begin
    }
    void prefill_end(const StoreDiagnosticStats& x) {
        decode_ = x; decoded_ = true;
        emit("prefill_end", x, 0, true);    // also decode_begin; last prompt feed belongs to prefill
    }
    void end(const StoreDiagnosticStats& x, int64_t outputs, bool ok) {
        emit(decoded_ ? "decode_end" : "prefill_abort", x, outputs, ok);
        emit("request_end", x, outputs, ok);
    }
};
}  // namespace strata::forkwarm
