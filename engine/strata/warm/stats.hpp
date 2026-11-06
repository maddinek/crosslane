// Field adapter for the unchanged a5 boundary writer; observations are cumulative.
#pragma once
#include <cstdint>
namespace strata::forkwarm {
struct Stats {
 int64_t tokens=0,experts=0,cpu_experts=0,stream_experts=0,ram_experts=0;
 int64_t store_hits=0,store_misses=0,store_prefetched=0,store_pf_used=0,store_pf_wasted=0;
 int64_t store_evictions=0,store_bytes=0,store_fails=0,store_zc=0;
 int64_t store_slots=0,store_slots_max=0,store_zc_visible=0;
 double store_wait_ms=0,store_busy_ms=0,cpu_ms=0,gpu_wait_ms=0,engram_ms=0;
};
struct StoreDiagnosticStats {Stats model;int64_t no_slot=0,cancelled=0,put=0,slot_bytes=0;};
}
