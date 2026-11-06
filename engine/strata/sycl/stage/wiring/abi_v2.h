/* Version 2 (ADDITIVE): compute-stage Backend over the C boundary. v1 (abi.h, strata_stage_open_v1,
 * strata_stage_api_v1) is unchanged and still exported by the same DSO; v2 is a second factory symbol
 * with its own struct. Same rules as v1: serial caller per context, every callback catches all
 * exceptions, nonzero status = failure with error filled, failure poisons the context (later calls
 * fail until begin()). No STL, C++ ownership, events or device pointers cross; Plan/Token/Packet
 * cross as codec.hpp byte records. All copies are synchronous: success means local completion.
 * Pinned host memory is allocated by THIS context (pinned/free_pinned) and only bytes cross runtimes.
 *
 * Lifetime rule: every pinned allocation must be released with free_pinned BEFORE destroy(), and the DSO
 * must stay loaded until destroy() returned. The host loader (loader_v2.hpp) enforces this with shared
 * ownership: each pinned allocation's deleter holds a reference to the context+library handle, so
 * destroy()/dlclose run only after the last allocation is freed. destroy() also releases any allocation
 * still registered, as a last resort; using such memory afterwards is undefined.
 * Factory: out/error/version/api_bytes are validated before anything is written to *out (a smaller,
 * older caller struct is never overrun); *out is zeroed only once those checks pass.
 * describe(): "<name> range=[a,b) device_bytes=N ... plan_footprint=M cap=C" after a successful begin().
 * Direct import_boundary callers must validate packet identity, epoch, step and checksum themselves:
 * the backend checks only edges, extents and position-derived valid rows (stage::Runtime does the rest).
 */
#ifndef STRATA_STAGE_ABI_V2_H
#define STRATA_STAGE_ABI_V2_H
#include "../abi.h"
#ifdef __cplusplus
extern "C" {
#endif
#define STRATA_STAGE_ABI_V2 2u
#define STRATA_STAGE_BACKEND_FACTORY "strata_stage_open_backend_v2"
typedef struct strata_stage_backend_api_v2 {
    uint32_t abi_version; /* STRATA_STAGE_ABI_V2 */
    uint32_t struct_bytes;
    uint32_t backend; /* STRATA_STAGE_HIP | STRATA_STAGE_SYCL */
    uint32_t reserved;
    void* context;
    /* plan: codec::encode(Plan). Loads/validates only layers [first,end) and zeroes all request state. */
    int32_t (*begin)(void*, const uint8_t* plan, uint64_t plan_bytes, uint32_t first, uint32_t end,
                     uint64_t request, strata_stage_error_v1*);
    int32_t (*pinned)(void*, uint64_t bytes, void** out, strata_stage_error_v1*);
    int32_t (*free_pinned)(void*, void* host, strata_stage_error_v1*);
    /* cancel: address of the caller's lock-free 1-byte std::atomic<bool>; read with acquire loads only. */
    int32_t (*execute)(void*, const uint8_t* token, uint64_t token_bytes, const void* cancel,
                       strata_stage_error_v1*);
    /* host must be a base pointer from this context's pinned(); returns after the local D2H completed. */
    int32_t (*export_boundary)(void*, const uint8_t* token, uint64_t token_bytes, void* host,
                               uint64_t host_bytes, uint64_t* written, strata_stage_error_v1*);
    int32_t (*committed_rows)(void*, const uint8_t* token, uint64_t token_bytes, uint64_t* rows,
                              uint64_t capacity, uint64_t* count, strata_stage_error_v1*);
    int32_t (*import_boundary)(void*, const uint8_t* token, uint64_t token_bytes, const void* host,
                               uint64_t host_bytes, const uint8_t* packet, uint64_t packet_bytes,
                               uint64_t* read, strata_stage_error_v1*);
    /* Tail only: full-vocabulary F32 logits after the local D2H. count = vocab. */
    int32_t (*logits)(void*, const uint8_t* token, uint64_t token_bytes, float* out, uint64_t capacity,
                      uint64_t* count, strata_stage_error_v1*);
    int32_t (*prepare_commit)(void*, uint64_t step, strata_stage_error_v1*);
    int32_t (*commit)(void*, uint64_t step, strata_stage_error_v1*);
    void (*invalidate)(void*);
    void (*drain)(void*);
    int32_t (*describe)(void*, char*, uint64_t, strata_stage_error_v1*);
    void (*destroy)(void*);
} strata_stage_backend_api_v2;
/* config: NUL-terminated "key=value\n" lines (pack=<dir>, max_context=<n>, dump_s=<file>,
 * dump_layers=<a,b>, cpu_threads=<n>, expert_placement=<file|all-cpu>). Unknown keys reject. */
typedef int32_t (*strata_stage_backend_factory_v2)(uint32_t version, uint32_t api_bytes, uint32_t device_index,
                                                  const char* config, strata_stage_backend_api_v2* out,
                                                  strata_stage_error_v1* error);
#if defined(__GNUC__)
__attribute__((visibility("default")))
#endif
int32_t strata_stage_open_backend_v2(uint32_t version, uint32_t api_bytes, uint32_t device_index, const char* config,
                                     strata_stage_backend_api_v2* out, strata_stage_error_v1* error);
#ifdef __cplusplus
}
#endif
#endif
