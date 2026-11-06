/* Version 1: Linux same-process transport ABI. Validated pointers, serial caller.
 * All allocations belong to the creating context; free them before destroy.
 * Copies are synchronous: success means local queue/event completion, including
 * async errors. Host pointers must be base pointers from THIS context's pinned().
 * Device pointers are opaque owner-local handles; never dereference in the host
 * or pass to another context. Only bytes in host allocations cross runtimes.
 * Every callback catches all exceptions. No C++ types or ownership cross ABI.
 */
#ifndef STRATA_STAGE_ABI_H
#define STRATA_STAGE_ABI_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define STRATA_STAGE_ABI_V1 1u
#define STRATA_STAGE_HIP 1u
#define STRATA_STAGE_SYCL 2u
#define STRATA_STAGE_FACTORY "strata_stage_open_v1"
typedef struct strata_stage_error_v1 {
    int32_t code;
    char message[512];
} strata_stage_error_v1;
typedef struct strata_stage_api_v1 {
    uint32_t abi_version;
    uint32_t struct_bytes;
    uint32_t backend;
    uint32_t reserved;
    void* context;
    int32_t (*pinned)(void*, uint64_t, void**, strata_stage_error_v1*);
    int32_t (*allocate)(void*, uint64_t, void**, strata_stage_error_v1*);
    int32_t (*free_host)(void*, void*, strata_stage_error_v1*);
    int32_t (*free_device)(void*, void*, strata_stage_error_v1*);
    int32_t (*to_host)(void*, void* host, const void* device, uint64_t, strata_stage_error_v1*);
    int32_t (*to_device)(void*, void* device, const void* host, uint64_t, strata_stage_error_v1*);
    int32_t (*drain)(void*, strata_stage_error_v1*);
    int32_t (*describe)(void*, char*, uint64_t, strata_stage_error_v1*);
    void (*destroy)(void*);
} strata_stage_api_v1;
/* out/error must be nonnull; out is zeroed on failure. Nonzero status is failure.
 * Requested version/size must match exactly; future ABI gets another symbol.
 * No callback may be called concurrently on the same context. */
typedef int32_t (*strata_stage_factory_v1)(uint32_t, uint32_t, uint32_t,
                                         strata_stage_api_v1*, strata_stage_error_v1*);
#if defined(__GNUC__)
__attribute__((visibility("default")))
#endif
int32_t strata_stage_open_v1(uint32_t version, uint32_t api_bytes, uint32_t device_index,
                            strata_stage_api_v1* out, strata_stage_error_v1* error);
#ifdef __cplusplus
}
#endif
#endif
