#ifndef TFMBS_DEVICE_H
#define TFMBS_DEVICE_H

#include <stddef.h>
#include <stdint.h>
#include "tfmbs.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    long zero_skips;
    long total_ops;
    int lanes_used;
    double sim_cycle_reduction;
    size_t pool_used;
    size_t pool_total;
    int eviction_count;
} fabric_metrics_t;

typedef void* fabric_handle_t;

void* fabric_alloc(size_t size);
void fabric_free(void* ptr);
int fabric_memcpy_to(void* dest_fabric, const void* src_host, size_t size, int pack_pt5);
int fabric_memcpy_from(void* dest_host, const void* src_fabric, size_t size, int unpack_pt5);
int fabric_exec_gemv(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols);

// Cooperative API (Phase 16)
void fabric_register_weight(void* ptr, size_t size);

// Async API (Phase 8)
fabric_handle_t fabric_exec_gemv_async(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols);
int fabric_wait(fabric_handle_t handle);

// Low-level TFD submission (Phase 10)
int fabric_submit_tfd(tfmbs_tfd_t* tfd);

void fabric_get_metrics(fabric_metrics_t* out_metrics);
int is_fabric_ptr(const void* ptr);

#ifdef __cplusplus
}
#endif

#endif // TFMBS_DEVICE_H
