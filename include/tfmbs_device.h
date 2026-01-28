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

    // Phase 18 Semantic Metrics
    uint64_t active_ops;
    uint64_t mem_reads;
    uint64_t mem_writes;
    uint64_t broadcasts;
    uint64_t residency_hits;
    uint64_t residency_misses;
    uint64_t tile_local_reuse;

    long     cycles;
    double   fabric_cost;
    double   semantic_efficiency;
    double   economic_efficiency;
} fabric_metrics_t;

typedef void* fabric_handle_t;

void* fabric_alloc(size_t size);
void fabric_free(void* ptr);
int fabric_memcpy_to(void* dest_fabric, const void* src_host, size_t size, int pack_pt5);
int fabric_memcpy_from(void* dest_host, const void* src_fabric, size_t size, int unpack_pt5);
int fabric_exec_gemv(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols);
int fabric_exec_lstm(void* weight_ptr, void* input_ptr, void* output_ptr, int hidden_size, int input_size);

// Cooperative API (Phase 16)
void fabric_register_weight(void* ptr, size_t size);

// Async API (Phase 8)
fabric_handle_t fabric_exec_gemv_async(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols);
fabric_handle_t fabric_exec_lstm_async(void* weight_ptr, void* input_ptr, void* output_ptr, int hidden_size, int input_size);
fabric_handle_t fabric_exec_attn_async(void* q_ptr, void* k_ptr, void* v_ptr, void* o_ptr, int seq_len, int head_dim);
fabric_handle_t fabric_exec_conv3d_async(void* weight_ptr, void* input_ptr, void* output_ptr, int out_c, int in_c, int dhw);
int fabric_wait(fabric_handle_t handle);

// Phase 18 Persistence & Telemetry
void fabric_lstm_bind(void* weight_ptr, void* state_ptr, uint8_t tile_mask);
fabric_handle_t fabric_exec_lstm_persistent_async(void* weight_ptr, void* input_ptr, void* state_ptr, int h_size, int i_size, uint8_t tile_mask);
void fabric_dump_metrics_csv(const char* path);
void fabric_dump_economic_csv(const char* path);

// Low-level TFD submission (Phase 10)
int fabric_submit_tfd(tfmbs_tfd_t* tfd);

void fabric_get_metrics(fabric_metrics_t* out_metrics);
int is_fabric_ptr(const void* ptr);

#ifdef __cplusplus
}
#endif

#endif // TFMBS_DEVICE_H
