#ifndef TFMBS_DEVICE_H
#define TFMBS_DEVICE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    long zero_skips;
    long total_ops;
    int lanes_used;
    double sim_cycle_reduction;
} fabric_metrics_t;

void* fabric_alloc(size_t size);
void fabric_free(void* ptr);
int fabric_memcpy_to(void* dest_fabric, const void* src_host, size_t size, int pack_pt5);
int fabric_memcpy_from(void* dest_host, const void* src_fabric, size_t size, int unpack_pt5);
int fabric_exec_gemv(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols);
void fabric_get_metrics(fabric_metrics_t* out_metrics);
int is_fabric_ptr(const void* ptr);

#ifdef __cplusplus
}
#endif

#endif // TFMBS_DEVICE_H
