#ifndef TFMBS_API_H
#define TFMBS_API_H

#include <stddef.h>
#include <stdint.h>
#include "tfmbs_device.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Represents a resident tensor on the Ternary Fabric.
 */
typedef struct {
    void* device_ptr;
    size_t size;
    uint32_t flags;
} tfmbs_tensor_t;

/**
 * @brief Bind a host buffer to the Fabric, establishing residency.
 * @param host_ptr Source data
 * @param size Number of elements (trits)
 * @param pack_pt5 Whether to pack RAW to PT-5
 * @return Tensor object
 */
tfmbs_tensor_t tfmbs_tensor_bind(void* host_ptr, size_t size, int pack_pt5);

/**
 * @brief Release a resident tensor.
 */
void tfmbs_tensor_release(tfmbs_tensor_t* tensor);

/**
 * @brief High-level GEMM operation.
 */
fabric_handle_t tfmbs_gemm(tfmbs_tensor_t* weight, tfmbs_tensor_t* input, tfmbs_tensor_t* output, int rows, int cols);

/**
 * @brief Single-step LSTM update.
 * Assumes weight contains 4 gates [4*h, i+h]
 */
fabric_handle_t tfmbs_lstm_step(tfmbs_tensor_t* weight, tfmbs_tensor_t* input, tfmbs_tensor_t* output, int h_size, int i_size);

/**
 * @brief Explicit synchronization point.
 */
void tfmbs_sync(fabric_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif // TFMBS_API_H
