#ifndef TFMBS_DEVICE_H
#define TFMBS_DEVICE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Allocate memory in the Fabric pool.
 * Returns a handle/pointer to Fabric-resident memory.
 */
void* fabric_alloc(size_t size);

/**
 * Free memory in the Fabric pool.
 */
void fabric_free(void* ptr);

/**
 * Copy data from Host RAM to Fabric Memory.
 * If pack_pt5 is non-zero, it assumes src_host contains raw ternary values (-1, 0, 1)
 * and packs them into PT-5 format in the Fabric.
 * If pack_pt5 is zero, it performs a raw byte copy.
 */
int fabric_memcpy_to(void* dest_fabric, const void* src_host, size_t size, int pack_pt5);

/**
 * Copy data from Fabric Memory to Host RAM.
 * If unpack_pt5 is non-zero, it assumes dest_fabric contains PT-5 data
 * and unpacks it into raw ternary values (-1, 0, 1) in the Host buffer.
 */
int fabric_memcpy_from(void* dest_host, const void* src_fabric, size_t size, int unpack_pt5);

/**
 * Execute a Ternary General Matrix-Vector multiplication.
 * y = W * x
 * @param weight_ptr Fabric-resident weight matrix (PT-5 packed)
 * @param input_ptr Fabric-resident input vector (PT-5 packed)
 * @param output_ptr Fabric-resident output vector (binary int32_t)
 */
int fabric_exec_gemv(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols);

/**
 * Helper to check if a pointer belongs to the Fabric pool.
 */
int is_fabric_ptr(const void* ptr);

#ifdef __cplusplus
}
#endif

#endif // TFMBS_DEVICE_H
