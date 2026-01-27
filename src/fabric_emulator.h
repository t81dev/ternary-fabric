#ifndef FABRIC_EMULATOR_H
#define FABRIC_EMULATOR_H

#include <stddef.h>
#include "tfmbs_device.h"

void* emu_fabric_alloc(size_t size);
void emu_fabric_free(void* ptr);
int emu_fabric_memcpy_to(void* dest_fabric, const void* src_host, size_t size, int pack_pt5);
int emu_fabric_memcpy_from(void* dest_host, const void* src_fabric, size_t size, int unpack_pt5);
int emu_fabric_exec_gemv(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols);
fabric_handle_t emu_fabric_exec_gemv_async(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols);
void emu_fabric_wait(fabric_handle_t handle);
void emu_fabric_get_metrics(fabric_metrics_t* out_metrics);
int emu_is_fabric_ptr(const void* ptr);

#endif
