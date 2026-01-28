#ifndef FABRIC_EMULATOR_H
#define FABRIC_EMULATOR_H

#include <stddef.h>
#include "tfmbs_device.h"

void emu_fabric_init();
void* emu_fabric_alloc(size_t size);
void* emu_fabric_alloc_id(int fabric_id, size_t size);
void emu_fabric_free(void* ptr);
int emu_fabric_memcpy_to(void* dest_fabric, const void* src_host, size_t size, int pack_pt5);
int emu_fabric_memcpy_from(void* dest_host, const void* src_fabric, size_t size, int unpack_pt5);
int emu_fabric_exec_gemv(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols, uint8_t tile_mask);
int emu_fabric_exec_lstm(void* weight_ptr, void* input_ptr, void* output_ptr, int hidden_size, int input_size, uint8_t tile_mask);
fabric_handle_t emu_fabric_exec_gemv_async(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols, uint8_t tile_mask);
fabric_handle_t emu_fabric_exec_gemv_async_id(int fabric_id, void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols, uint8_t tile_mask);
fabric_handle_t emu_fabric_exec_lstm_async(void* weight_ptr, void* input_ptr, void* output_ptr, int hidden_size, int input_size, uint8_t tile_mask);
fabric_handle_t emu_fabric_exec_lstm_async_id(int fabric_id, void* weight_ptr, void* input_ptr, void* output_ptr, int hidden_size, int input_size, uint8_t tile_mask);
int emu_fabric_wait(fabric_handle_t handle);
void emu_fabric_get_metrics(fabric_metrics_t* out_metrics);
void emu_fabric_get_metrics_id(int fabric_id, fabric_metrics_t* out_metrics);
int emu_is_fabric_ptr(const void* ptr);

void emu_fabric_lstm_bind(void* weight_ptr, void* state_ptr, uint8_t tile_mask);
fabric_handle_t emu_fabric_exec_lstm_persistent_async(void* weight_ptr, void* input_ptr, void* state_ptr, int h_size, int i_size, uint8_t tile_mask);
void emu_fabric_dump_metrics_csv(const char* path);
void emu_fabric_dump_economic_csv(const char* path);

int emu_fabric_inter_copy(int src_id, int dst_id, void* src_ptr, void* dst_ptr, size_t size);

#endif
