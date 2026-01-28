#include <stdlib.h>
#include <string.h>
#include "tfmbs_api.h"

tfmbs_tensor_t tfmbs_tensor_bind(void* host_ptr, size_t size, int pack_pt5) {
    tfmbs_tensor_t t;
    t.size = size;
    t.flags = 0;

    // Allocate on fabric
    size_t alloc_size = pack_pt5 ? (size + 4) / 5 : size;
    t.device_ptr = fabric_alloc(alloc_size);

    if (t.device_ptr && host_ptr) {
        fabric_memcpy_to(t.device_ptr, host_ptr, size, pack_pt5);
    }

    return t;
}

void tfmbs_tensor_release(tfmbs_tensor_t* tensor) {
    if (tensor && tensor->device_ptr) {
        fabric_free(tensor->device_ptr);
        tensor->device_ptr = NULL;
    }
}

fabric_handle_t tfmbs_gemm(tfmbs_tensor_t* weight, tfmbs_tensor_t* input, tfmbs_tensor_t* output, int rows, int cols) {
    if (!weight || !input || !output) return NULL;
    return fabric_exec_gemv_async(weight->device_ptr, input->device_ptr, output->device_ptr, rows, cols);
}

fabric_handle_t tfmbs_lstm_step(tfmbs_tensor_t* weight, tfmbs_tensor_t* input, tfmbs_tensor_t* output, int h_size, int i_size) {
    if (!weight || !input || !output) return NULL;
    return fabric_exec_lstm_async(weight->device_ptr, input->device_ptr, output->device_ptr, h_size, i_size);
}

fabric_handle_t tfmbs_attn(tfmbs_tensor_t* q, tfmbs_tensor_t* k, tfmbs_tensor_t* v, tfmbs_tensor_t* o, int seq_len, int head_dim) {
    if (!q || !k || !v || !o) return NULL;
    return fabric_exec_attn_async(q->device_ptr, k->device_ptr, v->device_ptr, o->device_ptr, seq_len, head_dim);
}

fabric_handle_t tfmbs_conv3d(tfmbs_tensor_t* weight, tfmbs_tensor_t* input, tfmbs_tensor_t* output, int in_c, int out_c, int dhw) {
    if (!weight || !input || !output) return NULL;
    return fabric_exec_conv3d_async(weight->device_ptr, input->device_ptr, output->device_ptr, out_c, in_c, dhw);
}

void tfmbs_lstm_bind(tfmbs_tensor_t* weight, tfmbs_tensor_t* state, uint8_t tile_mask) {
    if (weight && state) {
        fabric_lstm_bind(weight->device_ptr, state->device_ptr, tile_mask);
    }
}

fabric_handle_t tfmbs_lstm_step_async(tfmbs_tensor_t* weight, tfmbs_tensor_t* input, tfmbs_tensor_t* state, int h_size, int i_size, uint8_t tile_mask) {
    if (!weight || !input || !state) return NULL;
    return fabric_exec_lstm_persistent_async(weight->device_ptr, input->device_ptr, state->device_ptr, h_size, i_size, tile_mask);
}

void tfmbs_sync(fabric_handle_t handle) {
    if (handle) {
        fabric_wait(handle);
    }
}

void tfmbs_dump_metrics_csv(const char* path) {
    fabric_dump_metrics_csv(path);
}

void tfmbs_dump_economic_csv(const char* path) {
    fabric_dump_economic_csv(path);
}
