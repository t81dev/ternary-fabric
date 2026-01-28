#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tfmbs_api.h"

int main() {
    printf("\n--- [Temporal] Persistent LSTM Benchmark ---\n");
    int h_size = 512, i_size = 512;
    int rows = 4 * h_size, cols = i_size + h_size;

    int8_t* host_w = malloc(rows * cols);
    memset(host_w, 1, rows * cols);

    tfmbs_tensor_t w = tfmbs_tensor_bind(host_w, rows * cols, 1);
    // Hidden state
    tfmbs_tensor_t state = tfmbs_tensor_bind(NULL, h_size * sizeof(int32_t), 0);
    tfmbs_tensor_t input = tfmbs_tensor_bind(NULL, i_size, 1);

    // Phase 18: Bind
    tfmbs_lstm_bind(&w, &state, 0x0F);

    for (int t = 0; t < 10; t++) {
        fabric_handle_t h = tfmbs_lstm_step_async(&w, &input, &state, h_size, i_size, 0x0F);
        tfmbs_sync(h);

        fabric_metrics_t m;
        fabric_get_metrics(&m);
        printf("Step %d: Cost: %.2f, Efficiency: %.4f, Hits: %lu\n",
               t, m.fabric_cost, m.semantic_efficiency, m.residency_hits);
        tfmbs_dump_metrics_csv("temporal_metrics.csv");
    }

    tfmbs_tensor_release(&w); tfmbs_tensor_release(&state); tfmbs_tensor_release(&input);
    free(host_w);
    return 0;
}
