#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tfmbs_device.h"
#include "tfmbs_api.h"

void run_stress(float sparsity, int use_batching) {
    int rows = 512;
    int cols = 1024;
    int size = rows * cols;
    int num_steps = 10;

    void* w = fabric_alloc(size);
    void* i = fabric_alloc(cols);
    void* o = fabric_alloc(rows * sizeof(int32_t));

    int8_t* host_w = malloc(size);
    int8_t* host_i = malloc(cols);
    memset(host_i, 1, cols);

    // Initialize with high sparsity
    for (int j = 0; j < size; j++) {
        host_w[j] = ((float)rand() / RAND_MAX < (1.0 - sparsity)) ? 1 : 0;
    }
    fabric_memcpy_to(w, host_w, size, 1);
    fabric_memcpy_to(i, host_i, cols, 1);

    printf("\n[Sparse-Stress] Sparsity: %.1f%%, Batching: %s\n", sparsity * 100.0, use_batching ? "ON" : "OFF");

    fabric_handle_t handles[num_steps];

    for (int s = 0; s < num_steps; s++) {
        handles[s] = fabric_exec_gemv_async(w, i, o, rows, cols);
        if (!use_batching) {
            fabric_wait(handles[s]);
            fabric_dump_economic_csv("sparse_stress_economic.csv");
            fabric_dump_metrics_csv("sparse_stress_metrics.csv");
        }
    }

    if (use_batching) {
        for (int s = 0; s < num_steps; s++) {
            fabric_wait(handles[s]);
            fabric_dump_economic_csv("sparse_stress_economic.csv");
            fabric_dump_metrics_csv("sparse_stress_metrics.csv");
        }
    }

    fabric_free(w); fabric_free(i); fabric_free(o);
    free(host_w); free(host_i);
}

int main() {
    srand(42);
    // Sparsity levels: 95%, 98%, 99%
    float levels[] = {0.95, 0.98, 0.99};

    for (int l = 0; l < 3; l++) {
        run_stress(levels[l], 0); // Batching OFF
        run_stress(levels[l], 1); // Batching ON
    }

    return 0;
}
