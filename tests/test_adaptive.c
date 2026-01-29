#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include "tfmbs_device.h"

void run_gemv(int rows, int cols, float sparsity) {
    int8_t* w_host = malloc(rows * cols);
    int8_t* i_host = malloc(cols);
    int32_t* o_host = malloc(rows * sizeof(int32_t));

    for (int i = 0; i < rows * cols; i++) {
        if ((float)rand() / RAND_MAX < sparsity) w_host[i] = 0;
        else w_host[i] = (rand() % 2) ? 1 : -1;
    }
    for (int i = 0; i < cols; i++) i_host[i] = (rand() % 2) ? 1 : -1;

    void* w_fab = fabric_alloc((rows * cols + 4) / 5);
    void* i_fab = fabric_alloc((cols + 4) / 5);
    void* o_fab = fabric_alloc(rows * sizeof(int32_t));

    fabric_memcpy_to(w_fab, w_host, rows * cols, 1);
    fabric_memcpy_to(i_fab, i_host, cols, 1);

    fabric_exec_gemv(w_fab, i_fab, o_fab, rows, cols);

    fabric_memcpy_from(o_host, o_fab, rows * sizeof(int32_t), 0);

    fabric_free(w_fab);
    fabric_free(i_fab);
    fabric_free(o_fab);
    free(w_host);
    free(i_host);
    free(o_host);
}

int main() {
    printf("--- TFMBS Phase 26: Adaptive Runtime Agent Test ---\n");

    // Set environment for adaptive policy
    setenv("TFMBS_ADAPTIVE_POLICY", "sparsity", 1);
    setenv("TFMBS_SPARSITY_THRESHOLD", "0.5", 1);
    setenv("TFMBS_DEBUG", "1", 1);

    // 1. Run with high sparsity (should offload to fabric)
    printf("\n[Step 1] Running with HIGH sparsity (0.8). Expecting Fabric Offload...\n");
    for (int i = 0; i < 5; i++) {
        run_gemv(64, 64, 0.8f);
    }

    fabric_metrics_t m;
    fabric_get_metrics(&m);
    printf("Current Metrics: Offloads=%lu, Fallbacks=%lu\n", m.offload_count, m.fallback_count);
    assert(m.offload_count > 0);

    // 2. Run with low sparsity (should eventually fallback to CPU)
    printf("\n[Step 2] Running with LOW sparsity (0.1). Expecting eventual CPU Fallback...\n");
    for (int i = 0; i < 20; i++) {
        run_gemv(64, 64, 0.1f);
    }

    fabric_get_metrics(&m);
    printf("Current Metrics: Offloads=%lu, Fallbacks=%lu\n", m.offload_count, m.fallback_count);
    assert(m.fallback_count > 0);

    // 3. Run with high sparsity again (should eventually probe and return to fabric)
    printf("\n[Step 3] Running with HIGH sparsity again (0.9). Expecting eventual return to Fabric...\n");
    uint64_t prev_offloads = m.offload_count;
    for (int i = 0; i < 30; i++) {
        run_gemv(64, 64, 0.9f);
    }

    fabric_get_metrics(&m);
    printf("Current Metrics: Offloads=%lu, Fallbacks=%lu\n", m.offload_count, m.fallback_count);
    assert(m.offload_count > prev_offloads);

    printf("\nAdaptive Runtime Agent Test: PASSED\n");
    return 0;
}
