#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tfmbs_device.h"

void benchmark_tile_saturation() {
    printf("\n--- [Layer 1] Tile Saturation Test ---\n");
    int rows = 1024;
    int cols = 1024;

    void* w = fabric_alloc(rows * cols);
    void* i = fabric_alloc(cols);
    void* o = fabric_alloc(rows * sizeof(int32_t));

    int8_t* host_w = malloc(rows * cols);
    int8_t* host_i = malloc(cols);
    memset(host_w, 1, rows * cols); // All ones to avoid zero-skips
    memset(host_i, 1, cols);

    fabric_memcpy_to(w, host_w, rows * cols, 1);
    fabric_memcpy_to(i, host_i, cols, 1);

    for (int mask = 1; mask <= 15; mask = (mask << 1) | 1) {
        char mask_str[10];
        snprintf(mask_str, sizeof(mask_str), "0x%02x", mask);
        setenv("FABRIC_TILE_MASK", mask_str, 1);

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        fabric_handle_t h = fabric_exec_gemv_async(w, i, o, rows, cols);
        fabric_wait(h);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double diff = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

        fabric_metrics_t m;
        fabric_get_metrics(&m);
        printf("Mask 0x%02x: %.6f s, Zero-Skips: %ld, Pool: %zu MB\n",
               mask, diff, m.zero_skips, m.pool_used / (1024*1024));
    }

    fabric_free(w); fabric_free(i); fabric_free(o);
    free(host_w); free(host_i);
}

void benchmark_zero_skip_density() {
    printf("\n--- [Layer 1] Zero-Skip Density Curves ---\n");
    int size = 1024 * 512;
    void* w = fabric_alloc(size);
    void* i = fabric_alloc(1024);
    void* o = fabric_alloc(512 * sizeof(int32_t));

    int8_t* host_w = malloc(size);
    int8_t* host_i = malloc(1024);
    memset(host_i, 1, 1024);

    float densities[] = {0.0, 0.25, 0.5, 0.75, 0.9, 1.0};
    for (int d = 0; d < 6; d++) {
        float density = densities[d];
        for (int j = 0; j < size; j++) {
            host_w[j] = ((float)rand() / RAND_MAX > density) ? 1 : 0;
        }
        fabric_memcpy_to(w, host_w, size, 1);
        fabric_memcpy_to(i, host_i, 1024, 1);

        fabric_handle_t h = fabric_exec_gemv_async(w, i, o, 512, 1024);
        fabric_wait(h);

        fabric_metrics_t m;
        fabric_get_metrics(&m);
        printf("Density %.2f: Skip Ratio %.1f%%\n", density, m.sim_cycle_reduction);
    }

    fabric_free(w); fabric_free(i); fabric_free(o);
    free(host_w); free(host_i);
}

int main() {
    benchmark_tile_saturation();
    benchmark_zero_skip_density();
    return 0;
}
