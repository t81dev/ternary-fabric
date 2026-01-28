#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tfmbs_api.h"

void benchmark_gemm_kernel() {
    printf("\n--- [Layer 2] GEMM Kernel Benchmark ---\n");
    int rows = 1024, cols = 1024;

    int8_t* host_w = malloc(rows * cols);
    for(int j=0; j<rows*cols; j++) host_w[j] = (rand()%3)-1;

    // Using high-level API
    tfmbs_tensor_t w = tfmbs_tensor_bind(host_w, rows * cols, 1);
    tfmbs_tensor_t i = tfmbs_tensor_bind(NULL, cols, 1);
    tfmbs_tensor_t o = tfmbs_tensor_bind(NULL, rows * sizeof(int32_t), 0);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for(int n=0; n<10; n++) {
        fabric_handle_t h = tfmbs_gemm(&w, &i, &o, rows, cols);
        tfmbs_sync(h);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double diff = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("10x GEMM %dx%d: %.6f s (%.2f GFLOPS effective)\n",
           rows, cols, diff, (rows*cols*2.0*10.0)/diff/1e9);

    tfmbs_tensor_release(&w); tfmbs_tensor_release(&i); tfmbs_tensor_release(&o);
    free(host_w);
}

void benchmark_lstm_kernel() {
    printf("\n--- [Layer 2] LSTM Kernel Benchmark ---\n");
    int h_size = 512;
    int i_size = 512;
    int rows = 4 * h_size;
    int cols = i_size + h_size;

    int8_t* host_w = malloc(rows * cols);
    memset(host_w, 0, rows * cols);
    for(int j=0; j<rows*cols; j++) if(rand()%10 == 0) host_w[j] = (rand()%3)-1;

    tfmbs_tensor_t w = tfmbs_tensor_bind(host_w, rows * cols, 1);
    tfmbs_tensor_t i = tfmbs_tensor_bind(NULL, cols, 1);
    tfmbs_tensor_t o = tfmbs_tensor_bind(NULL, rows * sizeof(int32_t), 0);

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for(int n=0; n<10; n++) {
        fabric_handle_t h = tfmbs_lstm_step(&w, &i, &o, h_size, i_size);
        tfmbs_sync(h);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double diff = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    fabric_metrics_t m;
    fabric_get_metrics(&m);
    printf("10x T-LSTM Step (H=%d, I=%d): %.6f s\n", h_size, i_size, diff);
    printf("Last Step Metrics: %ld skips, %.1f%% cycle reduction\n", m.zero_skips, m.sim_cycle_reduction);

    tfmbs_tensor_release(&w); tfmbs_tensor_release(&i); tfmbs_tensor_release(&o);
    free(host_w);
}

int main() {
    benchmark_gemm_kernel();
    benchmark_lstm_kernel();
    return 0;
}
