#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tfmbs_api.h"

void benchmark_gemm_kernel() {
    printf("\n--- [GEMM] Kernel Benchmark ---\n");
    int rows = 1024, cols = 1024;

    int8_t* host_w = malloc(rows * cols);
    for(int j=0; j<rows*cols; j++) host_w[j] = (rand()%3)-1;

    tfmbs_tensor_t w = tfmbs_tensor_bind(host_w, rows * cols, 1);
    tfmbs_tensor_t i = tfmbs_tensor_bind(NULL, cols, 1);
    tfmbs_tensor_t o = tfmbs_tensor_bind(NULL, rows * sizeof(int32_t), 0);

    for(int n=0; n<5; n++) {
        fabric_handle_t h = tfmbs_gemm(&w, &i, &o, rows, cols);
        tfmbs_sync(h);

        fabric_metrics_t m;
        fabric_get_metrics(&m);
        printf("Run %d: Cost: %.2f, Efficiency: %.4f, Residency Hits: %lu\n",
               n, m.fabric_cost, m.semantic_efficiency, m.residency_hits);
        tfmbs_dump_metrics_csv("gemm_metrics.csv");
    }

    tfmbs_tensor_release(&w); tfmbs_tensor_release(&i); tfmbs_tensor_release(&o);
    free(host_w);
}

int main() {
    benchmark_gemm_kernel();
    return 0;
}
