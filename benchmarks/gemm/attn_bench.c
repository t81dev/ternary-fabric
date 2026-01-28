#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tfmbs_api.h"

void benchmark_attn_kernel() {
    printf("\n--- [ATTN] Kernel Benchmark ---\n");
    int seq_len = 512, head_dim = 64;

    tfmbs_tensor_t q = tfmbs_tensor_bind(NULL, seq_len * head_dim, 1);
    tfmbs_tensor_t k = tfmbs_tensor_bind(NULL, seq_len * head_dim, 1);
    tfmbs_tensor_t v = tfmbs_tensor_bind(NULL, seq_len * head_dim, 1);
    tfmbs_tensor_t o = tfmbs_tensor_bind(NULL, seq_len * head_dim * sizeof(int32_t), 0);

    for(int n=0; n<5; n++) {
        fabric_handle_t h = tfmbs_attn(&q, &k, &v, &o, seq_len, head_dim);
        tfmbs_sync(h);

        fabric_metrics_t m;
        fabric_get_metrics(&m);
        printf("Run %d: Cost: %.2f, Efficiency: %.4f, Residency Hits: %lu\n",
               n, m.fabric_cost, m.semantic_efficiency, m.residency_hits);
    }

    tfmbs_tensor_release(&q); tfmbs_tensor_release(&k); tfmbs_tensor_release(&v); tfmbs_tensor_release(&o);
}

int main() {
    benchmark_attn_kernel();
    return 0;
}
