#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tfmbs_api.h"

void benchmark_conv3d_kernel() {
    printf("\n--- [CONV3D] Kernel Benchmark ---\n");
    int in_c = 64, out_c = 64, dhw = 3*3*3; // Small 3D kernel volume

    tfmbs_tensor_t w = tfmbs_tensor_bind(NULL, out_c * in_c * dhw, 1);
    tfmbs_tensor_t i = tfmbs_tensor_bind(NULL, in_c * 10 * 10 * 10, 1);
    tfmbs_tensor_t o = tfmbs_tensor_bind(NULL, out_c * sizeof(int32_t), 0);

    for(int n=0; n<5; n++) {
        fabric_handle_t h = tfmbs_conv3d(&w, &i, &o, in_c, out_c, dhw);
        tfmbs_sync(h);

        fabric_metrics_t m;
        fabric_get_metrics(&m);
        printf("Run %d: Cost: %.2f, Efficiency: %.4f, Residency Hits: %lu\n",
               n, m.fabric_cost, m.semantic_efficiency, m.residency_hits);
    }

    tfmbs_tensor_release(&w); tfmbs_tensor_release(&i); tfmbs_tensor_release(&o);
}

int main() {
    benchmark_conv3d_kernel();
    return 0;
}
