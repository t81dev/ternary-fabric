#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tfmbs_device.h"
#include "tfmbs_api.h"

void benchmark_zero_skip_density() {
    printf("\n--- [Synthetic] Zero-Skip Density Curves ---\n");
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
        printf("Density %.2f: Skip Ratio %.1f%%, Eff: %.4f, Cost: %.2f\n",
               density, m.sim_cycle_reduction, m.semantic_efficiency, m.fabric_cost);
    }

    fabric_free(w); fabric_free(i); fabric_free(o);
    free(host_w); free(host_i);
}

int main() {
    benchmark_zero_skip_density();
    tfmbs_dump_metrics_csv("density_metrics.csv");
    return 0;
}
