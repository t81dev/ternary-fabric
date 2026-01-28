#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "tfmbs_device.h"
#include "tfmbs_api.h"

void print_p18_metrics(const char* label) {
    fabric_metrics_t m;
    fabric_get_metrics(&m);
    printf("[%s] Cycles: %ld, Cost: %.2f, Eff: %.4f, Hits: %lu, Miss: %lu\n",
           label, m.cycles, m.fabric_cost, m.semantic_efficiency, m.residency_hits, m.residency_misses);
}

void benchmark_tile_saturation() {
    printf("\n--- [Synthetic] Tile Saturation Test ---\n");
    int rows = 1024;
    int cols = 1024;

    void* w = fabric_alloc(rows * cols);
    void* i = fabric_alloc(cols);
    void* o = fabric_alloc(rows * sizeof(int32_t));

    int8_t* host_w = malloc(rows * cols);
    int8_t* host_i = malloc(cols);
    memset(host_w, 1, rows * cols);
    memset(host_i, 1, cols);

    fabric_memcpy_to(w, host_w, rows * cols, 1);
    fabric_memcpy_to(i, host_i, cols, 1);

    for (int mask = 1; mask <= 15; mask = (mask << 1) | 1) {
        char mask_str[10];
        snprintf(mask_str, sizeof(mask_str), "0x%02x", mask);
        setenv("FABRIC_TILE_MASK", mask_str, 1);

        fabric_handle_t h = fabric_exec_gemv_async(w, i, o, rows, cols);
        fabric_wait(h);

        char label[20];
        snprintf(label, sizeof(label), "Mask 0x%02x", mask);
        print_p18_metrics(label);
    }

    fabric_free(w); fabric_free(i); fabric_free(o);
    free(host_w); free(host_i);
}

int main() {
    benchmark_tile_saturation();
    tfmbs_dump_metrics_csv("saturation_metrics.csv");
    return 0;
}
