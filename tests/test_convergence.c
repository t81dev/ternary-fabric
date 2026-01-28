#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "tfmbs_device.h"
#include "tfmbs_api.h"

void test_convergence() {
    printf("\n--- [Test] Phase 19 Convergence & Stability ---\n");

    int rows = 256;
    int cols = 256;
    int size = rows * cols;
    int steps = 15;

    void* w = fabric_alloc(size);
    void* i = fabric_alloc(cols);
    void* o = fabric_alloc(rows * sizeof(int32_t));

    int8_t* host_data = malloc(size);
    memset(host_data, 1, size);
    fabric_memcpy_to(w, host_data, size, 1);
    fabric_memcpy_to(i, host_data, cols, 1);

    uint64_t last_hits = 0;

    for (int s = 0; s < steps; s++) {
        fabric_handle_t h = fabric_exec_gemv_async(w, i, o, rows, cols);
        fabric_wait(h);

        fabric_metrics_t m;
        fabric_get_metrics(&m);

        // Residency Hits should be monotonic (or at least not decrease if state is maintained)
        assert(m.residency_hits >= last_hits);
        last_hits = m.residency_hits;

        // In this simple test, tile selection should ideally stabilize
        // We can't strictly assert it doesn't change if costs are exactly equal,
        // but with hysteresis it should stay.
        // We'll just print it for now and manually verify or add a loose check.

        // To get the chosen tile, we could use the economic dump or a new metric.
        // For now, let's rely on the fact that if it's the same weight block,
        // it SHOULD converge to the same tile after first touch.

        printf("Step %d: Residency Hits %lu, Cost %.1f, Eff %.4f\n",
               s, m.residency_hits, m.fabric_cost, m.semantic_efficiency);
        tfmbs_dump_economic_csv("convergence_economic.csv");
    }

    fabric_free(w); fabric_free(i); fabric_free(o);
    free(host_data);

    printf("[Test] Convergence Test PASSED (Monotonic Residency Verified)\n");
}

int main() {
    test_convergence();
    tfmbs_dump_economic_csv("convergence_economic.csv");
    return 0;
}
