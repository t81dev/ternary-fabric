#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "tfmbs_device.h"

int main() {
    printf("Starting Eviction Test...\n");

    // Allocate 3 blocks of 50MB each (Total 150MB, should trigger eviction since pool is 128MB)
    size_t size = 50 * 1024 * 1024;

    void* p1 = fabric_alloc(size);
    assert(p1 != NULL);
    printf("Allocated P1: %p\n", p1);

    void* p2 = fabric_alloc(size);
    assert(p2 != NULL);
    printf("Allocated P2: %p\n", p2);

    // At this point, ~100MB used. Pool is 128MB.
    // Next allocation of 50MB should trigger eviction of P1 (the LRU).

    void* p3 = fabric_alloc(size);
    assert(p3 != NULL);
    printf("Allocated P3: %p\n", p3);

    fabric_metrics_t metrics;
    fabric_get_metrics(&metrics);
    printf("Pool Used: %zu, Evictions: %d\n", metrics.pool_used, metrics.eviction_count);

    assert(metrics.eviction_count >= 1);

    // P1 was evicted, so its space should have been reused by P3 (potentially).
    // In our implementation, p3 might be the same as p1 if they matched size.
    printf("P1: %p, P3: %p\n", p1, p3);
    assert(p3 == p1); // Should reuse p1's slot after eviction and coalescing

    // Verify P2 is still "resident" (in the sense that we haven't reused its memory yet)
    // Actually, fabric_alloc doesn't tell us if it's resident, but metrics do.

    printf("Eviction Test Passed!\n");
    return 0;
}
