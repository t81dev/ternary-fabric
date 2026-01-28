#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "tfmbs_device.h"

void test_multi_fabric() {
    printf("\n--- [Test] Phase 21 Multi-Fabric Orchestration ---\n");

    int rows = 512;
    int cols = 512;
    int size = rows * cols;

    // Allocate buffers. Due to round-robin, these should land on different fabrics.
    void* w1 = fabric_alloc(size);
    void* w2 = fabric_alloc(size);
    void* i = fabric_alloc(cols);
    void* o1 = fabric_alloc(rows * sizeof(int32_t));
    void* o2 = fabric_alloc(rows * sizeof(int32_t));

    int8_t* host_data = malloc(size);
    memset(host_data, 1, size);
    fabric_memcpy_to(w1, host_data, size, 1);
    fabric_memcpy_to(w2, host_data, size, 1);
    fabric_memcpy_to(i, host_data, cols, 1);

    printf("Executing sequence of kernels across multiple fabrics...\n");

    // Submit a mix of kernels. Orchestrator should distribute them and handle transfers.
    fabric_handle_t h1 = fabric_exec_gemv_async(w1, i, o1, rows, cols);
    fabric_handle_t h2 = fabric_exec_gemv_async(w2, i, o2, rows, cols);

    // Dependent task (Mock Fusion): uses o1 as input
    fabric_handle_t h3 = fabric_exec_gemv_async(w2, o1, o2, rows, rows);

    fabric_wait(h1);
    fabric_wait(h2);
    fabric_wait(h3);

    printf("Dumping Multi-Fabric Economic Metrics...\n");
    fabric_dump_economic_csv("multi_fabric_test.csv");

    fabric_free(w1); fabric_free(w2); fabric_free(i); fabric_free(o1); fabric_free(o2);
    free(host_data);

    printf("[Test] Phase 21 Multi-Fabric Test complete. Check multi_fabric_test.csv for orchestration results.\n");
}

int main() {
    // Force 2 fabrics for this test
    setenv("TFMBS_NUM_FABRICS", "2", 1);
    test_multi_fabric();
    return 0;
}
