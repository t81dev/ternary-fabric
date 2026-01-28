#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>
#include "tfmbs_device.h"

void test_learning() {
    printf("\n--- [Test] Phase 20 Adaptive Learning ---\n");

    int rows = 512;
    int cols = 512;
    int size = rows * cols;
    int iterations = 100;

    void* w = fabric_alloc(size);
    void* i = fabric_alloc(cols);
    void* o = fabric_alloc(rows * sizeof(int32_t));

    int8_t* host_data = malloc(size);
    memset(host_data, 1, size);
    // Fill with some sparsity to make it interesting
    for (int j = 0; j < size; j++) if (j % 3 == 0) host_data[j] = 0;

    fabric_memcpy_to(w, host_data, size, 1);
    fabric_memcpy_to(i, host_data, cols, 1);

    printf("Running %d iterations to observe parameter tuning...\n", iterations);

    for (int it = 0; it < iterations; it++) {
        // Submit several tasks to allow batching
        fabric_handle_t handles[8];
        for(int b=0; b<8; b++) handles[b] = fabric_exec_gemv_async(w, i, o, rows, cols);
        for(int b=0; b<8; b++) fabric_wait(handles[b]);

        if (it % 20 == 0) {
            fabric_dump_economic_csv("learning_test.csv");
            printf("Iteration %d complete.\n", it);
        }
    }

    fabric_dump_economic_csv("learning_test.csv");
    fabric_free(w); fabric_free(i); fabric_free(o);
    free(host_data);

    printf("[Test] Phase 20 Learning Test complete. Check learning_test.csv for parameter evolution.\n");
}

int main() {
    test_learning();
    return 0;
}
