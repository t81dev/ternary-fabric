#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include "tfmbs_device.h"

int main() {
    printf("Starting Async Test...\n");

    // Allocate buffers
    void* w = fabric_alloc(100000);
    void* i = fabric_alloc(100000);
    void* o = fabric_alloc(100000);

    // Fill with dummy data
    int8_t host_w[100000];
    int8_t host_i[100000];
    for(int k=0; k<100000; k++) { host_w[k] = 1; host_i[k] = 1; }

    fabric_memcpy_to(w, host_w, 100000, 1);
    fabric_memcpy_to(i, host_i, 100000, 1);

    printf("Launching Async GEMV...\n");
    fabric_handle_t handle = fabric_exec_gemv_async(w, i, o, 100, 100);
    assert(handle != NULL);

    printf("Doing some 'host' work while Fabric computes...\n");
    for(int j=0; j<10; j++) {
        printf("Host working... %d\n", j);
        usleep(10000);
    }

    printf("Waiting for result...\n");
    fabric_wait(handle);

    printf("Verifying result...\n");
    int32_t results[100];
    fabric_memcpy_from(results, o, 100 * sizeof(int32_t), 0);

    // Each row of 100 elements should be 100 (since w=1, x=1)
    for(int r=0; r<100; r++) {
        if (results[r] != 100) {
            printf("Mismatch at row %d: %d\n", r, results[r]);
            exit(1);
        }
    }

    printf("Async Test Passed!\n");
    return 0;
}
