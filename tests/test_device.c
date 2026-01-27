#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include "tfmbs_device.h"

int main() {
    printf("Testing TFMBS Device Emulator...\n");

    // 1. Test Allocation
    int8_t* weight_host = (int8_t*)malloc(15);
    for(int i=0; i<15; i++) weight_host[i] = (i % 3) - 1;

    void* weight_fabric = fabric_alloc(3); // 15 trits = 3 bytes PT-5
    assert(weight_fabric != NULL);
    assert(is_fabric_ptr(weight_fabric));

    // 2. Test Memcpy To (with packing)
    fabric_memcpy_to(weight_fabric, weight_host, 15, 1);

    // 3. Test Memcpy From (with unpacking)
    int8_t* weight_unpacked = (int8_t*)malloc(15);
    fabric_memcpy_from(weight_unpacked, weight_fabric, 15, 1);

    for(int i=0; i<15; i++) {
        assert(weight_unpacked[i] == weight_host[i]);
    }
    printf("Memcpy PT-5 Packing/Unpacking: PASSED\n");

    // 4. Test GEMV (Mock)
    // 1x15 * 15x1
    int8_t* input_host = (int8_t*)malloc(15);
    for(int i=0; i<15; i++) input_host[i] = 1;

    void* input_fabric = fabric_alloc(3);
    fabric_memcpy_to(input_fabric, input_host, 15, 1);

    void* output_fabric = fabric_alloc(4); // 1 int32_t

    fabric_exec_gemv(weight_fabric, input_fabric, output_fabric, 1, 15);

    int32_t result;
    fabric_memcpy_from(&result, output_fabric, sizeof(int32_t), 0);

    int32_t expected = 0;
    for(int i=0; i<15; i++) expected += weight_host[i] * input_host[i];

    printf("GEMV Result: %d (Expected: %d)\n", result, expected);
    assert(result == expected);
    printf("GEMV Mock Execution: PASSED\n");

    free(weight_host);
    free(weight_unpacked);
    free(input_host);

    printf("All Device Emulator Tests PASSED!\n");
    return 0;
}
