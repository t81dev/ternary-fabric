#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "tfmbs.h"

/**
 * @brief Unpacks PT-5 balanced ternary data.
 */
void unpack_pt5_reference(const uint8_t* packed, int8_t* trits, uint32_t count) {
    for (uint32_t i = 0; i < count; i++) {
        uint32_t byte_idx = i / 5;
        uint32_t trit_in_byte = i % 5;
        uint8_t byte_val = packed[byte_idx];
        for (uint32_t p = 0; p < trit_in_byte; p++) byte_val /= 3;
        uint8_t unsigned_trit = byte_val % 3;
        // Mapping: 0->0, 1->1, 2->-1 (Balanced Ternary)
        trits[i] = (unsigned_trit == 2) ? -1 : (int8_t)unsigned_trit;
    }
}

/**
 * @brief Reference Ternary Dot Product (T-MAC)
 */
int32_t ternary_dot_product(const int8_t* weights, const int8_t* inputs, uint32_t count) {
    int32_t accumulator = 0;
    for (uint32_t i = 0; i < count; i++) {
        accumulator += (weights[i] * inputs[i]);
    }
    return accumulator;
}

int main(int argc, char** argv) {
    uint32_t depth = 1000000;
    if (argc > 1) depth = atoi(argv[1]);

    printf("🧪 TFMBS Reference C-Implementation Benchmark\n");
    printf("Depth: %u trits\n", depth);

    // 1. Generate random trits and pack them
    size_t packed_size = (depth + 4) / 5;
    uint8_t* packed_weights = malloc(packed_size);
    uint8_t* packed_inputs = malloc(packed_size);

    // Fill with some dummy data
    for(size_t i=0; i<packed_size; i++) {
        packed_weights[i] = rand() % 243;
        packed_inputs[i] = rand() % 243;
    }

    // 2. Start Timing
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // 3. Dehydrate (Unpack) and Execute (MAC)
    int8_t* w_trits = malloc(depth);
    int8_t* i_trits = malloc(depth);

    unpack_pt5_reference(packed_weights, w_trits, depth);
    unpack_pt5_reference(packed_inputs, i_trits, depth);

    int32_t result = ternary_dot_product(w_trits, i_trits, depth);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Total Ops: 1 Mul + 1 Add per trit = 2 ops
    double gops = (depth * 2.0) / duration / 1e9;

    printf("Result: %d\n", result);
    printf("⏱️  Execution Time: %.6f seconds\n", duration);
    printf("📊 Throughput: %.4f GOPS\n", gops);

    free(packed_weights);
    free(packed_inputs);
    free(w_trits);
    free(i_trits);

    return 0;
}
