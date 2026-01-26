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

/**
 * @brief Reference Ternary GEMM (Matrix-Vector Multiply)
 * Y = W * X
 * @param weights Interleaved matrix [cols][rows] if matching SIMD lanes,
 *                or flat [rows * cols]
 */
void ternary_gemm_reference(const int8_t* weights, const int8_t* inputs, int32_t* outputs, uint32_t rows, uint32_t cols) {
    for (uint32_t r = 0; r < rows; r++) {
        int32_t acc = 0;
        for (uint32_t c = 0; c < cols; c++) {
            // Standard row-major: weights[r * cols + c]
            // Fabric-friendly (Interleaved): weights[c * rows + r]
            // We'll use row-major for the reference and handle mapping in the loader/driver.
            acc += (int32_t)weights[r * cols + c] * (int32_t)inputs[c];
        }
        outputs[r] = acc;
    }
}

int main(int argc, char** argv) {
    uint32_t depth = 1000000;
    int mode = 0; // 0: Dot, 1: GEMM
    if (argc > 1) depth = atoi(argv[1]);
    if (argc > 2) mode = atoi(argv[2]);

    printf("🧪 TFMBS Reference C-Implementation Benchmark (%s)\n", mode == 0 ? "Dot" : "GEMM");
    printf("Depth/Size: %u\n", depth);

    if (mode == 0) {
        // --- DOT PRODUCT MODE ---
        size_t packed_size = (depth + 4) / 5;
        uint8_t* packed_weights = malloc(packed_size);
        uint8_t* packed_inputs = malloc(packed_size);

        for(size_t i=0; i<packed_size; i++) {
            packed_weights[i] = rand() % 243;
            packed_inputs[i] = rand() % 243;
        }

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        int8_t* w_trits = malloc(depth);
        int8_t* i_trits = malloc(depth);
        unpack_pt5_reference(packed_weights, w_trits, depth);
        unpack_pt5_reference(packed_inputs, i_trits, depth);

        int32_t result = ternary_dot_product(w_trits, i_trits, depth);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        double gops = (depth * 2.0) / duration / 1e9;

        printf("Result: %d\n", result);
        printf("⏱️  Execution Time: %.6f seconds\n", duration);
        printf("📊 Throughput: %.4f GOPS\n", gops);

        free(packed_weights); free(packed_inputs); free(w_trits); free(i_trits);
    } else {
        // --- GEMM MODE ---
        // For simplicity, we'll do [rows=15] x [cols=depth]
        uint32_t rows = 15;
        uint32_t cols = depth;
        int8_t* w_mat = malloc(rows * cols);
        int8_t* i_vec = malloc(cols);
        int32_t* o_vec = malloc(rows * sizeof(int32_t));

        for(uint32_t i=0; i<rows*cols; i++) w_mat[i] = (rand() % 3) - 1;
        for(uint32_t i=0; i<cols; i++) i_vec[i] = (rand() % 3) - 1;

        struct timespec start, end;
        clock_gettime(CLOCK_MONOTONIC, &start);

        ternary_gemm_reference(w_mat, i_vec, o_vec, rows, cols);

        clock_gettime(CLOCK_MONOTONIC, &end);
        double duration = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
        double gops = (rows * cols * 2.0) / duration / 1e9;

        printf("GEMM 15x%u completed.\n", cols);
        printf("⏱️  Execution Time: %.6f seconds\n", duration);
        printf("📊 Throughput: %.4f GOPS\n", gops);

        free(w_mat); free(i_vec); free(o_vec);
    }

    return 0;
}
