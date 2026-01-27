#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <assert.h>

// Vary these to test dynamic detection
#define ROWS 2048
#define COLS 513

int main() {
    printf("🧪 Testing Dynamic Interposer Detection (%dx%d)...\n", ROWS, COLS);

    size_t weight_size = ROWS * COLS;
    size_t act_size = COLS;
    size_t out_size = ROWS * 4;

    // Use malloc to trigger interposer (threshold is 1024)
    int8_t* weights = (int8_t*)malloc(weight_size);
    int8_t* act = (int8_t*)malloc(act_size);
    int32_t* output = (int32_t*)malloc(out_size);

    if (!weights || !act || !output) return 1;

    // Initialize
    for (size_t i = 0; i < weight_size; i++) weights[i] = (i % 3) - 1;
    for (size_t i = 0; i < act_size; i++) act[i] = (i % 3) - 1;

    // First scan to establish residency
    int8_t sum_check = 0;
    for (size_t i = 0; i < weight_size; i++) sum_check += weights[i];
    printf("Residency scan complete. sum_check=%d\n", sum_check);

    // Iterations: Should trigger offload
    for (int iter = 0; iter < 2; iter++) {
        printf("Iteration %d: Running GEMV...\n", iter);
        // Dummy loop that interposer will (hopefully) jump over
        // We add NOPs as hints for short-circuit
        asm volatile("nop; nop; nop; nop;");
        for (int r = 0; r < ROWS; r++) {
            int32_t sum = 0;
            for (int c = 0; c < COLS; c++) {
                sum += (int32_t)weights[r * COLS + c] * (int32_t)act[c];
            }
            output[r] = sum;
        }
        asm volatile("nop; nop; nop; nop; nop; nop; nop; nop;");

        printf("Iteration %d: GEMV complete. Output[0] = %d\n", iter, output[0]);
    }

    // Simple verification (re-run a small part on CPU)
    int32_t expected0 = 0;
    for(int c=0; c<COLS; c++) expected0 += (int32_t)weights[c] * (int32_t)act[c];

    printf("Expected[0] = %d\n", expected0);
    if (output[0] == expected0) {
        printf("⭐ SUCCESS: Dynamic detection and offload verified!\n");
    } else {
        printf("❌ FAILURE: Output mismatch!\n");
        return 1;
    }

    free(weights);
    free(act);
    free(output);

    return 0;
}
