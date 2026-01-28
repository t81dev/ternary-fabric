#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define ROWS 512
#define COLS 512
#define ITERATIONS 1

int main() {
    size_t weight_size = 1048576; // 1MB
    size_t act_size = 100000;
    size_t out_size = 50000;

    int8_t* weights = (int8_t*)mmap(NULL, weight_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int8_t* act = (int8_t*)mmap(NULL, act_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    int32_t* output = (int32_t*)mmap(NULL, out_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

    if (weights == MAP_FAILED || act == MAP_FAILED || output == MAP_FAILED) return 1;

    for (size_t i = 0; i < weight_size; i++) weights[i] = (i % 3) - 1;
    for (size_t i = 0; i < act_size; i++) act[i] = (i % 3) - 1;

    for (int iter = 0; iter < ITERATIONS; iter++) {
        printf("Iteration %d...\n", iter);
        asm volatile("nop; nop; nop; nop;");
        for (int r = 0; r < ROWS; r++) {
            int32_t sum = 0;
            for (int c = 0; c < COLS; c++) {
                sum += (int32_t)weights[r * COLS + c] * (int32_t)act[c];
            }
            output[r] = sum;
        }
        asm volatile("nop; nop; nop; nop; nop; nop; nop; nop;");

        printf("Iteration %d: Row 0: %d\n", iter, output[0]);
    }
    return 0;
}
