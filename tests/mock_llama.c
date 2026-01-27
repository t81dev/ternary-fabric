#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

int main() {
    printf("--- Mock llama.cpp Starting ---\n");

    // 1. Allocate weights (2MB > 1MB threshold)
    size_t weight_size = 2 * 1024 * 1024;
    printf("Allocating weights (%zu bytes)...\n", weight_size);
    char* weights = (char*)malloc(weight_size);
    if (!weights) {
        printf("Failed to allocate weights\n");
        return 1;
    }

    // 2. Initialize weights
    printf("Initializing weights...\n");
    memset(weights, 1, weight_size);

    // 3. Allocate activation (1KB < 1MB threshold)
    size_t act_size = 1024;
    printf("Allocating activations (%zu bytes)...\n", act_size);
    char* act = (char*)malloc(act_size);
    if (!act) {
        printf("Failed to allocate activations\n");
        return 1;
    }
    memset(act, 2, act_size);

    // 4. Perform "compute"
    printf("Performing compute loop...\n");
    long sum = 0;
    for (size_t i = 0; i < 1000; i++) {
        sum += weights[i] * act[i];
    }

    printf("Result: %ld (Expected: 2000)\n", sum);

    // 5. Verify data integrity in Fabric
    // If weights were redirected, they should still be readable.
    if (weights[500] != 1) {
        printf("DATA CORRUPTION in weights!\n");
    } else {
        printf("Data integrity verified.\n");
    }

    free(weights);
    free(act);

    // 6. Test mmap redirection
    printf("Testing mmap redirection...\n");
    size_t mmap_size = 4 * 1024 * 1024;
    void* mmap_ptr = mmap(NULL, mmap_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (mmap_ptr == MAP_FAILED) {
        printf("mmap failed\n");
    } else {
        printf("mmap succeeded at %p\n", mmap_ptr);
        munmap(mmap_ptr, mmap_size);
    }

    printf("--- Mock llama.cpp Finished ---\n");
    return 0;
}
