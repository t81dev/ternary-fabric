#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include "tfmbs_device.h"

#define FABRIC_POOL_SIZE (128 * 1024 * 1024) // 128 MB for emulation

static uint8_t* g_fabric_pool = NULL;
static size_t g_fabric_used = 0;

static void init_fabric_pool() {
    if (!g_fabric_pool) {
        g_fabric_pool = (uint8_t*)mmap(NULL, FABRIC_POOL_SIZE, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (g_fabric_pool == MAP_FAILED) {
            perror("[TFMBS-Device] Failed to mmap Fabric pool");
            exit(1);
        }
        __builtin_memset(g_fabric_pool, 0, FABRIC_POOL_SIZE);
        g_fabric_used = 0;
        printf("[TFMBS-Device] Fabric pool initialized at %p (%d MB)\n", g_fabric_pool, FABRIC_POOL_SIZE / (1024*1024));
    }
}

void* fabric_alloc(size_t size) {
    init_fabric_pool();

    // Alignment to 64 bytes
    size_t aligned_size = (size + 63) & ~63;

    if (g_fabric_used + aligned_size > FABRIC_POOL_SIZE) {
        fprintf(stderr, "[TFMBS-Device] Out of Fabric Memory! Requested %zu, used %zu/%d\n",
                aligned_size, g_fabric_used, FABRIC_POOL_SIZE);
        return NULL;
    }

    void* ptr = g_fabric_pool + g_fabric_used;
    g_fabric_used += aligned_size;
    return ptr;
}

void fabric_free(void* ptr) {
    // Basic mock doesn't support individual frees yet
    (void)ptr;
}

int is_fabric_ptr(const void* ptr) {
    if (!g_fabric_pool) return 0;
    return (uint8_t*)ptr >= g_fabric_pool && (uint8_t*)ptr < (g_fabric_pool + FABRIC_POOL_SIZE);
}

static uint8_t pack_trits_to_byte(const int8_t trits[5]) {
    uint8_t byte_val = 0;
    uint8_t p3 = 1;
    for (int i = 0; i < 5; i++) {
        uint8_t unsigned_trit = trits[i] + 1; // map -1,0,1 to 0,1,2
        byte_val += (unsigned_trit * p3);
        p3 *= 3;
    }
    return byte_val;
}

static void unpack_byte_to_trits(uint8_t byte_val, int8_t trits[5]) {
    for (int i = 0; i < 5; i++) {
        uint8_t unsigned_trit = byte_val % 3;
        trits[i] = (int8_t)unsigned_trit - 1;
        byte_val /= 3;
    }
}

int fabric_memcpy_to(void* dest_fabric, const void* src_host, size_t size, int pack_pt5) {
    if (!is_fabric_ptr(dest_fabric)) return -1;

    if (pack_pt5) {
        size_t num_trits = size;
        size_t num_bytes = (num_trits + 4) / 5;
        const int8_t* src = (const int8_t*)src_host;
        uint8_t* dest = (uint8_t*)dest_fabric;

        for (size_t i = 0; i < num_bytes; i++) {
            int8_t trits[5] = {0, 0, 0, 0, 0};
            for (int j = 0; j < 5; j++) {
                if (i * 5 + j < num_trits) trits[j] = src[i * 5 + j];
            }
            dest[i] = pack_trits_to_byte(trits);
        }
    } else {
        __builtin_memcpy(dest_fabric, src_host, size);
    }
    return 0;
}

int fabric_memcpy_from(void* dest_host, const void* src_fabric, size_t size, int unpack_pt5) {
    if (!is_fabric_ptr(src_fabric)) return -1;

    if (unpack_pt5) {
        size_t num_trits = size;
        size_t num_bytes = (num_trits + 4) / 5;
        const uint8_t* src = (const uint8_t*)src_fabric;
        int8_t* dest = (int8_t*)dest_host;

        for (size_t i = 0; i < num_bytes; i++) {
            int8_t trits[5];
            unpack_byte_to_trits(src[i], trits);
            for (int j = 0; j < 5; j++) {
                if (i * 5 + j < num_trits) dest[i * 5 + j] = trits[j];
            }
        }
    } else {
        __builtin_memcpy(dest_host, src_fabric, size);
    }
    return 0;
}

int fabric_exec_gemv(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols) {
    if (!is_fabric_ptr(weight_ptr) || !is_fabric_ptr(input_ptr) || !is_fabric_ptr(output_ptr)) {
        fprintf(stderr, "[TFMBS-Device] GEMV Error: Non-fabric pointer provided\n");
        return -1;
    }

    int8_t* w_trits = (int8_t*)malloc(rows * cols);
    int8_t* i_trits = (int8_t*)malloc(cols);
    int32_t* results = (int32_t*)output_ptr;

    if (!w_trits || !i_trits) {
        free(w_trits); free(i_trits);
        return -1;
    }

    // Unpack weights
    uint8_t* w_packed = (uint8_t*)weight_ptr;
    for (int i = 0; i < (rows * cols + 4) / 5; i++) {
        int8_t trits[5];
        unpack_byte_to_trits(w_packed[i], trits);
        for (int j = 0; j < 5; j++) {
            if (i * 5 + j < rows * cols) w_trits[i * 5 + j] = trits[j];
        }
    }

    // Unpack inputs
    uint8_t* i_packed = (uint8_t*)input_ptr;
    for (int i = 0; i < (cols + 4) / 5; i++) {
        int8_t trits[5];
        unpack_byte_to_trits(i_packed[i], trits);
        for (int j = 0; j < 5; j++) {
            if (i * 5 + j < cols) i_trits[i * 5 + j] = trits[j];
        }
    }

    // GEMV
    for (int r = 0; r < rows; r++) {
        int32_t acc = 0;
        for (int c = 0; c < cols; c++) {
            acc += (int32_t)w_trits[r * cols + c] * (int32_t)i_trits[c];
        }
        results[r] = acc;
    }

    free(w_trits);
    free(i_trits);
    return 0;
}
