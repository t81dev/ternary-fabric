#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include "tfmbs_device.h"

#define FABRIC_POOL_SIZE (128 * 1024 * 1024) // 128 MB for emulation

static uint8_t* g_fabric_pool = NULL;
static size_t g_fabric_used = 0;
static fabric_metrics_t g_last_metrics = {0, 0, 0, 0.0};

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

    // Alignment to page size for mprotect compatibility in the interposer
    size_t ps = 4096;
    size_t aligned_size = (size + ps - 1) & ~(ps - 1);

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
            uint8_t byte_val = 0;
            uint8_t p3 = 1;
            for (int j = 0; j < 5; j++) {
                int8_t trit = (i * 5 + j < num_trits) ? src[i * 5 + j] : 0;
                byte_val += (trit + 1) * p3;
                p3 *= 3;
            }
            dest[i] = byte_val;
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

static int8_t g_w_trits[2000000];
static int8_t g_i_trits[2000000];

int fabric_exec_gemv(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols) {
    if (!is_fabric_ptr(weight_ptr) || !is_fabric_ptr(input_ptr) || !is_fabric_ptr(output_ptr)) {
        return -1;
    }

    // Unpack weights
    uint8_t* w_packed = (uint8_t*)weight_ptr;
    for (int i = 0; i < (rows * cols + 4) / 5; i++) {
        int8_t trits[5];
        unpack_byte_to_trits(w_packed[i], trits);
        for (int j = 0; j < 5; j++) {
            if (i * 5 + j < rows * cols) g_w_trits[i * 5 + j] = trits[j];
        }
    }

    // Unpack inputs
    uint8_t* i_packed = (uint8_t*)input_ptr;
    for (int i = 0; i < (cols + 4) / 5; i++) {
        int8_t trits[5];
        unpack_byte_to_trits(i_packed[i], trits);
        for (int j = 0; j < 5; j++) {
            if (i * 5 + j < cols) g_i_trits[i * 5 + j] = trits[j];
        }
    }

    // Reset Metrics
    g_last_metrics.zero_skips = 0;
    g_last_metrics.total_ops = (long)rows * cols;
    g_last_metrics.lanes_used = 15;

    // GEMV with Zero-Skip Emulation
    int32_t* results = (int32_t*)output_ptr;
    for (int r = 0; r < rows; r++) {
        int32_t acc = 0;
        for (int c = 0; c < cols; c++) {
            int8_t w = g_w_trits[r * cols + c];
            int8_t x = g_i_trits[c];
            if (w == 0 || x == 0) {
                g_last_metrics.zero_skips++;
            } else {
                acc += (int32_t)w * (int32_t)x;
            }
        }
        results[r] = acc;
    }

    g_last_metrics.sim_cycle_reduction = (double)g_last_metrics.zero_skips / g_last_metrics.total_ops * 100.0;
    return 0;
}

void fabric_get_metrics(fabric_metrics_t* out_metrics) {
    if (out_metrics) {
        *out_metrics = g_last_metrics;
    }
}
