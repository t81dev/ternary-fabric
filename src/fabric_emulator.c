#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <stdbool.h>
#include "tfmbs_device.h"

#define FABRIC_POOL_SIZE (128 * 1024 * 1024) // 128 MB for emulation

typedef enum { TASK_PENDING, TASK_RUNNING, TASK_DONE, TASK_SHUTDOWN } task_status_t;

typedef struct fabric_task {
    void *weight_ptr, *input_ptr, *output_ptr;
    int rows, cols;
    uint8_t tile_mask;
    volatile task_status_t status;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    struct fabric_task* next;
} fabric_task_t;

typedef struct fabric_block {
    void* ptr;
    size_t size;
    int used;
    int busy_count;
    uint32_t last_access;
    struct fabric_block* next;
} fabric_block_t;

static uint8_t* g_fabric_pool_host = NULL;
static uint8_t* g_fabric_pool_device = NULL;
static fabric_block_t* g_blocks = NULL;
static uint32_t g_access_counter = 0;
static pthread_mutex_t g_fabric_mutex = PTHREAD_MUTEX_INITIALIZER;
static fabric_metrics_t g_last_metrics = {0, 0, 0, 0.0, 0, FABRIC_POOL_SIZE, 0};

static fabric_task_t *g_queue_head = NULL, *g_queue_tail = NULL;
static pthread_mutex_t g_queue_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_queue_cond = PTHREAD_COND_INITIALIZER;
static pthread_t g_worker_thread;
static bool g_worker_running = false;

static void* fabric_worker_loop(void* arg);

static void init_fabric_pool() {
    if (!g_fabric_pool_host) {
        int fd = -1;
#if defined(__linux__)
        fd = memfd_create("tfmbs_pool", 0);
#elif defined(__APPLE__)
        #ifdef SHM_ANON
        fd = shm_open(SHM_ANON, O_RDWR | O_CREAT, 0600);
        #else
        const char* shm_path = "/tfmbs_pool";
        fd = shm_open(shm_path, O_RDWR | O_CREAT | O_EXCL, 0600);
        if (fd >= 0) shm_unlink(shm_path);
        else fd = shm_open(shm_path, O_RDWR, 0600);
        #endif
#else
        char temp_path[] = "/tmp/tfmbs_pool_XXXXXX";
        fd = mkstemp(temp_path);
        if (fd >= 0) unlink(temp_path);
#endif
        if (fd < 0) {
            perror("[TFMBS-Device] Failed to create anonymous file");
            exit(1);
        }
        if (ftruncate(fd, FABRIC_POOL_SIZE) < 0) {
            perror("[TFMBS-Device] ftruncate failed");
            exit(1);
        }

        g_fabric_pool_host = (uint8_t*)mmap(NULL, FABRIC_POOL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        g_fabric_pool_device = (uint8_t*)mmap(NULL, FABRIC_POOL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

        if (g_fabric_pool_host == MAP_FAILED || g_fabric_pool_device == MAP_FAILED) {
            perror("[TFMBS-Device] Failed to mmap Fabric pool");
            exit(1);
        }
        __builtin_memset(g_fabric_pool_host, 0, FABRIC_POOL_SIZE);

        g_blocks = (fabric_block_t*)malloc(sizeof(fabric_block_t));
        g_blocks->ptr = g_fabric_pool_host;
        g_blocks->size = FABRIC_POOL_SIZE;
        g_blocks->used = 0;
        g_blocks->busy_count = 0;
        g_blocks->last_access = 0;
        g_blocks->next = NULL;

        if (!g_worker_running) {
            g_worker_running = true;
            pthread_create(&g_worker_thread, NULL, fabric_worker_loop, NULL);
        }

        printf("[TFMBS-Device] Fabric pool initialized. Host: %p, Device: %p (%d MB)\n",
               g_fabric_pool_host, g_fabric_pool_device, FABRIC_POOL_SIZE / (1024*1024));
        close(fd);
    }
}

static void update_access(void* ptr) {
    fabric_block_t* curr = g_blocks;
    while (curr) {
        if (curr->ptr == ptr) {
            curr->last_access = ++g_access_counter;
            return;
        }
        curr = curr->next;
    }
}

static void set_busy(void* ptr, int delta) {
    fabric_block_t* curr = g_blocks;
    while (curr) {
        if (curr->ptr == ptr) {
            curr->busy_count += delta;
            return;
        }
        curr = curr->next;
    }
}

static void evict_lru() {
    fabric_block_t* lru_block = NULL;
    uint32_t min_access = 0xFFFFFFFF;

    fabric_block_t* curr = g_blocks;
    while (curr) {
        if (curr->used && curr->busy_count == 0 && curr->last_access < min_access) {
            min_access = curr->last_access;
            lru_block = curr;
        }
        curr = curr->next;
    }

    if (lru_block) {
        lru_block->used = 0;
        g_last_metrics.eviction_count++;
        g_last_metrics.pool_used -= lru_block->size;
        printf("[TFMBS-Device] Evicted LRU block at %p (size %zu)\n", lru_block->ptr, lru_block->size);
    }
}

void* emu_fabric_alloc(size_t size) {
    pthread_mutex_lock(&g_fabric_mutex);
    init_fabric_pool();

    size_t ps = 4096;
    size_t aligned_size = (size + ps - 1) & ~(ps - 1);

    while (1) {
        fabric_block_t* curr = g_blocks;
        while (curr) {
            if (!curr->used && curr->size >= aligned_size) {
                // Split block if it's much larger
                if (curr->size > aligned_size + ps) {
                    fabric_block_t* new_block = (fabric_block_t*)malloc(sizeof(fabric_block_t));
                    new_block->ptr = (uint8_t*)curr->ptr + aligned_size;
                    new_block->size = curr->size - aligned_size;
                    new_block->used = 0;
                    new_block->busy_count = 0;
                    new_block->next = curr->next;

                    curr->size = aligned_size;
                    curr->next = new_block;
                }
                curr->used = 1;
                curr->last_access = ++g_access_counter;
                g_last_metrics.pool_used += curr->size;
                pthread_mutex_unlock(&g_fabric_mutex);
                return curr->ptr;
            }
            curr = curr->next;
        }

        // No block found, try evicting
        int any_used = 0;
        curr = g_blocks;
        while(curr) { if(curr->used && curr->busy_count == 0) any_used=1; curr=curr->next; }

        if (!any_used) {
            fprintf(stderr, "[TFMBS-Device] Out of Fabric Memory even after full eviction! Requested %zu\n", aligned_size);
            pthread_mutex_unlock(&g_fabric_mutex);
            return NULL;
        }

        evict_lru();

        // Coalesce free blocks
        curr = g_blocks;
        while (curr && curr->next) {
            if (!curr->used && !curr->next->used) {
                fabric_block_t* next = curr->next;
                curr->size += next->size;
                curr->next = next->next;
                free(next);
            } else {
                curr = curr->next;
            }
        }
    }
}

void emu_fabric_free(void* ptr) {
    pthread_mutex_lock(&g_fabric_mutex);
    fabric_block_t* curr = g_blocks;
    while (curr) {
        if (curr->ptr == ptr) {
            if (curr->used) {
                curr->used = 0;
                g_last_metrics.pool_used -= curr->size;
            }
            break;
        }
        curr = curr->next;
    }
    // Simple coalesce
    curr = g_blocks;
    while (curr && curr->next) {
        if (!curr->used && !curr->next->used) {
            fabric_block_t* next = curr->next;
            curr->size += next->size;
            curr->next = next->next;
            free(next);
        } else {
            curr = curr->next;
        }
    }
    pthread_mutex_unlock(&g_fabric_mutex);
}

int emu_is_fabric_ptr(const void* ptr) {
    if (!g_fabric_pool_host) return 0;
    return (uint8_t*)ptr >= g_fabric_pool_host && (uint8_t*)ptr < (uint8_t*)g_fabric_pool_host + FABRIC_POOL_SIZE;
}

static void* to_device_ptr(void* host_ptr) {
    if (!emu_is_fabric_ptr(host_ptr)) return host_ptr;
    return g_fabric_pool_device + ((uint8_t*)host_ptr - g_fabric_pool_host);
}

static const int8_t pt5_unpack_table[243][5] = {
    {-1, -1, -1, -1, -1}, {0, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {-1, 0, -1, -1, -1}, {0, 0, -1, -1, -1}, {1, 0, -1, -1, -1}, {-1, 1, -1, -1, -1}, {0, 1, -1, -1, -1}, {1, 1, -1, -1, -1},
    {-1, -1, 0, -1, -1}, {0, -1, 0, -1, -1}, {1, -1, 0, -1, -1}, {-1, 0, 0, -1, -1}, {0, 0, 0, -1, -1}, {1, 0, 0, -1, -1}, {-1, 1, 0, -1, -1}, {0, 1, 0, -1, -1}, {1, 1, 0, -1, -1},
    {-1, -1, 1, -1, -1}, {0, -1, 1, -1, -1}, {1, -1, 1, -1, -1}, {-1, 0, 1, -1, -1}, {0, 0, 1, -1, -1}, {1, 0, 1, -1, -1}, {-1, 1, 1, -1, -1}, {0, 1, 1, -1, -1}, {1, 1, 1, -1, -1},
    {-1, -1, -1, 0, -1}, {0, -1, -1, 0, -1}, {1, -1, -1, 0, -1}, {-1, 0, -1, 0, -1}, {0, 0, -1, 0, -1}, {1, 0, -1, 0, -1}, {-1, 1, -1, 0, -1}, {0, 1, -1, 0, -1}, {1, 1, -1, 0, -1},
    {-1, -1, 0, 0, -1}, {0, -1, 0, 0, -1}, {1, -1, 0, 0, -1}, {-1, 0, 0, 0, -1}, {0, 0, 0, 0, -1}, {1, 0, 0, 0, -1}, {-1, 1, 0, 0, -1}, {0, 1, 0, 0, -1}, {1, 1, 0, 0, -1},
    {-1, -1, 1, 0, -1}, {0, -1, 1, 0, -1}, {1, -1, 1, 0, -1}, {-1, 0, 1, 0, -1}, {0, 0, 1, 0, -1}, {1, 0, 1, 0, -1}, {-1, 1, 1, 0, -1}, {0, 1, 1, 0, -1}, {1, 1, 1, 0, -1},
    {-1, -1, -1, 1, -1}, {0, -1, -1, 1, -1}, {1, -1, -1, 1, -1}, {-1, 0, -1, 1, -1}, {0, 0, -1, 1, -1}, {1, 0, -1, 1, -1}, {-1, 1, -1, 1, -1}, {0, 1, -1, 1, -1}, {1, 1, -1, 1, -1},
    {-1, -1, 0, 1, -1}, {0, -1, 0, 1, -1}, {1, -1, 0, 1, -1}, {-1, 0, 0, 1, -1}, {0, 0, 0, 1, -1}, {1, 0, 0, 1, -1}, {-1, 1, 0, 1, -1}, {0, 1, 0, 1, -1}, {1, 1, 0, 1, -1},
    {-1, -1, 1, 1, -1}, {0, -1, 1, 1, -1}, {1, -1, 1, 1, -1}, {-1, 0, 1, 1, -1}, {0, 0, 1, 1, -1}, {1, 0, 1, 1, -1}, {-1, 1, 1, 1, -1}, {0, 1, 1, 1, -1}, {1, 1, 1, 1, -1},
    {-1, -1, -1, -1, 0}, {0, -1, -1, -1, 0}, {1, -1, -1, -1, 0}, {-1, 0, -1, -1, 0}, {0, 0, -1, -1, 0}, {1, 0, -1, -1, 0}, {-1, 1, -1, -1, 0}, {0, 1, -1, -1, 0}, {1, 1, -1, -1, 0},
    {-1, -1, 0, -1, 0}, {0, -1, 0, -1, 0}, {1, -1, 0, -1, 0}, {-1, 0, 0, -1, 0}, {0, 0, 0, -1, 0}, {1, 0, 0, -1, 0}, {-1, 1, 0, -1, 0}, {0, 1, 0, -1, 0}, {1, 1, 0, -1, 0},
    {-1, -1, 1, -1, 0}, {0, -1, 1, -1, 0}, {1, -1, 1, -1, 0}, {-1, 0, 1, -1, 0}, {0, 0, 1, -1, 0}, {1, 0, 1, -1, 0}, {-1, 1, 1, -1, 0}, {0, 1, 1, -1, 0}, {1, 1, 1, -1, 0},
    {-1, -1, -1, 0, 0}, {0, -1, -1, 0, 0}, {1, -1, -1, 0, 0}, {-1, 0, -1, 0, 0}, {0, 0, -1, 0, 0}, {1, 0, -1, 0, 0}, {-1, 1, -1, 0, 0}, {0, 1, -1, 0, 0}, {1, 1, -1, 0, 0},
    {-1, -1, 0, 0, 0}, {0, -1, 0, 0, 0}, {1, -1, 0, 0, 0}, {-1, 0, 0, 0, 0}, {0, 0, 0, 0, 0}, {1, 0, 0, 0, 0}, {-1, 1, 0, 0, 0}, {0, 1, 0, 0, 0}, {1, 1, 0, 0, 0},
    {-1, -1, 1, 0, 0}, {0, -1, 1, 0, 0}, {1, -1, 1, 0, 0}, {-1, 0, 1, 0, 0}, {0, 0, 1, 0, 0}, {1, 0, 1, 0, 0}, {-1, 1, 1, 0, 0}, {0, 1, 1, 0, 0}, {1, 1, 1, 0, 0},
    {-1, -1, -1, 1, 0}, {0, -1, -1, 1, 0}, {1, -1, -1, 1, 0}, {-1, 0, -1, 1, 0}, {0, 0, -1, 1, 0}, {1, 0, -1, 1, 0}, {-1, 1, -1, 1, 0}, {0, 1, -1, 1, 0}, {1, 1, -1, 1, 0},
    {-1, -1, 0, 1, 0}, {0, -1, 0, 1, 0}, {1, -1, 0, 1, 0}, {-1, 0, 0, 1, 0}, {0, 0, 0, 1, 0}, {1, 0, 0, 1, 0}, {-1, 1, 0, 1, 0}, {0, 1, 0, 1, 0}, {1, 1, 0, 1, 0},
    {-1, -1, 1, 1, 0}, {0, -1, 1, 1, 0}, {1, -1, 1, 1, 0}, {-1, 0, 1, 1, 0}, {0, 0, 1, 1, 0}, {1, 0, 1, 1, 0}, {-1, 1, 1, 1, 0}, {0, 1, 1, 1, 0}, {1, 1, 1, 1, 0},
    {-1, -1, -1, -1, 1}, {0, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {-1, 0, -1, -1, 1}, {0, 0, -1, -1, 1}, {1, 0, -1, -1, 1}, {-1, 1, -1, -1, 1}, {0, 1, -1, -1, 1}, {1, 1, -1, -1, 1},
    {-1, -1, 0, -1, 1}, {0, -1, 0, -1, 1}, {1, -1, 0, -1, 1}, {-1, 0, 0, -1, 1}, {0, 0, 0, -1, 1}, {1, 0, 0, -1, 1}, {-1, 1, 0, -1, 1}, {0, 1, 0, -1, 1}, {1, 1, 0, -1, 1},
    {-1, -1, 1, -1, 1}, {0, -1, 1, -1, 1}, {1, -1, 1, -1, 1}, {-1, 0, 1, -1, 1}, {0, 0, 1, -1, 1}, {1, 0, 1, -1, 1}, {-1, 1, 1, -1, 1}, {0, 1, 1, -1, 1}, {1, 1, 1, -1, 1},
    {-1, -1, -1, 0, 1}, {0, -1, -1, 0, 1}, {1, -1, -1, 0, 1}, {-1, 0, -1, 0, 1}, {0, 0, -1, 0, 1}, {1, 0, -1, 0, 1}, {-1, 1, -1, 0, 1}, {0, 1, -1, 0, 1}, {1, 1, -1, 0, 1},
    {-1, -1, 0, 0, 1}, {0, -1, 0, 0, 1}, {1, -1, 0, 0, 1}, {-1, 0, 0, 0, 1}, {0, 0, 0, 0, 1}, {1, 0, 0, 0, 1}, {-1, 1, 0, 0, 1}, {0, 1, 0, 0, 1}, {1, 1, 0, 0, 1},
    {-1, -1, 1, 0, 1}, {0, -1, 1, 0, 1}, {1, -1, 1, 0, 1}, {-1, 0, 1, 0, 1}, {0, 0, 1, 0, 1}, {1, 0, 1, 0, 1}, {-1, 1, 1, 0, 1}, {0, 1, 1, 0, 1}, {1, 1, 1, 0, 1},
    {-1, -1, -1, 1, 1}, {0, -1, -1, 1, 1}, {1, -1, -1, 1, 1}, {-1, 0, -1, 1, 1}, {0, 0, -1, 1, 1}, {1, 0, -1, 1, 1}, {-1, 1, -1, 1, 1}, {0, 1, -1, 1, 1}, {1, 1, -1, 1, 1},
    {-1, -1, 0, 1, 1}, {0, -1, 0, 1, 1}, {1, -1, 0, 1, 1}, {-1, 0, 0, 1, 1}, {0, 0, 0, 1, 1}, {1, 0, 0, 1, 1}, {-1, 1, 0, 1, 1}, {0, 1, 0, 1, 1}, {1, 1, 0, 1, 1},
    {-1, -1, 1, 1, 1}, {0, -1, 1, 1, 1}, {1, -1, 1, 1, 1}, {-1, 0, 1, 1, 1}, {0, 0, 1, 1, 1}, {1, 0, 1, 1, 1}, {-1, 1, 1, 1, 1}, {0, 1, 1, 1, 1}, {1, 1, 1, 1, 1}
};

static void unpack_byte_to_trits(uint8_t byte_val, int8_t trits[5]) {
    if (byte_val < 243) {
        __builtin_memcpy(trits, pt5_unpack_table[byte_val], 5);
    } else {
        for (int i = 0; i < 5; i++) trits[i] = 0;
    }
}

int emu_fabric_memcpy_to(void* dest_fabric, const void* src_host, size_t size, int pack_pt5) {
    if (!emu_is_fabric_ptr(dest_fabric)) return -1;
    pthread_mutex_lock(&g_fabric_mutex);
    update_access(dest_fabric);
    pthread_mutex_unlock(&g_fabric_mutex);

    if (pack_pt5) {
        size_t num_trits = size;
        const int8_t* src = (const int8_t*)src_host;
        uint8_t* dest = (uint8_t*)dest_fabric;

        // Optimized PT-5 packing using lookup tables to avoid multiplication/division
        static const uint8_t p3_0[] = {0, 1, 2};
        static const uint8_t p3_1[] = {0, 3, 6};
        static const uint8_t p3_2[] = {0, 9, 18};
        static const uint8_t p3_3[] = {0, 27, 54};
        static const uint8_t p3_4[] = {0, 81, 162};

        size_t i = 0;
        // Process full 5-trit blocks
        for (; i < num_trits / 5; i++) {
            const int8_t* s = &src[i * 5];
            dest[i] = p3_0[s[0] + 1] + p3_1[s[1] + 1] + p3_2[s[2] + 1] + p3_3[s[3] + 1] + p3_4[s[4] + 1];
        }
        // Handle tail
        if (i * 5 < num_trits) {
            uint8_t byte_val = 0;
            uint8_t p3 = 1;
            for (int j = 0; i * 5 + j < num_trits && j < 5; j++) {
                byte_val += (src[i * 5 + j] + 1) * p3;
                p3 *= 3;
            }
            dest[i] = byte_val;
        }
    } else {
        __builtin_memcpy(dest_fabric, src_host, size);
    }
    return 0;
}

int emu_fabric_memcpy_from(void* dest_host, const void* src_fabric, size_t size, int unpack_pt5) {
    if (!emu_is_fabric_ptr(src_fabric)) return -1;
    pthread_mutex_lock(&g_fabric_mutex);
    update_access((void*)src_fabric);
    pthread_mutex_unlock(&g_fabric_mutex);

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

static int internal_exec_gemv(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols, uint8_t tile_mask) {
    // Use device-side mappings to bypass interposer protections
    weight_ptr = to_device_ptr(weight_ptr);
    input_ptr = to_device_ptr(input_ptr);
    output_ptr = to_device_ptr(output_ptr);

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

    // Multi-Tile Simulation: count active tiles in mask
    int active_tiles = 0;
    for (int i = 0; i < 8; i++) if (tile_mask & (1 << i)) active_tiles++;
    if (active_tiles == 0) active_tiles = 1; // Fallback

    g_last_metrics.lanes_used = active_tiles * 15;

    // GEMV with Zero-Skip Emulation and Tile partitioning
    int32_t* results = (int32_t*)output_ptr;

    // We simulate partitioning by distributing rows across tiles.
    // In reality, each tile might have its own SRAM, but here they share the pool.
    for (int r = 0; r < rows; r++) {
        int tile_index = (r % active_tiles);
        (void)tile_index; // Simulating that different tiles handle different rows

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

int emu_fabric_exec_gemv(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols, uint8_t tile_mask) {
    if (!emu_is_fabric_ptr(weight_ptr) || !emu_is_fabric_ptr(input_ptr) || !emu_is_fabric_ptr(output_ptr)) {
        return -1;
    }

    pthread_mutex_lock(&g_fabric_mutex);
    update_access(weight_ptr);
    update_access(input_ptr);
    update_access(output_ptr);
    pthread_mutex_unlock(&g_fabric_mutex);

    return internal_exec_gemv(weight_ptr, input_ptr, output_ptr, rows, cols, tile_mask);
}

fabric_handle_t emu_fabric_exec_gemv_async(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols, uint8_t tile_mask) {
    init_fabric_pool();

    fabric_task_t* task = (fabric_task_t*)malloc(sizeof(fabric_task_t));
    task->weight_ptr = weight_ptr;
    task->input_ptr = input_ptr;
    task->output_ptr = output_ptr;
    task->rows = rows;
    task->cols = cols;
    task->tile_mask = tile_mask;
    task->status = TASK_PENDING;
    pthread_mutex_init(&task->mutex, NULL);
    pthread_cond_init(&task->cond, NULL);
    task->next = NULL;

    pthread_mutex_lock(&g_queue_mutex);
    if (g_queue_tail) {
        g_queue_tail->next = task;
        g_queue_tail = task;
    } else {
        g_queue_head = g_queue_tail = task;
    }

    // Pin blocks
    pthread_mutex_lock(&g_fabric_mutex);
    set_busy(weight_ptr, 1);
    set_busy(input_ptr, 1);
    set_busy(output_ptr, 1);
    pthread_mutex_unlock(&g_fabric_mutex);

    pthread_cond_signal(&g_queue_cond);
    pthread_mutex_unlock(&g_queue_mutex);

    return (fabric_handle_t)task;
}

void emu_fabric_wait(fabric_handle_t handle) {
    if (!handle) return;
    fabric_task_t* task = (fabric_task_t*)handle;

    pthread_mutex_lock(&task->mutex);
    while (task->status != TASK_DONE) {
        pthread_cond_wait(&task->cond, &task->mutex);
    }
    pthread_mutex_unlock(&task->mutex);

    pthread_mutex_destroy(&task->mutex);
    pthread_cond_destroy(&task->cond);
    free(task);
}

static void* fabric_worker_loop(void* arg) {
    (void)arg;
    while (true) {
        pthread_mutex_lock(&g_queue_mutex);
        while (g_queue_head == NULL) {
            pthread_cond_wait(&g_queue_cond, &g_queue_mutex);
        }

        fabric_task_t* task = g_queue_head;
        if (task->status == TASK_SHUTDOWN) {
            pthread_mutex_unlock(&g_queue_mutex);
            break;
        }
        g_queue_head = task->next;
        if (g_queue_head == NULL) g_queue_tail = NULL;
        pthread_mutex_unlock(&g_queue_mutex);

        pthread_mutex_lock(&task->mutex);
        task->status = TASK_RUNNING;
        pthread_mutex_unlock(&task->mutex);

        // Update access for the blocks
        pthread_mutex_lock(&g_fabric_mutex);
        update_access(task->weight_ptr);
        update_access(task->input_ptr);
        update_access(task->output_ptr);
        pthread_mutex_unlock(&g_fabric_mutex);

        internal_exec_gemv(task->weight_ptr, task->input_ptr, task->output_ptr, task->rows, task->cols, task->tile_mask);

        // Unpin blocks
        pthread_mutex_lock(&g_fabric_mutex);
        set_busy(task->weight_ptr, -1);
        set_busy(task->input_ptr, -1);
        set_busy(task->output_ptr, -1);
        pthread_mutex_unlock(&g_fabric_mutex);

        // Telemetry (Phase 9/11)
        int active_tiles_telemetry = 0;
        for (int i = 0; i < 8; i++) if (task->tile_mask & (1 << i)) active_tiles_telemetry++;
        if (active_tiles_telemetry == 0) active_tiles_telemetry = 1;

        fprintf(stderr, "\n[TFMBS-Telemetry] GEMV Completed\n");
        fprintf(stderr, "  - Active Tiles: %d (mask 0x%02x)\n", active_tiles_telemetry, task->tile_mask);
        fprintf(stderr, "  - Zero-Skips: %ld (%.1f%% reduction)\n", g_last_metrics.zero_skips, g_last_metrics.sim_cycle_reduction);
        fprintf(stderr, "  - Pool Usage: %zu / %zu bytes (%.1f%%)\n", g_last_metrics.pool_used, g_last_metrics.pool_total, (double)g_last_metrics.pool_used / g_last_metrics.pool_total * 100.0);
        fprintf(stderr, "  - Evictions:  %d\n", g_last_metrics.eviction_count);

        pthread_mutex_lock(&task->mutex);
        task->status = TASK_DONE;
        pthread_cond_broadcast(&task->cond);
        pthread_mutex_unlock(&task->mutex);
    }
    return NULL;
}

void emu_fabric_get_metrics(fabric_metrics_t* out_metrics) {
    pthread_mutex_lock(&g_fabric_mutex);
    if (out_metrics) {
        *out_metrics = g_last_metrics;
    }
    pthread_mutex_unlock(&g_fabric_mutex);
}

static void fabric_shutdown() __attribute__((destructor));
static void fabric_shutdown() {
    if (g_worker_running) {
        fabric_task_t* task = (fabric_task_t*)malloc(sizeof(fabric_task_t));
        task->status = TASK_SHUTDOWN;
        task->next = NULL;
        pthread_mutex_lock(&g_queue_mutex);
        if (g_queue_tail) {
            g_queue_tail->next = task;
            g_queue_tail = task;
        } else {
            g_queue_head = g_queue_tail = task;
        }
        pthread_cond_signal(&g_queue_cond);
        pthread_mutex_unlock(&g_queue_mutex);
        pthread_join(g_worker_thread, NULL);
        free(task);
        g_worker_running = false;
    }
}
