#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <pthread.h>
#include <stdbool.h>
#include "tfmbs_device.h"

#define FABRIC_POOL_SIZE (128 * 1024 * 1024) // 128 MB for emulation
#define MAX_TILES 8

typedef enum { TASK_PENDING, TASK_RUNNING, TASK_DONE, TASK_SHUTDOWN } task_status_t;
typedef enum { KERNEL_GEMV, KERNEL_LSTM, KERNEL_LSTM_PERSISTENT } kernel_type_t;

typedef struct fabric_task {
    void *weight_ptr, *input_ptr, *output_ptr;
    int rows, cols;
    kernel_type_t kernel;
    uint32_t exec_hints;
    uint8_t tile_mask;
    double projected_cost;
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
    uint32_t access_count;
    uint32_t residency_hits;
    int tile_id;    /* Phase 18: -1 for global, 0-N for specific tile */
    int pinned;     /* Phase 18: prevent eviction */
    int resident;   /* Phase 18: whether it's currently on fabric */
    struct fabric_block* next;
} fabric_block_t;

static uint8_t* g_fabric_pool_host = NULL;
static uint8_t* g_fabric_pool_device = NULL;
static fabric_block_t* g_blocks = NULL;
static uint32_t g_access_counter = 0;
static pthread_mutex_t g_fabric_mutex = PTHREAD_MUTEX_INITIALIZER;
static fabric_metrics_t g_last_metrics = {.pool_total = FABRIC_POOL_SIZE};
static fabric_metrics_t g_tile_metrics[MAX_TILES] = {0};

// Phase 19 Economic Introspection
static double g_last_projected_cost = 0;
static double g_last_residency_rebate = 0;
static int g_last_chosen_tile_id = -1;
static uint8_t g_last_tile_mask = 0;
static char g_last_kernel_name[32] = "NONE";
static char g_last_eviction_scores[512] = "";

// Phase 20: Learning & Self-Tuning
typedef struct {
    double weight_cost;
    double mem_read_cost;
    double mem_write_cost;
    double broadcast_cost;
    double residency_miss_cost;
} tfmbs_projection_params_t;

typedef struct {
    double tile_kernel_mult[MAX_TILES][4]; // GEMV, LSTM, LSTM-P, (unused)
    int    kernel_exec_count[MAX_TILES][4];
    double eviction_freq_weight;
    double eviction_age_weight;
    double eviction_success_weight;
    int    dynamic_batch_size;
    double avg_efficiency_ema;
    double avg_throughput_ema;
} tfmbs_learning_state_t;

static tfmbs_projection_params_t g_proj_params = {
    .weight_cost = 1.0,
    .mem_read_cost = 5.0,
    .mem_write_cost = 8.0,
    .broadcast_cost = 2.0,
    .residency_miss_cost = 6.0
};

static tfmbs_learning_state_t g_learn_state = {
    .tile_kernel_mult = {
        {1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0}, {1.0, 1.0, 1.0, 1.0}
    },
    .eviction_freq_weight = 1.0,
    .eviction_age_weight = 1.0,
    .eviction_success_weight = 2.0,
    .dynamic_batch_size = 8,
    .avg_efficiency_ema = 0.0,
    .avg_throughput_ema = 0.0
};

static inline double tfmbs_compute_cost(fabric_metrics_t *m, uint32_t hints) {
    double cost =
        m->active_ops       * 1.0 +
        m->mem_reads        * 5.0 +
        m->mem_writes       * 8.0 +
        m->broadcasts       * 2.0 +
        m->residency_misses * 6.0;

    if (hints & TFMBS_HINT_FUSED) cost *= 0.7;
    return cost;
}

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
        srand(time(NULL));

        g_blocks = (fabric_block_t*)malloc(sizeof(fabric_block_t));
        g_blocks->ptr = g_fabric_pool_host;
        g_blocks->size = FABRIC_POOL_SIZE;
        g_blocks->used = 0;
        g_blocks->busy_count = 0;
        g_blocks->last_access = 0;
        g_blocks->access_count = 0;
        g_blocks->residency_hits = 0;
        g_blocks->tile_id = -1;
        g_blocks->pinned = 0;
        g_blocks->resident = 0;
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

static fabric_block_t* find_block(void* ptr) {
    fabric_block_t* curr = g_blocks;
    while (curr) {
        if (curr->ptr == ptr) return curr;
        curr = curr->next;
    }
    return NULL;
}

static void update_access(void* ptr) {
    fabric_block_t* curr = g_blocks;
    while (curr) {
        if (curr->ptr == ptr) {
            curr->last_access = ++g_access_counter;
            curr->access_count++;
            return;
        }
        curr = curr->next;
    }
}

static double projected_cost(int tile_id, kernel_type_t kernel, void* w_ptr, void* i_ptr, void* o_ptr, double *out_rebate) {
    fabric_block_t *bw = find_block(w_ptr);
    fabric_block_t *bi = find_block(i_ptr);
    (void)o_ptr;

    double cost = 0.0;
    double rebate = 0.0;

    // Estimate components of cost
    int w_miss = (bw && bw->resident && bw->tile_id == tile_id) ? 0 : 1;
    int i_miss = (bi && bi->resident && bi->tile_id == tile_id) ? 0 : 1;

    // Phase 20: Use adaptive projection parameters
    // We estimate the components of the Ground Truth cost
    double est_active_ops = 1000.0; // Mock average active ops per tile
    double est_mem_reads = 0.0;
    if (w_miss) est_mem_reads += 500.0; // Mock weight size
    if (i_miss) est_mem_reads += 100.0; // Mock input size

    cost += est_active_ops * g_proj_params.weight_cost;
    cost += est_mem_reads * g_proj_params.mem_read_cost;
    if (w_miss || i_miss) cost += g_proj_params.residency_miss_cost * 10.0;

    // Rebate for residency hits (legacy style for out_rebate telemetry)
    if (!w_miss) rebate += 50.0;
    if (!i_miss) rebate += 10.0;
    if (out_rebate) *out_rebate = rebate;

    // Favor tiles with better historical efficiency for this kernel
    cost *= g_learn_state.tile_kernel_mult[tile_id][kernel];

    // Hysteresis (Phase 19): Prefer tile already assigned to the weight block
    if (bw && bw->tile_id == tile_id) {
        cost -= 0.5; // Small epsilon to prevent economic jitter
    }

    return cost;
}

static uint8_t tfmbs_select_tiles(kernel_type_t kernel, void* w_ptr, void* i_ptr, void* o_ptr, int desired_count) {
    if (desired_count <= 0) desired_count = 1;
    if (desired_count > MAX_TILES) desired_count = MAX_TILES;

    double costs[MAX_TILES];
    double rebates[MAX_TILES];
    int tile_indices[MAX_TILES];
    for (int i = 0; i < MAX_TILES; i++) {
        costs[i] = projected_cost(i, kernel, w_ptr, i_ptr, o_ptr, &rebates[i]);
        tile_indices[i] = i;
    }

    // Sort tiles by cost (Ascending: lower cost first)
    for (int i = 0; i < MAX_TILES - 1; i++) {
        for (int j = 0; j < MAX_TILES - i - 1; j++) {
            if (costs[j] > costs[j+1]) {
                double tc = costs[j]; costs[j] = costs[j+1]; costs[j+1] = tc;
                double tr = rebates[j]; rebates[j] = rebates[j+1]; rebates[j+1] = tr;
                int ti = tile_indices[j]; tile_indices[j] = tile_indices[j+1]; tile_indices[j+1] = ti;
            }
        }
    }

    // Update introspection
    g_last_chosen_tile_id = tile_indices[0];
    g_last_projected_cost = costs[0];
    g_last_residency_rebate = rebates[0];

    uint8_t mask = 0;
    for (int i = 0; i < desired_count; i++) {
        mask |= (1 << tile_indices[i]);
    }
    g_last_tile_mask = mask;

    if (getenv("TFMBS_DEBUG")) {
        fprintf(stderr, "[TFMBS-Device] Scheduled %d tiles. Best tile: %d (cost %.2f), mask 0x%02x\n",
                desired_count, tile_indices[0], costs[0], mask);
    }
    return mask;
}

static void track_residency(fabric_block_t* b, uint8_t tile_mask) {
    if (!b) return;
    bool hit_any = false;
    int first_tile = -1;

    for (int i = 0; i < MAX_TILES; i++) {
        if (tile_mask & (1 << i)) {
            if (first_tile == -1) first_tile = i;
            if (b->resident && (b->tile_id == i || b->tile_id == -1)) {
                g_tile_metrics[i].residency_hits++;
                b->residency_hits++;
                hit_any = true;
            } else {
                g_tile_metrics[i].residency_misses++;
            }
        }
    }

    if (hit_any) {
        g_last_metrics.residency_hits++;
    } else {
        g_last_metrics.residency_misses++;
        b->resident = 1;
        if (b->tile_id < 0 && first_tile != -1) {
            b->tile_id = first_tile;
            if (getenv("TFMBS_DEBUG")) {
                fprintf(stderr, "[TFMBS-Device] Block at %p assigned to Tile %d (first-touch)\n", b->ptr, first_tile);
            }
        }
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

static double block_policy_score(fabric_block_t* b) {
    if (b->pinned) return 1e18;

    uint32_t now = g_access_counter;
    uint32_t age = now - b->last_access;

    // Phase 20: Adaptive Eviction Scoring
    // Score = W_freq * freq + W_recency / (age + 1) + W_success * success_rate
    double freq = (double)b->access_count;
    double recency = 1.0 / (age + 1.0);
    double success_rate = (b->access_count > 0) ? (double)b->residency_hits / b->access_count : 0;

    double score = g_learn_state.eviction_freq_weight * freq +
                   g_learn_state.eviction_age_weight * recency * 10.0 +
                   g_learn_state.eviction_success_weight * success_rate * 100.0;

    return score;
}

static void evict_policy() {
    fabric_block_t* victim = NULL;
    double min_score = 1e20;

    fabric_block_t* curr = g_blocks;
    while (curr) {
        if (curr->used && curr->busy_count == 0 && !curr->pinned) {
            double score = block_policy_score(curr);
            if (score < min_score) {
                min_score = score;
                victim = curr;
            }
        }
        curr = curr->next;
    }

    if (victim) {
        victim->used = 0;
        victim->resident = 0;
        g_last_metrics.eviction_count++;
        g_last_metrics.pool_used -= victim->size;

        char score_str[64];
        snprintf(score_str, sizeof(score_str), "%.4f ", min_score);
        strncat(g_last_eviction_scores, score_str, sizeof(g_last_eviction_scores) - strlen(g_last_eviction_scores) - 1);

        printf("[TFMBS-Device] Policy-Evicted block at %p (score %.4f, size %zu)\n", victim->ptr, min_score, victim->size);
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
                    new_block->tile_id = -1;
                    new_block->pinned = 0;
                    new_block->access_count = 0;
                    new_block->residency_hits = 0;
                    new_block->resident = 0;
                    new_block->next = curr->next;

                    curr->size = aligned_size;
                    curr->next = new_block;
                }
                curr->used = 1;
                curr->resident = 1;
                curr->tile_id = -1;
                curr->pinned = 0;
                curr->last_access = ++g_access_counter;
                curr->access_count = 1;
                curr->residency_hits = 0;
                g_last_metrics.pool_used += curr->size;
                pthread_mutex_unlock(&g_fabric_mutex);
                return curr->ptr;
            }
            curr = curr->next;
        }

        // No block found, try evicting
        int any_evictable = 0;
        curr = g_blocks;
        while(curr) { if(curr->used && curr->busy_count == 0) any_evictable=1; curr=curr->next; }

        if (!any_evictable) {
            // Check if there are ANY free blocks (even if too small)
            int any_free = 0;
            curr = g_blocks;
            while(curr) { if(!curr->used) any_free=1; curr=curr->next; }

            if (!any_free) {
                fprintf(stderr, "[TFMBS-Device] Out of Fabric Memory: All blocks are BUSY. Requested %zu\n", aligned_size);
                pthread_mutex_unlock(&g_fabric_mutex);
                return NULL;
            }
        }

        if (any_evictable) {
            evict_policy();
        } else {
            // We have some free blocks but they are not large enough or fragmented
            // and we can't evict anything else.
            fprintf(stderr, "[TFMBS-Device] Out of Fabric Memory: Fragmentation or Busy blocks. Requested %zu\n", aligned_size);
            pthread_mutex_unlock(&g_fabric_mutex);
            return NULL;
        }

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
                curr->resident = 0;
                curr->pinned = 0;
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
    fabric_block_t *b = find_block(dest_fabric);
    if (b) b->resident = 1;
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

static int8_t g_w_trits[8000000];
static int8_t g_i_trits[8000000];

static int internal_exec_lstm(void* weight_ptr, void* input_ptr, void* output_ptr, int h_size, int i_size, uint8_t tile_mask, int persistent, uint32_t hints) {
    strncpy(g_last_kernel_name, persistent ? "LSTM-P" : "LSTM", sizeof(g_last_kernel_name));
    void* dev_weight_ptr = to_device_ptr(weight_ptr);
    void* dev_input_ptr = to_device_ptr(input_ptr);
    void* dev_output_ptr = to_device_ptr(output_ptr);

    // Simulation of 4 gates (i, f, g, o)
    // Weights: [4 * h_size, i_size + h_size]
    int rows = 4 * h_size;
    int cols = i_size + h_size;

    int active_tiles = 0;
    for (int i = 0; i < 8; i++) if (tile_mask & (1 << i)) active_tiles++;
    if (active_tiles == 0) active_tiles = 1;

    pthread_mutex_lock(&g_fabric_mutex);
    fabric_block_t *bw = find_block(weight_ptr);
    fabric_block_t *bi = find_block(input_ptr);
    fabric_block_t *bo = find_block(output_ptr);

    track_residency(bw, tile_mask);
    track_residency(bi, tile_mask);
    track_residency(bo, tile_mask);
    pthread_mutex_unlock(&g_fabric_mutex);

    // Unpack weights (limited for simulation)
    uint8_t* w_packed = (uint8_t*)dev_weight_ptr;
    for (int i = 0; i < (rows * cols + 4) / 5 && i * 5 < 8000000; i++) {
        int8_t trits[5];
        unpack_byte_to_trits(w_packed[i], trits);
        for (int j = 0; j < 5; j++) {
            if (i * 5 + j < rows * cols) g_w_trits[i * 5 + j] = trits[j];
        }
    }

    if (persistent) {
        // Unpack x from input_ptr (i_size trits)
        uint8_t* x_packed = (uint8_t*)dev_input_ptr;
        for (int i = 0; i < (i_size + 4) / 5 && i * 5 < 8000000; i++) {
            int8_t trits[5];
            unpack_byte_to_trits(x_packed[i], trits);
            for (int j = 0; j < 5; j++) {
                if (i * 5 + j < i_size) g_i_trits[i * 5 + j] = trits[j];
            }
        }
        // Take h_prev from output_ptr (state) - mock simple copy
        // For simulation, we'll assume h_prev is already in the tail of g_i_trits
        // In a real persistent kernel, this would be a tile-local read.
        int32_t* h_prev = (int32_t*)dev_output_ptr;
        for (int j = 0; j < h_size; j++) {
            // Simplified: treat previous output as hidden state trits for simulation
            g_i_trits[i_size + j] = (h_prev[j] > 0) ? 1 : (h_prev[j] < 0 ? -1 : 0);
        }
    } else {
        // Unpack inputs (x and h_prev concatenated from input_ptr)
        uint8_t* i_packed = (uint8_t*)dev_input_ptr;
        for (int i = 0; i < (cols + 4) / 5 && i * 5 < 8000000; i++) {
            int8_t trits[5];
            unpack_byte_to_trits(i_packed[i], trits);
            for (int j = 0; j < 5; j++) {
                if (i * 5 + j < cols) g_i_trits[i * 5 + j] = trits[j];
            }
        }
    }

    g_last_metrics.zero_skips = 0;
    g_last_metrics.total_ops = (long)rows * cols;

    // Phase 18: Optimize mem_reads if weights/state are pinned/resident
    g_last_metrics.mem_reads = (cols / 5); // At least read the input
    if (bw && bw->pinned && bw->resident) {
        // Gates are resident, no mem_read cost for them
    } else {
        g_last_metrics.mem_reads += (rows * cols) / 5;
    }

    g_last_metrics.mem_writes = rows * 4;

    // Distribute basic metrics to tiles
    for (int i = 0; i < MAX_TILES; i++) {
        if (tile_mask & (1 << i)) {
            g_tile_metrics[i].total_ops += g_last_metrics.total_ops / active_tiles;
            g_tile_metrics[i].mem_reads += g_last_metrics.mem_reads / active_tiles;
            g_tile_metrics[i].mem_writes += g_last_metrics.mem_writes / active_tiles;
        }
    }
    g_last_metrics.cycles = rows + cols; // Rough estimation

    int32_t* results = (int32_t*)dev_output_ptr;
    for (int r = 0; r < rows; r++) {
        int32_t acc = 0;
        for (int c = 0; c < cols; c++) {
            int8_t w = g_w_trits[r * cols + c];
            int8_t x = g_i_trits[c];
            if (w == 0 || x == 0) g_last_metrics.zero_skips++;
            else acc += (int32_t)w * (int32_t)x;
        }
        results[r] = acc;
    }

    g_last_metrics.active_ops = g_last_metrics.total_ops - g_last_metrics.zero_skips;

    // Distribute active ops and zero skips
    for (int i = 0; i < MAX_TILES; i++) {
        if (tile_mask & (1 << i)) {
            g_tile_metrics[i].active_ops += g_last_metrics.active_ops / active_tiles;
            g_tile_metrics[i].zero_skips += g_last_metrics.zero_skips / active_tiles;
            g_tile_metrics[i].cycles += g_last_metrics.cycles / active_tiles;
        }
    }

    g_last_metrics.sim_cycle_reduction = (double)g_last_metrics.zero_skips / g_last_metrics.total_ops * 100.0;
    g_last_metrics.fabric_cost = tfmbs_compute_cost(&g_last_metrics, hints);

    // Update tile efficiency
    for (int i = 0; i < MAX_TILES; i++) {
        if (tile_mask & (1 << i)) {
            g_tile_metrics[i].fabric_cost = tfmbs_compute_cost(&g_tile_metrics[i], hints);
            if (g_tile_metrics[i].total_ops > 0)
                g_tile_metrics[i].semantic_efficiency = (double)g_tile_metrics[i].active_ops / g_tile_metrics[i].total_ops;
            if (g_tile_metrics[i].fabric_cost > 0)
                g_tile_metrics[i].economic_efficiency = (double)g_tile_metrics[i].active_ops / g_tile_metrics[i].fabric_cost;
        }
    }

    if (g_last_metrics.total_ops > 0)
        g_last_metrics.semantic_efficiency = (double)g_last_metrics.active_ops / g_last_metrics.total_ops;
    else
        g_last_metrics.semantic_efficiency = 0.0;

    if (g_last_metrics.fabric_cost > 0)
        g_last_metrics.economic_efficiency = (double)g_last_metrics.active_ops / g_last_metrics.fabric_cost;
    else
        g_last_metrics.economic_efficiency = 0.0;

    return 0;
}

static int internal_exec_gemv(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols, uint8_t tile_mask, uint32_t hints) {
    strncpy(g_last_kernel_name, "GEMV", sizeof(g_last_kernel_name));
    // Use device-side mappings to bypass interposer protections
    void* dev_weight_ptr = to_device_ptr(weight_ptr);
    void* dev_input_ptr = to_device_ptr(input_ptr);
    void* dev_output_ptr = to_device_ptr(output_ptr);

    // Multi-Tile Simulation: count active tiles in mask
    int active_tiles = 0;
    for (int i = 0; i < 8; i++) if (tile_mask & (1 << i)) active_tiles++;
    if (active_tiles == 0) active_tiles = 1; // Fallback

    pthread_mutex_lock(&g_fabric_mutex);
    fabric_block_t *bw = find_block(weight_ptr);
    fabric_block_t *bi = find_block(input_ptr);
    fabric_block_t *bo = find_block(output_ptr);

    // Residency and Broadcast tracking
    track_residency(bw, tile_mask);
    track_residency(bi, tile_mask);
    track_residency(bo, tile_mask);

    if (active_tiles > 1) {
        g_last_metrics.broadcasts++;
        for (int i = 0; i < MAX_TILES; i++) {
            if (tile_mask & (1 << i)) g_tile_metrics[i].broadcasts++;
        }
        g_last_metrics.tile_local_reuse += (active_tiles - 1);
        for (int i = 0; i < MAX_TILES; i++) {
            if (tile_mask & (1 << i)) g_tile_metrics[i].tile_local_reuse++; // Approximate
        }
    }
    pthread_mutex_unlock(&g_fabric_mutex);

    // Unpack weights
    uint8_t* w_packed = (uint8_t*)dev_weight_ptr;
    for (int i = 0; i < (rows * cols + 4) / 5; i++) {
        int8_t trits[5];
        unpack_byte_to_trits(w_packed[i], trits);
        for (int j = 0; j < 5; j++) {
            if (i * 5 + j < rows * cols) g_w_trits[i * 5 + j] = trits[j];
        }
    }

    // Unpack inputs
    uint8_t* i_packed = (uint8_t*)dev_input_ptr;
    for (int i = 0; i < (cols + 4) / 5; i++) {
        int8_t trits[5];
        unpack_byte_to_trits(i_packed[i], trits);
        for (int j = 0; j < 5; j++) {
            if (i * 5 + j < cols) g_i_trits[i * 5 + j] = trits[j];
        }
    }

    // Reset Metrics for this run
    g_last_metrics.zero_skips = 0;
    g_last_metrics.total_ops = (long)rows * cols;
    g_last_metrics.lanes_used = active_tiles * 15;
    g_last_metrics.mem_reads = (rows * cols) / 5 + cols / 5;
    g_last_metrics.mem_writes = rows * 4;
    g_last_metrics.cycles = (rows * cols) / g_last_metrics.lanes_used;

    // Distribute metrics to tiles
    for (int i = 0; i < MAX_TILES; i++) {
        if (tile_mask & (1 << i)) {
            g_tile_metrics[i].total_ops += g_last_metrics.total_ops / active_tiles;
            g_tile_metrics[i].mem_reads += g_last_metrics.mem_reads / active_tiles;
            g_tile_metrics[i].mem_writes += g_last_metrics.mem_writes / active_tiles;
        }
    }

    // GEMV with Zero-Skip Emulation and Tile partitioning
    int32_t* results = (int32_t*)dev_output_ptr;

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

    g_last_metrics.active_ops = g_last_metrics.total_ops - g_last_metrics.zero_skips;

    // Distribute active ops and zero skips
    for (int i = 0; i < MAX_TILES; i++) {
        if (tile_mask & (1 << i)) {
            g_tile_metrics[i].active_ops += g_last_metrics.active_ops / active_tiles;
            g_tile_metrics[i].zero_skips += g_last_metrics.zero_skips / active_tiles;
            g_tile_metrics[i].cycles += g_last_metrics.cycles / active_tiles;
        }
    }

    g_last_metrics.sim_cycle_reduction = (double)g_last_metrics.zero_skips / g_last_metrics.total_ops * 100.0;
    g_last_metrics.fabric_cost = tfmbs_compute_cost(&g_last_metrics, hints);

    // Update tile efficiency
    for (int i = 0; i < MAX_TILES; i++) {
        if (tile_mask & (1 << i)) {
            g_tile_metrics[i].fabric_cost = tfmbs_compute_cost(&g_tile_metrics[i], hints);
            if (g_tile_metrics[i].total_ops > 0)
                g_tile_metrics[i].semantic_efficiency = (double)g_tile_metrics[i].active_ops / g_tile_metrics[i].total_ops;
            if (g_tile_metrics[i].fabric_cost > 0)
                g_tile_metrics[i].economic_efficiency = (double)g_tile_metrics[i].active_ops / g_tile_metrics[i].fabric_cost;
        }
    }

    if (g_last_metrics.total_ops > 0)
        g_last_metrics.semantic_efficiency = (double)g_last_metrics.active_ops / g_last_metrics.total_ops;
    else
        g_last_metrics.semantic_efficiency = 0.0;

    if (g_last_metrics.fabric_cost > 0)
        g_last_metrics.economic_efficiency = (double)g_last_metrics.active_ops / g_last_metrics.fabric_cost;
    else
        g_last_metrics.economic_efficiency = 0.0;

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

    int desired_tiles = 0;
    for (int i = 0; i < 8; i++) if (tile_mask & (1 << i)) desired_tiles++;
    if (desired_tiles == 0) desired_tiles = 1;
    uint8_t scheduled_mask = tfmbs_select_tiles(KERNEL_GEMV, weight_ptr, input_ptr, output_ptr, desired_tiles);

    return internal_exec_gemv(weight_ptr, input_ptr, output_ptr, rows, cols, scheduled_mask, 0);
}

int emu_fabric_exec_lstm(void* weight_ptr, void* input_ptr, void* output_ptr, int h_size, int i_size, uint8_t tile_mask) {
    if (!emu_is_fabric_ptr(weight_ptr) || !emu_is_fabric_ptr(input_ptr) || !emu_is_fabric_ptr(output_ptr)) return -1;

    int desired_tiles = 0;
    for (int i = 0; i < 8; i++) if (tile_mask & (1 << i)) desired_tiles++;
    if (desired_tiles == 0) desired_tiles = 1;
    uint8_t scheduled_mask = tfmbs_select_tiles(KERNEL_LSTM, weight_ptr, input_ptr, output_ptr, desired_tiles);

    return internal_exec_lstm(weight_ptr, input_ptr, output_ptr, h_size, i_size, scheduled_mask, 0, 0);
}

fabric_handle_t emu_fabric_exec_gemv_async(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols, uint8_t tile_mask) {
    init_fabric_pool();

    // Cost-Aware Scheduling
    int desired_tiles = 0;
    for (int i = 0; i < 8; i++) if (tile_mask & (1 << i)) desired_tiles++;
    if (desired_tiles == 0) desired_tiles = 1;
    uint8_t scheduled_mask = tfmbs_select_tiles(KERNEL_GEMV, weight_ptr, input_ptr, output_ptr, desired_tiles);

    fabric_task_t* task = (fabric_task_t*)malloc(sizeof(fabric_task_t));
    task->weight_ptr = weight_ptr;
    task->input_ptr = input_ptr;
    task->output_ptr = output_ptr;
    task->rows = rows;
    task->cols = cols;
    task->kernel = KERNEL_GEMV;
    task->exec_hints = 0; // Default
    task->tile_mask = scheduled_mask;
    task->projected_cost = g_last_projected_cost;
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

fabric_handle_t emu_fabric_exec_lstm_async(void* weight_ptr, void* input_ptr, void* output_ptr, int h_size, int i_size, uint8_t tile_mask) {
    init_fabric_pool();

    int desired_tiles = 0;
    for (int i = 0; i < 8; i++) if (tile_mask & (1 << i)) desired_tiles++;
    if (desired_tiles == 0) desired_tiles = 1;
    uint8_t scheduled_mask = tfmbs_select_tiles(KERNEL_LSTM, weight_ptr, input_ptr, output_ptr, desired_tiles);

    fabric_task_t* task = (fabric_task_t*)malloc(sizeof(fabric_task_t));
    task->weight_ptr = weight_ptr;
    task->input_ptr = input_ptr;
    task->output_ptr = output_ptr;
    task->rows = h_size;
    task->cols = i_size;
    task->kernel = KERNEL_LSTM;
    task->exec_hints = 0;
    task->tile_mask = scheduled_mask;
    task->projected_cost = g_last_projected_cost;
    task->status = TASK_PENDING;
    pthread_mutex_init(&task->mutex, NULL);
    pthread_cond_init(&task->cond, NULL);
    task->next = NULL;

    pthread_mutex_lock(&g_queue_mutex);
    if (g_queue_tail) { g_queue_tail->next = task; g_queue_tail = task; }
    else { g_queue_head = g_queue_tail = task; }

    pthread_mutex_lock(&g_fabric_mutex);
    set_busy(weight_ptr, 1); set_busy(input_ptr, 1); set_busy(output_ptr, 1);
    pthread_mutex_unlock(&g_fabric_mutex);

    pthread_cond_signal(&g_queue_cond);
    pthread_mutex_unlock(&g_queue_mutex);
    return (fabric_handle_t)task;
}

void emu_fabric_lstm_bind(void* weight_ptr, void* state_ptr, uint8_t tile_mask) {
    pthread_mutex_lock(&g_fabric_mutex);
    fabric_block_t *bw = find_block(weight_ptr);
    fabric_block_t *bs = find_block(state_ptr);
    if (bw) { bw->pinned = 1; bw->tile_id = tile_mask; bw->resident = 1; }
    if (bs) { bs->pinned = 1; bs->tile_id = tile_mask; bs->resident = 1; }
    pthread_mutex_unlock(&g_fabric_mutex);
}

fabric_handle_t emu_fabric_exec_lstm_persistent_async(void* weight_ptr, void* input_ptr, void* state_ptr, int h_size, int i_size, uint8_t tile_mask) {
    init_fabric_pool();

    int desired_tiles = 0;
    for (int i = 0; i < 8; i++) if (tile_mask & (1 << i)) desired_tiles++;
    if (desired_tiles == 0) desired_tiles = 1;
    uint8_t scheduled_mask = tfmbs_select_tiles(KERNEL_LSTM_PERSISTENT, weight_ptr, input_ptr, state_ptr, desired_tiles);

    fabric_task_t* task = (fabric_task_t*)malloc(sizeof(fabric_task_t));
    task->weight_ptr = weight_ptr;
    task->input_ptr = input_ptr;
    task->output_ptr = state_ptr; // state is also output
    task->rows = h_size;
    task->cols = i_size;
    task->kernel = KERNEL_LSTM_PERSISTENT;
    task->exec_hints = 0;
    task->tile_mask = scheduled_mask;
    task->projected_cost = g_last_projected_cost;
    task->status = TASK_PENDING;
    pthread_mutex_init(&task->mutex, NULL);
    pthread_cond_init(&task->cond, NULL);
    task->next = NULL;

    pthread_mutex_lock(&g_queue_mutex);
    if (g_queue_tail) { g_queue_tail->next = task; g_queue_tail = task; }
    else { g_queue_head = g_queue_tail = task; }

    pthread_mutex_lock(&g_fabric_mutex);
    set_busy(weight_ptr, 1); set_busy(input_ptr, 1); set_busy(state_ptr, 1);
    pthread_mutex_unlock(&g_fabric_mutex);

    pthread_cond_signal(&g_queue_cond);
    pthread_mutex_unlock(&g_queue_mutex);
    return (fabric_handle_t)task;
}

static int g_economic_step = 0;

void emu_fabric_dump_economic_csv(const char* path) {
    int exists = access(path, F_OK) == 0;
    FILE* f = fopen(path, "a");
    if (!f) return;
    if (!exists) {
        fprintf(f, "step,kernel,tile_mask,chosen_tile_id,projected_cost,residency_rebate,batch_size,weight_cost,mem_read_cost,mem_write_cost,broadcast_cost,residency_miss_cost,tile_mult,eviction_weights,eviction_scores\n");
    }
    pthread_mutex_lock(&g_fabric_mutex);
    // Remove trailing space from eviction scores
    size_t len = strlen(g_last_eviction_scores);
    if (len > 0 && g_last_eviction_scores[len-1] == ' ') g_last_eviction_scores[len-1] = '\0';

    // Get current tile multiplier for the last chosen tile and kernel
    double current_mult = 1.0;
    if (g_last_chosen_tile_id >= 0) {
        // Find kernel index
        int k_idx = 0;
        if (strcmp(g_last_kernel_name, "LSTM") == 0) k_idx = 1;
        else if (strcmp(g_last_kernel_name, "LSTM-P") == 0) k_idx = 2;
        current_mult = g_learn_state.tile_kernel_mult[g_last_chosen_tile_id][k_idx];
    }

    fprintf(f, "%d,%s,0x%02x,%d,%.2f,%.2f,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.4f,\"%.2f;%.2f;%.2f\",\"%s\"\n",
            g_economic_step++, g_last_kernel_name, g_last_tile_mask, g_last_chosen_tile_id,
            g_last_projected_cost, g_last_residency_rebate, g_learn_state.dynamic_batch_size,
            g_proj_params.weight_cost, g_proj_params.mem_read_cost, g_proj_params.mem_write_cost,
            g_proj_params.broadcast_cost, g_proj_params.residency_miss_cost,
            current_mult,
            g_learn_state.eviction_freq_weight, g_learn_state.eviction_age_weight, g_learn_state.eviction_success_weight,
            g_last_eviction_scores);

    // Reset for next step
    g_last_eviction_scores[0] = '\0';
    pthread_mutex_unlock(&g_fabric_mutex);
    fclose(f);
}

static void update_learning(fabric_task_t* task, fabric_metrics_t* metrics) {
    // 2.1 Adaptive Cost Coefficients: Minimize delta between projected and actual cost
    double delta = metrics->fabric_cost - task->projected_cost;
    double lr = 0.05;

    pthread_mutex_lock(&g_fabric_mutex);
    g_proj_params.weight_cost         += lr * delta * 0.05;
    g_proj_params.mem_read_cost       += lr * delta * 0.1;
    g_proj_params.mem_write_cost      += lr * delta * 0.1;
    g_proj_params.broadcast_cost      += lr * delta * 0.05;
    g_proj_params.residency_miss_cost += lr * delta * 0.2;

    // Bounds: 0.1x to 5.0x of Phase 19 defaults
    if (g_proj_params.weight_cost < 0.1) g_proj_params.weight_cost = 0.1;
    if (g_proj_params.weight_cost > 5.0) g_proj_params.weight_cost = 5.0;
    if (g_proj_params.mem_read_cost < 0.5) g_proj_params.mem_read_cost = 0.5;
    if (g_proj_params.mem_read_cost > 25.0) g_proj_params.mem_read_cost = 25.0;
    if (g_proj_params.mem_write_cost < 0.8) g_proj_params.mem_write_cost = 0.8;
    if (g_proj_params.mem_write_cost > 40.0) g_proj_params.mem_write_cost = 40.0;
    if (g_proj_params.broadcast_cost < 0.2) g_proj_params.broadcast_cost = 0.2;
    if (g_proj_params.broadcast_cost > 10.0) g_proj_params.broadcast_cost = 10.0;
    if (g_proj_params.residency_miss_cost < 0.6) g_proj_params.residency_miss_cost = 0.6;
    if (g_proj_params.residency_miss_cost > 30.0) g_proj_params.residency_miss_cost = 30.0;

    // 2.2 Dynamic Scheduler Weighting: Learn tile-kernel preferences
    int k_idx = task->kernel;
    for (int i = 0; i < MAX_TILES; i++) {
        if (task->tile_mask & (1 << i)) {
            g_learn_state.kernel_exec_count[i][k_idx]++;

            // Adjust multiplier based on efficiency vs EMA
            double current_eff = metrics->economic_efficiency;
            if (g_learn_state.avg_efficiency_ema == 0) g_learn_state.avg_efficiency_ema = current_eff;

            if (current_eff > g_learn_state.avg_efficiency_ema) {
                // Better than average: favor this tile-kernel pair (decrease cost multiplier)
                g_learn_state.tile_kernel_mult[i][k_idx] *= 0.98;
            } else {
                // Worse than average: penalize (increase cost multiplier)
                g_learn_state.tile_kernel_mult[i][k_idx] *= 1.02;
            }

            // Apply specified decay (0.99 every 10 executions) - simplified to every execution for smoothness
            if (g_learn_state.kernel_exec_count[i][k_idx] % 10 == 0) {
                // Return multiplier towards 1.0
                g_learn_state.tile_kernel_mult[i][k_idx] = 1.0 + (g_learn_state.tile_kernel_mult[i][k_idx] - 1.0) * 0.99;
            }

            // Bounds for multiplier
            if (g_learn_state.tile_kernel_mult[i][k_idx] < 0.5) g_learn_state.tile_kernel_mult[i][k_idx] = 0.5;
            if (g_learn_state.tile_kernel_mult[i][k_idx] > 2.0) g_learn_state.tile_kernel_mult[i][k_idx] = 2.0;
        }
    }

    // 2.3 Eviction Policy Self-Tuning
    if (metrics->residency_misses > 0) {
        // We are missing too much, increase protection for successful/frequent blocks
        g_learn_state.eviction_success_weight += 0.01 * metrics->residency_misses;
        g_learn_state.eviction_freq_weight += 0.01;
    } else if (metrics->residency_hits > 10) {
        // High hit rate, maybe we can afford to be more aggressive (decay weights)
        g_learn_state.eviction_success_weight *= 0.999;
        g_learn_state.eviction_freq_weight *= 0.999;
    }

    // Update Global EMA
    g_learn_state.avg_efficiency_ema = 0.9 * g_learn_state.avg_efficiency_ema + 0.1 * metrics->economic_efficiency;
    double current_throughput = (metrics->cycles > 0) ? (double)metrics->active_ops / metrics->cycles : 0;
    g_learn_state.avg_throughput_ema = 0.9 * g_learn_state.avg_throughput_ema + 0.1 * current_throughput;

    pthread_mutex_unlock(&g_fabric_mutex);

    if (getenv("TFMBS_ECONOMIC_LOG")) {
        emu_fabric_dump_economic_csv(getenv("TFMBS_ECONOMIC_LOG"));
    }
}

static void update_batch_tuning(int last_batch_size) {
    static double last_score = 0;
    static int last_size = 8;

    pthread_mutex_lock(&g_fabric_mutex);
    // Composite score: 0.7 * efficiency + 0.3 * normalized_throughput
    // Throughput is roughly 0-60 lanes, so normalize by 60.
    double current_score = 0.7 * g_learn_state.avg_efficiency_ema + 0.3 * (g_learn_state.avg_throughput_ema / 60.0);

    if (last_score > 0 && last_batch_size != last_size) {
        if (current_score > last_score) {
            // Direction was good
            if (last_batch_size > last_size) g_learn_state.dynamic_batch_size++;
            else g_learn_state.dynamic_batch_size--;
        } else if (current_score < last_score) {
            // Direction was bad
            if (last_batch_size > last_size) g_learn_state.dynamic_batch_size--;
            else g_learn_state.dynamic_batch_size++;
        }
    }

    if (g_learn_state.dynamic_batch_size < 1) g_learn_state.dynamic_batch_size = 1;
    if (g_learn_state.dynamic_batch_size > 32) g_learn_state.dynamic_batch_size = 32;

    last_score = current_score;
    last_size = last_batch_size;
    pthread_mutex_unlock(&g_fabric_mutex);
}

void emu_fabric_dump_metrics_csv(const char* path) {
    int exists = access(path, F_OK) == 0;
    FILE* f = fopen(path, "a");
    if (!f) return;
    if (!exists) {
        fprintf(f, "zero_skips,total_ops,active_ops,mem_reads,mem_writes,broadcasts,residency_hits,residency_misses,tile_local_reuse,cycles,fabric_cost,semantic_efficiency,economic_efficiency\n");
    }
    pthread_mutex_lock(&g_fabric_mutex);
    fprintf(f, "%ld,%ld,%lu,%lu,%lu,%lu,%lu,%lu,%lu,%ld,%.2f,%.4f,%.4f\n",
            g_last_metrics.zero_skips, g_last_metrics.total_ops, g_last_metrics.active_ops,
            g_last_metrics.mem_reads, g_last_metrics.mem_writes, g_last_metrics.broadcasts,
            g_last_metrics.residency_hits, g_last_metrics.residency_misses,
            g_last_metrics.tile_local_reuse,
            g_last_metrics.cycles, g_last_metrics.fabric_cost,
            g_last_metrics.semantic_efficiency, g_last_metrics.economic_efficiency);
    pthread_mutex_unlock(&g_fabric_mutex);
    fclose(f);
}

int emu_fabric_wait(fabric_handle_t handle) {
    if (!handle) return -1;
    fabric_task_t* task = (fabric_task_t*)handle;

    pthread_mutex_lock(&task->mutex);
    while (task->status != TASK_DONE) {
        pthread_cond_wait(&task->cond, &task->mutex);
    }
    pthread_mutex_unlock(&task->mutex);

    pthread_mutex_destroy(&task->mutex);
    pthread_cond_destroy(&task->cond);
    free(task);
    return 0;
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

        // Batch tasks of the same type
        fabric_task_t* batch[32];
        int batch_size = 0;
        int max_b = g_learn_state.dynamic_batch_size;

        // Phase 20: Batch size exploration (5% chance)
        if (rand() % 100 < 5) {
            max_b += (rand() % 2 == 0) ? 1 : -1;
            if (max_b < 1) max_b = 1;
            if (max_b > 32) max_b = 32;
        }

        while (g_queue_head && batch_size < max_b) {
            if (g_queue_head->status == TASK_SHUTDOWN) break;
            if (batch_size > 0 && g_queue_head->kernel != batch[0]->kernel) break;

            batch[batch_size] = g_queue_head;
            g_queue_head = g_queue_head->next;
            if (g_queue_head == NULL) g_queue_tail = NULL;
            batch_size++;
        }
        pthread_mutex_unlock(&g_queue_mutex);

        for (int b = 0; b < batch_size; b++) {
            fabric_task_t* t = batch[b];
            pthread_mutex_lock(&t->mutex);
            t->status = TASK_RUNNING;
            pthread_mutex_unlock(&t->mutex);

            // Update access
            pthread_mutex_lock(&g_fabric_mutex);
            update_access(t->weight_ptr);
            update_access(t->input_ptr);
            update_access(t->output_ptr);
            pthread_mutex_unlock(&g_fabric_mutex);

            if (t->kernel == KERNEL_LSTM) {
                internal_exec_lstm(t->weight_ptr, t->input_ptr, t->output_ptr, t->rows, t->cols, t->tile_mask, 0, t->exec_hints);
            } else if (t->kernel == KERNEL_LSTM_PERSISTENT) {
                internal_exec_lstm(t->weight_ptr, t->input_ptr, t->output_ptr, t->rows, t->cols, t->tile_mask, 1, t->exec_hints);
            } else {
                internal_exec_gemv(t->weight_ptr, t->input_ptr, t->output_ptr, t->rows, t->cols, t->tile_mask, t->exec_hints);
            }

            // Phase 20: Feedback Loop
            update_learning(t, &g_last_metrics);

            // Simulate overlapping memory + compute in batch (Pipelining)
            if (batch_size > 1) {
                g_last_metrics.cycles = (long)(g_last_metrics.cycles * 0.85);
            }

            // Unpin blocks
            pthread_mutex_lock(&g_fabric_mutex);
            set_busy(t->weight_ptr, -1);
            set_busy(t->input_ptr, -1);
            set_busy(t->output_ptr, -1);
            pthread_mutex_unlock(&g_fabric_mutex);

            // Telemetry
            if (b == batch_size - 1) { // Only log once per batch for brevity
                int active_tiles_telemetry = 0;
                for (int i = 0; i < 8; i++) if (t->tile_mask & (1 << i)) active_tiles_telemetry++;
                if (active_tiles_telemetry == 0) active_tiles_telemetry = 1;

                fprintf(stderr, "\n[TFMBS-Telemetry] Batch of %d %s(s) Completed\n",
                        batch_size, t->kernel == KERNEL_LSTM ? "LSTM" : (t->kernel == KERNEL_LSTM_PERSISTENT ? "LSTM-Persistent" : "GEMV"));
                fprintf(stderr, "  - Last Active Tiles: %d (mask 0x%02x)\n", active_tiles_telemetry, t->tile_mask);
                fprintf(stderr, "  - Zero-Skips: %ld (%.1f%% reduction)\n", g_last_metrics.zero_skips, g_last_metrics.sim_cycle_reduction);
                fprintf(stderr, "  - Cycles:     %ld (Cost: %.1f, Sem-Eff: %.2f, Econ-Eff: %.2f)\n",
                        g_last_metrics.cycles, g_last_metrics.fabric_cost,
                        g_last_metrics.semantic_efficiency, g_last_metrics.economic_efficiency);
                fprintf(stderr, "  - Residency:  Hits: %lu, Misses: %lu\n", g_last_metrics.residency_hits, g_last_metrics.residency_misses);
                fprintf(stderr, "  - Pool Usage: %zu / %zu bytes (%.1f%%)\n", g_last_metrics.pool_used, g_last_metrics.pool_total, (double)g_last_metrics.pool_used / g_last_metrics.pool_total * 100.0);
                fprintf(stderr, "  - Evictions:  %d\n", g_last_metrics.eviction_count);
            }

            pthread_mutex_lock(&t->mutex);
            t->status = TASK_DONE;
            pthread_cond_broadcast(&t->cond);
            pthread_mutex_unlock(&t->mutex);
        }
        // Phase 20: Tune batching
        update_batch_tuning(batch_size);
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
