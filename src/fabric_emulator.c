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
#include "fabric_emulator.h"

#define FABRIC_POOL_SIZE (128 * 1024 * 1024)
#define MAX_TILES 8

typedef enum {
    TASK_PENDING,
    TASK_RUNNING,
    TASK_DONE,
    TASK_SHUTDOWN
} task_status_t;

typedef enum {
    KERNEL_GEMV,
    KERNEL_LSTM,
    KERNEL_LSTM_PERSISTENT,
    KERNEL_TRANSFER
} kernel_type_t;

typedef struct fabric_task {
    void *weight_ptr;
    void *input_ptr;
    void *output_ptr;
    int rows;
    int cols;
    kernel_type_t kernel;
    uint32_t exec_hints;
    uint8_t tile_mask;
    double projected_cost;
    volatile task_status_t status;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    struct fabric_task* next;
    int src_fabric_id;
} fabric_task_t;

typedef struct fabric_block {
    void* ptr;
    size_t size;
    int used;
    int busy_count;
    uint32_t last_access;
    uint32_t access_count;
    uint32_t residency_hits;
    int tile_id;
    int pinned;
    int resident;
    struct fabric_block* next;
} fabric_block_t;

typedef struct {
    double weight_cost;
    double mem_read_cost;
    double mem_write_cost;
    double broadcast_cost;
    double residency_miss_cost;
} tfmbs_projection_params_t;

typedef struct {
    double tile_kernel_mult[MAX_TILES][4];
    int    kernel_exec_count[MAX_TILES][4];
    double eviction_freq_weight;
    double eviction_age_weight;
    double eviction_success_weight;
    int    dynamic_batch_size;
    double avg_efficiency_ema;
    double avg_throughput_ema;
} tfmbs_learning_state_t;

typedef struct fabric_instance {
    int id;
    uint8_t* pool_host;
    uint8_t* pool_device;
    fabric_block_t* blocks;
    uint32_t access_counter;
    pthread_mutex_t mutex;
    fabric_metrics_t last_metrics;
    fabric_metrics_t tile_metrics[MAX_TILES];

    double last_projected_cost;
    double last_residency_rebate;
    int last_chosen_tile_id;
    uint8_t last_tile_mask;
    char last_kernel_name[32];
    char last_eviction_scores[512];

    tfmbs_projection_params_t proj_params;
    tfmbs_learning_state_t learn_state;

    fabric_task_t *queue_head, *queue_tail;
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_cond;
    pthread_t worker_thread;
    bool worker_running;

    int pipeline_depth;
} fabric_instance_t;

static fabric_instance_t* g_fabrics = NULL;
static int g_num_fabrics = 0;
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;

static void* fabric_worker_loop(void* arg);

static void init_fabric_instance(fabric_instance_t* inst, int id) {
    inst->id = id;
    int fd = -1;
    char name[64];
    snprintf(name, sizeof(name), "tfmbs_pool_%d", id);

#if defined(__linux__)
    fd = memfd_create(name, 0);
#else
    char temp_path[] = "/tmp/tfmbs_pool_XXXXXX";
    fd = mkstemp(temp_path);
    if (fd >= 0) unlink(temp_path);
#endif

    if (fd >= 0) {
        if (ftruncate(fd, FABRIC_POOL_SIZE) < 0) {
            perror("ftruncate failed");
        }
    }

    inst->pool_host = (uint8_t*)mmap(NULL, FABRIC_POOL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    inst->pool_device = (uint8_t*)mmap(NULL, FABRIC_POOL_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    if (inst->pool_host != MAP_FAILED) {
        __builtin_memset(inst->pool_host, 0, FABRIC_POOL_SIZE);
    }

    inst->blocks = (fabric_block_t*)calloc(1, sizeof(fabric_block_t));
    inst->blocks->ptr = (inst->pool_host == MAP_FAILED) ? NULL : inst->pool_host;
    inst->blocks->size = FABRIC_POOL_SIZE;
    inst->access_counter = 0;
    pthread_mutex_init(&inst->mutex, NULL);
    inst->last_metrics.pool_total = FABRIC_POOL_SIZE;

    inst->proj_params.weight_cost = 1.0;
    inst->proj_params.mem_read_cost = 5.0;
    inst->proj_params.mem_write_cost = 8.0;
    inst->proj_params.broadcast_cost = 2.0;
    inst->proj_params.residency_miss_cost = 6.0;

    for (int i = 0; i < MAX_TILES; i++) {
        for (int k = 0; k < 4; k++) {
            inst->learn_state.tile_kernel_mult[i][k] = 1.0;
        }
    }
    inst->learn_state.eviction_freq_weight = 1.0;
    inst->learn_state.eviction_age_weight = 1.0;
    inst->learn_state.eviction_success_weight = 2.0;
    inst->learn_state.dynamic_batch_size = 8;
    inst->learn_state.avg_efficiency_ema = 0.0;
    inst->learn_state.avg_throughput_ema = 0.0;

    inst->queue_head = inst->queue_tail = NULL;
    pthread_mutex_init(&inst->queue_mutex, NULL);
    pthread_cond_init(&inst->queue_cond, NULL);
    inst->worker_running = true;
    inst->pipeline_depth = 1;
    pthread_create(&inst->worker_thread, NULL, fabric_worker_loop, inst);

    if (fd >= 0) close(fd);
}

void emu_fabric_init() {
    pthread_mutex_lock(&g_init_mutex);
    if (g_fabrics) {
        pthread_mutex_unlock(&g_init_mutex);
        return;
    }
    const char* num_env = getenv("TFMBS_NUM_FABRICS");
    g_num_fabrics = num_env ? atoi(num_env) : 2;
    if (g_num_fabrics < 1) g_num_fabrics = 1;
    if (g_num_fabrics > 16) g_num_fabrics = 16;

    g_fabrics = (fabric_instance_t*)calloc(g_num_fabrics, sizeof(fabric_instance_t));
    srand(time(NULL));
    for (int i = 0; i < g_num_fabrics; i++) {
        init_fabric_instance(&g_fabrics[i], i);
    }
    pthread_mutex_unlock(&g_init_mutex);
}

static fabric_instance_t* get_inst(int id) {
    emu_fabric_init();
    return &g_fabrics[id % g_num_fabrics];
}

static fabric_instance_t* find_inst_by_ptr(const void* ptr) {
    emu_fabric_init();
    for (int i = 0; i < g_num_fabrics; i++) {
        if ((uint8_t*)ptr >= g_fabrics[i].pool_host && (uint8_t*)ptr < g_fabrics[i].pool_host + FABRIC_POOL_SIZE) {
            return &g_fabrics[i];
        }
    }
    return NULL;
}

static fabric_block_t* find_block(fabric_instance_t* inst, void* ptr) {
    fabric_block_t* curr = inst->blocks;
    while (curr) {
        if (curr->ptr == ptr) return curr;
        curr = curr->next;
    }
    return NULL;
}

static void update_access(fabric_instance_t* inst, void* ptr) {
    fabric_block_t* curr = find_block(inst, ptr);
    if (curr) {
        curr->last_access = ++inst->access_counter;
        curr->access_count++;
    }
}

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

static double projected_cost_inst(fabric_instance_t* inst, int tile_id, kernel_type_t kernel, void* w_ptr, void* i_ptr, void* o_ptr __attribute__((unused)), double *out_rebate) {
    fabric_block_t *bw = find_block(inst, w_ptr);
    fabric_block_t *bi = find_block(inst, i_ptr);

    int w_miss = (bw && bw->resident && bw->tile_id == tile_id) ? 0 : 1;
    int i_miss = (bi && bi->resident && bi->tile_id == tile_id) ? 0 : 1;

    double cost = 1000.0 * inst->proj_params.weight_cost + (w_miss ? 500.0 : 0.0 + i_miss ? 100.0 : 0.0) * inst->proj_params.mem_read_cost;
    if (w_miss || i_miss) cost += inst->proj_params.residency_miss_cost * 10.0;

    if (out_rebate) *out_rebate = (w_miss ? 0.0 : 50.0) + (i_miss ? 0.0 : 10.0);

    cost *= inst->learn_state.tile_kernel_mult[tile_id][kernel];

    if (bw && bw->tile_id == tile_id) {
        cost -= 0.5;
    }

    return cost;
}

static uint8_t tfmbs_select_tiles_inst(fabric_instance_t* inst, kernel_type_t kernel, void* w_ptr, void* i_ptr, void* o_ptr, int desired_count) {
    double costs[MAX_TILES];
    int indices[MAX_TILES];

    for (int i = 0; i < MAX_TILES; i++) {
        costs[i] = projected_cost_inst(inst, i, kernel, w_ptr, i_ptr, o_ptr, NULL);
        indices[i] = i;
    }

    for (int i = 0; i < MAX_TILES - 1; i++) {
        for (int j = 0; j < MAX_TILES - i - 1; j++) {
            if (costs[j] > costs[j+1]) {
                double tc = costs[j]; costs[j] = costs[j+1]; costs[j+1] = tc;
                int ti = indices[j]; indices[j] = indices[j+1]; indices[j+1] = ti;
            }
        }
    }

    inst->last_chosen_tile_id = indices[0];
    inst->last_projected_cost = costs[0];

    uint8_t mask = 0;
    int count = (desired_count > MAX_TILES) ? MAX_TILES : desired_count;
    if (count < 1) count = 1;

    for (int i = 0; i < count; i++) {
        mask |= (1 << indices[i]);
    }

    inst->last_tile_mask = mask;
    return mask;
}

static void track_residency_inst(fabric_instance_t* inst, fabric_block_t* b, uint8_t tile_mask) {
    if (!b) return;
    bool hit_any = false;
    int first_tile = -1;

    for (int i = 0; i < MAX_TILES; i++) {
        if (tile_mask & (1 << i)) {
            if (first_tile == -1) first_tile = i;
            if (b->resident && (b->tile_id == i || b->tile_id == -1)) {
                inst->tile_metrics[i].residency_hits++;
                b->residency_hits++;
                hit_any = true;
            } else {
                inst->tile_metrics[i].residency_misses++;
            }
        }
    }

    if (hit_any) {
        inst->last_metrics.residency_hits++;
    } else {
        inst->last_metrics.residency_misses++;
        b->resident = 1;
        if (b->tile_id < 0) b->tile_id = first_tile;
    }
}

void* emu_fabric_alloc_id(int fabric_id, size_t size) {
    fabric_instance_t* inst = get_inst(fabric_id);
    pthread_mutex_lock(&inst->mutex);
    size_t ps = 4096;
    size_t aligned_size = (size + ps - 1) & ~(ps - 1);
    fabric_block_t* curr = inst->blocks;
    while (curr) {
        if (!curr->used && curr->size >= aligned_size) {
            if (curr->size > aligned_size + ps) {
                fabric_block_t* nb = (fabric_block_t*)calloc(1, sizeof(fabric_block_t));
                nb->ptr = (uint8_t*)curr->ptr + aligned_size;
                nb->size = curr->size - aligned_size;
                nb->next = curr->next;
                curr->size = aligned_size;
                curr->next = nb;
            }
            curr->used = 1;
            curr->resident = 1;
            curr->last_access = ++inst->access_counter;
            curr->access_count = 1;
            inst->last_metrics.pool_used += curr->size;
            pthread_mutex_unlock(&inst->mutex);
            return curr->ptr;
        }
        curr = curr->next;
    }
    pthread_mutex_unlock(&inst->mutex);
    return NULL;
}

void* emu_fabric_alloc(size_t size) {
    return emu_fabric_alloc_id(0, size);
}

void emu_fabric_free(void* ptr) {
    fabric_instance_t* inst = find_inst_by_ptr(ptr);
    if (!inst) return;
    pthread_mutex_lock(&inst->mutex);
    fabric_block_t* curr = inst->blocks;
    while (curr) {
        if (curr->ptr == ptr) {
            if (curr->used) {
                curr->used = 0;
                inst->last_metrics.pool_used -= curr->size;
            }
            break;
        }
        curr = curr->next;
    }
    pthread_mutex_unlock(&inst->mutex);
}

int emu_is_fabric_ptr(const void* ptr) {
    return find_inst_by_ptr(ptr) != NULL;
}

static void* to_device_ptr(fabric_instance_t* inst, void* hp) {
    if (hp && find_inst_by_ptr(hp) == inst) {
        return inst->pool_device + ((uint8_t*)hp - inst->pool_host);
    }
    return hp;
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

static void unpack_byte_to_trits(uint8_t bv, int8_t ts[5]) {
    if (bv < 243) __builtin_memcpy(ts, pt5_unpack_table[bv], 5);
    else memset(ts, 0, 5);
}

int emu_fabric_memcpy_to(void* df, const void* sh, size_t sz, int pk) {
    fabric_instance_t* inst = find_inst_by_ptr(df);
    if (!inst) return -1;
    pthread_mutex_lock(&inst->mutex);
    update_access(inst, df);
    fabric_block_t *b = find_block(inst, df);
    if (b) b->resident = 1;
    pthread_mutex_unlock(&inst->mutex);

    if (pk) {
        size_t nt = sz;
        const int8_t* s = (const int8_t*)sh;
        uint8_t* d = (uint8_t*)df;
        static const uint8_t p3_0[] = {0, 1, 2}, p3_1[] = {0, 3, 6}, p3_2[] = {0, 9, 18}, p3_3[] = {0, 27, 54}, p3_4[] = {0, 81, 162};
        for (size_t i = 0; i < nt / 5; i++) {
            d[i] = p3_0[s[i*5]+1] + p3_1[s[i*5+1]+1] + p3_2[s[i*5+2]+1] + p3_3[s[i*5+3]+1] + p3_4[s[i*5+4]+1];
        }
    } else __builtin_memcpy(df, sh, sz);
    return 0;
}

int emu_fabric_memcpy_from(void* dh, const void* sf, size_t sz, int upk) {
    fabric_instance_t* inst = find_inst_by_ptr(sf);
    if (!inst) return -1;
    if (upk) {
        size_t nt = sz;
        const uint8_t* s = (const uint8_t*)sf;
        int8_t* d = (int8_t*)dh;
        for (size_t i = 0; i < (nt+4)/5; i++) {
            int8_t ts[5];
            unpack_byte_to_trits(s[i], ts);
            for (int j=0; j<5; j++) if (i*5+j < nt) d[i*5+j] = ts[j];
        }
    } else __builtin_memcpy(dh, sf, sz);
    return 0;
}

static int8_t g_w_trits[8000000], g_i_trits[8000000];

static int internal_exec_lstm(fabric_instance_t* inst, void* weight_ptr, void* input_ptr, void* output_ptr, int h_size, int i_size, uint8_t tile_mask, int persistent, uint32_t hints) {
    strncpy(inst->last_kernel_name, persistent ? "LSTM-P" : "LSTM", 32);
    void* dev_weight_ptr = to_device_ptr(inst, weight_ptr);
    void* dev_input_ptr = to_device_ptr(inst, input_ptr);
    void* dev_output_ptr = to_device_ptr(inst, output_ptr);
    int rows = 4 * h_size;
    int cols = i_size + h_size;

    int active_tiles = 0;
    for (int i = 0; i < 8; i++) if (tile_mask & (1 << i)) active_tiles++;
    if (active_tiles == 0) active_tiles = 1;

    pthread_mutex_lock(&inst->mutex);
    fabric_block_t *bw = find_block(inst, weight_ptr);
    fabric_block_t *bi = find_block(inst, input_ptr);
    fabric_block_t *bo = find_block(inst, output_ptr);
    track_residency_inst(inst, bw, tile_mask);
    track_residency_inst(inst, bi, tile_mask);
    track_residency_inst(inst, bo, tile_mask);
    pthread_mutex_unlock(&inst->mutex);

    uint8_t* w_packed = (uint8_t*)dev_weight_ptr;
    for (int i = 0; i < (rows * cols + 4) / 5 && i * 5 < 8000000; i++) {
        int8_t trits[5]; unpack_byte_to_trits(w_packed[i], trits);
        for (int j = 0; j < 5; j++) if (i * 5 + j < rows * cols) g_w_trits[i * 5 + j] = trits[j];
    }
    if (persistent) {
        uint8_t* x_packed = (uint8_t*)dev_input_ptr;
        for (int i = 0; i < (i_size + 4) / 5 && i * 5 < 8000000; i++) {
            int8_t trits[5]; unpack_byte_to_trits(x_packed[i], trits);
            for (int j = 0; j < 5; j++) if (i * 5 + j < i_size) g_i_trits[i * 5 + j] = trits[j];
        }
        int32_t* h_prev = (int32_t*)dev_output_ptr;
        for (int j = 0; j < h_size; j++) g_i_trits[i_size + j] = (h_prev[j] > 0) ? 1 : (h_prev[j] < 0 ? -1 : 0);
    } else {
        uint8_t* i_packed = (uint8_t*)dev_input_ptr;
        for (int i = 0; i < (cols + 4) / 5 && i * 5 < 8000000; i++) {
            int8_t trits[5]; unpack_byte_to_trits(i_packed[i], trits);
            for (int j = 0; j < 5; j++) if (i * 5 + j < cols) g_i_trits[i * 5 + j] = trits[j];
        }
    }
    inst->last_metrics.zero_skips = 0;
    inst->last_metrics.total_ops = (long)rows * cols;
    inst->last_metrics.mem_reads = (cols / 5);
    if (bw && bw->pinned && bw->resident) {} else inst->last_metrics.mem_reads += (rows * cols) / 5;
    inst->last_metrics.mem_writes = rows * 4;

    int32_t* results = (int32_t*)dev_output_ptr;
    for (int r = 0; r < rows; r++) {
        int32_t acc = 0;
        for (int c = 0; c < cols; c++) {
            int8_t w = g_w_trits[r * cols + c];
            int8_t x = g_i_trits[c];
            if (w == 0 || x == 0) inst->last_metrics.zero_skips++;
            else acc += (int32_t)w * (int32_t)x;
        }
        results[r] = acc;
    }
    inst->last_metrics.active_ops = inst->last_metrics.total_ops - inst->last_metrics.zero_skips;
    inst->last_metrics.fabric_cost = tfmbs_compute_cost(&inst->last_metrics, hints);
    if (inst->last_metrics.fabric_cost > 0)
        inst->last_metrics.economic_efficiency = (double)inst->last_metrics.active_ops / inst->last_metrics.fabric_cost;
    return 0;
}

static int internal_exec_gemv(fabric_instance_t* inst, void* w_ptr, void* i_ptr, void* o_ptr, int rows, int cols, uint8_t tile_mask, uint32_t hints) {
    strncpy(inst->last_kernel_name, "GEMV", 32);
    void* dw = to_device_ptr(inst, w_ptr);
    void* di = to_device_ptr(inst, i_ptr);
    void* do_ = to_device_ptr(inst, o_ptr);
    pthread_mutex_lock(&inst->mutex);
    track_residency_inst(inst, find_block(inst, w_ptr), tile_mask);
    track_residency_inst(inst, find_block(inst, i_ptr), tile_mask);
    pthread_mutex_unlock(&inst->mutex);

    uint8_t* wp = (uint8_t*)dw;
    for (int i=0; i<(rows*cols+4)/5; i++) {
        int8_t ts[5]; unpack_byte_to_trits(wp[i], ts);
        for (int j=0; j<5; j++) if (i*5+j < rows*cols) g_w_trits[i*5+j] = ts[j];
    }
    uint8_t* ip = (uint8_t*)di;
    for (int i=0; i<(cols+4)/5; i++) {
        int8_t ts[5]; unpack_byte_to_trits(ip[i], ts);
        for (int j=0; j<5; j++) if (i*5+j < cols) g_i_trits[i*5+j] = ts[j];
    }
    inst->last_metrics.total_ops = (long)rows * cols;
    inst->last_metrics.zero_skips = 0;
    int32_t* res = (int32_t*)do_;
    for (int r=0; r<rows; r++) {
        int32_t acc=0;
        for (int c=0; c<cols; c++) {
            int8_t w=g_w_trits[r*cols+c], x=g_i_trits[c];
            if (!w || !x) inst->last_metrics.zero_skips++;
            else acc += (int32_t)w * (int32_t)x;
        }
        res[r] = acc;
    }
    inst->last_metrics.active_ops = inst->last_metrics.total_ops - inst->last_metrics.zero_skips;
    inst->last_metrics.fabric_cost = tfmbs_compute_cost(&inst->last_metrics, hints);
    if (inst->last_metrics.fabric_cost > 0)
        inst->last_metrics.economic_efficiency = (double)inst->last_metrics.active_ops / inst->last_metrics.fabric_cost;
    return 0;
}

static void update_learning(fabric_instance_t* inst, fabric_task_t* task, fabric_metrics_t* metrics) {
    double delta = metrics->fabric_cost - task->projected_cost;
    double lr = 0.05;
    pthread_mutex_lock(&inst->mutex);
    inst->proj_params.weight_cost         += lr * delta * 0.05;
    inst->proj_params.mem_read_cost       += lr * delta * 0.1;
    inst->proj_params.mem_write_cost      += lr * delta * 0.1;
    inst->proj_params.broadcast_cost      += lr * delta * 0.05;
    inst->proj_params.residency_miss_cost += lr * delta * 0.2;

    int k_idx = task->kernel;
    for (int i = 0; i < MAX_TILES; i++) {
        if (task->tile_mask & (1 << i)) {
            inst->learn_state.kernel_exec_count[i][k_idx]++;
            double current_eff = metrics->economic_efficiency;
            if (inst->learn_state.avg_efficiency_ema == 0) inst->learn_state.avg_efficiency_ema = current_eff;
            if (current_eff > inst->learn_state.avg_efficiency_ema) inst->learn_state.tile_kernel_mult[i][k_idx] *= 0.98;
            else inst->learn_state.tile_kernel_mult[i][k_idx] *= 1.02;
        }
    }
    inst->learn_state.avg_efficiency_ema = 0.9 * inst->learn_state.avg_efficiency_ema + 0.1 * metrics->economic_efficiency;
    pthread_mutex_unlock(&inst->mutex);
}

static void update_batch_tuning(fabric_instance_t* inst, int last_batch_size) {
    pthread_mutex_lock(&inst->mutex);
    // Simple batch tuning logic restored
    if (inst->learn_state.avg_efficiency_ema > 0.5) inst->learn_state.dynamic_batch_size++;
    else if (inst->learn_state.avg_efficiency_ema < 0.2) inst->learn_state.dynamic_batch_size--;
    if (inst->learn_state.dynamic_batch_size < 1) inst->learn_state.dynamic_batch_size = 1;
    if (inst->learn_state.dynamic_batch_size > 32) inst->learn_state.dynamic_batch_size = 32;
    pthread_mutex_unlock(&inst->mutex);
}

int emu_fabric_exec_gemv(void* w, void* i, void* o, int r, int c, uint8_t tm) {
    fabric_instance_t* inst = find_inst_by_ptr(w);
    if (!inst) return -1;
    uint8_t sm = tfmbs_select_tiles_inst(inst, KERNEL_GEMV, w, i, o, 4);
    return internal_exec_gemv(inst, w, i, o, r, c, sm, 0);
}

fabric_handle_t emu_fabric_exec_gemv_async_id(int fid, void* w, void* i, void* o, int r, int c, uint8_t tm) {
    fabric_instance_t* inst = get_inst(fid);
    fabric_task_t* t = (fabric_task_t*)calloc(1, sizeof(fabric_task_t));
    t->weight_ptr=w; t->input_ptr=i; t->output_ptr=o; t->rows=r; t->cols=c; t->kernel=KERNEL_GEMV; t->tile_mask=tm; t->status=TASK_PENDING;
    t->projected_cost = inst->last_projected_cost;
    pthread_mutex_init(&t->mutex, NULL); pthread_cond_init(&t->cond, NULL);
    pthread_mutex_lock(&inst->queue_mutex);
    if (inst->queue_tail) inst->queue_tail->next = t; else inst->queue_head = t;
    inst->queue_tail = t;
    pthread_cond_signal(&inst->queue_cond);
    pthread_mutex_unlock(&inst->queue_mutex);
    return (fabric_handle_t)t;
}

fabric_handle_t emu_fabric_exec_gemv_async(void* w, void* i, void* o, int r, int c, uint8_t tm) {
    fabric_instance_t* inst = find_inst_by_ptr(w);
    return emu_fabric_exec_gemv_async_id(inst ? inst->id : 0, w, i, o, r, c, tm);
}

int emu_fabric_wait(fabric_handle_t h) {
    if (!h) return -1;
    fabric_task_t* t = (fabric_task_t*)h;
    pthread_mutex_lock(&t->mutex);
    while (t->status != TASK_DONE) pthread_cond_wait(&t->cond, &t->mutex);
    pthread_mutex_unlock(&t->mutex);
    pthread_mutex_destroy(&t->mutex);
    pthread_cond_destroy(&t->cond);
    free(t);
    return 0;
}

int emu_fabric_inter_copy(int sid, int did, void* sp, void* dp, size_t sz) {
    (void)sid; (void)did;
    __builtin_memcpy(dp, sp, sz);
    return 0;
}

void emu_fabric_dump_economic_csv(const char* p) {
    emu_fabric_init();
    FILE* f = fopen(p, "a");
    if (!f) return;
    fprintf(f, "step,fid,kernel,projected_cost,efficiency\n");
    for (int i=0; i<g_num_fabrics; i++) {
        fprintf(f, "0,%d,%s,%.2f,%.4f\n", i, g_fabrics[i].last_kernel_name, g_fabrics[i].last_projected_cost, g_fabrics[i].last_metrics.economic_efficiency);
    }
    fclose(f);
}

static void* fabric_worker_loop(void* arg) {
    fabric_instance_t* inst = (fabric_instance_t*)arg;

    typedef struct {
        fabric_task_t* tasks[32];
        int size;
    } batch_stage_t;

    batch_stage_t prefetch_stage = {.size = 0};
    batch_stage_t execute_stage = {.size = 0};
    batch_stage_t commit_stage = {.size = 0};

    while (inst->worker_running) {
        // Stage 3: Commit
        for (int b = 0; b < commit_stage.size; b++) {
            fabric_task_t* t = commit_stage.tasks[b];
            pthread_mutex_lock(&t->mutex);
            t->status = TASK_DONE;
            pthread_cond_broadcast(&t->cond);
            pthread_mutex_unlock(&t->mutex);
        }
        commit_stage.size = 0;

        // Stage 2: Execute
        if (execute_stage.size > 0) {
            for (int b = 0; b < execute_stage.size; b++) {
                fabric_task_t* t = execute_stage.tasks[b];
                pthread_mutex_lock(&inst->mutex);
                update_access(inst, t->weight_ptr); update_access(inst, t->input_ptr); update_access(inst, t->output_ptr);
                pthread_mutex_unlock(&inst->mutex);

                if (t->kernel == KERNEL_GEMV) internal_exec_gemv(inst, t->weight_ptr, t->input_ptr, t->output_ptr, t->rows, t->cols, t->tile_mask, t->exec_hints);
                else if (t->kernel == KERNEL_LSTM) internal_exec_lstm(inst, t->weight_ptr, t->input_ptr, t->output_ptr, t->rows, t->cols, t->tile_mask, 0, t->exec_hints);
                else if (t->kernel == KERNEL_LSTM_PERSISTENT) internal_exec_lstm(inst, t->weight_ptr, t->input_ptr, t->output_ptr, t->rows, t->cols, t->tile_mask, 1, t->exec_hints);
                else if (t->kernel == KERNEL_TRANSFER) emu_fabric_inter_copy(t->src_fabric_id, inst->id, t->input_ptr, t->output_ptr, t->rows);

                update_learning(inst, t, &inst->last_metrics);
            }
            memcpy(&commit_stage, &execute_stage, sizeof(batch_stage_t));
            execute_stage.size = 0;
            update_batch_tuning(inst, commit_stage.size);
        }

        // Stage 1: Pre-fetch
        if (prefetch_stage.size > 0) {
            usleep(100);
            memcpy(&execute_stage, &prefetch_stage, sizeof(batch_stage_t));
            prefetch_stage.size = 0;
        }

        pthread_mutex_lock(&inst->queue_mutex);
        if (inst->queue_head == NULL && execute_stage.size == 0 && commit_stage.size == 0) {
             pthread_cond_wait(&inst->queue_cond, &inst->queue_mutex);
        }

        if (inst->queue_head) {
            int max_b = inst->learn_state.dynamic_batch_size;
            while (inst->queue_head && prefetch_stage.size < max_b) {
                fabric_task_t* t = inst->queue_head;
                if (t->status == TASK_SHUTDOWN) { inst->worker_running = false; break; }
                if (prefetch_stage.size > 0 && t->kernel != prefetch_stage.tasks[0]->kernel) break;

                t->status = TASK_RUNNING;
                prefetch_stage.tasks[prefetch_stage.size++] = t;
                inst->queue_head = t->next;
                if (inst->queue_head == NULL) inst->queue_tail = NULL;
            }
            if (inst->last_metrics.semantic_efficiency > 0.5) inst->pipeline_depth = 3;
            else inst->pipeline_depth = 1;
        }
        pthread_mutex_unlock(&inst->queue_mutex);
    }
    return NULL;
}

void emu_fabric_get_metrics(fabric_metrics_t* m) { emu_fabric_init(); if (m) *m = g_fabrics[0].last_metrics; }
void emu_fabric_dump_metrics_csv(const char* p) { emu_fabric_dump_economic_csv(p); }
void emu_fabric_lstm_bind(void* w, void* s, uint8_t tm) { (void)w;(void)s;(void)tm; }
fabric_handle_t emu_fabric_exec_lstm_async_id(int fid, void* w, void* i, void* o, int h, int is, uint8_t tm) {
    fabric_instance_t* inst = get_inst(fid);
    fabric_task_t* t = (fabric_task_t*)calloc(1, sizeof(fabric_task_t));
    t->weight_ptr=w; t->input_ptr=i; t->output_ptr=o; t->rows=h; t->cols=is; t->kernel=KERNEL_LSTM; t->tile_mask=tm; t->status=TASK_PENDING;
    pthread_mutex_init(&t->mutex, NULL); pthread_cond_init(&t->cond, NULL);
    pthread_mutex_lock(&inst->queue_mutex); if (inst->queue_tail) inst->queue_tail->next = t; else inst->queue_head = t; inst->queue_tail = t;
    pthread_cond_signal(&inst->queue_cond); pthread_mutex_unlock(&inst->queue_mutex);
    return (fabric_handle_t)t;
}
fabric_handle_t emu_fabric_exec_lstm_async(void* w, void* i, void* o, int h, int is, uint8_t tm) { fabric_instance_t* inst = find_inst_by_ptr(w); return emu_fabric_exec_lstm_async_id(inst?inst->id:0, w, i, o, h, is, tm); }
fabric_handle_t emu_fabric_exec_lstm_persistent_async(void* w, void* i, void* s, int h, int is, uint8_t tm) {
    fabric_instance_t* inst = find_inst_by_ptr(w);
    int fid = inst ? inst->id : 0;
    fabric_instance_t* target = get_inst(fid);
    fabric_task_t* t = (fabric_task_t*)calloc(1, sizeof(fabric_task_t));
    t->weight_ptr=w; t->input_ptr=i; t->output_ptr=s; t->rows=h; t->cols=is; t->kernel=KERNEL_LSTM_PERSISTENT; t->tile_mask=tm; t->status=TASK_PENDING;
    pthread_mutex_init(&t->mutex, NULL); pthread_cond_init(&t->cond, NULL);
    pthread_mutex_lock(&target->queue_mutex); if (target->queue_tail) target->queue_tail->next = t; else target->queue_head = t; target->queue_tail = t;
    pthread_cond_signal(&target->queue_cond); pthread_mutex_unlock(&target->queue_mutex);
    return (fabric_handle_t)t;
}
int emu_fabric_exec_lstm(void* w, void* i, void* o, int h, int is, uint8_t tm) {
    fabric_instance_t* inst = find_inst_by_ptr(w);
    if (!inst) return -1;
    uint8_t sm = tfmbs_select_tiles_inst(inst, KERNEL_LSTM, w, i, o, 4);
    return internal_exec_lstm(inst, w, i, o, h, is, sm, 0, 0);
}
