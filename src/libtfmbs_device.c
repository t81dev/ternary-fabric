#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include "tfmbs_device.h"
#include "fabric_emulator.h"
#include "tfmbs_driver.h"
#include "fabric_net.h"
#include "../include/uapi_tfmbs.h"

static int g_tfmbs_fd = -1;
static int g_initialized = 0;

// Phase 21 & 25: Global Orchestrator
typedef enum { OK_GEMV, OK_LSTM, OK_LSTM_P, OK_ATTN, OK_CONV3D } orch_kernel_t;

typedef struct orch_task {
    orch_kernel_t type;
    void *w, *i, *o, *a;
    int r, c, av;
    uint8_t tile_mask;
    int node_id; // Phase 25
    volatile int dispatched;
    fabric_handle_t emu_handle;
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    struct orch_task* next;
} orch_task_t;

static orch_task_t *g_orch_head = NULL, *g_orch_tail = NULL;
static pthread_mutex_t g_orch_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t g_orch_cond = PTHREAD_COND_INITIALIZER;
static pthread_t g_orch_thread;
static int g_orch_running = 0;

static int g_num_fabrics = 2;
static int g_last_dispatched_fid = 0;
static int g_this_node_id = 0;

// Phase 26: Adaptive Runtime Agent
typedef enum {
    TFMBS_POLICY_OFFLOAD_ALL,
    TFMBS_POLICY_FALLBACK_ALL,
    TFMBS_POLICY_ADAPTIVE_SPARSITY
} tfmbs_policy_t;

static tfmbs_policy_t g_adaptive_policy = TFMBS_POLICY_OFFLOAD_ALL;
static float g_sparsity_threshold = 0.3f;
static float g_ema_alpha = 0.2f;
static float g_avg_sparsity_ema = 0.66f; // Start with typical ternary sparsity
static uint64_t g_fallback_count = 0;
static uint64_t g_offload_count = 0;
static int g_fallback_streak = 0;
#define FALLBACK_PROBE_INTERVAL 10

#define MAX_BUFFERS 2048
static struct {
    void* ptr;
    int fabric_id;
    size_t size;
} g_residency_map[MAX_BUFFERS];
static int g_residency_count = 0;

static void* orchestrator_loop(void* arg);
static int estimate_task_cost(orch_task_t* task, int node_id);

static void init_device() {
    if (g_initialized) return;

    const char* node_env = getenv("TFMBS_NODE_ID");
    if (node_env) g_this_node_id = atoi(node_env);
    fabric_net_init(g_this_node_id);

    const char* num_env = getenv("TFMBS_NUM_FABRICS");
    if (num_env) g_num_fabrics = atoi(num_env);

    if (!g_orch_running) {
        g_orch_running = 1;
        pthread_create(&g_orch_thread, NULL, orchestrator_loop, NULL);
    }

    const char* policy_env = getenv("TFMBS_ADAPTIVE_POLICY");
    if (policy_env) {
        if (strcmp(policy_env, "offload") == 0) g_adaptive_policy = TFMBS_POLICY_OFFLOAD_ALL;
        else if (strcmp(policy_env, "fallback") == 0) g_adaptive_policy = TFMBS_POLICY_FALLBACK_ALL;
        else if (strcmp(policy_env, "sparsity") == 0) g_adaptive_policy = TFMBS_POLICY_ADAPTIVE_SPARSITY;
    }

    const char* thresh_env = getenv("TFMBS_SPARSITY_THRESHOLD");
    if (thresh_env) g_sparsity_threshold = atof(thresh_env);

    const char* alpha_env = getenv("TFMBS_EMA_ALPHA");
    if (alpha_env) g_ema_alpha = atof(alpha_env);

    const char* hw = getenv("FABRIC_HARDWARE_PATH");
    if (hw && hw[0] == '1') {
        g_tfmbs_fd = tfmbs_dev_open("/dev/tfmbs", 0);
        if (g_tfmbs_fd >= 0) {
            printf("[TFMBS-Device] Using Hardware Path (Mock Device)\n");
        } else {
            fprintf(stderr, "[TFMBS-Device] Failed to open hardware path, falling back to Emulator\n");
        }
    }
    g_initialized = 1;
}

static void register_residency(void* ptr, int fabric_id, size_t size) {
    pthread_mutex_lock(&g_orch_mutex);
    for (int i=0; i<g_residency_count; i++) {
        if (g_residency_map[i].ptr == ptr) {
            g_residency_map[i].fabric_id = fabric_id;
            pthread_mutex_unlock(&g_orch_mutex);
            return;
        }
    }
    if (g_residency_count < MAX_BUFFERS) {
        g_residency_map[g_residency_count].ptr = ptr;
        g_residency_map[g_residency_count].fabric_id = fabric_id;
        g_residency_map[g_residency_count].size = size;
        g_residency_count++;
    }
    pthread_mutex_unlock(&g_orch_mutex);
}

static int get_residency(void* ptr, size_t *out_size) {
    int fid = -1;
    pthread_mutex_lock(&g_orch_mutex);
    for (int i=0; i<g_residency_count; i++) {
        if (g_residency_map[i].ptr == ptr) {
            fid = g_residency_map[i].fabric_id;
            if (out_size) *out_size = g_residency_map[i].size;
            break;
        }
    }
    pthread_mutex_unlock(&g_orch_mutex);
    return fid;
}

void* fabric_alloc_id(int fabric_id, size_t size) {
    init_device();
    if (g_tfmbs_fd >= 0) {
        tfmbs_ioc_alloc_t args = { .size = size };
        if (tfmbs_dev_ioctl(g_tfmbs_fd, TFMBS_IOC_ALLOC, &args) == 0) return (void*)args.addr;
    }
    void* ptr = emu_fabric_alloc_id(fabric_id, size);
    if (ptr) register_residency(ptr, fabric_id, size);
    return ptr;
}

void* fabric_alloc(size_t size) {
    init_device();
    // Smart allocation: pick fabric with least number of buffers for now
    // In a more complex emulator, we'd check real pool usage.
    static int next_fid = 0;
    int fid = next_fid;
    next_fid = (next_fid + 1) % g_num_fabrics;
    return fabric_alloc_id(fid, size);
}

void fabric_free(void* ptr) {
    init_device();
    if (g_tfmbs_fd >= 0) {
        tfmbs_ioc_free_t args = { .addr = (uint64_t)ptr };
        tfmbs_dev_ioctl(g_tfmbs_fd, TFMBS_IOC_FREE, &args);
        return;
    }
    emu_fabric_free(ptr);
    pthread_mutex_lock(&g_orch_mutex);
    for (int i=0; i<g_residency_count; i++) {
        if (g_residency_map[i].ptr == ptr) {
            g_residency_map[i] = g_residency_map[g_residency_count-1];
            g_residency_count--;
            break;
        }
    }
    pthread_mutex_unlock(&g_orch_mutex);
}

int is_fabric_ptr(const void* ptr) { return emu_is_fabric_ptr(ptr); }

int fabric_memcpy_to(void* dest_fabric, const void* src_host, size_t size, int pack_pt5) {
    init_device();
    if (g_tfmbs_fd >= 0) {
        tfmbs_ioc_memcpy_to_t args = { .dest_addr = (uint64_t)dest_fabric, .src_host_ptr = src_host, .size = size, .pack_pt5 = pack_pt5 };
        return tfmbs_dev_ioctl(g_tfmbs_fd, TFMBS_IOC_MEMCPY_TO, &args);
    }
    return emu_fabric_memcpy_to(dest_fabric, src_host, size, pack_pt5);
}

int fabric_memcpy_from(void* dest_host, const void* src_fabric, size_t size, int unpack_pt5) {
    init_device();
    if (g_tfmbs_fd >= 0) {
        tfmbs_ioc_memcpy_from_t args = { .dest_host_ptr = dest_host, .src_addr = (uint64_t)src_fabric, .size = size, .unpack_pt5 = unpack_pt5 };
        return tfmbs_dev_ioctl(g_tfmbs_fd, TFMBS_IOC_MEMCPY_FROM, &args);
    }
    return emu_fabric_memcpy_from(dest_host, src_fabric, size, unpack_pt5);
}

int fabric_exec_gemv(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols) {
    init_device();
    fabric_handle_t h = fabric_exec_gemv_async(weight_ptr, input_ptr, output_ptr, rows, cols);
    return fabric_wait(h);
}

int fabric_exec_lstm(void* weight_ptr, void* input_ptr, void* output_ptr, int h_size, int i_size) {
    init_device();
    fabric_handle_t h = fabric_exec_lstm_async(weight_ptr, input_ptr, output_ptr, h_size, i_size);
    return fabric_wait(h);
}

fabric_handle_t fabric_exec_gemv_async_id(int fabric_id, void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols) {
    init_device();
    return emu_fabric_exec_gemv_async_id(fabric_id, weight_ptr, input_ptr, output_ptr, rows, cols, 0x0F);
}

fabric_handle_t fabric_exec_gemv_async(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols) {
    init_device();
    uint8_t tile_mask = 0x0F;
    const char* mask_env = getenv("FABRIC_TILE_MASK");
    if (mask_env) tile_mask = (uint8_t)strtol(mask_env, NULL, 0);

    orch_task_t* task = malloc(sizeof(orch_task_t));
    task->type = OK_GEMV; task->w = weight_ptr; task->i = input_ptr; task->o = output_ptr;
    task->r = rows; task->c = cols; task->tile_mask = tile_mask;
    task->node_id = g_this_node_id;
    const char* node_target = getenv("TFMBS_TARGET_NODE");
    if (node_target) task->node_id = atoi(node_target);

    task->dispatched = 0; task->emu_handle = NULL;
    pthread_mutex_init(&task->mutex, NULL); pthread_cond_init(&task->cond, NULL); task->next = NULL;

    // Adaptive Partitioning: Decide node based on cost
    if (g_this_node_id == 0 && getenv("TFMBS_ENABLE_PARTITIONING")) {
        int cost_local = estimate_task_cost(task, 0);
        int cost_remote = estimate_task_cost(task, 1);
        if (cost_remote < cost_local) task->node_id = 1;
    }

    pthread_mutex_lock(&g_orch_mutex);
    if (g_orch_tail) { g_orch_tail->next = task; g_orch_tail = task; }
    else g_orch_head = g_orch_tail = task;
    pthread_cond_signal(&g_orch_cond); pthread_mutex_unlock(&g_orch_mutex);
    return (fabric_handle_t)task;
}

fabric_handle_t fabric_exec_attn_async(void* q_ptr, void* k_ptr, void* v_ptr, void* o_ptr, int seq_len, int head_dim) {
    init_device();
    uint8_t tile_mask = 0x0F;
    const char* mask_env = getenv("FABRIC_TILE_MASK");
    if (mask_env) tile_mask = (uint8_t)strtol(mask_env, NULL, 0);

    orch_task_t* task = malloc(sizeof(orch_task_t));
    task->type = OK_ATTN; task->w = q_ptr; task->i = k_ptr; task->a = v_ptr; task->o = o_ptr;
    task->r = seq_len; task->c = head_dim; task->tile_mask = tile_mask;
    task->dispatched = 0; task->emu_handle = NULL;
    pthread_mutex_init(&task->mutex, NULL); pthread_cond_init(&task->cond, NULL); task->next = NULL;
    pthread_mutex_lock(&g_orch_mutex);
    if (g_orch_tail) { g_orch_tail->next = task; g_orch_tail = task; }
    else g_orch_head = g_orch_tail = task;
    pthread_cond_signal(&g_orch_cond); pthread_mutex_unlock(&g_orch_mutex);
    return (fabric_handle_t)task;
}

fabric_handle_t fabric_exec_conv3d_async(void* weight_ptr, void* input_ptr, void* output_ptr, int out_c, int in_c, int dhw) {
    init_device();
    uint8_t tile_mask = 0x0F;
    const char* mask_env = getenv("FABRIC_TILE_MASK");
    if (mask_env) tile_mask = (uint8_t)strtol(mask_env, NULL, 0);

    orch_task_t* task = malloc(sizeof(orch_task_t));
    task->type = OK_CONV3D; task->w = weight_ptr; task->i = input_ptr; task->o = output_ptr;
    task->r = out_c; task->c = in_c; task->av = dhw;
    task->tile_mask = tile_mask;
    task->dispatched = 0; task->emu_handle = NULL;
    pthread_mutex_init(&task->mutex, NULL); pthread_cond_init(&task->cond, NULL);
    pthread_mutex_lock(&g_orch_mutex);
    if (g_orch_tail) { g_orch_tail->next = task; g_orch_tail = task; }
    else g_orch_head = g_orch_tail = task;
    pthread_cond_signal(&g_orch_cond); pthread_mutex_unlock(&g_orch_mutex);
    return (fabric_handle_t)task;
}

fabric_handle_t fabric_exec_lstm_async_id(int fabric_id, void* weight_ptr, void* input_ptr, void* output_ptr, int h_size, int i_size) {
    init_device();
    return emu_fabric_exec_lstm_async_id(fabric_id, weight_ptr, input_ptr, output_ptr, h_size, i_size, 0x0F);
}

fabric_handle_t fabric_exec_lstm_async(void* weight_ptr, void* input_ptr, void* output_ptr, int h_size, int i_size) {
    init_device();
    uint8_t tile_mask = 0x0F;
    const char* mask_env = getenv("FABRIC_TILE_MASK");
    if (mask_env) tile_mask = (uint8_t)strtol(mask_env, NULL, 0);

    orch_task_t* task = malloc(sizeof(orch_task_t));
    task->type = OK_LSTM; task->w = weight_ptr; task->i = input_ptr; task->o = output_ptr;
    task->r = h_size; task->c = i_size; task->tile_mask = tile_mask;
    task->dispatched = 0; task->emu_handle = NULL;
    pthread_mutex_init(&task->mutex, NULL); pthread_cond_init(&task->cond, NULL); task->next = NULL;
    pthread_mutex_lock(&g_orch_mutex);
    if (g_orch_tail) { g_orch_tail->next = task; g_orch_tail = task; }
    else g_orch_head = g_orch_tail = task;
    pthread_cond_signal(&g_orch_cond); pthread_mutex_unlock(&g_orch_mutex);
    return (fabric_handle_t)task;
}

void fabric_lstm_bind(void* weight_ptr, void* state_ptr, uint8_t tile_mask) {
    init_device();
    emu_fabric_lstm_bind(weight_ptr, state_ptr, tile_mask);
}

fabric_handle_t fabric_exec_lstm_persistent_async(void* weight_ptr, void* input_ptr, void* state_ptr, int h_size, int i_size, uint8_t tile_mask) {
    init_device();
    orch_task_t* task = malloc(sizeof(orch_task_t));
    task->type = OK_LSTM_P; task->w = weight_ptr; task->i = input_ptr; task->o = state_ptr;
    task->r = h_size; task->c = i_size; task->tile_mask = tile_mask;
    task->dispatched = 0; task->emu_handle = NULL;
    pthread_mutex_init(&task->mutex, NULL); pthread_cond_init(&task->cond, NULL); task->next = NULL;
    pthread_mutex_lock(&g_orch_mutex);
    if (g_orch_tail) { g_orch_tail->next = task; g_orch_tail = task; }
    else g_orch_head = g_orch_tail = task;
    pthread_cond_signal(&g_orch_cond); pthread_mutex_unlock(&g_orch_mutex);
    return (fabric_handle_t)task;
}

void fabric_dump_metrics_csv(const char* path) { emu_fabric_dump_metrics_csv(path); }
void fabric_dump_economic_csv(const char* path) { emu_fabric_dump_economic_csv(path); }

int fabric_wait(fabric_handle_t handle) {
    init_device();
    orch_task_t* task = (orch_task_t*)handle;
    pthread_mutex_lock(&task->mutex);
    while (!task->dispatched) pthread_cond_wait(&task->cond, &task->mutex);
    pthread_mutex_unlock(&task->mutex);
    int res = 0;
    if (task->emu_handle) {
        res = emu_fabric_wait(task->emu_handle);
        // Post-execution: Update EMA from metrics
        fabric_metrics_t m;
        emu_fabric_get_metrics_id(g_last_dispatched_fid, &m);
        if (m.total_ops > 0) {
            float current_sparsity = (float)m.zero_skips / m.total_ops;
            g_avg_sparsity_ema = (1.0f - g_ema_alpha) * g_avg_sparsity_ema + g_ema_alpha * current_sparsity;
        }
    }
    pthread_mutex_destroy(&task->mutex); pthread_cond_destroy(&task->cond); free(task);
    return res;
}

int fabric_submit_tfd(tfmbs_tfd_t* tfd) {
    init_device();
    if (g_tfmbs_fd >= 0) return tfmbs_dev_ioctl(g_tfmbs_fd, TFMBS_IOC_SUBMIT, tfd);
    printf("[TFMBS-Device] Emulator TFD Submit (Partial): Base=0x%lx, Kernel=0x%02x\n", tfd->base_addr, (int)(tfd->exec_hints & TFMBS_HINT_KERNEL_MASK));
    return 0;
}

void fabric_get_metrics(fabric_metrics_t* out_metrics) {
    init_device();
    if (!out_metrics) return;

    if (g_tfmbs_fd >= 0) {
        tfmbs_ioc_metrics_t ioc_m;
        if (tfmbs_dev_ioctl(g_tfmbs_fd, TFMBS_IOC_GET_METRICS, &ioc_m) == 0) {
            out_metrics->zero_skips = ioc_m.zero_skips;
            out_metrics->total_ops = ioc_m.total_ops;
            out_metrics->active_ops = ioc_m.active_ops;
            out_metrics->mem_reads = ioc_m.mem_reads;
            out_metrics->mem_writes = ioc_m.mem_writes;
            out_metrics->residency_hits = ioc_m.residency_hits;
            out_metrics->residency_misses = ioc_m.residency_misses;
            out_metrics->fallback_count = g_fallback_count;
            out_metrics->offload_count = g_offload_count;
            out_metrics->pool_used = ioc_m.pool_used;
            out_metrics->pool_total = ioc_m.pool_total;
            return;
        }
    }

    fabric_metrics_t m;
    emu_fabric_get_metrics_id(g_last_dispatched_fid, &m);
    m.fallback_count = g_fallback_count;
    m.offload_count = g_offload_count;
    *out_metrics = m;
}

static int should_offload_adaptive(orch_task_t* task) {
    (void)task;
    if (g_adaptive_policy == TFMBS_POLICY_OFFLOAD_ALL) return 1;
    if (g_adaptive_policy == TFMBS_POLICY_FALLBACK_ALL) return 0;

    if (g_adaptive_policy == TFMBS_POLICY_ADAPTIVE_SPARSITY) {
        // If EMA is below threshold, fallback to CPU, but probe occasionally
        if (g_avg_sparsity_ema < g_sparsity_threshold) {
            if (g_fallback_streak < FALLBACK_PROBE_INTERVAL) {
                if (getenv("TFMBS_DEBUG")) printf("[TFMBS-Agent] Adaptive Fallback: EMA sparsity %.2f < threshold %.2f (streak %d)\n", g_avg_sparsity_ema, g_sparsity_threshold, g_fallback_streak);
                return 0;
            } else {
                if (getenv("TFMBS_DEBUG")) printf("[TFMBS-Agent] Probing Fabric despite low EMA sparsity %.2f\n", g_avg_sparsity_ema);
                return 1;
            }
        }
    }
    return 1;
}

static void cpu_fallback_gemv(orch_task_t* task) {
    if (getenv("TFMBS_DEBUG")) printf("[TFMBS-Agent] Executing CPU Fallback for GEMV (%dx%d)\n", task->r, task->c);
    // Unpack weights if they are fabric pointers and packed
    // For simplicity in this emulator-based agent, we'll use a helper to read back and compute
    int8_t* w_trits = malloc(task->r * task->c);
    int8_t* i_trits = malloc(task->c);
    int32_t* o_res = (int32_t*)task->o;

    fabric_memcpy_from(w_trits, task->w, task->r * task->c, 1);
    fabric_memcpy_from(i_trits, task->i, task->c, 1);

    for (int r = 0; r < task->r; r++) {
        int32_t acc = 0;
        for (int c = 0; c < task->c; c++) {
            acc += (int32_t)w_trits[r * task->c + c] * (int32_t)i_trits[c];
        }
        o_res[r] = acc;
    }

    free(w_trits);
    free(i_trits);
}

static int estimate_task_cost(orch_task_t* task, int node_id) {
    float sparsity = g_avg_sparsity_ema;
    int lanes = 60;
    int comm_penalty = (node_id != g_this_node_id) ? 5000 : 0;
    long active_ops = (long)(task->r * task->c * (1.0f - sparsity));
    return (int)(active_ops / lanes) + comm_penalty;
}

static void ensure_resident(void* ptr, int best_fid) {
    size_t size = 0;
    int current_fid = get_residency(ptr, &size);
    if (current_fid != -1 && current_fid != best_fid) {
        void* new_ptr = emu_fabric_alloc_id(best_fid, size);
        if (new_ptr) {
            emu_fabric_inter_copy(current_fid, best_fid, ptr, new_ptr, size);
            register_residency(ptr, best_fid, size);
        }
    }
}

static void* orchestrator_loop(void* arg) {
    (void)arg;
    while (g_orch_running) {
        pthread_mutex_lock(&g_orch_mutex);
        while (g_orch_head == NULL) pthread_cond_wait(&g_orch_cond, &g_orch_mutex);
        orch_task_t* task = g_orch_head;
        g_orch_head = g_orch_head->next;
        if (g_orch_head == NULL) g_orch_tail = NULL;

        orch_task_t* lookahead[5]; int la_count = 0;
        if (!getenv("TFMBS_DISABLE_LOOKAHEAD")) {
            orch_task_t* curr = g_orch_head;
            while (curr && la_count < 5) { lookahead[la_count++] = curr; curr = curr->next; }
        }
        pthread_mutex_unlock(&g_orch_mutex);

        // Predictive Scheduler: Window 5
        // If a lookahead task uses the same weight as current task, definitely keep it here.
        int best_fid = 0;
        size_t w_size = 0;
        int w_fid = get_residency(task->w, &w_size);
        if (w_fid != -1) best_fid = w_fid;
        else {
            // Assign based on input residency if weight is new
            int i_fid = get_residency(task->i, NULL);
            if (i_fid != -1) best_fid = i_fid;
            else {
                static int rr = 0;
                best_fid = rr; rr = (rr + 1) % g_num_fabrics;
            }
        }

        // Hot-state pre-loading & Kernel Fusion: check lookahead
        for (int i=0; i<la_count; i++) {
            if (lookahead[i]->w == task->w) {
                // Future task needs same weights, reinforce best_fid
            }
            if (lookahead[i]->i == task->o || lookahead[i]->w == task->o) {
                // Future task depends on CURRENT output.
                // Prioritize keeping this task on best_fid to avoid transfer for lookahead[i].
                if (getenv("TFMBS_DEBUG")) printf("[TFMBS-Orch] Kernel Fusion detected: task producing %p for lookahead[%d]\n", task->o, i);
            }
        }

        // Automatic Transfers
        ensure_resident(task->w, best_fid);
        ensure_resident(task->i, best_fid);

        // Dispatch to Local Emulator, Remote Node, or Fallback
        g_last_dispatched_fid = best_fid;
        int offload = should_offload_adaptive(task);

        if (task->node_id != g_this_node_id) {
            // Phase 25: Remote RDMA Dispatch
            if (getenv("TFMBS_DEBUG")) printf("[TFMBS-Orch] Dispatching task to Remote Node %d via RDMA\n", task->node_id);
            // Simulated: pack task and send
            fabric_net_send(task->node_id, task, sizeof(orch_task_t));
            // In a real system we'd wait for completion over the wire
            task->emu_handle = NULL;
        } else if (offload) {
            g_offload_count++;
            g_fallback_streak = 0;
            if (task->type == OK_GEMV) task->emu_handle = emu_fabric_exec_gemv_async_id(best_fid, task->w, task->i, task->o, task->r, task->c, task->tile_mask);
            else if (task->type == OK_LSTM) task->emu_handle = emu_fabric_exec_lstm_async_id(best_fid, task->w, task->i, task->o, task->r, task->c, task->tile_mask);
            else if (task->type == OK_LSTM_P) task->emu_handle = emu_fabric_exec_lstm_persistent_async(task->w, task->i, task->o, task->r, task->c, task->tile_mask);
            else if (task->type == OK_ATTN) task->emu_handle = emu_fabric_exec_attn_async_id(best_fid, task->w, task->i, task->a, task->o, task->r, task->c, task->tile_mask);
            else if (task->type == OK_CONV3D) task->emu_handle = emu_fabric_exec_conv3d_async_id(best_fid, task->w, task->i, task->o, task->r, task->c, task->av, task->tile_mask);
        } else {
            g_fallback_count++;
            g_fallback_streak++;
            if (task->type == OK_GEMV) cpu_fallback_gemv(task);
            // Other kernels could have fallbacks too, but GEMV is the focus for Phase 26
            task->emu_handle = NULL; // Signal that it's already done
        }

        // Update output residency
        size_t out_sz = (task->type == OK_ATTN) ? (size_t)task->r * task->c * 4 : (size_t)task->r * 4;
        register_residency(task->o, best_fid, out_sz);

        pthread_mutex_lock(&task->mutex);
        task->dispatched = 1;
        pthread_cond_broadcast(&task->cond);
        pthread_mutex_unlock(&task->mutex);
    }
    return NULL;
}
