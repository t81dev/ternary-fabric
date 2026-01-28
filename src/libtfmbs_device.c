#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>
#include <unistd.h>
#include "tfmbs_device.h"
#include "fabric_emulator.h"
#include "tfmbs_driver.h"
#include "../include/uapi_tfmbs.h"

static int g_tfmbs_fd = -1;
static int g_initialized = 0;

// Phase 21: Global Orchestrator
typedef enum { OK_GEMV, OK_LSTM, OK_LSTM_P } orch_kernel_t;

typedef struct orch_task {
    orch_kernel_t type;
    void *w, *i, *o;
    int r, c;
    uint8_t tile_mask;
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

#define MAX_BUFFERS 2048
static struct {
    void* ptr;
    int fabric_id;
    size_t size;
} g_residency_map[MAX_BUFFERS];
static int g_residency_count = 0;

static void* orchestrator_loop(void* arg);

static void init_device() {
    if (g_initialized) return;
    const char* num_env = getenv("TFMBS_NUM_FABRICS");
    if (num_env) g_num_fabrics = atoi(num_env);

    if (!g_orch_running) {
        g_orch_running = 1;
        pthread_create(&g_orch_thread, NULL, orchestrator_loop, NULL);
    }

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
    orch_task_t* task = malloc(sizeof(orch_task_t));
    task->type = OK_GEMV; task->w = weight_ptr; task->i = input_ptr; task->o = output_ptr;
    task->r = rows; task->c = cols; task->tile_mask = 0x0F;
    task->dispatched = 0; task->emu_handle = NULL;
    pthread_mutex_init(&task->mutex, NULL); pthread_cond_init(&task->cond, NULL); task->next = NULL;
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
    orch_task_t* task = malloc(sizeof(orch_task_t));
    task->type = OK_LSTM; task->w = weight_ptr; task->i = input_ptr; task->o = output_ptr;
    task->r = h_size; task->c = i_size; task->tile_mask = 0x0F;
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
    int res = emu_fabric_wait(task->emu_handle);
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
    emu_fabric_get_metrics(out_metrics);
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
        orch_task_t* curr = g_orch_head;
        while (curr && la_count < 5) { lookahead[la_count++] = curr; curr = curr->next; }
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

        // Dispatch to emulator
        if (task->type == OK_GEMV) task->emu_handle = emu_fabric_exec_gemv_async_id(best_fid, task->w, task->i, task->o, task->r, task->c, task->tile_mask);
        else if (task->type == OK_LSTM) task->emu_handle = emu_fabric_exec_lstm_async_id(best_fid, task->w, task->i, task->o, task->r, task->c, task->tile_mask);
        else if (task->type == OK_LSTM_P) task->emu_handle = emu_fabric_exec_lstm_persistent_async(task->w, task->i, task->o, task->r, task->c, task->tile_mask);

        // Update output residency
        register_residency(task->o, best_fid, (task->type == OK_GEMV ? task->r * 4 : task->r * 4));

        pthread_mutex_lock(&task->mutex);
        task->dispatched = 1;
        pthread_cond_broadcast(&task->cond);
        pthread_mutex_unlock(&task->mutex);
    }
    return NULL;
}
