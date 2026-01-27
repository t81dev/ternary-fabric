#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "tfmbs_device.h"
#include "fabric_emulator.h"
#include "tfmbs_driver.h"
#include "../include/uapi_tfmbs.h"

static int g_tfmbs_fd = -1;
static int g_initialized = 0;

static void init_device() {
    if (g_initialized) return;
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

void* fabric_alloc(size_t size) {
    init_device();
    if (g_tfmbs_fd >= 0) {
        tfmbs_ioc_alloc_t args = { .size = size };
        if (tfmbs_dev_ioctl(g_tfmbs_fd, TFMBS_IOC_ALLOC, &args) == 0) {
            return (void*)args.addr;
        }
    }
    return emu_fabric_alloc(size);
}

void fabric_free(void* ptr) {
    init_device();
    if (g_tfmbs_fd >= 0) {
        tfmbs_ioc_free_t args = { .addr = (uint64_t)ptr };
        tfmbs_dev_ioctl(g_tfmbs_fd, TFMBS_IOC_FREE, &args);
        return;
    }
    emu_fabric_free(ptr);
}

int is_fabric_ptr(const void* ptr) {
    // Both driver and emulator share the same "fabric" memory space in this mock
    return emu_is_fabric_ptr(ptr);
}

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
    if (g_tfmbs_fd >= 0) {
        // For simplicity, we use the async path in the ioctl and wait immediately
        fabric_handle_t h = fabric_exec_gemv_async(weight_ptr, input_ptr, output_ptr, rows, cols);
        fabric_wait(h);
        return 0;
    }
    return emu_fabric_exec_gemv(weight_ptr, input_ptr, output_ptr, rows, cols);
}

fabric_handle_t fabric_exec_gemv_async(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols) {
    init_device();
    if (g_tfmbs_fd >= 0) {
        tfmbs_ioc_submit_gemv_t args = { .weight_addr = (uint64_t)weight_ptr, .input_addr = (uint64_t)input_ptr, .output_addr = (uint64_t)output_ptr, .rows = rows, .cols = cols };
        if (tfmbs_dev_ioctl(g_tfmbs_fd, TFMBS_IOC_SUBMIT_GEMV, &args) == 0) {
            return (fabric_handle_t)args.handle;
        }
        return NULL;
    }
    return emu_fabric_exec_gemv_async(weight_ptr, input_ptr, output_ptr, rows, cols);
}

void fabric_wait(fabric_handle_t handle) {
    init_device();
    if (g_tfmbs_fd >= 0) {
        tfmbs_ioc_wait_t args = { .handle = (uint64_t)handle };
        tfmbs_dev_ioctl(g_tfmbs_fd, TFMBS_IOC_WAIT, &args);
        return;
    }
    emu_fabric_wait(handle);
}

void fabric_get_metrics(fabric_metrics_t* out_metrics) {
    init_device();
    if (g_tfmbs_fd >= 0 && out_metrics) {
        tfmbs_ioc_metrics_t args;
        if (tfmbs_dev_ioctl(g_tfmbs_fd, TFMBS_IOC_GET_METRICS, &args) == 0) {
            out_metrics->zero_skips = args.zero_skips;
            out_metrics->total_ops = args.total_ops;
            out_metrics->pool_used = args.pool_used;
            out_metrics->pool_total = args.pool_total;
            out_metrics->eviction_count = args.evictions;
            // sim_cycle_reduction can be calculated
            if (args.total_ops > 0)
                out_metrics->sim_cycle_reduction = (double)args.zero_skips / args.total_ops * 100.0;
            return;
        }
    }
    emu_fabric_get_metrics(out_metrics);
}
