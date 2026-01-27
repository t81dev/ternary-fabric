#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "tfmbs_device.h"
#include "fabric_emulator.h"
#include "../include/uapi_tfmbs.h"

/**
 * @brief Mock implementation of a TFMBS kernel driver.
 * In a real system, this would be a Linux kernel module.
 * Here, it wraps the userspace libtfmbs_device emulation.
 */

int tfmbs_dev_open(const char* path, int flags) {
    (void)flags;
    if (strcmp(path, "/dev/tfmbs") == 0) {
        printf("[TFMBS-Driver] Device /dev/tfmbs opened.\n");
        return 100; // Mock file descriptor
    }
    errno = ENOENT;
    return -1;
}

int tfmbs_dev_ioctl(int fd, unsigned long request, void* arg) {
    if (fd != 100) {
        errno = EBADF;
        return -1;
    }

    switch (request) {
        case TFMBS_IOC_ALLOC: {
            tfmbs_ioc_alloc_t* a = (tfmbs_ioc_alloc_t*)arg;
            a->addr = (uint64_t)emu_fabric_alloc(a->size);
            return a->addr ? 0 : -ENOMEM;
        }
        case TFMBS_IOC_FREE: {
            tfmbs_ioc_free_t* f = (tfmbs_ioc_free_t*)arg;
            emu_fabric_free((void*)f->addr);
            return 0;
        }
        case TFMBS_IOC_MEMCPY_TO: {
            tfmbs_ioc_memcpy_to_t* m = (tfmbs_ioc_memcpy_to_t*)arg;
            return emu_fabric_memcpy_to((void*)m->dest_addr, m->src_host_ptr, m->size, m->pack_pt5);
        }
        case TFMBS_IOC_MEMCPY_FROM: {
            tfmbs_ioc_memcpy_from_t* m = (tfmbs_ioc_memcpy_from_t*)arg;
            return emu_fabric_memcpy_from(m->dest_host_ptr, (void*)m->src_addr, m->size, m->unpack_pt5);
        }
        case TFMBS_IOC_SUBMIT_GEMV: {
            tfmbs_ioc_submit_gemv_t* s = (tfmbs_ioc_submit_gemv_t*)arg;
            s->handle = (uint64_t)emu_fabric_exec_gemv_async((void*)s->weight_addr, (void*)s->input_addr, (void*)s->output_addr, s->rows, s->cols, s->tile_mask);
            return s->handle ? 0 : -EIO;
        }
        case TFMBS_IOC_WAIT: {
            tfmbs_ioc_wait_t* w = (tfmbs_ioc_wait_t*)arg;
            emu_fabric_wait((fabric_handle_t)w->handle);
            return 0;
        }
        case TFMBS_IOC_GET_METRICS: {
            tfmbs_ioc_metrics_t* m = (tfmbs_ioc_metrics_t*)arg;
            fabric_metrics_t fm;
            emu_fabric_get_metrics(&fm);
            m->zero_skips = fm.zero_skips;
            m->total_ops = fm.total_ops;
            m->pool_used = (uint32_t)fm.pool_used;
            m->pool_total = (uint32_t)fm.pool_total;
            m->evictions = fm.eviction_count;
            return 0;
        }
        case TFMBS_IOC_GET_INFO: {
            tfmbs_ioc_device_info_t* i = (tfmbs_ioc_device_info_t*)arg;
            i->num_tiles = 4;
            i->lanes_per_tile = 15;
            i->total_pool_size = 128 * 1024 * 1024;
            return 0;
        }
        case TFMBS_IOC_SUBMIT: {
            tfmbs_tfd_t* tfd = (tfmbs_tfd_t*)arg;
            uint8_t kernel = tfd->exec_hints & TFMBS_HINT_KERNEL_MASK;
            printf("[TFMBS-Driver] TFD Submitted: Base=0x%lx, Kernel=0x%02x, Tiles=0x%02x\n",
                   tfd->base_addr, kernel, tfd->tile_mask);
            // In a real driver, this would kick off the hardware DMA/execution
            return 0;
        }
        default:
            errno = EINVAL;
            return -1;
    }
}

int tfmbs_dev_close(int fd) {
    if (fd == 100) {
        printf("[TFMBS-Driver] Device /dev/tfmbs closed.\n");
        return 0;
    }
    errno = EBADF;
    return -1;
}
