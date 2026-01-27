/*
 * TFMBS: Ternary Fabric Memory & Bus Specification
 * UAPI: User-Kernel Interface (Mock)
 *
 * This header defines the ioctl-based interface for interacting with
 * the /dev/tfmbs device.
 */

#ifndef TFMBS_UAPI_H
#define TFMBS_UAPI_H

#include <stdint.h>
#include "tfmbs.h"

#ifdef __linux__
#include <linux/ioctl.h>
#else
// Minimal ioctl macro mocks for non-linux environments
#define _IOWR(type, nr, size) (0x80000000 | (sizeof(size) << 16) | (type << 8) | nr)
#define _IOW(type, nr, size)  (0x40000000 | (sizeof(size) << 16) | (type << 8) | nr)
#define _IOR(type, nr, size)  (0x20000000 | (sizeof(size) << 16) | (type << 8) | nr)
#endif

#define TFMBS_IOC_MAGIC 'T'

typedef struct {
    uint64_t size;
    uint64_t addr; // OUT: Device address
} tfmbs_ioc_alloc_t;

typedef struct {
    uint64_t addr;
} tfmbs_ioc_free_t;

typedef struct {
    uint64_t dest_addr;
    const void* src_host_ptr;
    uint64_t size;
    int pack_pt5;
} tfmbs_ioc_memcpy_to_t;

typedef struct {
    void* dest_host_ptr;
    uint64_t src_addr;
    uint64_t size;
    int unpack_pt5;
} tfmbs_ioc_memcpy_from_t;

typedef struct {
    uint64_t weight_addr;
    uint64_t input_addr;
    uint64_t output_addr;
    uint32_t rows;
    uint32_t cols;
    uint8_t  tile_mask;
    uint64_t handle; // OUT: Async task handle
} tfmbs_ioc_submit_gemv_t;

typedef struct {
    uint64_t handle;
} tfmbs_ioc_wait_t;

typedef struct {
    uint64_t zero_skips;
    uint64_t total_ops;
    uint32_t pool_used;
    uint32_t pool_total;
    uint32_t evictions;
} tfmbs_ioc_metrics_t;

typedef struct {
    uint32_t num_tiles;
    uint32_t lanes_per_tile;
    uint64_t total_pool_size;
} tfmbs_ioc_device_info_t;

#define TFMBS_IOC_ALLOC        _IOWR(TFMBS_IOC_MAGIC, 0x01, tfmbs_ioc_alloc_t)
#define TFMBS_IOC_FREE         _IOW(TFMBS_IOC_MAGIC, 0x02, tfmbs_ioc_free_t)
#define TFMBS_IOC_MEMCPY_TO    _IOW(TFMBS_IOC_MAGIC, 0x03, tfmbs_ioc_memcpy_to_t)
#define TFMBS_IOC_MEMCPY_FROM  _IOW(TFMBS_IOC_MAGIC, 0x04, tfmbs_ioc_memcpy_from_t)
#define TFMBS_IOC_SUBMIT_GEMV  _IOWR(TFMBS_IOC_MAGIC, 0x05, tfmbs_ioc_submit_gemv_t)
#define TFMBS_IOC_WAIT         _IOW(TFMBS_IOC_MAGIC, 0x06, tfmbs_ioc_wait_t)
#define TFMBS_IOC_GET_METRICS  _IOR(TFMBS_IOC_MAGIC, 0x07, tfmbs_ioc_metrics_t)
#define TFMBS_IOC_GET_INFO     _IOR(TFMBS_IOC_MAGIC, 0x08, tfmbs_ioc_device_info_t)
#define TFMBS_IOC_SUBMIT       _IOW(TFMBS_IOC_MAGIC, 0x09, tfmbs_tfd_t)

#endif /* TFMBS_UAPI_H */
