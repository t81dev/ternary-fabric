#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "tfmbs_driver.h"
#include "uapi_tfmbs.h"

int main() {
    printf("🧪 Testing Phase 10 Mock Driver Interface...\n");

    int fd = tfmbs_dev_open("/dev/tfmbs", 0);
    assert(fd >= 0);

    // Test Alloc
    tfmbs_ioc_alloc_t alloc_args = { .size = 1024 };
    int ret = tfmbs_dev_ioctl(fd, TFMBS_IOC_ALLOC, &alloc_args);
    assert(ret == 0);
    assert(alloc_args.addr != 0);
    printf("✅ Alloc successful: 0x%lx\n", alloc_args.addr);

    // Test Memcpy To
    int8_t host_data[5] = {1, 0, -1, 1, 0};
    tfmbs_ioc_memcpy_to_t to_args = {
        .dest_addr = alloc_args.addr,
        .src_host_ptr = host_data,
        .size = 5,
        .pack_pt5 = 1
    };
    ret = tfmbs_dev_ioctl(fd, TFMBS_IOC_MEMCPY_TO, &to_args);
    assert(ret == 0);
    printf("✅ Memcpy To (PT-5) successful\n");

    // Test Memcpy From
    int8_t back_data[5];
    tfmbs_ioc_memcpy_from_t from_args = {
        .dest_host_ptr = back_data,
        .src_addr = alloc_args.addr,
        .size = 5,
        .unpack_pt5 = 1
    };
    ret = tfmbs_dev_ioctl(fd, TFMBS_IOC_MEMCPY_FROM, &from_args);
    assert(ret == 0);
    for(int i=0; i<5; i++) assert(back_data[i] == host_data[i]);
    printf("✅ Memcpy From (PT-5) verified\n");

    // Test Metrics
    tfmbs_ioc_metrics_t metrics;
    ret = tfmbs_dev_ioctl(fd, TFMBS_IOC_GET_METRICS, &metrics);
    assert(ret == 0);
    printf("✅ Metrics retrieved: Pool %u/%u\n", metrics.pool_used, metrics.pool_total);

    // Test Info
    tfmbs_ioc_device_info_t info;
    ret = tfmbs_dev_ioctl(fd, TFMBS_IOC_GET_INFO, &info);
    assert(ret == 0);
    assert(info.num_tiles == 4);
    printf("✅ Device Info: %u tiles, %u lanes, %lu bytes pool\n", info.num_tiles, info.lanes_per_tile, info.total_pool_size);

    // Test Free
    tfmbs_ioc_free_t free_args = { .addr = alloc_args.addr };
    ret = tfmbs_dev_ioctl(fd, TFMBS_IOC_FREE, &free_args);
    assert(ret == 0);
    printf("✅ Free successful\n");

    tfmbs_dev_close(fd);
    printf("⭐ Phase 10 Driver Mock Validation PASSED\n");

    return 0;
}
