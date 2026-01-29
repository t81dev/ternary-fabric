#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "tfmbs_driver.h"
#include "uapi_tfmbs.h"

int main() {
    int fd = tfmbs_dev_open("/dev/tfmbs", 0);
    assert(fd != -1);

    struct tfmbs_dma_desc desc;
    desc.src_addr = 0x1000;
    desc.dst_addr = 0x2000;
    desc.len = 64;
    desc.flags = 0x1 | 0x4; // Host->Device, Interrupt on complete
    desc.next = 0;

    tfmbs_ioc_submit_dma_t args;
    args.descs = &desc;
    args.count = 1;

    printf("[Test] Submitting DMA descriptor...\n");
    int res = tfmbs_dev_ioctl(fd, TFMBS_IOC_SUBMIT_DMA, &args);
    assert(res == 0);
    printf("[Test] DMA submission successful.\n");

    tfmbs_dev_close(fd);
    return 0;
}
