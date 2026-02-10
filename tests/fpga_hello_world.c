#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "tfmbs.h"

// Basic memory-mapped register access for the TFMBS Fabric on Zynq
#define TFMBS_PHYS_ADDR 0x43C00000 // Typical AXI Lite base for PL
#define TFMBS_REG_SIZE  0x10000

int main() {
    printf("--- TFMBS FPGA Hello World ---\n");

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("open /dev/mem");
        return 1;
    }

    uint32_t *regs = mmap(NULL, TFMBS_REG_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, TFMBS_PHYS_ADDR);
    if (regs == MAP_FAILED) {
        perror("mmap");
        return 1;
    }

    printf("Fabric identified. ID: 0x%08X\n", regs[0]);

    // Setup a tiny GEMV (example trits)
    // In a real test, we would load PT5 packed trits into the fabric SRAM
    printf("Loading GEMV kernel...\n");

    // Trigger execution (dummy trigger for now)
    regs[1] = 0x1; // Start bit

    // Wait for completion
    while (regs[2] & 0x1) {
        usleep(100);
    }

    uint32_t zero_skips = regs[3];
    printf("Execution complete.\n");
    printf("Zero-Skips Captured: %u\n", zero_skips);

    munmap(regs, TFMBS_REG_SIZE);
    close(fd);
    return 0;
}
