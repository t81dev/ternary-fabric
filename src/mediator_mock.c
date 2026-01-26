#include <stdio.h>
#include <stdbool.h>
#include <inttypes.h>
#include "../include/tfmbs.h"

// Mock internal fabric state
typedef struct {
    bool hardware_ready;
    uint32_t active_tfd_count;
} fabric_state_t;

fabric_state_t core_state = { .hardware_ready = true, .active_tfd_count = 0 };

/**
 * @brief The Mediator "Validation" Logic
 * Ensures the TFD follows the rules defined in TERNARY_MEMORY_BUS.md
 */
tfmbs_status_t validate_tfd(tfmbs_tfd_t *tfd) {
    if (tfd->version != 0x01) return TFMBS_STATUS_ERR_ADDR;
    if (tfd->base_addr == 0) return TFMBS_STATUS_ERR_ADDR;
    
    // PT-5 requires frames to be multiples of 5 trits in this mock
    if (tfd->packing_fmt == TFMBS_PACKING_PT5 && (tfd->frame_len % 5 != 0)) {
        return TFMBS_STATUS_ERR_PACK;
    }

    return TFMBS_STATUS_OK;
}

/**
 * @brief The Mediator "Execution" Entry Point
 * Simulates the INTERCONNECT.md handshake.
 */
tfmbs_status_t tfmbs_submit_tfd(tfmbs_tfd_t *tfd) {
    printf("[Mediator] Receiving TFD for address 0x%lx...\n", tfd->base_addr);

    // 1. Validation Stage
    tfmbs_status_t status = validate_tfd(tfd);
    if (status != TFMBS_STATUS_OK) {
        printf("[Mediator] Rejecting TFD: Error Code %d\n", status);
        return status;
    }

    // 2. Hydration Stage (Simulated DMA)
    printf("[Mediator] State: HYDRATING... (Reading %d trits via PT-5)\n", tfd->frame_len);

    // 3. Transformation Stage (AI Engine)
    if (tfd->exec_hints & 0x01) { // Let's say hint 0x01 is "AI Dot Product"
        printf("[Mediator] State: ACTIVE. Engaging AI Engine T-MAC units.\n");
    }

    // 4. Commit Stage
    printf("[Mediator] State: COMPLETE. Signaling Host via EVENT_DONE.\n");
    return TFMBS_STATUS_OK;
}

int main() {
    // Example: Submitting a valid TFD
    tfmbs_tfd_t my_frame = {
        .base_addr = 0xDEADBEEF, // Mock address
        .frame_len = 100,
        .packing_fmt = TFMBS_PACKING_PT5,
        .exec_hints = 0x01,
        .version = 0x01
    };

    tfmbs_status_t result = tfmbs_submit_tfd(&my_frame);
    
    if (result == TFMBS_STATUS_OK) {
        printf("Fabric operation successful.\n");
    }

    return 0;
}