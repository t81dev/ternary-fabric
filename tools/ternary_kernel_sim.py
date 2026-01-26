#include <stdio.h>
#include <stdlib.h>
#include "../include/tfmbs.h"

/**
 * @brief Reference implementation of PT-5 encoding.
 * Packs 5 balanced trits into one byte.
 */
uint8_t pack_trits_to_byte(int8_t trits[5]) {
    uint8_t byte_val = 0;
    uint8_t p3 = 1; // Powers of 3: 1, 3, 9, 27, 81

    for (int i = 0; i < 5; i++) {
        // Shift balanced (-1, 0, 1) to (0, 1, 2)
        uint8_t unsigned_trit = trits[i] + 1;
        byte_val += (unsigned_trit * p3);
        p3 *= 3;
    }
    return byte_val;
}

int main() {
    // 1. Prepare raw balanced ternary data (15 trits)
    int8_t raw_data[15] = {1, 0, -1, 1, 1,   0, 0, 0, -1, -1,   1, 1, 1, 1, 1};
    
    // 2. Allocate binary buffer (15 trits / 5 trits-per-byte = 3 bytes)
    uint8_t *buffer = (uint8_t *)malloc(3);

    // 3. Pack data into the buffer
    for (int i = 0; i < 3; i++) {
        buffer[i] = pack_trits_to_byte(&raw_data[i * 5]);
    }

    // 4. Define the Ternary Frame Descriptor (TFD)
    tfmbs_tfd_t my_frame = {
        .base_addr = (uintptr_t)buffer, // Host-side pointer for emulation
        .frame_len = 15,
        .packing_fmt = TFMBS_PACKING_PT5,
        .lane_count = 1,
        .lane_stride = 1,
        .flags = TFMBS_FLAG_READ | TFMBS_FLAG_PINNED,
        .version = 0x01
    };

    // 5. Output results for verification
    printf("TFD Initialized for Frame at 0x%lx\n", my_frame.base_addr);
    printf("Packed Bytes: [0x%02X, 0x%02X, 0x%02X]\n", buffer[0], buffer[1], buffer[2]);

    free(buffer);
    return 0;
}