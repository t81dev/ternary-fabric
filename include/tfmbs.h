/*
 * TFMBS: Ternary Fabric Memory & Bus Specification
 * Version: 0.1 (Draft)
 * * This header defines the binary interface for interacting with 
 * the ternary-native memory and interconnect fabric.
 */

#ifndef TERNARY_FABRIC_H
#define TERNARY_FABRIC_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Ternary Packing Formats
 * Defines how trits are mapped into binary storage.
 */
typedef enum {
    TFMBS_PACKING_PT5  = 0x01, // 5 trits per 8-bit byte (Balanced Ternary)
    TFMBS_PACKING_T2B  = 0x02, // 2 bits per trit (Unpacked/Simple)
    TFMBS_PACKING_RAW  = 0x03  // Implementation-specific high-density
} tfmbs_packing_t;

/**
 * @brief Ternary Frame Descriptor (TFD)
 * The primary ABI boundary between Binary Host and Ternary Fabric.
 * Must be 64-bit aligned for hardware DMA compatibility.
 */
typedef struct {
    uint64_t base_addr;    /* Physical fabric-local address of frame      */
    uint32_t frame_len;    /* Total number of trits in the frame          */
    uint16_t packing_fmt;  /* See tfmbs_packing_t                         */
    uint16_t lane_count;   /* Number of parallel SIMD lanes               */
    uint32_t lane_stride;  /* Trit distance between elements in a lane    */
    uint32_t flags;        /* Permissions, Coherence, and Priority flags  */
    uint32_t exec_hints;   /* Accelerator-specific kernel optimization    */
    uint8_t  version;      /* TFMBS Version (0x01 for this draft)         */
    uint8_t  _reserved[7]; /* Reserved for future alignment               */
} tfmbs_tfd_t;

/**
 * @brief TFD Flags
 */
#define TFMBS_FLAG_READ       (1 << 0)
#define TFMBS_FLAG_WRITE      (1 << 1)
#define TFMBS_FLAG_COHERENT   (1 << 2) // Request hardware-managed coherence
#define TFMBS_FLAG_CRITICAL   (1 << 3) // Fail if hints are not understood
#define TFMBS_FLAG_PINNED     (1 << 4) // Memory is guaranteed non-pageable

/**
 * @brief Status Codes
 */
typedef enum {
    TFMBS_STATUS_OK       = 0,
    TFMBS_STATUS_PENDING  = 1,
    TFMBS_STATUS_ERR_ADDR = 2, // Alignment or Bounds error
    TFMBS_STATUS_ERR_PACK = 3, // Invalid packing encountered
    TFMBS_STATUS_ERR_BUSY = 4  // Fabric resource contention
} tfmbs_status_t;

#ifdef __cplusplus
}
#endif

#endif /* TERNARY_FABRIC_H */