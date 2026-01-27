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
 * @brief Standard Kernel IDs
 * Assigned to the `exec_hints` lower bits or a dedicated field.
 */
typedef enum {
    TFMBS_KERNEL_NOP      = 0x00, // No operation
    TFMBS_KERNEL_DOT      = 0x01, // Dot Product (Accumulate weight * input)
    TFMBS_KERNEL_ADD      = 0x02, // Element-wise Addition
    TFMBS_KERNEL_MUL      = 0x03, // Element-wise Multiplication
    TFMBS_KERNEL_CONV2D   = 0x04, // 2D Convolution (requires specific stride)
    TFMBS_KERNEL_MAXPOOL  = 0x05, // Ternary Max Pooling
    TFMBS_KERNEL_TGEMM    = 0x06, // Ternary Matrix-Multiply
    TFMBS_KERNEL_CONV3D   = 0x07, // 3D Convolution
    TFMBS_KERNEL_LSTM     = 0x08, // Ternary LSTM
    TFMBS_KERNEL_ATTN     = 0x09  // Ternary Attention
} tfmbs_kernel_t;

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
    uint8_t  tile_mask;    /* Multi-tile selection mask                   */
    uint8_t  _reserved[6]; /* Reserved for future alignment               */
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
 * @brief Execution Hints Bitmask
 */
#define TFMBS_HINT_KERNEL_MASK     0x000000FF
#define TFMBS_HINT_QUANT_SHIFT     8
#define TFMBS_HINT_QUANT_MASK      0x0000FF00
#define TFMBS_HINT_BIAS_EN         (1 << 16)
#define TFMBS_HINT_ZERO_SKIP_EN    (1 << 17)
#define TFMBS_HINT_FREE_NEG_EN     (1 << 18)
#define TFMBS_HINT_WEIGHT_BRDCST   (1 << 19)

/* T-CONV Specific Hints */
#define TFMBS_HINT_STRIDE_MASK     0x00300000 // Bits 21:20
#define TFMBS_HINT_STRIDE_SHIFT    20
#define TFMBS_HINT_PAD_MASK        0x00C00000 // Bits 23:22
#define TFMBS_HINT_PAD_SHIFT       22
#define TFMBS_HINT_KSIZE_MASK      0x03000000 // Bits 25:24
#define TFMBS_HINT_KSIZE_SHIFT     24
#define TFMBS_HINT_DILATION_EN     (1 << 26)

/* T-POOL Specific Hints */
#define TFMBS_HINT_POOL_WIN_MASK   0x18000000 // Bits 28:27
#define TFMBS_HINT_POOL_WIN_SHIFT  27
#define TFMBS_HINT_POOL_OP_MASK    0x60000000 // Bits 30:29
#define TFMBS_HINT_POOL_OP_SHIFT   29

#define TFMBS_POOL_OP_MAX          0x0
#define TFMBS_POOL_OP_MIN          0x1
#define TFMBS_POOL_OP_AVG          0x2

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