# 05: TFD & Execution Hints

The **Ternary Frame Descriptor (TFD)** is the single source of truth for a task submitted to the fabric. In software, it is represented as a struct; in hardware, its fields are mapped to AXI registers.

## 1. TFD Structure (`tfmbs.h`)

```c
typedef struct {
    uint64_t base_addr;    /* Physical fabric-local address of frame      */
    uint32_t frame_len;    /* Total number of trits in the frame          */
    uint16_t packing_fmt;  /* See tfmbs_packing_t (e.g., PT-5)            */
    uint16_t lane_count;   /* Number of parallel SIMD lanes               */
    uint32_t lane_stride;  /* Trit distance between elements in a lane    */
    uint32_t flags;        /* Permissions, Coherence, and Paging          */
    uint32_t exec_hints;   /* Accelerator-specific kernel optimization    */
    uint8_t  version;      /* TFMBS Version                               */
    uint8_t  tile_mask;    /* Multi-tile selection mask                   */
    uint8_t  _reserved[6];
} tfmbs_tfd_t;
```

## 2. TFD Flags

The `flags` field manages memory behavior and priority:

*   **`0x01` (READ):** Application is allowed to read from this frame.
*   **`0x02` (WRITE):** Application is allowed to write to this frame.
*   **`0x08` (CRITICAL):** Abort if execution hints are not understood by the fabric.
*   **`0x10` (PINNED):** Prevents the frame from being evicted by the LRU paging system (Phase 7).

## 3. Execution Hints Bit-Field

The `exec_hints` register is a 32-bit field that configures the hardware kernel and enables optimizations.

```text
Bits  | Name           | Description
------|----------------|---------------------------------------------------
07:00 | KERNEL_ID      | 01:DOT, 04:CONV2D, 05:MAXPOOL, 06:TGEMM, 07:CONV3D, 08:LSTM, 09:ATTN
16    | BIAS_EN        | Enable post-accumulation bias / State persistence
17    | ZERO_SKIP_EN   | Enable **Zero-Skip** (clock-gating)
18    | FREE_NEG_EN    | Enable **Free-Negation** (sign inversion)
19    | WEIGHT_BRDCST  | Optimization hint for shared weights
21:20 | STRIDE         | (T-CONV) Stride size (00:1, 01:2, etc.)
23:22 | PAD            | (T-CONV) Padding size
25:24 | KSIZE          | (T-CONV) Kernel size (00:1x1, 01:3x3)
28:27 | POOL_WIN       | (T-POOL) Window size
30:29 | POOL_OP        | (T-POOL) 00:MAX, 01:MIN, 10:AVG
```

## 4. Kernel ID Mapping

| ID | Kernel | Function |
| --- | --- | --- |
| `0x01` | `DOT` | Standard vector dot product. |
| `0x03` | `MUL` | Element-wise multiplication (Result is not accumulated). |
| `0x04` | `CONV2D`| 2D Convolution (Uses stride/pad hints). |
| `0x05` | `MAXPOOL`| Ternary Max Pooling. |
| `0x06` | `TGEMM` | Matrix multiplication (Native RTL). |
| `0x07` | `CONV3D`| 3D Convolution (Native RTL). |
| `0x08` | `LSTM`  | Ternary LSTM (Native RTL). |
| `0x09` | `ATTN`  | Ternary Attention (Native RTL). |

## 5. Packing Formats (`packing_fmt`)

*   **`0x01` (PT-5):** 5 balanced trits per byte. This is the primary format supported by the hardware unpackers.
*   **`0x02` (T2B):** 2 bits per trit (Simple encoding). Used mainly for debugging or low-density storage.
