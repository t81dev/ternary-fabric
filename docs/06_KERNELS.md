# 06: Kernel Reference

The Ternary Fabric supports several specialized hardware kernels. Each kernel interprets the frame data differently based on the `exec_hints`.

## 1. T-GEMM (Ternary General Matrix Multiply)
**Kernel ID:** `0x06`

T-GEMM is optimized for high-throughput matrix operations.
*   **Data Layout:** Weights should be stored in column-major format to align with the vectorized lanes.
*   **Broadcast:** Use `tile_mask` and `WEIGHT_BRDCST` hint to reuse a single weight matrix across multiple input streams on different tiles.

## 2. T-CONV2D (Ternary Convolution)
**Kernel ID:** `0x04`

Designed for CNN acceleration.
*   **Striding:** Controlled by `exec_hints[21:20]`. The hardware automatically skips input addresses based on the stride.
*   **Padding:** Supports zero-padding via `exec_hints[23:22]`.
*   **Kernel Size:** Optimized for 1x1 and 3x3 kernels.

## 3. T-POOL (Ternary Pooling)
**Kernel ID:** `0x05`

Supports Max, Min, and Average pooling.
*   **Window Size:** Set via `exec_hints[28:27]`.
*   **Operations:**
    *   `00`: MAX (Find highest trit value).
    *   `01`: MIN (Find lowest trit value).
    *   `10`: AVG (Sum and divide, or shift).

## 4. Basic DOT & MUL
**Kernel IDs:** `0x01` (DOT), `0x03` (MUL)

*   **DOT:** Performs $Acc = \sum (W_i \times I_i)$.
*   **MUL:** Performs $Out_i = W_i \times I_i$. The result registers contain the last product for each lane.

## 5. Accelerated Ternary Kernels (Phase 15 Maturation)

These kernels have been promoted from software reference to full RTL acceleration. They leverage the vectorized ternary lanes for high-throughput execution.

### T-CONV3D
**Kernel ID:** `0x07`
**Status:** Accelerated
Extends the 2D convolution logic with a 3D spatial traversal. The hardware automatically calculates squared-stride offsets for memory addressing.

### T-LSTM
**Kernel ID:** `0x08`
**Status:** Accelerated
Supports recurrent ternary operations. Optimized with hardware state-management, allowing the accumulator to persist across multiple frame descriptors when the `BIAS_EN` hint is set.

### T-ATTENTION
**Kernel ID:** `0x09`
**Status:** Accelerated
Implements the core multi-head attention projection mechanism using ternary-quantized Query, Key, and Value vectors. Supports persistent state for key-value caching optimization.

## 6. Summary Table of Hints

| Kernel | Required Hints | Optional Hints |
| --- | --- | --- |
| TGEMM | - | Zero-Skip, Broadcast |
| CONV2D | Stride, KSize | Pad, Dilation |
| MAXPOOL| Pool_Win, Pool_Op | - |
| CONV3D | Stride | Zero-Skip |
| LSTM   | - | BIAS_EN (State Persistence) |
| ATTN   | - | BIAS_EN (State Persistence) |
