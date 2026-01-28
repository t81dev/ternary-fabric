# 03: Hardware Architecture

The Ternary Fabric is a vectorized SIMD accelerator designed for integration into binary systems. It acts as an **Execution Adjacency** co-processor, sitting on the system bus and appearing as a memory-mapped peripheral. In Phase 21, multiple **Fabric Instances** can be orchestrated to scale system-level throughput.

## 1. Multi-Tile Topology

Each **Fabric Instance** is composed of multiple independent **Tiles** that operate in lock-step. This allows for massive parallelization of workloads.

### Global Frame Controller
The Frame Controller acts as the "brain" of the fabric instance. It translates the **TFD** into a stream of control signals (Start, Opcode, Stride) that are broadcast to all active tiles.

### Tile Masking & Selection
Workloads can be targeted at specific tiles using the `tile_mask` (found in the **TFD** and the CONTROL register).
*   **Selective Execution:** If a tile's mask bit is `0`, it remains idle, conserving power.
*   **Independent Data, Shared Control:** All active tiles receive the same control signals but operate on their private SRAM data.

```text
       AXI4-Lite Control Plane / AXI-Stream DMA
                  |
        +---------V------------------+
        |     Frame Controller       | (Global Control)
        +---------|------------------+
                  | (Shared Control Bus: Start, Op, Stride)
        +---------+---------+---------+---------+
        |         |         |         |         |
    +---V---+ +---V---+ +---V---+ +---V---+     |
    | Tile 0| | Tile 1| | Tile 2| | Tile 3| ... | (Up to NUM_TILES)
    +-------+ +-------+ +-------+ +-------+     |
        |         |         |         |         |
        +---------+---------+---------+---------+
                  |
        (Aggregated Vector Results / Counters)
```

## 2. Anatomy of a Tile

Each **Tile** contains its own private high-speed memory and a vector of **Ternary Lanes**.

*   **Private SRAM:** Dual-bank memory. Bank A stores Weights, Bank B stores Inputs (Activations).
*   **PT-5 Unpackers:** Combinatorial logic that hydrates 8-bit bytes into 15 independent 2-bit signed trits.
*   **Vector Engine:** A collection of 15 **Ternary Lanes** that perform math in parallel.

## 3. The Ternary Lane (Processing Element)

The heart of the fabric is the **Ternary Lane**. It replaces the standard multiplier with high-efficiency gated logic.

```text
Weight (2b) --+       Input (2b) --+
              |                    |
        +-----V--------------------V-----+
        |       Zero-Skip Logic          | (Clock Gate)
        +-----|--------------------|-----+
              |                    |
        +-----V--------------------V-----+
        |       Sign-Flip Logic          | (Inverter)
        +-----|--------------------|-----+
              |                    |
        +-----V--------------------V-----+
        |       32-bit Accumulator       |
        +--------------------------------+
```

### Zero-Skip Logic
If either the Weight or Input trit is `0`, the accumulator's clock-enable is suppressed. This prevents any toggling in the 32-bit adder, effectively reducing the dynamic power of that lane to near-zero for that cycle and suppressing unnecessary memory access.

### Sign-Flip Logic
Since trits are restricted to $\{-1, 0, 1\}$, multiplication is simplified:
*   If both are non-zero and have the same sign, the result is $+1$ (Increment Accumulator).
*   If they have different signs, the result is $-1$ (Decrement Accumulator).

### Native Kernel Support (Phase 15)
The Vector Engine includes specialized logic for advanced kernels:
*   **T-Conv3D:** Automatic squared-stride offset calculation for 3D spatial traversal.
*   **T-LSTM / T-Attention:** When the `BIAS_EN` hint is set, the accumulator persists its value across frame descriptors, enabling efficient state management.

## 4. Pipeline Execution Flow

1.  **Submission:** The Host writes a **TFD** to the AXI registers.
2.  **Hydration:** The Frame Controller generates addresses. SRAM data is fetched and passed through **PT-5** unpackers.
3.  **Transformation:** **Ternary Lanes** perform the Sign-Flip and Accumulate operation.
4.  **Commit:** Once the `frame_len` is reached, the `done` bit is set, and results are available for the host.

## 5. Memory Model & System Boundary

The interaction between the binary host and the ternary fabric occurs at the memory boundary, utilizing the **PT-5** encoding format.

### PT-5 Hydration Pipeline
Data is stored and transmitted across the AXI bus in packed 8-bit bytes (5 trits per byte).
*   **Dehydration (Hardware):** Occurs at the entry point of each **Tile**. Combinatorial logic expands each 8-bit byte into 15 independent 2-bit signed trits in a single clock cycle. This incurs **zero pipeline latency**.
*   **Hydration (Software/Host):** When the host prepares weights or inputs, it must pack them into **PT-5**. This achieves a **37.5% reduction in bus bandwidth** compared to using 2 bits per trit.

## 6. ASIC Readiness

The fabric is designed for easy portability:
*   **No Multipliers:** Reduces area and timing pressure.
*   **Behavioral SRAM Wrappers:** Easily replaced with memory macros from any foundry.
*   **Clock Gating:** Hardware-native **Zero-Skip** provides fine-grained power gating hooks.
