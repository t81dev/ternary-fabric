# 03: Hardware Architecture

The Ternary Fabric is a vectorized SIMD accelerator designed for integration into binary systems. It acts as an **Execution Adjacency** co-processor, sitting on the system bus and appearing as a memory-mapped peripheral.

## 1. Multi-Tile Topology

The fabric is composed of multiple independent **Tiles** that operate in lock-step. This allows for massive parallelization of workloads.

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

Each Tile contains its own private high-speed memory and a vector of Processing Elements.

*   **Private SRAM:** Dual-bank memory. Bank A stores Weights, Bank B stores Inputs (Activations).
*   **PT-5 Unpackers:** Combinatorial logic that hydrates 8-bit bytes into 15 independent 2-bit signed trits.
*   **Vector Engine:** A collection of 15 TPE lanes that perform math in parallel.

## 3. The TPE Lane (Processing Element)

The heart of the fabric is the Ternary Processing Element (TPE). It replaces the standard multiplier with high-efficiency logic.

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
If either the Weight or Input trit is `0`, the accumulator's clock-enable is suppressed. This prevents any toggling in the 32-bit adder, effectively reducing the dynamic power of that lane to near-zero for that cycle.

### Sign-Flip Logic
Since trits are restricted to $\{-1, 0, 1\}$, multiplication is trivial:
*   If both are non-zero and have the same sign, the result is $+1$ (Increment Accumulator).
*   If they have different signs, the result is $-1$ (Decrement Accumulator/Two's Complement).

## 4. Pipeline Execution Flow

1.  **Submission:** The Host writes a TFD to the AXI registers.
2.  **Hydration:** The Frame Controller generates addresses. SRAM data is fetched and passed through PT-5 unpackers.
3.  **Transformation:** TPE lanes perform the Sign-Flip and Accumulate operation.
4.  **Commit:** Once the `frame_len` is reached, the `done` bit is set, and results are available for the host to read.

## 5. ASIC Readiness

The fabric is designed for easy portability:
*   **No Multipliers:** Reduces area and timing pressure.
*   **Behavioral SRAM Wrappers:** Easily replaced with memory macros from any foundry (TSMC, GlobalFoundries, etc.).
*   **Clock Gating:** Hardware-native Zero-Skip provides fine-grained power gating hooks.
