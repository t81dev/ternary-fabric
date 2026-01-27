# 00: Project Overview

**Ternary Fabric** is a high-performance co-processor designed to accelerate AI and signal processing workloads by utilizing **Balanced Ternary** arithmetic. By moving beyond traditional binary logic, the fabric eliminates the need for expensive hardware multipliers and enables massive energy savings through data-aware optimizations.

## 1. Core Innovation

Traditional binary accelerators spend a significant portion of their power and area on multipliers. Ternary Fabric replaces these with simple **Multiplexer-Accumulators**:

*   **Sign-Flip Logic:** Multiplication by $+1$ or $-1$ is reduced to a conditional addition or subtraction.
*   **Zero-Skip Hardware:** If either the weight or input is $0$, the accumulator is clock-gated. Power is only consumed for non-zero operations, making it ideal for sparse models like Large Language Models (LLMs).

## 2. Technical Pillars

### Binary Sovereignty
The fabric is designed as a specialized data plane attached to a binary host (e.g., a Zynq SoC or an x86 host with FPGA acceleration). The host manages scheduling, memory allocation, and high-level logic, while the fabric handles the heavy lifting of ternary math.

### The Hydration Pipeline
To save bandwidth, data is stored in a high-density **PT-5** format (5 trits per 8-bit byte, 95.1% efficiency). Upon reaching the Processing Elements (TPEs), it is "hydrated" into a 2-bit signed logic format for single-cycle execution.

### Multi-Tile Scalability
The architecture is parameterized to scale from a single tile to a large matrix of tiles. Each tile operates in lock-step, sharing a frame controller but maintaining private SRAM for local data storage and high-throughput parallel execution.

## 3. Project Status (Phase 9)

The project has reached the **Transparent Acceleration** milestone (Phase 9), enabling zero-patch integration with binary applications:

*   **Phases 1-5:** Development of the core RTL, PT-5 codec, and initial "Fabric Illusion" compute offloading.
*   **Phase 6:** Multi-tile scaling, broadcast weight support, and verification of **Zero-Skip** efficiency (~64-76%).
*   **Phase 7:** Implementation of **LRU-based Paging & Eviction**, allowing models larger than the physical Fabric memory pool to be executed transparently.
*   **Phase 8:** Introduction of **Asynchronous Pipelining**, utilizing a background worker thread to overlap host processing with Fabric execution.
*   **Phase 9 (Current):** Integration of real-time **Telemetry & Performance** dashboards, providing visibility into skip rates, pool usage, and async queue status.

## 4. Experimental Kernels

Beyond the core hardware kernels, the fabric currently supports experimental reference implementations for **T-Conv3D**, **T-LSTM**, and **T-Attention**. These are wired into the `pytfmbs` API and TFMBS header, allowing for software-level verification and architectural exploration before full RTL acceleration.

## 5. Key Terminology

*   **Trit:** A ternary digit $\{-1, 0, 1\}$.
*   **TFD:** Ternary Frame Descriptor. The control structure used to submit tasks to the fabric.
*   **PT-5:** The packing format used for ternary data on the bus.
*   **Zero-Skip:** Hardware optimization that suppresses clocking for zero-value multiplications.
*   **Free Negation:** An optimization where weights can be negated "for free" in hardware during execution.
