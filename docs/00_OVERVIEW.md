# 00: Project Overview

**Ternary Fabric** is a high-performance co-processor designed to accelerate AI and signal processing workloads by utilizing **Balanced Ternary** arithmetic. By moving beyond traditional binary logic, the fabric eliminates the need for expensive hardware multipliers and enables massive energy savings through data-aware optimizations.

## 1. Core Innovation

Traditional binary accelerators spend a significant portion of their power and area on multipliers. The Ternary Fabric replaces these with simple **Multiplexer-Accumulators**:

*   **Sign-Flip Logic:** Multiplication by $+1$ or $-1$ is reduced to a conditional addition or subtraction.
*   **Zero-Skip Hardware:** If either the weight or input is $0$, the accumulator is clock-gated. Power is only consumed for non-zero operations, making it ideal for sparse models like Large Language Models (LLMs).

## 2. Technical Pillars

### Binary Sovereignty
The fabric is designed as a specialized data plane attached to a binary host (e.g., a Zynq SoC or an x86 host with FPGA acceleration). The host manages scheduling, memory allocation, and high-level logic, while the fabric handles the heavy lifting of ternary math.

### The Hydration Pipeline
To save bandwidth, data is stored in a high-density **PT-5** format (5 trits per 8-bit byte, 95.1% efficiency). Upon reaching the Processing Elements (TPEs), it is "hydrated" into a 2-bit signed logic format for single-cycle execution.

### Multi-Tile Scalability
The architecture is parameterized to scale from a single tile to a large matrix of tiles. Each tile operates in lock-step, sharing a frame controller but maintaining private SRAM for local data storage and high-throughput parallel execution.

## 3. Project Status (Phase 6b)

The project has evolved through several key phases:

*   **Phases 1-2:** Specification of the TFMBS ABI and development of the PT-5 codec.
*   **Phase 3:** RTL design of the Vector Engine and TPE lanes.
*   **Phase 4:** Integration with Python via `pytfmbs` and development of the quantization toolkit.
*   **Phase 5:** Implementation of hardware profiling counters and dynamic lane masking.
*   **Phase 6a:** Extension of the kernel library (T-CONV, T-POOL) and DMA support.
*   **Phase 6b (Current):** Multi-tile scaling, broadcast weight support, and ASIC-ready SRAM wrappers. This phase marks the **production-core** readiness of the multi-tile fabric.

## 4. Experimental Kernels

Beyond the core hardware kernels, the fabric currently supports experimental reference implementations for **T-Conv3D**, **T-LSTM**, and **T-Attention**. These are wired into the `pytfmbs` API and TFMBS header, allowing for software-level verification and architectural exploration before full RTL acceleration.

## 5. Key Terminology

*   **Trit:** A ternary digit $\{-1, 0, 1\}$.
*   **TFD:** Ternary Frame Descriptor. The control structure used to submit tasks to the fabric.
*   **PT-5:** The packing format used for ternary data on the bus.
*   **Zero-Skip:** Hardware optimization that suppresses clocking for zero-value multiplications.
*   **Free Negation:** An optimization where weights can be negated "for free" in hardware during execution.
