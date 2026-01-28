# 00: Project Overview

**Ternary Fabric** is a high-performance co-processor designed to accelerate AI and signal processing workloads by utilizing **Balanced Ternary** arithmetic. By moving beyond traditional binary logic, the fabric eliminates the need for expensive hardware multipliers and enables massive energy savings through data-aware optimizations.

## 1. Core Innovation

Traditional binary accelerators spend a significant portion of their power and area on multipliers. Ternary Fabric replaces these with simple **Gated Accumulators**:

*   **Sign-Flip Logic:** Multiplication by $+1$ or $-1$ is reduced to a conditional addition or subtraction.
*   **Zero-Skip Hardware:** If either the weight or input is $0$, the accumulator and memory access are clock-gated. Power is only consumed for non-zero operations, making it ideal for sparse models like Large Language Models (LLMs).

## 2. Technical Pillars

### Binary Sovereignty
The fabric is designed as a specialized data plane attached to a binary host (e.g., a Zynq SoC or an x86 host with FPGA acceleration). The host manages scheduling, memory allocation, and high-level logic, while the fabric handles the heavy lifting of ternary math.

### The Hydration Pipeline
To save bandwidth, data is stored in a high-density **PT-5** format (5 trits per 8-bit byte, 95.1% efficiency). Upon reaching the **Ternary Lanes**, it is "hydrated" into a 2-bit signed logic format for single-cycle execution.

### Multi-Tile Scalability
The architecture is parameterized to scale from a single **Tile** to a large matrix of tiles. Each tile operates in lock-step, sharing a frame controller but maintaining private SRAM for local data storage and high-throughput parallel execution.

### Predictive Multi-Fabric Orchestration (Phase 21)
The system coordinates multiple **Fabric Instances** via a **Global Orchestrator**, utilizing predictive scheduling and kernel fusion to maximize efficiency across independent co-processors.

## 3. Project Status (Phase 21) ✅

The project has evolved through several key milestones:

*   **Phases 0–9:** Specification, Emulation, Interposition, Residency, and Telemetry.
*   **Phases 10–14:** Hardware Path (Mock), Multi-Tile Scaling, PyTorch Integration, and GGUF support.
*   **Phase 15:** Native RTL acceleration for CONV3D, LSTM, and Attention kernels.
*   **Phases 18–20:** Workload Metrics, Data-Driven Adaptation, and Learning/Self-Tuning.
*   **Phase 21:** Predictive Multi-Fabric Orchestration.

## 4. Accelerated Hardware Kernels

The fabric supports native RTL-accelerated kernels for:
*   **T-GEMM:** General Matrix Multiply for LLM offloading.
*   **T-Conv3D:** 3D Convolution with squared-stride address calculation.
*   **T-LSTM:** Recurrent ternary operations with hardware-managed state persistence.
*   **T-Attention:** Multi-head attention projections with persistent key-value caching support.

## 5. Key Terminology

*   **Trit:** A balanced-ternary digit $\{-1, 0, 1\}$.
*   **TFD:** **Ternary Frame Descriptor**. The control structure used to submit tasks to the fabric.
*   **PT-5:** The high-density packing format used for ternary data on the bus.
*   **Zero-Skip:** Hardware optimization that suppresses clocking and memory access for zero-value operands.
*   **Fabric Cost:** A cycle-aware metric accounting for active operations and memory weightings.
*   **Economic Efficiency:** Ratio of active operations to total **Fabric Cost**.
*   **Residency Hit:** Finding required weights already in the Fabric memory pool.
