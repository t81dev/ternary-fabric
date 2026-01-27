# DMA Strategy for Multi-Fabric Scaling (Phase 11)

This document outlines the high-performance data movement strategy for scaling Ternary Fabric workloads across multiple co-processors.

## 1. DMA Engine Architecture

The Ternary Fabric uses an **AXI Direct Memory Access (DMA)** engine to transfer PT-5 packed data between Host RAM and the private SRAMs of each Tile.

*   **Scatter-Gather (SG) Support:** The driver manages SG lists to allow large model weights, which may be fragmented in virtual memory, to be streamed into the fabric as a single logical transfer.
*   **Burst Transfers:** Optimized for 128-bit or 256-bit AXI bursts to saturate the system bus.

## 2. Multi-Fabric Distribution

In systems with multiple Fabric instances (e.g., a PCIe card with 4 Fabrics), the following strategies are employed:

### Weight Partitioning
Large matrices are split across Fabrics. The DMA controller supports **multicast** transfers where a single weight buffer in RAM is broadcast to multiple Fabric instances simultaneously, reducing bus contention.

### Overlapped Execution (Double Buffering)
To maximize throughput, the system employs a double-buffering scheme:
1.  **Fabric Execution:** Fabric $N$ processes the current data frame from SRAM Bank A.
2.  **DMA Inflow:** The DMA engine concurrently loads the next data frame into SRAM Bank B.
3.  **Bank Swap:** Upon completion, the roles swap with zero pipeline stall.

## 3. Reduction Strategy

For large-scale T-GEMM operations spanning multiple Fabrics:
*   **Partial Sums:** Each Fabric computes a partial result based on its local weight partition.
*   **Host Aggregation:** Results are DMA-ed back to Host RAM where a final reduction step is performed using binary SIMD (e.g., AVX or NEON).
*   **Autonomous Reduction (Future):** Future revisions may include an Inter-Fabric Bus (IFB) for hardware-native reduction without host intervention.

## 4. Bandwidth Efficiency

By using **PT-5 packing**, the DMA engine achieves **95.1% utilization** of the raw bus bandwidth for ternary data, significantly outperforming 2-bit-per-trit packing (62.5% efficiency).
