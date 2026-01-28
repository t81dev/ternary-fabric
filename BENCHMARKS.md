# Ternary Fabric Benchmarks & Metrics

This document serves as the authoritative record for all performance benchmarks, resource utilization, and economic metrics for the Ternary Fabric project.

---

## 📖 Glossary of Terms

| Term | Definition |
| :--- | :--- |
| **Trit** | A balanced-ternary digit $\{-1, 0, 1\}$. |
| **TFD** | **Ternary Frame Descriptor**: The primary control structure for task submission. |
| **PT-5** | A high-density packing format encoding 5 trits into 8 bits (95.1% efficiency). |
| **Zero-Skip** | Hardware optimization that suppresses clocking and memory access for zero-value operands. |
| **Residency Hit** | Occurs when a required memory block is already present in the Fabric memory pool. |
| **Fabric Cost** | Cycle-aware metric: `(active_ops * 1.0) + (mem_reads * 5.0) + (mem_writes * 8.0) + (broadcasts * 2.0) + (residency_misses * 6.0)`. |
| **Semantic Efficiency** | Ratio of active operations to total operations (`active_ops / total_ops`). |
| **Economic Efficiency** | Ratio of active operations to total Fabric Cost (`active_ops / fabric_cost`). |
| **Ternary Lane** | The smallest parallel compute unit in the fabric. |
| **Tile** | A group of 15 Ternary Lanes sharing a single frame controller. |
| **Fabric Instance** | An independent processing entity with its own memory pool and worker thread. |

---

## 🚀 Performance Summary

### Layer 1: Synthetic (Hardware Limits)
*Measured using the Phase 21 Emulator at 250 MHz.*

| Configuration | Lanes | GOPS (Peak) | GOPS (Effective @ 50% Sparsity) | Zero-Skip Reduction |
| :--- | :--- | :--- | :--- | :--- |
| **Single Tile** | 15 | 7.5 | ~12.5 | 65-70% |
| **Aggregated (4 Tiles)** | 60 | 30.0 | ~50.0 | 66% |
| **High-Density (Projected)** | 1024 | 512.0 | ~850.0 | 65-72% |

### Layer 2: Kernel (Primitive Performance)
*Workload: GEMV / GEMM (1024x1024), T-LSTM (H=512, I=512)*

| Kernel | Time (10 iterations) | Cycles | Sparsity | Semantic Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **T-GEMM** | 0.076 s | 17,476 | 65% | 0.35 |
| **T-LSTM** | 0.072 s | 3,072 | 90% | 0.10 |

### Layer 3: Application (End-to-End)
*Workload: `mock_llama` (8 GEMV batches)*

| Metric | Measured Value |
| :--- | :--- |
| **Zero-Skip Reduction** | 95.0% |
| **Residency Hit Rate** | 100% (after initial hydration) |
| **Economic Efficiency** | 0.05 |
| **Pool Usage** | 0.5 MB / 128 MB (0.4%) |

---

## 🏗️ Hardware Synthesis (XC7Z020)

| Resource | Aggregated (4 Tiles) | % of Zynq-7000 |
| :--- | :--- | :--- |
| **LUTs** | ~14,000 | ~26% |
| **Flip-Flops** | ~24,000 | ~22% |
| **BRAM (36Kb)** | 16 | ~11% |
| **DSPs** | 0 | 0% |

---

## 📉 Multi-Fabric Orchestration (Phase 21)

| Metric | 1 Fabric | 2 Fabrics | 4 Fabrics (Projected) |
| :--- | :--- | :--- | :--- |
| **Throughput (GOPS)** | 30.0 | 60.0 | 120.0 |
| **Scheduling Overhead** | < 1% | < 2% | ~3% |
| **Lookahead Window** | N/A | 5 Kernels | 5 Kernels |

---

## 📝 Discrepancies & Notes
- **GOPS Calculation:** Peak GOPS are calculated based on a 250 MHz clock. Effective GOPS account for the throughput boost provided by Zero-Skip logic in sparse regimes.
- **Economic Efficiency:** The low efficiency (0.05) in application benchmarks is due to the small batch size and high memory-to-compute ratio in initial testing. This is expected to improve with larger model workloads.
- **Hardware vs Emulator:** Emulator results assume zero-latency memory access for PT-5 hydration, which may vary on physical hardware based on AXI bus contention.
