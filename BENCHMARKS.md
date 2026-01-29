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
| **Dense Trit SRAM** | Optimized storage packing 20 trits into 32 bits (99% efficiency). |

---

## 🚀 Performance Summary (Phase 21-25)

### Layer 1: Synthetic (Hardware Limits)
*Measured using the Phase 21 Emulator at 250 MHz.*

| Configuration | Lanes | GOPS (Peak) | GOPS (Effective @ 50% Sparsity) | Zero-Skip Reduction |
| :--- | :--- | :--- | :--- | :--- |
| **Single Tile** | 15 | 7.5 | ~15.0 | 65-70% |
| **Aggregated (4 Tiles)** | 60 | 30.0 | ~60.0 | 66% |
| **High-Density (Projected)** | 1024 | 512.0 | ~1000.0 | 65-72% |

### Layer 2: Kernel (Primitive Performance)
*Workload: 1024x1024 Kernels. Baselines: CPU (Reference C), Phase 20 (No Lookahead).*

| Kernel | Fabric (Phase 21) | Fabric (Phase 20) | CPU Baseline | Semantic Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **T-GEMM** | **7.5 GOPS** | 7.4 GOPS | 2.51 GOPS | 0.66 |
| **T-LSTM** | **7.5 GOPS** | 7.2 GOPS | 0.17 GOPS (Dot) | 0.66 |
| **T-Attention** | **7.5 GOPS** | 7.3 GOPS | 0.81 GOPS (est) | 0.34 |
| **T-Conv3D** | **7.5 GOPS** | 7.4 GOPS | 1.15 GOPS (est) | 0.30 |

*Note: Fabric GOPS are simulated based on a 250 MHz clock model in the emulator. CPU benchmarks are measured on the host system.*

### Layer 3: Application (End-to-End)
*Workload: `mock_llama` (8 GEMV batches)*

| Metric | Fabric (Phase 21) | CPU Baseline | Improvement |
| :--- | :--- | :--- | :--- |
| **Execution Time** | Accelerated (Bypass) | 0.012 s | **~10x (Projected)** |
| **Zero-Skip Reduction** | 95.0% | 0% | N/A |
| **Residency Hit Rate** | 100% | N/A | N/A |
| **Economic Efficiency** | 0.05 | N/A | N/A |

---

## 🏗️ Hardware Synthesis (XC7Z020)

| Resource | Aggregated (4 Tiles) | % of Zynq-7000 | Status |
| :--- | :--- | :--- | :--- |
| **LUTs** | ~14,000 | ~26% | Measured |
| **Flip-Flops** | ~24,000 | ~22% | Measured |
| **BRAM (36Kb)** | 16 | ~11% | Measured |
| **DSPs** | 0 | 0% | Measured |

---

## 📉 Multi-Fabric Orchestration (Phase 21 & 25)

| Metric | 1 Fabric | 2 Fabrics | 4 Fabrics | Multi-Node (RDMA) |
| :--- | :--- | :--- | :--- |
| **Throughput (GOPS)** | 30.0 | 60.0 | 120.0 | ~120.0 (Scale-out) |
| **Scheduling Overhead** | < 0.5% | < 1.2% | ~2.5% | ~4.0% |
| **Lookahead Window** | N/A | 5 Kernels | 5 Kernels | 5 Kernels |
| **Residency Hits (Global)** | 85% | 92% | 95% | 90% (Distributed) |

---

## 💎 Dense Ternary Storage (Phase 24)

| Format | Trits per 32-bit Word | Bits per Trit | Physical Efficiency |
| :--- | :--- | :--- | :--- |
| **Naive (2-bit)** | 16 | 2.0 | 79.2% |
| **PT-5 (8-bit)** | 18 | 1.77 | 94.5% |
| **Dense (PT-20)** | **20** | **1.6** | **99.1%** |

---

## 🧠 Hybrid Execution & Adaptive Agent (Phase 26)

The Adaptive Runtime Agent dynamically switches between Ternary Fabric and CPU Fallback based on real-time sparsity telemetry.

| Scenario | Sparsity | Path | Effective GOPS | Efficiency Uplift |
| :--- | :--- | :--- | :--- | :--- |
| **High Sparsity** | > 80% | **Fabric** | ~30.0 (per node) | 3.0x vs Dense |
| **Medium Sparsity** | 50% | **Fabric** | ~15.0 (per node) | 2.0x vs Dense |
| **Low Sparsity** | < 30% | **CPU Fallback** | Host-Limited | ~1.5x vs Stalled Fabric |

### Adaptive Performance (Stress Test)
*Measured using `tests/test_adaptive` with `TFMBS_SPARSITY_THRESHOLD=0.5`.*

- **Initial State:** Sparsity 0.8 -> **Fabric** (Offload count increases).
- **Transition:** Sparsity drops to 0.1 -> **CPU Fallback** (Fallback count increases).
- **Recovery:** Sparsity returns to 0.9 -> **Fabric** (Detected via periodic probe).

---

## 📝 Discrepancies & Notes
- **Baseline Data:** CPU baselines are measured using the `src/reference_tfmbs.c` implementation and `tests/mock_llama.c`.
- **Phase 20 Baseline:** Simulated by setting `TFMBS_DISABLE_LOOKAHEAD=1`, which disables the predictive Global Orchestrator.
- **Emulator Constraints:** Application-level interposer benchmarks were verified via unit tests; sustained application-level profiling in this environment uses projected scaling based on kernel performance due to environment-specific signal handling latency.
- **Raw Logs:** Detailed execution logs for all configurations can be found in `benchmarks/logs/`.
