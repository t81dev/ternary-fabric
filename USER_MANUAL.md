# Ternary Fabric User Manual (v0.2)

Welcome to the **Ternary Fabric** user manual. This documentation is designed to help hardware designers, software developers, and researchers understand and utilize the ternary-native acceleration substrate.

## 📖 Table of Contents

1.  **[Project Overview](docs/00_OVERVIEW.md)**
    What is the Ternary Fabric? Design philosophy, core innovation, and Phase 21 status.
2.  **[Installation & Setup](docs/01_INSTALL.md)**
    Dependencies, build instructions, and environment setup.
3.  **[Quick Start Guide](docs/02_QUICK_START.md)**
    **Start Here!** Run your first ternary operation in minutes using the Python API.
4.  **[Hardware Architecture](docs/03_HARDWARE.md)**
    Deep dive into Multi-Tile Topology, **Ternary Lanes**, and the hydration pipeline.
5.  **[Memory Map & Registers](docs/04_MEMORY_MAP.md)**
    AXI address mapping, register definitions, and multi-tile SRAM regions.
6.  **[TFD & Execution Hints](docs/05_TFD.md)**
    The binary interface specification, kernel flags, and packing formats.
7.  **[Kernel Reference](docs/06_KERNELS.md)**
    Usage details for accelerated kernels (T-GEMM, T-CONV3D, T-LSTM, T-ATTN).
8.  **[Software API Guide](docs/07_API.md)**
    Comprehensive guide to the `pytfmbs` Python library.
9.  **[Profiling & Optimization](docs/08_PROFILING.md)**
    Measuring **Zero-Skip** effectiveness and tuning for multi-tile performance.
10. **[How-To Tutorials](docs/09_TUTORIALS.md)**
    Step-by-step guides for quantization, multi-tile scaling, and DMA loading.
11. **[Appendices](docs/10_APPENDICES.md)**
    Acronyms, **PT-5** details, and verification reports.
12. **[Multi-Tile Scaling](docs/11_MULTI_TILE.md)**
    Details on multi-tile topology and masking.
13. **[PyTorch Integration](docs/12_PYTORCH.md)**
    Using the Ternary Fabric within the PyTorch deep learning framework.
14. **[GGUF Model Optimizations](docs/14_GGUF.md)**
    Optimizing llama.cpp and GGUF workflows.
15. **[Data-Driven Adaptation](docs/19_DATA_DRIVEN_ADAPTATION.md)**
    Details on Phase 19 cost-aware scheduling and adaptive residency.
16. **[Learning & Self-Tuning](docs/20_LEARNING_AND_SELF_TUNING.md)**
    Phase 20 documentation on adaptive cost modeling and scheduler weighting.
17. **[Predictive Multi-Fabric Orchestration](docs/21_MULTI_FABRIC_ORCHESTRATION.md)**
    Phase 21 documentation on global orchestration and predictive scheduling.
18. **[Strategy Roadmap](docs/ROADMAP.md)**
    The project roadmap detailing completed and future phases.
19. **[Benchmarks & Metrics](BENCHMARKS.md)**
    Authoritative record for performance, resources, and economic metrics.

---

## 🛠️ Enabling Fabric Acceleration

The Ternary Fabric can accelerate applications like `llama.cpp` without source code modifications using the `libtfmbs_intercept.so` interposer.

### Usage
The easiest way to run an application with Fabric acceleration is using the `tfmbs-run` tool:
```bash
./tools/tfmbs-run ./my_app
```

### Environment Variables
*   `FABRIC_SHORT_CIRCUIT=1`: Enables the interposer to bypass CPU compute loops once weights are resident in the Fabric.
*   `TFMBS_DEBUG=1`: Enables detailed real-time logging of memory registry changes, GEMV offloads, and async waits.
*   `TFMBS_VALIDATE=1`: Bit-exact comparison of Fabric results against a CPU reference for every operation.
*   `TFMBS_NUM_FABRICS=n`: Configures the number of independent **Fabric Instances** (Phase 21).

---

## ⚠️ Best Practices & Limitations

### When NOT to use Fabric
- **Small Matrices:** GEMV operations smaller than 1024 elements may incur more overhead than they save.
- **High-Precision Requirements:** Fabric is optimized for ternary weights. High-precision (FP32/BF16) weights will be quantized/compressed, which may impact accuracy.
- **Frequent Writes:** If you frequently write to weight buffers, the interposer will constantly need to re-pack them into **PT-5** format.

## 📊 Interpreting Telemetry

When running with the interposer or Python API, the Fabric provides real-time telemetry:

*   **Zero-Skips:** The number of operations eliminated because an operand was zero. Typically 50-75% for LLM workloads.
*   **Semantic Efficiency:** The ratio of useful operations (`active_ops`) to total operations.
*   **Economic Efficiency:** The ratio of useful operations to total **Fabric Cost** (weighted sum of ops, memory, and residency misses).
*   **Adaptive Coefficients:** (Phase 20) Real-time values for `weight_cost`, `mem_read_cost`, etc., reflecting the scheduler's learned cost model.
*   **Pool Usage:** Current consumption of the 128MB emulation pool.
*   **Eviction Events:** Increments when the policy frees space for new weight frames in the residency pool.
*   **Dynamic Batch Size:** (Phase 20) Shows the current auto-tuned batch size optimized for the current workload.

---

## 🌐 Predictive Multi-Fabric Orchestration (Phase 21)

Starting with Phase 20, the Ternary Fabric is a self-tuning co-processor. It automatically optimizes its own internal parameters based on measured performance.

## 🌐 Multi-Fabric Orchestration (Phase 21)

Phase 21 elevates the TFMBS from a single adaptive co-processor to a **multi-fabric orchestration layer**. It provides proactive, system-level efficiency management:

### Global Orchestration
Workloads are dynamically distributed across multiple isolated fabric instances. The system tracks buffer residency across all fabrics and automatically manages inter-fabric data movement (transfers) to minimize latency.

### Predictive Scheduling (Lookahead)
The orchestrator uses a **lookahead window of 5 kernels** to anticipate future task requirements. It selects the optimal fabric for the current task by considering where the weights will be needed next, effectively implementing **hot-state anticipation**.

### Cross-Fabric Fusion
The scheduler identifies dependent task sequences (e.g., GEMV output feeding an LSTM gate update) and prioritizes keeping them on the same fabric. This virtual "macro-kernel" approach reduces repeated hydration and inter-fabric communication.

### Adaptive Multi-Stage Pipeline
Each fabric manages a three-stage asynchronous pipeline:
1.  **Pre-fetch:** Handles buffer hydration, PT-5 packing, and inter-fabric transfers.
2.  **Execute:** Performs the native ternary kernel computation.
3.  **Commit:** Finalizes results and signals task completion.

The pipeline depth automatically adjusts based on workload density—extending for throughput on dense kernels and shortening for low latency on sparse workloads.

---

## 🧠 Memory Management

### LRU Paging
The Fabric emulator manages a fixed 128MB pool of "Fabric Memory". If an allocation exceeds this or the pool is full, the system uses a **Least Recently Used (LRU)** policy to evict resident **PT-5** frames. Evicted frames are transparently re-hydrated if accessed again by the host.

### Asynchronous Execution
Compute tasks are submitted to a background worker thread. The host application only blocks if it attempts to access a memory buffer that has a pending Fabric operation.

---

## 🚀 Key Examples
Check out the `examples/` directory for runnable scripts:
*   `quick_start.py`: Basic T-GEMM operation.
*   `multi_tile_tgemm.py`: Multi-tile and broadcast demonstration.
*   `dma_loader_demo.py`: Using the AXI-Stream DMA path.
*   `profiling_example.py`: Extracting hardware performance counters.
*   `tests/test_phase21_multi_fabric.c`: Verifying global orchestration and cross-fabric dependency management.

---
© 2026 Ternary Fabric Project. All rights reserved.
