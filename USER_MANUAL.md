# Ternary Fabric User Manual (v0.2)

Welcome to the **Ternary Fabric** user manual. This documentation is designed to help hardware designers, software developers, and researchers understand and utilize the ternary-native acceleration substrate.

## 📖 Table of Contents

1.  **[Project Overview](docs/00_OVERVIEW.md)**
    What is the Ternary Fabric? Design philosophy, core innovation, and Phase 15 status.
2.  **[Installation & Setup](docs/01_INSTALL.md)**
    Dependencies, build instructions, and environment setup.
3.  **[Quick Start Guide](docs/02_QUICK_START.md)**
    **Start Here!** Run your first ternary operation in minutes using the Python API.
4.  **[Hardware Architecture](docs/03_HARDWARE.md)**
    Deep dive into Multi-Tile Topology, TPE Lanes, and the hydration pipeline.
5.  **[Memory Map & Registers](docs/04_MEMORY_MAP.md)**
    AXI address mapping, register definitions, and multi-tile SRAM regions.
6.  **[TFD & Execution Hints](docs/05_TFD.md)**
    The binary interface specification, kernel flags, and packing formats.
7.  **[Kernel Reference](docs/06_KERNELS.md)**
    Usage details for accelerated kernels (T-GEMM, T-CONV3D, T-LSTM, T-ATTN).
8.  **[Software API Guide](docs/07_API.md)**
    Comprehensive guide to the `pytfmbs` Python library.
9.  **[Profiling & Optimization](docs/08_PROFILING.md)**
    Measuring Zero-Skip effectiveness and tuning for multi-tile performance.
10. **[How-To Tutorials](docs/09_TUTORIALS.md)**
    Step-by-step guides for quantization, multi-tile scaling, and DMA loading.
11. **[Appendices](docs/10_APPENDICES.md)**
    Acronyms, PT-5 details, and Phase 15 verification reports.
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
17. **[Strategy Roadmap](docs/ROADMAP.md)**
    The project roadmap detailing completed and future phases.

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

### Cooperative Mode (Explicit Registration)
If you want to avoid the overhead of the "First Scan" heuristic, you can explicitly register weight buffers by calling `fabric_register_weight(ptr, size)` from your application.

---

## ⚠️ Best Practices & Limitations

### When NOT to use Fabric
- **Small Matrices:** GEMV operations smaller than 1024 elements may incur more overhead than they save.
- **High-Precision Requirements:** Fabric is optimized for ternary weights. High-precision (FP32/BF16) weights will be quantized/compressed, which may impact accuracy.
- **Frequent Writes:** If you frequently write to weight buffers, the interposer will constantly need to re-pack them into PT-5 format.

## 📊 Interpreting Telemetry

When running with the interposer or Python API, the Fabric provides real-time telemetry:

*   **Zero-Skips:** The number of operations eliminated because an operand was zero. Typically 50-75% for LLM workloads.
*   **Semantic Efficiency:** The ratio of useful operations (`active_ops`) to total operations.
*   **Economic Efficiency:** The ratio of useful operations to total `fabric_cost` (weighted sum of ops, memory, and residency misses).
*   **Adaptive Coefficients:** (Phase 20) Real-time values for `weight_cost`, `mem_read_cost`, etc., reflecting the scheduler's learned cost model.
*   **Pool Usage:** Current consumption of the 128MB emulation pool.
*   **Eviction Events:** Increments when the policy (Frequency + Age + Success) frees space for new weight frames.
*   **Dynamic Batch Size:** (Phase 20) Shows the current auto-tuned batch size optimized for the current workload.

---

## 🤖 Adaptive Learning (Phase 20)

Starting with Phase 20, the Ternary Fabric is a self-tuning co-processor. It automatically optimizes its own internal parameters based on measured performance:

### Self-Tuning Cost Model
The scheduler uses a **Hill-Climbing** algorithm to adjust its cost projection coefficients. If the actual `fabric_cost` differs from the `projected_cost`, the fabric updates its internal model to ensure more accurate tile selection in the future.

### Dynamic Scheduler Weighting
Tiles "learn" which kernels they are most efficient at executing. A tile that consistently delivers higher **Economic Efficiency** for a specific kernel (e.g., LSTM) will be favored for that kernel in future scheduling decisions.

### Eviction Policy Optimization
The eviction policy is no longer fixed LRU. It now dynamically weights **Frequency**, **Recency (Age)**, and **Residency Success Rate**. This ensures that weights critical to maintaining high efficiency are protected from eviction.

### Temporal Auto-Tuning
The asynchronous command queue automatically adjusts its **Batch Size** (between 1 and 32) to maximize a composite score of efficiency and throughput. It occasionally "explores" different batch sizes to find the optimal throughput for the current sparsity regime.

---

## 🧠 Memory Management

### LRU Paging
The Fabric emulator manages a fixed 128MB pool of "Fabric Memory". If an allocation exceeds this or the pool is full, the system uses a **Least Recently Used (LRU)** policy to evict resident PT-5 frames. Evicted frames are transparently re-hydrated if accessed again by the host.

### Asynchronous Execution
Compute tasks (GEMV) are submitted to a background worker thread. The host application only blocks if it attempts to access a memory buffer that has a pending Fabric operation.

---

## 🚀 Key Examples
Check out the `examples/` directory for runnable scripts:
*   `quick_start.py`: Basic T-GEMM operation.
*   `multi_tile_tgemm.py`: Multi-tile and broadcast demonstration.
*   `dma_loader_demo.py`: Using the AXI-Stream DMA path.
*   `profiling_example.py`: Extracting hardware performance counters.
*   `tests/mock_llama.c`: Demonstrating `LD_PRELOAD` memory redirection and Phase 5+ features.

---
© 2026 Ternary Fabric Project. All rights reserved.
