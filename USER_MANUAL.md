# Ternary Fabric User Manual (v0.1)

Welcome to the **Ternary Fabric** user manual. This documentation is designed to help hardware designers, software developers, and researchers understand and utilize the ternary-native acceleration substrate.

## 📖 Table of Contents

1.  **[Project Overview](docs/00_OVERVIEW.md)**
    What is the Ternary Fabric? Design philosophy, core innovation, and Phase 6b status.
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
    Usage details for Production (T-GEMM, T-CONV) and Experimental (T-LSTM, T-ATTN) kernels.
8.  **[Software API Guide](docs/07_API.md)**
    Comprehensive guide to the `pytfmbs` Python library.
9.  **[Profiling & Optimization](docs/08_PROFILING.md)**
    Measuring Zero-Skip effectiveness and tuning for multi-tile performance.
10. **[How-To Tutorials](docs/09_TUTORIALS.md)**
    Step-by-step guides for quantization, multi-tile scaling, and DMA loading.
11. **[Appendices](docs/10_APPENDICES.md)**
    Acronyms, PT-5 details, and Phase 6b verification reports.
12. **[Strategy Roadmap](docs/ROADMAP.md)**
    The project roadmap detailing completed and future phases.

---

## 🛠️ Enabling Fabric Acceleration

The Ternary Fabric can accelerate applications like `llama.cpp` without source code modifications using the `libtfmbs_intercept.so` interposer.

### Usage
```bash
# Enable Fabric acceleration with CPU short-circuiting
export FABRIC_SHORT_CIRCUIT=1
LD_PRELOAD=./libtfmbs_intercept.so ./my_app
```

### Environment Variables
*   `FABRIC_SHORT_CIRCUIT=1`: Enables the interposer to bypass CPU compute loops once weights are resident in the Fabric.
*   `TFMBS_DEBUG=1`: Enables verbose logging of memory allocations and offloading events.

---

## 📊 Interpreting Telemetry (Phase 9)

When running with the interposer, the Fabric provides real-time telemetry to `stderr`:

*   **Zero-Skips:** The number of operations eliminated because an operand was zero. Typically 50-75% for LLM workloads.
*   **Pool Usage:** Current consumption of the 128MB emulation pool.
*   **Eviction Events:** Increments when the LRU policy frees space for new weight frames.
*   **Async Queue:** Shows in-flight GEMVs being processed by the background worker thread.

---

## 🧠 Memory Management (Phase 7 & 8)

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
