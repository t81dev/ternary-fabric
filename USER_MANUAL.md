# Ternary Fabric User Manual (v0.1)

Welcome to the **Ternary Fabric** user manual. This documentation is designed to help hardware designers, software developers, and researchers understand and utilize the ternary-native acceleration substrate.

## 📖 Table of Contents

1.  **[Project Overview](docs/00_OVERVIEW.md)**
    What is the Ternary Fabric? Design philosophy and core innovation.
2.  **[Installation & Setup](docs/01_INSTALL.md)**
    Dependencies, build instructions, and environment setup.
3.  **[Quick Start Guide](docs/02_QUICK_START.md)**
    Run your first ternary operation in minutes.
4.  **[Hardware Architecture](docs/03_HARDWARE.md)**
    Deep dive into Tiles, TPE Lanes, and the hydration pipeline.
5.  **[Memory Map & Registers](docs/04_MEMORY_MAP.md)**
    AXI address mapping and register definitions.
6.  **[TFD & Execution Hints](docs/05_TFD.md)**
    The binary interface specification and kernel flags.
7.  **[Kernel Reference](docs/06_KERNELS.md)**
    Usage details for T-GEMM, T-CONV, and T-POOL.
8.  **[Software API Guide](docs/07_API.md)**
    How to use the `pytfmbs` Python library.
9.  **[Profiling & Optimization](docs/08_PROFILING.md)**
    Measuring Zero-Skip effectiveness and performance tuning.
10. **[How-To Tutorials](docs/09_TUTORIALS.md)**
    Quantization, multi-tile scaling, and DMA loading.
11. **[Appendices](docs/10_APPENDICES.md)**
    Acronyms, PT-5 details, and verification reports.

---

## 🚀 Key Examples
Check out the `examples/` directory for runnable scripts:
*   `quick_start.py`: Basic T-GEMM operation.
*   `multi_tile_tgemm.py`: Multi-tile and broadcast demonstration.
*   `dma_loader_demo.py`: Using the AXI-Stream DMA path.
*   `profiling_example.py`: Extracting hardware performance counters.

---
© 2024 Ternary Fabric Project. All rights reserved.
