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
12. **[llama.cpp Acceleration Roadmap](archive/llama.cpp_roadmap.md)**
    Strategy for transparent device-level acceleration using Memory Interposition.

---

## 🚀 Key Examples
Check out the `examples/` directory for runnable scripts:
*   `quick_start.py`: Basic T-GEMM operation.
*   `multi_tile_tgemm.py`: Multi-tile and broadcast demonstration.
*   `dma_loader_demo.py`: Using the AXI-Stream DMA path.
*   `profiling_example.py`: Extracting hardware performance counters.
*   `tests/mock_llama.c`: Demonstrating `LD_PRELOAD` memory redirection.

---
© 2026 Ternary Fabric Project. All rights reserved.
