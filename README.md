# Ternary Fabric

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![TFMBS Version](https://img.shields.io/badge/TFMBS-v0.1--draft-blue.svg)](include/tfmbs.h)

**Ternary Fabric** is a ternary-native memory and interconnect co-processor designed to accelerate AI and signal processing workloads. It operates as an **execution adjacency** to a binary host, utilizing balanced-ternary semantics ({-1, 0, 1}) to eliminate multiplication overhead and enable fine-grained hardware optimizations.

---

## 🌟 Key Features & Highlights

*   **Multi-Tile Scaling:** Parameterized architecture (default **4 tiles**) with private SRAMs and a shared global frame controller.
*   **Vectorized SIMD:** Each tile features **15 parallel TPE lanes**, providing high-throughput hydration and execution.
*   **Hardware Optimizations:**
    *   **Zero-Skip:** Suppresses clock toggling when operands are zero, reducing dynamic power.
    *   **Free-Negation:** Zero-cost sign-flipping for weights/inputs.
    *   **Weight Broadcast:** Efficient multi-tile weight distribution for GEMM and CONV kernels.
*   **Hydrated Frames:** Data is stored in **PT-5** format (5 trits per byte) for **95.1% storage efficiency** and hydrated into 2-bit signed logic during execution.
*   **Rich Kernel Library:** Hardware-native support for **T-GEMM**, **T-CONV2D**, **T-MAXPOOL**, and **T-DOT**.
*   **Experimental Kernels:** Reference-backed support for **T-Conv3D**, **T-LSTM**, and **T-Attention** in the software layer.
*   **ASIC Ready:** Uses behavioral SRAM wrappers and standard AXI4-Lite interfaces, ready for synthesis in advanced process nodes.

---

## 🔭 Project Vision

The **Ternary Fabric** aims to redefine AI acceleration by treating ternary logic as a first-class citizen. Our vision is to provide a synthesizable, high-velocity hardware substrate that bridges the gap between binary-dominated host systems and the emergent field of Ternary Neural Networks (TNNs). By specializing in trits, we unlock optimizations (Zero-Skip, Free-Negation) that are physically impossible in binary arithmetic.

## 🚀 Quick Start & Benchmarks

The fastest way to get started is using the Python API.

1.  **Build the Python Extension:**
    ```bash
    make python_ext
    ```

2.  **Run Benchmarks:**
    Check the performance on your local system:
    ```bash
    python3 tools/benchmark_suite.py
    ```

3.  **Run an Example:**
    Explore the `examples/` directory for common use cases:
    ```bash
    # Basic T-GEMM operation
    python3 examples/quick_start.py

    # Multi-tile and weight broadcast demo
    python3 examples/multi_tile_tgemm.py

    # Profiling Zero-Skip effectiveness
    python3 examples/profiling_example.py
    ```

For detailed setup instructions, see **[Installation & Setup](docs/01_INSTALL.md)**.

---

## 🏗️ Architecture Summary

### Multi-Tile Topology
The fabric scales by tiling independent processing units that operate in lock-step via a shared Frame Controller.

```text
       AXI4-Lite Control Plane / AXI-Stream DMA
                  |
        +---------V------------------+
        |     Frame Controller       | (Global Control)
        +---------|------------------+
                  | (Shared Control Bus: Start, Op, Stride)
        +---------+---------+---------+---------+
        |         |         |         |         |
    +---V---+ +---V---+ +---V---+ +---V---+     |
    | Tile 0| | Tile 1| | Tile 2| | Tile 3| ... | (NUM_TILES=4)
    +-------+ +-------+ +-------+ +-------+     |
        |         |         |         |         |
        +---------+---------+---------+---------+
                  |
        (Aggregated Vector Results / Counters)
```

### The TPE Lane (Processing Element)
Each lane replaces power-hungry binary multipliers with simple gating and sign-flip logic.

```text
Weight (2b) --+       Input (2b) --+
              |                    |
        +-----V--------------------V-----+
        |       Zero-Skip Logic          | (Clock Gate)
        +-----|--------------------|-----+
              |                    |
        +-----V--------------------V-----+
        |       Sign-Flip Logic          | (Inverter)
        +-----|--------------------|-----+
              |                    |
        +-----V--------------------V-----+
        |       32-bit Accumulator       |
        +--------------------------------+
```

---

## 📊 Performance Metrics

The Ternary Fabric achieves extreme throughput by leveraging the zero-cost nature of ternary multiplication.

| Configuration | Lanes | Clock | Throughput (Peak) |
| :--- | :--- | :--- | :--- |
| **Per Tile** | 15 | 250 MHz | **7.5 GOPS** |
| **Aggregated Fabric (4 Tiles)** | 60 | 250 MHz | **30.0 GOPS** |
| **Projected (High-Density)** | 1024 | 250 MHz | **512.0 GOPS** |

*Note: 1 MAC (Multiply-Accumulate) is counted as 2 Operations. GOPS = Giga-Operations Per Second.*

---

## 📖 Documentation

*   **[Whitepaper](WHITE_PAPER.md):** Technical overview and benchmark results.
*   **[User Manual](USER_MANUAL.md):** The central landing page for all documentation.
*   **[Visual Architecture](docs/visuals.md):** Diagrams and workflows of the fabric.
*   **[TFMBS Specification](specs/):** Normative definitions for the Frame Model and Bus.
*   **[API Guide](docs/07_API.md):** Detailed documentation for the `pytfmbs` Python library.
*   **[Hardware Guide](docs/03_HARDWARE.md):** Deep dive into RTL and architecture.

---

## 🛠️ Roadmap Status: Phase 6b Complete

We have successfully implemented and verified the multi-tile scaling architecture.

*   ✅ **Phase 1-4:** Specification, ABI, RTL, and AXI Integration.
*   ✅ **Phase 5:** Kernel Extensions (T-CONV, T-POOL).
*   ✅ **Phase 6a/b:** Multi-tile Scaling, Weight Broadcast, and Profiling API.
*   🧪 **Experimental:** T-Conv3D, T-LSTM, and T-Attention kernels (Python reference).
*   📅 **Next Steps:** FPGA Deployment and TNN (Ternary Neural Network) model zoo integration.

---

## 🤝 Contributing

We welcome contributions! Please see **[CONTRIBUTING.md](CONTRIBUTING.md)** for our standards on ternary-native optimization and code quality.

---
© 2026 Ternary Fabric Project. All rights reserved.
