# Ternary Fabric

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://github.com/ternary-fabric/ternary-fabric/actions/workflows/ci.yml/badge.svg)](https://github.com/ternary-fabric/ternary-fabric/actions)
[![TFMBS Version](https://img.shields.io/badge/TFMBS-v0.1--draft-blue.svg)](include/tfmbs.h)

**Ternary Fabric** is a ternary-native memory and interconnect co-processor designed to accelerate AI and signal processing. By utilizing balanced-ternary semantics ({-1, 0, 1}), it eliminates multiplication overhead and enables fine-grained hardware optimizations like **Zero-Skip** and **Free-Negation**.

Ternary Fabric transparently intercepts AI workloads and executes sparse linear algebra using ternary hardware semantics to reduce power, memory traffic, and multiplication cost without rewriting models.

---

## ⚡ Quick Start & Prerequisites

### Prerequisites
*   **Python 3.8+**, **NumPy**, **Setuptools**, **GCC**.
*   **Optional:** Verilator, Icarus Verilog (for hardware simulation).
*   **Docker:** (Coming Soon) A pre-configured environment for development.

### Installation & Benchmarking
```bash
# 1. Build the Python extension and device library
make all

# 2. Run the benchmark suite to verify local performance
python3 tools/benchmark_suite.py

# 3. Explore a quick-start example
python3 examples/quick_start.py
```
For detailed setup instructions, see **[Installation & Setup](docs/01_INSTALL.md)**.

---

## 🤔 Why Ternary?

In traditional binary systems, multiplication is the most expensive operation in terms of area and power. Ternary computing ({-1, 0, 1}) turns multiplications into simple gated additions or subtractions.

### Performance vs. Sparsity
The Fabric thrives on sparsity. In typical LLM workloads with 50-70% zero-valued operands, the **Zero-Skip** hardware suppresses clock toggling, leading to massive power savings and effective throughput gains that binary accelerators cannot match.

```text
Effective Throughput
  ^
  |          / (Ternary Fabric with Zero-Skip)
  |         /
  |        /
  |-------/---- (Traditional Binary Accelerator)
  |      /
  |     /
  +---------------------------> Sparsity (%)
  0%         50%        100%
```

---

## 🚀 Real-World Use Case: `llama.cpp` Integration

Ternary Fabric provides **zero-patch acceleration** for `llama.cpp` through device-level memory interposition. It intercepts memory allocations and offloads GEMV operations to the fabric transparently.

```bash
# Enable Fabric acceleration with CPU short-circuiting
export FABRIC_SHORT_CIRCUIT=1
LD_PRELOAD=./libtfmbs_intercept.so ./llama-cli -m model.gguf
```

### Telemetry Insights
```text
[TFMBS-Telemetry] GEMV Completed
  - Zero-Skips: 172,401 (65.8% reduction in total operations)
  - Pool Usage: 84.2 MB / 128 MB (LRU Managed)
  - Async Queue: 0 in-flight (Non-blocking execution)
```

---

## 🏗️ Project State & Architecture

The project is currently in **Phase 15**, offering deep integration with PyTorch and GGUF, alongside native hardware acceleration for advanced kernels.

*   **Framework Layer:** Transparent PyTorch integration via `TFMBSLinear`, supporting automatic quantization and weight residency.
*   **Software Layer (Emulation):** High-fidelity multi-tile emulator with IOCTL support (`/dev/tfmbs`), GGUF v2/v3 parsing, and transparent `LD_PRELOAD` interposer.
*   **Hardware Layer (RTL):** ASIC-ready synthesizable RTL with native support for T-GEMM, T-CONV3D, T-LSTM, and T-Attention kernels.

| Configuration | Lanes | Clock | Throughput (Peak) |
| :--- | :--- | :--- | :--- |
| **Aggregated Fabric (4 Tiles)** | 60 | 250 MHz | **30.0 GOPS** |
| **Projected (High-Density)** | 1024 | 250 MHz | **512.0 GOPS** |

---

## 📖 Documentation

The **[User Manual](USER_MANUAL.md)** serves as the central landing page for all documentation.

*   **[Whitepaper](WHITE_PAPER.md):** Technical narrative, benchmarks, and comparison to related work.
*   **[Hardware Guide](docs/03_HARDWARE.md):** Deep dive into RTL, hydration, and TPE design.
*   **[Roadmap](docs/ROADMAP.md):** Future phases including Kernel Drivers (Phase 10) and Multi-Fabric scaling.

---

## 🤝 Contributing

We welcome contributions! Please see **[CONTRIBUTING.md](CONTRIBUTING.md)** for our standards on ternary-native optimization and code quality.

---
© 2026 Ternary Fabric Project. Licensed under the Apache License, Version 2.0.
