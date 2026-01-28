# Ternary Fabric

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://github.com/ternary-fabric/ternary-fabric/actions/workflows/ci.yml/badge.svg)](https://github.com/ternary-fabric/ternary-fabric/actions)
[![TFMBS Version](https://img.shields.io/badge/TFMBS-v0.1--draft-blue.svg)](include/tfmbs.h)

**Ternary Fabric** is a ternary-native memory and interconnect co-processor designed to accelerate AI and signal processing. By utilizing balanced-ternary semantics `{ -1, 0, +1 }`, it replaces general multiplication with gated add/subtract and sign logic, enabling fine-grained hardware optimizations like **Zero-Skip** and **Free-Negation**.

> **In one line:** Ternary Fabric transparently intercepts AI workloads and executes sparse linear algebra using ternary hardware semantics to reduce power, memory traffic, and multiplication cost without rewriting models.

---

## ⚡ Quick Start & Prerequisites

### Prerequisites

* **Python 3.8+**, **NumPy**, **Setuptools**, **GCC**
* **Optional:** Verilator, Icarus Verilog (for hardware simulation)
* **Docker:** (Coming Soon) A pre-configured environment for development

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

In traditional binary systems, multiplication dominates area, power, and routing cost. Balanced ternary `{ -1, 0, +1 }` converts many multiply paths into conditional add, subtract, or bypass operations.

This enables:

* **Zero-Skip:** No toggle, no fetch, no accumulate when operand = 0.
* **Free-Negation:** Sign inversion without multiplier hardware.
* **Sparse-first execution:** Control logic prioritizes semantic absence over arithmetic presence.

### Performance vs. Sparsity

The Fabric thrives on sparsity. In typical GGUF / LLM layers after ternary quantization:

* **55–72%** of operands become zero.
* Zero-Skip suppresses both compute *and memory toggle* for those lanes.
* Energy per operation drops super-linearly with sparsity.

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

Binary accelerators still pay fetch and multiply cost for zeros. Fabric does not.

---

## 🪞 The Fabric Illusion

Ternary Fabric operates as a **semantic execution substrate**, not a library rewrite.

It creates a *memory illusion layer* that:

* Intercepts allocations and GEMV calls.
* Migrates hot weights into ternary residency pools.
* Offloads compute onto Fabric tiles.
* Short-circuits CPU execution when Fabric execution is available.

From the application’s point of view, nothing changes — but execution migrates underneath the program.

This allows **zero-patch acceleration** of existing AI software.

---

## 🧠 How Fabric Executes a Layer

When an application executes a linear layer, Fabric reshapes execution beneath the program:

1. **Invocation**
   The application issues a GEMV / GEMM through PyTorch, GGUF, or native code.

2. **Interception**
   The Fabric interposer captures allocations and compute calls using the illusion layer (`LD_PRELOAD` / IOCTL).

3. **Residency Migration**
   Hot weights are migrated into ternary residency pools managed with LRU and tile locality.

4. **Quantization**
   Operands are converted into balanced ternary `{ -1, 0, +1 }` form.

5. **Sparse Scheduling**
   Tiles schedule lanes with **Zero-Skip**:

   * `0` → bypass
   * `+1` → add
   * `-1` → subtract

6. **Execution**
   Accumulators perform gated adds/subtracts with sign control instead of full multipliers.

7. **Return Illusion**
   Results are written back into host-visible memory while CPU execution is short-circuited when possible.

Conceptually:

```
Binary:   y = W · x
Fabric:   y = Σ (sign(wᵢ) * xᵢ) ,   wᵢ ∈ {-1,0,+1}
          skip if wᵢ = 0
```

Fabric accelerates not by computing faster, but by **not computing at all when semantics allow omission**.

---

## 🆚 What Makes Fabric Different

Ternary Fabric is not a conventional accelerator. It combines ternary arithmetic with memory illusion and sparse-first scheduling.

| Capability              | GPU              | TPU              | Binary NPU       | **Ternary Fabric**            |
| ----------------------- | ---------------- | ---------------- | ---------------- | ----------------------------- |
| Zero handling           | Multiplies       | Multiplies       | Multiplies       | **Skipped in hardware**       |
| Negation cost           | Extra op         | Extra op         | Extra op         | **Free sign flip**            |
| Sparse-native           | Partial          | Partial          | Partial          | **First-class**               |
| Memory illusion         | No               | No               | Limited          | **Transparent interposition** |
| Patch-free acceleration | No               | No               | No               | **Yes**                       |
| Residency pools         | Limited          | Limited          | Limited          | **Native ternary residency**  |
| Execution model         | Arithmetic-first | Arithmetic-first | Arithmetic-first | **Semantic-first**            |

Where GPUs accelerate arithmetic, Fabric accelerates **semantic absence**: zeros, signs, bypasses, and residency locality.

---

## 🚀 Real-World Use Case: `llama.cpp` Integration

Ternary Fabric provides transparent acceleration for `llama.cpp` through device-level memory interposition. It intercepts memory allocations and GEMV operations and offloads them to the fabric automatically.

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

Telemetry exposes sparsity, residency, and queue pressure in real time.

---

## 🏗️ Project State & Architecture

The project is currently in **Phase 15**, representing full-stack operation:

* PyTorch frontends
* GGUF ingestion
* Multi-tile emulation
* IOCTL device layer
* Interposition illusion
* RTL kernel parity

### Architecture Layers

* **Framework Layer**
  Transparent PyTorch integration via `TFMBSLinear`, supporting automatic quantization and weight residency.

* **Software Layer (Emulation)**
  High-fidelity multi-tile emulator with `/dev/tfmbs` IOCTL support, GGUF v2/v3 parsing, async queues, and `LD_PRELOAD` interposer.

* **Hardware Layer (RTL)**
  ASIC-ready synthesizable RTL with native support for T-GEMM, T-CONV3D, T-LSTM, and T-Attention kernels.

### Fabric Configurations

| Configuration                   | Lanes | Clock   | Throughput (Peak) |
| :------------------------------ | :---- | :------ | :---------------- |
| **Aggregated Fabric (4 Tiles)** | 60    | 250 MHz | **30.0 GOPS**     |
| **Projected (High-Density)**    | 1024  | 250 MHz | **512.0 GOPS**    |

*GOPS = ternary operations per second after Zero-Skip suppression.

---

## 📖 Documentation

The **[User Manual](USER_MANUAL.md)** is the central landing page.

* **[Whitepaper](WHITE_PAPER.md):** Technical narrative, benchmarks, and comparisons.
* **[Hardware Guide](docs/03_HARDWARE.md):** RTL, hydration, and TPE design.
* **[Roadmap](docs/ROADMAP.md):** Future phases including kernel drivers and multi-fabric scaling.

---

## 🤝 Contributing

We welcome contributions across research and implementation.

Areas of interest include:

* Ternary quantization strategies
* Sparsity scheduling
* Memory illusion design
* Kernel semantics
* Residency management
* Interposer robustness

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for standards on ternary-native optimization and code quality.

---

© 2026 Ternary Fabric Project. Licensed under the Apache License, Version 2.0.

---
