## ternary-fabric

This repository defines a **ternary-native memory and interconnect fabric** designed to accelerate compression, signal processing, and AI workloads within binary-dominated systems.

Unlike traditional research that seeks to replace binary CPUs, this fabric operates as a **specialized data plane** (execution adjacency) attached to a binary host, utilizing balanced-ternary semantics  to eliminate multiplication overhead and enable zero-skip hardware optimizations.

---

### 🏗️ Architecture at a Glance

The **Ternary Fabric Memory & Bus Specification (TFMBS)** is built on three pillars:

1. **Binary Sovereignty:** The binary host remains authoritative, managing memory allocation and task scheduling via **Ternary Frame Descriptors (TFDs)**.
2. **Vectorized Lanes:** Data is organized into parallel lanes, allowing SIMD-style execution natively in ternary.
3. **Balanced Encoding:** Uses the **PT-5** packing scheme, squeezing 5 trits into every 8-bit byte ( states), achieving 95% storage efficiency on standard binary hardware.

---

### 📂 Repository Structure

* `specs/`: Normative definitions for the Bus, Interconnect, and AI Engine.
* `include/`: The `tfmbs.h` C header—the single source of truth for the ABI.
* `tools/`: Python utilities for packing, unpacking, and simulating ternary kernels.
* `src/`: Reference mediator mocks for simulating the host-to-fabric handshake.
* `examples/`: Sample code for frame initialization and data packing.

---

### 🚀 Quick Start

#### 1. Requirements

* GCC (for C examples)
* Python 3.x (for tools and simulation)

#### 2. Build and Test

Run the automated test suite to validate the codecs and the mediator mock:

```bash
chmod +x test_suite.sh
./test_suite.sh

```

#### 3. Using the CLI

Convert human-readable ternary strings into packed binary frames:

```bash
# Pack a string into a binary frame
python3 tools/tf-cli.py pack "++0-0++0-0" -o my_data.tfrm

# Unpack and verify
python3 tools/tf-cli.py unpack my_data.tfrm

```

---

### 🗺️ Roadmap

The project is currently in **Phase 2: Tooling & ABI**. We are actively working on:

* Formalizing the Kernel Library.
* Developing RTL (Verilog) for Ternary Processing Elements (TPEs).
* Creating a "Ternary Hex Dump" visualizer.

See [roadmap.md](https://www.google.com/search?q=roadmap.md) for the full path to hardware silicon.

---

### 🤝 Contributing

We welcome contributions from system architects and RTL engineers. Please see [CONTRIBUTING.md](https://www.google.com/search?q=CONTRIBUTING.md) for our standards on Binary Sovereignty and TFD-first development.

---