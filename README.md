### ternary-fabric

This repository defines a **ternary-native memory and interconnect fabric** designed to accelerate compression, signal processing, and AI workloads within binary-dominated systems.

Unlike traditional research that seeks to replace binary CPUs, this fabric operates as a **specialized data plane** (execution adjacency) attached to a binary host, utilizing balanced-ternary semantics () to eliminate multiplication overhead and enable **Zero-Skip** hardware optimizations.

---

### 🏗️ Architecture at a Glance

The **Ternary Fabric Memory & Bus Specification (TFMBS)** is built on three pillars:

1. **Binary Sovereignty:** The host CPU manages task scheduling via **Ternary Frame Descriptors (TFDs)** over an AXI4-Lite control plane.
2. **Vectorized SIMD:** Hardware-native logic tiles **Ternary Processing Elements (TPEs)** into parallel lanes for massive throughput.
3. **Hydrated Frames:** Data is stored in **PT-5** (5 trits per byte) for 95.1% storage efficiency but "hydrated" into 2-bit signed logic for execution.

---

### 📊 Performance Metrics (Calculated)

The Ternary Fabric achieves extreme throughput by leveraging the zero-cost nature of ternary multiplication.

#### ⚡ Theoretical Throughput

Given a target FPGA clock of **250 MHz** and the current **15-lane** SIMD configuration:

* **Ops per Cycle:** 
* **Peak Throughput:**  (Giga-Operations Per Second).
* **Scaling:** Because TPE lanes are extremely small (no hardware multipliers), scaling to **1024 lanes** on a mid-range FPGA yields **512 GOPS** at significantly lower power than binary INT8.

#### 🔋 Energy Efficiency (Zero-Skip)

Unlike binary accelerators, the TPE lanes utilize **Zero-Skip Logic**:

* If either weight or input is `0`, the accumulator's clock-enable is suppressed.
* **Result:** Dynamic power consumption scales with data sparsity, perfect for sparse LLMs.

---

### 📂 Repository Structure

* `specs/`: Normative definitions for the Frame Model, Memory Bus, and AI Acceleration.
* `include/`: `tfmbs.h`—The C ABI and single source of truth for the fabric.
* `src/hw/`: Synthesizable Verilog RTL for the TPEs, Vector Engine, and AXI Interconnect.
* `src/pytfmbs/`: Python C-Extensions for hardware-level control.
* `tests/`: Verilator-based C++ testbenches for cycle-accurate hardware validation.
* `tools/`: Python utilities for **Quantization**, **PT-5 Packing**, and **Ternary Hex Dumps**.

---

### 🚀 Quick Start

#### 1. Hardware-Native Toolchain

Convert floating-point weights into ternary binary frames ready for the fabric:

```bash
# 1. Quantize weights to {-1, 0, 1}
python3 tools/quantize.py my_weights.npy -o weights.txt

# 2. Pack into PT-5 binary and generate TFD header
python3 tools/ternary_cli.py weights.txt --lanes 15 --kernel 1

# 3. Inspect the spatial mapping
python3 tools/txd.py weights.txt.tfrm

```

#### 2. Cycle-Accurate Benchmark

Verify the RTL using the Verilator-based hardware benchmark. This compiles the Verilog into a high-speed C++ executable to measure throughput:

```bash
# Compile and run the AXI-integrated hardware simulation
make benchmark_hw

```

---

### 🗺️ Roadmap Status: **Phase 4: Integration**

We have successfully bridged the gap between spec and silicon.

* **Phase 1 & 2:** Complete (Specification, ABI, and Tooling).
* **Phase 3 (Hardware):** Complete. RTL for TPEs, Vector Engine, and AXI-Lite Interconnect is verified.
* **Phase 4 (Integration):** **Current.** Focus on `pyTFMBS` orchestration and real-world TNN inference.

---

### 🤝 Contributing

We welcome contributions from system architects and RTL engineers. Please see `CONTRIBUTING.md` for our standards on **Binary Sovereignty** and **Zero-Skip logic** optimization.

---