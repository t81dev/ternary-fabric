### ternary-fabric

This repository defines a **ternary-native memory and interconnect fabric** designed to accelerate compression, signal processing, and AI workloads within binary-dominated systems.

Unlike traditional research that seeks to replace binary CPUs, this fabric operates as a **specialized data plane** (execution adjacency) attached to a binary host, utilizing balanced-ternary semantics () to eliminate multiplication overhead and enable **Zero-Skip** hardware optimizations.

---

### 🏗️ Architecture at a Glance

The **Ternary Fabric Memory & Bus Specification (TFMBS)** is built on three pillars:

1. **Binary Sovereignty:** The host CPU manages task scheduling via **Ternary Frame Descriptors (TFDs)**.
2. **Vectorized SIMD:** Hardware-native logic tiles **Ternary Processing Elements (TPEs)** into parallel lanes for massive throughput.
3. **Hydrated Frames:** Data is stored in **PT-5** (5 trits per byte) for 95% storage efficiency but "hydrated" into 2-bit signed logic for logic execution.

---

### 📂 Repository Structure

* `specs/`: Normative definitions for the Frame Model, Memory Bus, and AI Acceleration.
* `include/`: `tfmbs.h`—The C ABI and single source of truth for the fabric.
* `src/hw/`: **(New)** Synthesizable Verilog RTL for the TPEs, Vector Engine, and AXI Interconnect.
* `src/pytfmbs/`: **(New)** Python C-Extensions for hardware-level control.
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

# 3. Inspect the hydration mapping
python3 tools/txd.py weights.txt.tfrm

```

#### 2. RTL Simulation

The fabric is designed for FPGA deployment. You can verify the `ternary_fabric_top.v` using Icarus Verilog or Vivado.

---

### 🗺️ Roadmap Status: **Phase 3/4 Transition**

We have successfully bridged the gap between spec and silicon.

* **Phase 1 & 2:** Complete.
* **Phase 3 (Hardware):** RTL for TPEs and AXI-Lite Interconnect is complete. Ready for FPGA validation.
* **Phase 4 (Integration):** Initial Python bindings and quantization tools are live.

---

### 🤝 Contributing

We welcome contributions from system architects and RTL engineers. Please see `CONTRIBUTING.md` for our standards on **Binary Sovereignty** and **Zero-Skip logic** optimization.

---