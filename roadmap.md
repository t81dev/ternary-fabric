# Roadmap: ternary-fabric

This roadmap outlines the development phases for the Ternary Fabric Memory & Bus Specification (TFMBS). The goal is to move from normative definitions to a hardware-validated acceleration substrate.

## Phase 1: Specification & Emulation (Complete)

* [x] Finalize `TFMBS` v0.1 Core Specification.
* [x] Define `TFD` (Ternary Frame Descriptor) structure in `tfmbs.h`.
* [x] **Software Reference Library:** C-based PT-5 codec.
* [x] **Functional Simulator:** Python-based T-MAC validation.

## Phase 2: Tooling & ABI (Complete)

* [x] **Initial Fabric Mediator:** Mockup created for host-handshake simulation.
* [x] **Kernel Library:** Standard kernel IDs (Dot, Mul, etc.) mapped in `tfmbs.h`.
* [x] **Ternary-CLI:** Command-line tool (`ternary_cli.py`) for `.tfrm` and `.tfd` generation.
* [x] **Debug Visualizer:** Ternary Hex Dump (`txd.py`) for spatial visualization.

## Phase 3: Hardware Prototyping (RTL Complete)

* [x] **Ternary Processing Element (TPE):** RTL design with Zero-Skip and Sign-Flip logic.
* [x] **Vector Fabric:** Tiled SIMD array (`vector_engine.v`) per `FRAME_MODEL.md`.
* [x] **Bus Logic:** PT-5 Hydration/Unpacking logic for high-density memory traversal.
* [x] **DMA/Control Bridge:** AXI4-Lite control plane (`axi_interconnect_v1.v`) for TFD programming.
* [ ] **FPGA Validation:** Deploy on Xilinx/Intel FPGA to measure power and timing.

## Phase 4: Integration & AI Applications (Current)

* [x] **Python Bindings:** Create `pyTFMBS` C-Extension for hardware orchestration.
* [x] **Ternary Quantization Toolkit:** `quantize.py` for mapping FP32 weights to .
* [x] **Unified Build System:** `Makefile` and `setup.py` for full-stack deployment.
* [ ] **ASIC Path:** Design-rule check (DRC) for native ternary SRAM cells for maximum density.

---

## Updated Status of Files

| File | Status | Description |
| --- | --- | --- |
| `include/tfmbs.h` | ✅ | ABI Header with Kernel IDs and TFD struct. |
| `tools/ternary_cli.py` | ✅ | PT-5 Packing and TFD header generator. |
| `tools/quantize.py` | ✅ | FP32 to Ternary thresholding utility. |
| `src/hw/tpe_lane.v` | ✅ | Single lane ALU (Sign-flip logic). |
| `src/hw/fabric_top.v` | ✅ | Top-level AXI-integrated accelerator. |
| `src/pytfmbs/core.c` | ✅ | C-Extension for Python-to-Hardware MMIO. |
| `Makefile` | ✅ | Orchestrates C, Python, and Verilog builds. |

---