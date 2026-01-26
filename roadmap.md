# Roadmap: ternary-fabric

This roadmap outlines the development phases for the Ternary Fabric Memory & Bus Specification (TFMBS). The goal is to move from normative definitions to a hardware-validated acceleration substrate.

## Phase 1: Specification & Emulation (Complete)

* [x] Finalize `TFMBS` v0.1 Core Specification.
* [x] Define `TFD` (Ternary Frame Descriptor) structure in `tfmbs.h`.
* [x] **Software Reference Library:** C-based PT-5 codec (`examples/pt5_pack_example.c`).
* [x] **Functional Simulator:** Python-based T-MAC validation (`tools/ternary_kernel_sim.py`).

## Phase 2: Tooling & ABI (Current)

* [x] **Initial Fabric Mediator:** Mockup created in `src/mediator_mock.c` for host-handshake simulation.
* [ ] **Kernel Library:** Formalize a set of standard kernel IDs (0x01: Dot Product, 0x02: Conv2D) within `tfmbs.h`.
* [ ] **Ternary-CLI:** A command-line tool to convert `.txt` or `.csv` ternary data into `.tfrm` (packed binary) files for testing.
* [ ] **Debug Visualizer:** A "Ternary Hex Dump" tool to inspect frame contents in balanced format.

## Phase 3: Hardware Prototyping (FPGA)

* [ ] **Ternary Processing Element (TPE):** RTL design of a single lane capable of sign-flip addition.
* [ ] **Vector Fabric:** Scaling TPEs into a multi-lane SIMD array as defined in `FRAME_MODEL.md`.
* [ ] **DMA Bridge:** AXI4-Lite control for TFDs and AXI4-Full for data streaming (Interconnect v1.0).
* [ ] **FPGA Validation:** Deploy on Xilinx/Intel FPGA to measure actual "Zero-Skip" power savings.

## Phase 4: Integration & AI Applications

* [ ] **Python Bindings:** Create `pyTFMBS` to allow NumPy arrays to be offloaded to the fabric.
* [ ] **Ternary Quantization Toolkit:** Tools to shrink 32-bit Float models into balanced ternary weights.
* [ ] **Native Signal Suite:** Implementation of Ternary Fast Fourier Transform (T-FFT) kernels.
* [ ] **ASIC Path:** Design-rule check (DRC) for native ternary SRAM cells for maximum density.

---

## Current Status of Files

| File | Status | Description |
| --- | --- | --- |
| `README.md` | ✅ | Project Overview |
| `specs/*.md` | ✅ | Full Normative Specification Suite |
| `include/tfmbs.h` | ✅ | C ABI Header (Source of Truth) |
| `src/mediator_mock.c` | 🛠️ | Initial handshake simulator |
| `tools/*.py` | ✅ | Reference codecs and math simulators |
| `Makefile` | ✅ | Automated build system |
| `test_suite.sh` | ✅ | CI/CD validation script |

---