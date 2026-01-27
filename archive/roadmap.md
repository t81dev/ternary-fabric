## # Roadmap: ternary-fabric

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

## Phase 3: Hardware Prototyping (Complete)

* [x] **Ternary Processing Element (TPE):** RTL design with Zero-Skip and Sign-Flip logic.
* [x] **Vector Fabric:** Tiled SIMD array (`vector_engine.v`) with PT-5 unpackers.
* [x] **Bus Logic:** AXI4-Lite control plane and SRAM integration.
* [x] **Verilator Validation:** Cycle-accurate verification of AXI handshakes and MAC results.

## Phase 4: Integration & Toolchain (Complete)

* [x] **Python Bindings:** Create `pyTFMBS` C-Extension for hardware orchestration.
* [x] **Ternary Quantization Toolkit:** `quantize.py` for mapping FP32 weights.
* [x] **Unified Build System:** `Makefile` and `setup.py` for full-stack deployment.
* [x] **Free Negation & Zero-Skip:** Implemented in RTL and supported by `pytfmbs`.
* [x] **T-GEMM Kernel:** Multi-lane matrix multiplication kernel added to fabric.

## Phase 5: End-to-End Workflow & Profiling (Complete)

* [x] **End-to-End Inference:** Run complete `.tfrm` workloads from host to fabric.
* [x] **Performance Profiling:** Implement hardware counters for lane utilization and Zero-Skip effectiveness.
* [x] **Dynamic Lane Handling:** Support for per-lane bitmasking and partial-lane operations.
* [x] **T-GEMM Verification:** Validate hardware matrix operations against a C reference.
* [x] **Optional: DMA Loader:** Hardware-accelerated `.tfrm` streaming via AXI DMA.

## Phase 6a: Optimization & Extended Workloads (Complete)

* [x] **Hardware DMA Loader:** AXI-based streaming of `.tfrm` files directly into fabric SRAM.
* [x] **Extended Kernel Library:** Added T-CONV and T-POOL kernels to the fabric and mock.
* [x] **Profiling Enhancements:** Per-lane counters (active cycles), overflow flags, and burst latency tracking.
* [x] **End-to-End Regression Suite:** Automated `pytest` suite for kernel validation and counter verification.

## Phase 6b: Deployment & Scaling (Current)

* [/] **FPGA Validation:** Synthesizable RTL updated for multi-tile configuration. Fmax estimates pending.
* [x] **Multi-Tile Scaling:** Expanded `ternary_fabric_top.v` with parameterized `NUM_TILES`.
* [x] **Tile-Aware Control:** Implemented `tile_mask` support in TFD and RTL.
* [/] **ASIC Path:** Behavioral SRAM wrapper with clock-enable hooks for power gating.

---

## Updated Status of Files

| File | Status | Description |
| --- | --- | --- |
| `include/tfmbs.h` | ✅ | ABI Header with `tile_mask` in TFD. |
| `src/hw/ternary_fabric_top.v` | ✅ | Parameterized Multi-Tile Fabric Top. |
| `src/hw/axi_interconnect_v1.v` | ✅ | Tile-aware address decoder and broadcast logic. |
| `src/hw/ternary_sram_wrapper.v` | ✅ | ASIC-ready behavioral SRAM with power hooks. |
| `src/pytfmbs/core.c` | ✅ | Multi-tile mock simulation and extended API. |
| `docs/validation/phase6b/` | 📂 | FPGA synthesis and validation logs. |
| `tests/test_multi_tile.py` | 🛠️ | Multi-tile regression suite (In progress). |

---