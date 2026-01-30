# 🧭 Strategy Roadmap — Device-Level Fabric Acceleration

The Ternary Fabric project implements a "Fabric Illusion" where the accelerator acts as a semantic memory and compute substrate, transparently offloading AI workloads from a binary host.

---

## 🧱 Execution Model

```
Application (e.g. llama.cpp)
   ↓
Virtual Memory (OS)
   ↓
Fabric Interposer (LD_PRELOAD)
   ↓
Global Orchestrator (libtfmbs_device.so)
   ↓
Fabric Instance (Emulator / Hardware)
```

---

## 🏁 Completed Phases (0–25)

### Phase 0: Device Contract ✅
- Defined the normative ABI for TFMBS memory and execution.
- Deliverable: `specs/TFMBS_DEVICE_SPEC.md`.

### Phase 1: Bit-Exact Emulator ✅
- Software-side reference for PT-5 packing and Zero-Skip logic.
- Deliverable: `src/fabric_emulator.c`.

### Phase 2: Memory Interposition ✅
- Transparent redirection of host allocations (`malloc`, `mmap`).
- Deliverable: `bin/libtfmbs_intercept.so`.

### Phase 3: Pattern Recognition ✅
- Real-time detection of weight-loading and compute scans via `SIGSEGV` traps.
- Deliverable: Interposer-level heuristic engine.

### Phase 4: Weight Residency ✅
- Automatic migration and packing (RAW → PT-5) of resident weights.
- Deliverable: Residency pool management in `libtfmbs_device.so`.

### Phase 9: Integrated Telemetry ✅
- Real-time visibility into sparsity, residency, and cycle counts.
- Deliverable: Terminal dashboard and telemetry logging.

### Phase 10: Hardware Path (Mock) ✅
- Transition to kernel-space driver interface via IOCTLs.
- Deliverable: `/dev/tfmbs` mock driver and HAL.

### Phase 11: Multi-Tile Scaling ✅
- Workload partitioning across parallel tiles (15 lanes per tile).
- Deliverable: `tile_mask` support in execution kernels.

### Phase 12: PyTorch Integration ✅
- "Fabric Illusion" support for high-level deep learning frameworks.
- Deliverable: `TFMBSLinear` module and autograd functions.

### Phase 13: Large-Model Batching ✅
- Optimizing for models exceeding 70B+ parameters.
- Deliverable: Asynchronous pre-fetch and double-buffering.

### Phase 14: GGUF Model Support ✅
- Direct loading of quantized GGUF weight blocks into Fabric memory.
- Deliverable: `pytfmbs.gguf` reader and packer.

### Phase 15: Experimental Kernel Maturation ✅
- Promotion of reference kernels to full hardware acceleration (RTL).
- Deliverable: T-GEMM, T-Conv3D, T-LSTM, and T-Attention RTL.

### Phase 18: Workload Maturation & Measurement ✅
- Establishing a rigorous three-tier benchmark plane and cost model.
- Deliverable: **Fabric Cost** and **Economic Efficiency** KPIs.

### Phase 19: Data-Driven Adaptation ✅
- Using metrics to drive autonomous scheduling and hysteresis.
- Deliverable: `economic_metrics.csv` and cost-aware scheduling.

### Phase 20: Learning & Self-Tuning ✅
- Hill-climbing feedback loops for cost coefficients and batch sizes.
- Deliverable: Adaptive scheduler weighting and eviction priorities.

### Phase 21: Predictive Multi-Fabric Orchestration ✅
- System-level management across multiple isolated Fabric Instances.
- Deliverables:
  - **Global Orchestrator:** Dynamic task distribution.
  - **Predictive Scheduler:** 5-kernel lookahead mechanism.
  - **Cross-Fabric Fusion:** Dependency-aware locality optimization.
  - **Three-Stage Pipeline:** Pre-fetch -> Execute -> Commit.

### Phase 22: Physical FPGA Synthesis & Hardware Verification (Driver Layer) ✅
- Implementation of a realistic DMA ring-buffer driver.
- Deliverables:
  - **DMA Ring Buffer:** Producer-consumer model for descriptor processing.
  - **Asynchronous IOCTLs:** `TFMBS_IOC_SUBMIT_DMA` for high-throughput batching.

### Phase 23: TFMBS-MLIR Dialect (Definitions) ✅
- Establishing TFMBS as a first-class citizen in the MLIR ecosystem.
- Deliverables:
  - **MLIR Dialect:** `tfmbs` dialect ODS definition.
  - **Core Ops:** `pack`, `unpack`, `gemv`, and `transfer` operations.

### Phase 24: Native Ternary SRAM & Dense Packing ✅
- Optimizing the physical substrate for ternary density.
- Deliverables:
  - **Ternary SRAM Model:** Behavioral Verilog for PT-20 dense packing (99% efficiency).

### Phase 25: RDMA-based Multi-Node Scaling (Simulation) ✅
- Extending orchestration to disaggregated clusters via simulated network.
- Deliverables:
  - **Simulated RDMA:** Socket-based inter-process fabric communication.
  - **Multi-Node Orchestrator:** Node-aware task dispatching.

### Phase 26: Adaptive Runtime Agent & Hybrid Execution ✅
- Intelligent runtime layer that chooses optimal execution path (Fabric vs. CPU).
- Deliverables:
  - **Adaptive Runtime Agent:** Decision logic based on sparsity EMA.
  - **CPU Fallback Path:** Native host-side execution for low-sparsity kernels.
  - **Hysteresis & Probing:** Logic to prevent "stuck" fallback states.
  - **Extended Telemetry:** Tracking of offload vs. fallback ratios.

---

## 🚀 Upcoming Phases & Strategic Tracks

The roadmap is now organized into parallel tracks to accelerate hardware sovereignty, ecosystem integration, and large-scale deployment.

### 🛤️ Track A: Hardware Sovereignty (FPGA to ASIC)
*Priority: Highest immediate | Timeline: 0–12 months*

**Phase 22: Physical FPGA Synthesis & Hardware Verification (RTL Handoff)**
Moving beyond the "Mock" driver (Phase 10) to real bitstream execution on Xilinx Zynq-7000 (XC7Z020/XC7Z045) hardware.
- **Deliverables:**
  - **Validated RTL:** Synthesizable Verilog/SystemVerilog for TFMBS tiles and lanes.
  - **Hardware-in-the-Loop (HIL):** Integration with physical IOCTLs for kernel dispatch.
  - **Silicon Benchmarks:** Real-world measurements of power, thermal, and cycle counts on FPGA.

**Phase 24: Native Ternary SRAM & Custom Logic Gating (Physical Design)**
Optimizing the physical substrate for ternary density by moving away from standard binary SRAM blocks.
- **Deliverables:**
  - **Ternary SRAM Models:** SPICE/GDSII models for optimized 1.58-bit storage cells.
  - **Advanced Gating:** Fine-grained clock and power gating for Zero-Skip at the gate level.
  - **Refined RTL:** Optimized ternary arithmetic logic units (TALU) for high-frequency targets.

**Phase 27: ASIC Tape-out Readiness & High-Density Fabric**
Finalizing the architecture for physical fabrication (e.g., 7nm/12nm nodes).
- **Deliverables:**
  - **GDSII-Ready RTL:** Hardened RTL package for physical synthesis handoff.
  - **Power/Area Maps:** Comprehensive modeling for high-density (1024+ lane) configurations.
  - **ASIC Efficiency Proof:** Projected >500 GOPS/W performance metrics.

---

### 🛤️ Track B: Compiler & Ecosystem Integration
*Priority: High leverage for adoption | Timeline: 3–9 months*

**Phase 23: TFMBS-MLIR Dialect & Compiler Integration (Optimization & Lowering)**
Establishing TFMBS as a first-class citizen in the MLIR/LLVM ecosystem to enable transparent model portability.
- **Deliverables:**
  - **Lowering Passes:** Automated conversion from Torch-MLIR and ONNX to TFMBS kernels.
  - **Torch/ONNX fixtures:** `tools/torch_to_tfmbs.py` converts exported Torch/ONNX graphs into `tests/mlir/torch_tfmbs.mlir` that already carries `telemetry` metadata, which downstream passes lower into `linalg.matmul` with the same hints.
  - `pytfmbs.TFMBSLinear.telemetry_hint` mirrors the same dictionary (layer name, sparsity, tile mask) so Python front ends can emit the compile-time hints that the Adaptive Runtime Agent consumes later.
  - **CI regression:** GitHub Actions clones `llvm/llvm-project`, builds shared MLIR, runs `tools/torch_to_tfmbs.py`, and executes `tests/mlir/run_tfmbs_to_linalg.py` with the dialect plugin to guarantee `linalg.matmul` is present before every merge.
  - **Operator Fusion:** `TfmbsFusionPass` now collapses telemetry-aligned GEMV pairs into `tfmbs.fused_gemv`, the fusion pass is exercised by `tests/mlir/tfmbs_fusion.mlir`, and CI runs the fusion+lowering pipeline so fused kernels still output `linalg.matmul`.
  - **Multi-stage coverage:** CI also runs `tests/mlir/tfmbs_multi_fusion.mlir` to show the fusion pass covers longer pipelines while preserving telemetry hints.
  - **Adaptive dashboard:** `pytfmbs.AdaptiveRuntimeAgent` logs fusion telemetry so `tools/adaptive_dashboard.py` can compare runtime fusion_order/fusion_sparsity trends against the compiler hints for Phase 26 scheduling.
  - **Compiler Plan:** See `docs/compiler_track_plan.md` for the current MLIR dialect status, TableGen targets, and next work items while hardware verification is pending.

**Phase 26: Dynamic Semantic Scheduling & Precision Adaptation** ✅
Using real-time telemetry to adjust execution precision and semantic depth based on model layer sensitivity.
- **Deliverables:**
  - **Adaptive Runtime Agent:** Logic to dynamically switch between ternary and binary (fallback) paths.
  - **Sensitivity Telemetry:** Metrics to identify "hot" vs. "sensitive" layers during inference.
  - **Hybrid Execution Policies:** Optimized heuristics for balancing accuracy and efficiency. (Software layer complete).

---

### 🛤️ Track C: Extreme Scale & Distributed Orchestration
*Priority: Medium-term / Data-center focus | Timeline: 6–18 months*

**Phase 25: RDMA-based Multi-Node Scaling (Hardware RDMA)**
Extending Phase 21 orchestration to disaggregated clusters, targeting models that exceed single-device memory.
- **Deliverables:**
  - **Hardware RDMA:** FPGA-side RDMA engines for direct fabric-to-fabric transfer.
  - **Global Residency Map:** Tracking PT-5 weight residency across a distributed fabric.
  - **100B+ Model Support:** End-to-end inference for massive models (e.g., BitNet-100B) via networked fabrics.

---

## 🎯 Major Strategic Milestones

- **Milestone 1: The Silicon Reality (Q3 2026)**
  End-to-end ternary inference on physical FPGA with verified >50x efficiency vs. CPU baseline for 7B-scale models.
- **Milestone 2: The Data-Center Fabric (Q4 2026)**
  Seamless distributed execution of 70B+ models across multi-node TFMBS clusters.
- **Milestone 3: ASIC-Ready Architectural Handoff (2027)**
  Finalized architectural package for physical fabrication, targeting >500 GOPS/W.

---

## 📝 Discrepancies & Notes
- All phases up to 21 are fully implemented in the emulator.
- Physical hardware synthesis (Phase 15/21) is verified against the XC7Z020 target.
- Hardware path (Phase 10) remains in "Mock" state for host-side verification.
