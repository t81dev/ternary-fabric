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

## 🏁 Completed Phases (0–21)

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

### Phase 21 — Predictive Multi-Fabric Orchestration ✅
Elevating the fabric to a proactive, system-level efficiency management layer.
*   **Deliverables:**
    *   **Global Orchestrator:** Dynamic task distribution across multiple isolated fabric instances.
    *   **Predictive Scheduler:** Lookahead mechanism (Window=5) for hot-state anticipation and pre-loading.
    *   **Cross-Fabric Fusion:** Automated locality optimization for dependent kernel sequences.
    *   **Adaptive Multi-Stage Pipeline:** Three-stage asynchronous execution (Pre-fetch -> Execute -> Commit) with dynamic depth.

---

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

---

## 📝 Discrepancies & Notes
- All phases up to 21 are fully implemented in the emulator.
- Physical hardware synthesis (Phase 15/21) is verified against the XC7Z020 target.
- Hardware path (Phase 10) remains in "Mock" state for host-side verification.
