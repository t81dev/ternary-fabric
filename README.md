# Ternary Fabric

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://github.com/ternary-fabric/ternary-fabric/actions/workflows/ci.yml/badge.svg)](https://github.com/ternary-fabric/ternary-fabric/)
[![TFMBS Version](https://img.shields.io/badge/TFMBS-v0.1--draft-blue.svg)](include/tfmbs.h)

**Ternary Fabric** is a ternary-native memory and interconnect co-processor designed to accelerate AI and signal processing. By utilizing **Balanced Ternary** semantics `{ -1, 0, +1 }`, it replaces traditional multiplication with gated logic, enabling extreme hardware optimizations like **Zero-Skip**.

> **In one line:** Ternary Fabric transparently intercepts AI workloads and executes sparse linear algebra using ternary hardware semantics to reduce power, memory traffic, and compute cost without rewriting models.

---

## ⚡ Quick Start

```bash
# 1. Build the project libraries and binaries
make all

# 2. Run the Phase 21 Multi-Fabric verification test
./bin/test_phase21

# 3. Explore the authoritative metrics
cat BENCHMARKS.md

# 4. Compile the Tfmbs MLIR plugin and run the regression guard
./tools/run_tfmbs_regression.sh
```

### 📌 Reference integration

```bash
python3 tools/reference_integration.py
```

This script rebuilds `bin/mock_llama`, runs a CPU baseline, reruns it through `tools/tfmbs-run`, and writes the latency summary to `logs/reference_integration.csv` along with a small bar chart. The Fabric Illusion heuristics exercised here are captured in `docs/FABRIC_ILLUSION_CONTRACT.md`.

`tools/run_tfmbs_regression.sh` automates the `cmake -B build -G Ninja` / `ninja -C build tfmbs_plugin` flow, locates `mlir-opt` (defaults to `../llvm-project/build-shared/bin/mlir-opt`), and then runs `tests/mlir/run_tfmbs_to_linalg.py` with the built plugin; set `LLVM_DIR`, `MLIR_DIR`, `BUILD_DIR`, `MLIR_OPT`, or `TFMBS_PLUGIN` before invoking the script to override the defaults so your environment mirrors CI.

## 📣 Public Readiness

Before the hardware racks (XC7Z020/XC7Z045) return, reference [`docs/PUBLIC_READINESS.md`] for:
- The public-facing story, compiler/telemetry workflow, and measurable evidence you can highlight in talks or write-ups.  
- The offline tooling (`tools/run_tfmbs_regression.sh`, `tools/run_hw_dma_telemetry.sh`, `tools/capture_dma_telemetry.py`) that keeps DMA/telemetry, regression, and dashboard validation running without the FPGA.  
- Clear outreach notes that list what’s ready today and the remaining hardware verification tasks waiting on the FPGA testbeds.

---

## 🏗️ Architecture & Vision

Ternary Fabric operates as a **semantic execution substrate**, not a library rewrite. It creates a *memory illusion layer* that migrates hot weights into ternary residency pools and offloads compute onto parallel Fabric Tiles. The Fabric Illusion contract is now captured in `docs/FABRIC_ILLUSION_CONTRACT.md` so reviewers can inspect exactly which allocations, signal handlers, and heuristics are part of the intercept.

### Core Innovations
- **Zero-Skip:** Hardware suppression of clocking and memory access for zero-value operands.
- **PT-5 Packing:** High-density storage format encoding 5 trits into 8 bits (95.1% efficiency).
- **Multi-Fabric Orchestration:** Global task distribution with predictive scheduling and kernel fusion.
- **CPU Short-Circuiting:** Bypassing CPU compute loops once residency and offload are established.

---

## 🏗️ Project State & Architecture

The project is currently in **Phase 25**, extending the fabric to a **Distributed Multi-Node Orchestration**:

* **Global Orchestration:** Coordinate workloads across multiple distinct TFMBS fabrics and physical nodes.
* **Predictive Scheduling:** Use lookahead telemetry to anticipate bottlenecks and optimize hot-state residency.
* **Simulated RDMA:** Socket-based inter-process communication for scaling across clusters.
* **DMA Ring Buffer:** Realistic driver-level interaction with asynchronous descriptor queues.

### Architecture Layers

The project has completed **Phase 25 (Simulated RDMA Multi-Node Orchestration)**. Key deliverables include:

- **Global Orchestrator:** Dynamic workload distribution across multiple Fabric Instances.
- **Predictive Scheduler:** 5-kernel lookahead for residency anticipation and hot-state pre-loading.
- **Cross-Fabric Fusion:** Automated locality optimization to eliminate inter-fabric data movement.
- **Three-Stage Pipeline:** Asynchronous execution (Pre-fetch -> Execute -> Commit) with adaptive depth.
- **TFMBS-MLIR Dialect:** First-class compiler support for ternary-native ops (`gemv`, `pack`, `transfer`).
- **Dense Trit SRAM:** 99% efficient 1.58-bit storage packing.

### Performance at a Glance

| Configuration | Lanes | Peak GOPS | Zero-Skip Reduction |
| :--- | :--- | :--- | :--- |
| **Aggregated Fabric (4 Tiles)** | 60 | **30.0** | 66% |
| **Projected (High-Density)** | 1024 | **512.0** | 65–72% |

Detailed throughput, density, and zero-skip numbers come from the Phase 21–25 measurement plane documented in **[BENCHMARKS.md](BENCHMARKS.md)**. The same cycle-aware cost model and economic efficiency definitions appear in `docs/18_WORKLOADS_METRICS.md`, so you can map “zero skip” counts to Fabric Cost directly.

## 🧭 North-Star Metric

Our north-star KPI is **zero-skip MACs avoided per Fabric Cost** (`zero_skips / fabric_cost`), borrowing its topology from `docs/18_WORKLOADS_METRICS.md`. It keeps the story on “work avoided vs. the energy proxy we already report” instead of chasing peak GOPS, and the `tools/reference_integration.py` helper makes a simple chart that showcases the delta between the CPU baseline and the intercept run.

---

## 📖 Documentation Stack

- **[User Manual](USER_MANUAL.md):** Installation, LD_PRELOAD acceleration, and advanced features.
- **[Roadmap](docs/ROADMAP.md):** Phase-by-phase progression and deliverables.
- **[Whitepaper](WHITE_PAPER.md):** Technical narrative, architecture, and comparisons.
- **[Benchmarks](BENCHMARKS.md):** Authoritative performance and resource metrics.
- **[Fabric Illusion Contract](docs/FABRIC_ILLUSION_CONTRACT.md):** What the interposer intercepts, the guarantees it offers, and the safe failure modes we support.
- **[Phase 3 Integration Handoff](docs/PHASE3_INTEGRATION_HANDOFF.md):** Ecosystem handoff checklist, measurable signals, and reproducible evidence path.

## 🧪 Compiler Regression

- Exercise the tfmbs-to-linalg pipeline via the regression helper script:
  ```bash
  python3 tests/mlir/run_tfmbs_to_linalg.py \
    --mlir-opt=../llvm-project/build-shared/bin/mlir-opt \
    --plugin=build/libtfmbs_plugin.dylib
  ```
  The script loads the tfmbs plugin as a dialect plugin before running `--pass-pipeline="builtin.module(tfmbs-to-linalg)"` so the update guardrails detect when `linalg.matmul` disappears or when tfmbs ops leak into the lowered IR. Set `MLIR_OPT` for alternate builds or wrap this invocation into CI.
- **Comprehensive lit guard:** `ninja -C build check-tfmbs` runs the `tests/mlir/lit.cfg` suite (custom `tfmbs_fusion_regression.mlir`/`tfmbs_multi_fusion_regression.mlir`) that uses the same regression helper to ensure both single-stage and multi-stage fusion traces lower cleanly while preserving telemetry metadata.
- **CI coverage:** GitHub Actions now performs the shared-MLIR build, runs `tools/torch_to_tfmbs.py` to regenerate the telemetry-rich Torch fixture, and executes `tests/mlir/run_tfmbs_to_linalg.py` (default + Torch fixture) with the dialect plugin so every merge confirms `linalg.matmul` appears even when telemetry metadata is present.
- **Operator fusion regression:** The CI job additionally runs `tests/mlir/run_tfmbs_to_linalg.py --mlir=tests/mlir/tfmbs_fusion.mlir` so the new `tfmbs-fuse` pass can combine sequential GEMV kernels before lowering and still emit `linalg.matmul`.
- **Fusion dashboard:** Use `pytfmbs.AdaptiveRuntimeAgent.save_history("logs/adaptive_history.json")` after running workloads, then run `tools/adaptive_dashboard.py` to compare `fusion_order`/`fusion_sparsity` against the compiler hints stored in `tests/mlir/torch_tfmbs.mlir`.

**Track status:** Track B (compiler & MLIR) is now fully operational—dialect, passes, lit guard, regression wiring, and the dashboard loop all run off the shared LLVM/MLIR build—while Track A awaits access to the XC7Z020/XC7Z045 boards so the FPGA verification checklist and synthesis docs can be filled in.

---

## 📝 Discrepancies & Notes
- Phase 25 performance is verified via inter-process simulation; physical hardware synthesis is verified against the XC7Z020 target.
- Reported GOPS assume a 250 MHz target frequency for the fabric tiles.

---

© 2026 Ternary Fabric Project. Licensed under the Apache License, Version 2.0.
