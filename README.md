# Ternary Fabric

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Build Status](https://github.com/ternary-fabric/ternary-fabric/actions/workflows/ci.yml/badge.svg)](https://github.com/ternary-fabric/ternary-fabric/actions)
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
```

---

## 🏗️ Architecture & Vision

Ternary Fabric operates as a **semantic execution substrate**, not a library rewrite. It creates a *memory illusion layer* that migrates hot weights into ternary residency pools and offloads compute onto parallel Fabric Tiles.

### Core Innovations
- **Zero-Skip:** Hardware suppression of clocking and memory access for zero-value operands.
- **PT-5 Packing:** High-density storage format encoding 5 trits into 8 bits (95.1% efficiency).
- **Multi-Fabric Orchestration:** Global task distribution with predictive scheduling and kernel fusion.
- **CPU Short-Circuiting:** Bypassing CPU compute loops once residency and offload are established.

---

## 🏗️ Project State & Architecture

The project is currently in **Phase 21**, representing a **Predictive Multi-Fabric Orchestration layer**:

* **Global Orchestration:** Coordinate workloads across multiple distinct TFMBS fabrics.
* **Predictive Scheduling:** Use lookahead telemetry to anticipate bottlenecks and optimize hot-state residency.
* **Cross-Fabric Fusion:** Virtual macro-kernels reduce inter-fabric communication and repeated hydration.
* **Adaptive Pipeline Depth:** Multi-stage execution (Pre-fetch -> Execute -> Commit) with dynamic depth control.

### Architecture Layers

The project has completed **Phase 21 (Predictive Multi-Fabric Orchestration)**. Key deliverables include:

- **Global Orchestrator:** Dynamic workload distribution across multiple Fabric Instances.
- **Predictive Scheduler:** 5-kernel lookahead for residency anticipation and hot-state pre-loading.
- **Cross-Fabric Fusion:** Automated locality optimization to eliminate inter-fabric data movement.
- **Three-Stage Pipeline:** Asynchronous execution (Pre-fetch -> Execute -> Commit) with adaptive depth.

### Performance at a Glance

| Configuration | Lanes | Peak GOPS | Zero-Skip Reduction |
| :--- | :--- | :--- | :--- |
| **Aggregated Fabric (4 Tiles)** | 60 | **30.0** | 66% |
| **Projected (High-Density)** | 1024 | **512.0** | 65–72% |

*Detailed metrics and terminology can be found in **[BENCHMARKS.md](BENCHMARKS.md)**.*

---

## 📖 Documentation Stack

- **[User Manual](USER_MANUAL.md):** Installation, LD_PRELOAD acceleration, and advanced features.
- **[Roadmap](docs/ROADMAP.md):** Phase-by-phase progression and deliverables.
- **[Whitepaper](WHITE_PAPER.md):** Technical narrative, architecture, and comparisons.
- **[Benchmarks](BENCHMARKS.md):** Authoritative performance and resource metrics.

---

## 📝 Discrepancies & Notes
- Phase 21 performance is verified in the emulator; hardware path acceleration is currently in "Mock" mode (Phase 10).
- Reported GOPS assume a 250 MHz target frequency for the fabric tiles.

---

© 2026 Ternary Fabric Project. Licensed under the Apache License, Version 2.0.
