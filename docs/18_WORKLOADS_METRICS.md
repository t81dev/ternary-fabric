# Phase 18: Ternary Workload Maturation & Measurement Plane

Phase 18 transitions the Ternary Fabric from an "acceleration shim" to an operational co-processing environment with defined semantics and rigorous measurement.

## 🎯 Objectives

1.  **Establish Workload Realism:** Move beyond isolated GEMV to complex, stateful workloads (**T-LSTM**).
2.  **Define the Programming Model:** Create a formal host-side API for fabric orchestration.
3.  **Implement a Measurement Plane:** Build a three-tier benchmark stack and a cycle-aware cost model.

## 📊 Three-Tier Benchmarking

We define a hierarchical approach to performance validation (see **[BENCHMARKS.md](../BENCHMARKS.md)** for authoritative results):

### Layer 1: Synthetic (Hardware Semantics)
- **Goal:** Stress the hardware limits.
- **Benchmarks:** Tile saturation, broadcast efficiency, **Zero-Skip** density curves.
- **Metrics:** Ops/cycle, tile utilization %, memory traffic.

### Layer 2: Kernel (Primitive Performance)
- **Goal:** Compare ternary kernels against binary baselines.
- **Benchmarks:** GEMV, Attention block, LSTM step.
- **Metrics:** Latency, effective GOPS, **Residency Hit** rate.

### Layer 3: Application (End-to-End)
- **Goal:** Quantify real-world impact.
- **Benchmarks:** `mock_llama`, small RNN classifier.
- **Metrics:** End-to-end latency, tokens/sec, **Economic Efficiency**.

## 🧠 T-LSTM Maturation

The **T-LSTM** kernel is promoted from a "reference/mock" to a primary kernel path.
- **State Persistence:** Support for `BIAS_EN` driven hidden state persistence within tile-local SRAM.
- **Recurrent Scheduling:** Optimized multi-tile scheduling for sequential time-step processing.
- **Host Integration:** Explicit `tfmbs_lstm_step` API.

## 💎 Host API Surface (C/C++ Primitives)

Defining the "Ternary Offload ABI" via high-level primitives:
- `tfmbs_tensor_bind(ptr, size, flags)`: Register and prepare a tensor for fabric residency.
- `tfmbs_gemm(...)`: Direct GEMM offload.
- `tfmbs_lstm_step(...)`: Single-step recurrent update.
- `tfmbs_sync(handle)`: Explicit synchronization points.

## 📉 Cycle-Aware Cost Model

The emulator implements a synthetic **Fabric Cost** function to proxy energy and efficiency:
```
fabric_cost = (active_ops * 1.0) + (mem_reads * 5.0) + (mem_writes * 8.0) + (broadcasts * 2.0) + (residency_misses * 6.0)
```
- **KPI:** **Economic Efficiency** (active operations / **Fabric Cost**).
- **Semantic Efficiency:** Active operations / total operations (**Zero-Skip** weighted).
