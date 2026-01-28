# 🧭 Strategy Roadmap — Device-Level Fabric Acceleration

---

## 🧱 Core Architecture

Target illusion:

```
llama.cpp
   ↓
Virtual Memory (OS)
   ↓
Fabric Driver (kernel / userspace)
   ↓
TFMBS Device (PCIe / MMIO / DMA)
   ↓
Ternary Execution + Memory Fabric
```

llama.cpp believes it reads/writes RAM.
Fabric actually:

* compresses weights,
* keeps them resident,
* skips zeros,
* executes dot products internally.

This mirrors GPU Unified Memory / CXL.mem style systems.

---

## 🏁 Completed Phases

### Phase 0 — Define the Fabric Device Contract ✅
Defined the normative device contract for TFMBS.
*   **Deliverable:** `TFMBS_DEVICE_SPEC.md`
*   **Status:** Complete. ABI for memory + execution established.

### Phase 1 — Emulated Device (User-Space First) ✅
Created a bit-exact software emulator for the Fabric.
*   **Deliverable:** `libtfmbs_device.so`
*   **Status:** Complete. Supports PT-5 packing, skip logic, and SIMD execution.

### Phase 2 — Memory Interposition Layer ✅
Implemented transparent memory redirection for host applications.
*   **Deliverable:** `libtfmbs_intercept.so`
*   **Status:** Complete. Successfully intercepts `malloc`, `mmap`, and `memcpy` via `LD_PRELOAD`.

### Phase 3 — Pattern Recognition for Compute ✅
Detected weight-loading and GEMV-compute patterns in real-time.
*   **Deliverable:** Heuristic-based pattern matching in interposer.
*   **Status:** Complete. Uses `SIGSEGV` + `mprotect` to track access scans.

### Phase 4 — Weight Residency & Compression ✅
Automatic migration of weights to ternary-native formats.
*   **Deliverable:** Auto-packing pipeline (RAW → PT-5).
*   **Status:** Complete. Weights are compressed and kept resident in Fabric memory.

### Phase 5 — Execution Injection ✅
Redirected CPU compute loops to Fabric hardware kernels.
*   **Deliverable:** Transparent GEMV offloading with **CPU Short-Circuiting**.
*   **Status:** Complete. CPU execution is bypassed once residency is established.

### Phase 6 — Zero-Skip + SIMD Enablement ✅
Activated native ternary power-saving and throughput features.
*   **Deliverable:** Hardware-backed Zero-Skip and lane-parallel execution.
*   **Status:** Complete. Verified ~64-76% operation reduction in benchmarks.

### Phase 7 — Paging & Eviction ✅
Managed large models exceeding physical Fabric memory.
*   **Deliverable:** LRU-based block allocator.
*   **Status:** Complete.
    *   **LRU Policy:** Implemented a fixed 128MB Fabric pool using a page-aligned block allocator.
    *   **Eviction:** When the pool is exhausted, the Least Recently Used (LRU) non-busy block is evicted.
    *   **Rehydration:** Evicted PT-5 frames are transparently re-packed from host RAM upon the next access.
    *   **Safety:** Implemented `busy_count` pinning to prevent eviction of blocks currently in the async pipeline.

### Phase 8 — Asynchronous Pipelining ✅
Overlapped host processing with Fabric execution.
*   **Deliverable:** Command queue and background worker thread.
*   **Status:** Complete.
    *   **Worker Thread:** A dedicated background thread processes GEMV tasks from a thread-safe queue.
    *   **Non-blocking API:** `fabric_exec_gemv_async` returns immediately with a handle.
    *   **Implicit Sync:** The interposer uses `mprotect(PROT_NONE)` on output buffers and `SIGSEGV` traps to automatically call `fabric_wait` when the host attempts to read results.

### Phase 9 — Telemetry & Proof ✅
Real-time visibility into Fabric performance and efficiency.
*   **Deliverable:** Integrated terminal dashboard.
*   **Status:** Complete.
    *   **Metrics:** Real-time tracking of Zero-Skips (%), Pool Residency (MB), and total Eviction counts.
    *   **Reporting:** Automatic telemetry dump to `stderr` upon completion of each asynchronous task.
    *   **Validation:** Verified ~65% operation reduction on standard LLM GEMV patterns.

---

## 🛠️ Current & Future Phases

### Phase 10 — Hardware Path (Mock Device Interface) ✅
Transitioned from pure userspace emulation to a mock kernel-space driver interface.
* Exposed Fabric via IOCTL interface (`TFMBS_IOC_SUBMIT`).
* Implemented bit-exact Hardware Abstraction Layer (HAL).

### Phase 11 — Multi-Tile & Multi-GPU Scaling ✅
Scale execution across multiple Fabric tiles within a single device.
* Support for `tile_mask` in GEMV operations.
* Dynamic workload partitioning across active tiles (15-60 lanes).

### Phase 12 — Framework Integration (PyTorch/TF) ✅
Bring "Fabric Illusion" to high-level deep learning frameworks.
*   **Status:** Complete.
*   **Deliverable:** `src/pytfmbs/torch.py` and `TFMBSLinear` module.
*   **Features:** Custom `torch.autograd` functions for Fabric offload, automatic weight quantization, and residency management.

### Phase 13 — Large-Model Support & Multi-Layer Batching ✅
Optimizing for models exceeding 70B+ parameters.
*   **Status:** Complete.
*   **Deliverable:** `TFMBSSequential`, `prefetch()` API, and `run_batch` in `pytfmbs`.
*   **Features:** Asynchronous `submit`/`wait`, multi-layer pipelining, and double-buffering support.

### Phase 14 — GGUF Model Optimizations ✅
Deep integration with the GGUF file format and llama.cpp specific optimizations.
*   **Status:** Complete.
*   **Deliverable:** `src/pytfmbs/gguf.py` and `GGUFReader`.
*   **Features:** Direct loading of Q4_0 and F32 GGUF weight blocks into Fabric with automatic ternary conversion.

### Phase 15 — Experimental Kernel Maturation ✅
Promotion of reference kernels to full hardware acceleration.
*   **Status:** Complete.
*   **Deliverables:** Updated `frame_controller.v` and `ternary_lane_alu.v` with native support for 3D Convolution, LSTM, and Attention kernels.
*   **Features:** Squared-stride memory addressing for CONV3D and `BIAS_EN` driven state persistence for recurrent/attention workloads.

### Phase 18 — Ternary Workload Maturation & Measurement Plane ✅
Anchoring the fabric with workload realism and a formal programming model.
*   **Status:** Complete.
*   **Deliverables:**
    *   **Three-Tier Benchmark Suite:** Synthetic, Kernel, and Application-level measurement.
    *   **T-LSTM Promotion:** Native kernel path for recurrent stateful workloads.
    *   **Host API Surface:** Formal C/C++ primitives for fabric orchestration.
    *   **Cost Model:** Cycle-aware emulator tracking "fabric_cost" (ops + memory weighted).

### Phase 19 — Data-Driven Adaptation (Cost-Aware Fabric) ✅
Using Phase 18 metrics to drive autonomous fabric behavior and scheduling.
*   **Status:** Complete.
*   **Deliverables:**
    *   **Economic Introspection:** Exposing projected costs, rebates, and eviction scores via `economic_metrics.csv`.
    *   **Hysteresis Scheduling:** Stabilized tile selection via sticky-tile affinity and cost-smoothing.
    *   **Sparse-Regime Hardening:** Verified efficiency in 95-99% sparse regimes with dedicated stress benchmarks.
    *   **KPI Maturation:** Introduction of `Economic Efficiency` (Meaning / Cost) vs `Semantic Efficiency` (Meaning / Ops).

---

# 🔑 What This Strategy Gives You

✅ Zero llama.cpp modifications
✅ Fabric as memory + compute substrate
✅ Transparent acceleration
✅ Works with existing GGUF models
✅ Matches Fabric’s identity as *memory fabric*

Instead of being a “backend,” Fabric becomes **part of the machine**.

---
