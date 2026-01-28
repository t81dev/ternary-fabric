# Phase 21: Predictive Multi-Fabric Orchestration

This document details the architecture and implementation of the Phase 21 orchestration layer for the Ternary Fabric.

## 1. Overview
Phase 21 transitions the TFMBS from a single-device accelerator to a system-level orchestrated platform. It introduces a central Global Orchestrator that manages workload distribution across multiple Fabric instances.

## 2. Global Orchestrator
The Orchestrator resides in `src/libtfmbs_device.c` and manages a staging queue for all incoming tasks. It runs in a dedicated background thread.

### Residency Tracking
The system maintains a global residency map (`g_residency_map`) that tracks which fabric instance currently holds a valid PT-5 representation of each memory buffer.

### Dispatch Heuristics
1. **Locality First:** If a buffer required by a task is already resident on a specific fabric, that fabric is prioritized.
2. **Load Balancing:** If residency is not established, tasks are distributed using a round-robin or least-loaded strategy.
3. **Automated Transfer:** If the orchestrator decides to move a task to a different fabric than where its data resides, it automatically inserts `KERNEL_TRANSFER` operations into the pipeline.

## 3. Predictive Scheduling (Lookahead)
The orchestrator maintains a **Lookahead Window of 5 kernels**. By inspecting future tasks, the scheduler can:
- **Anticipate Hot-State:** Pre-load weights onto a fabric that will be used by multiple upcoming tasks.
- **Minimize Transfers:** Avoid moving a buffer if a future task on the same fabric will need it.

## 4. Cross-Fabric Kernel Fusion
The scheduler detects dependencies between kernels (e.g., Task B uses the output of Task A). It prioritizes keeping these tasks on the same fabric to form a "virtual macro-kernel," eliminating the need for expensive inter-fabric memory movement.

## 5. Multi-Stage Pipeline
Each Fabric instance operates a three-stage pipeline:

| Stage | Action |
| :--- | :--- |
| **Pre-fetch** | Hydrates RAW buffers, packs them into PT-5, or performs inter-fabric copies. |
| **Execute** | Runs the native kernel (GEMV, LSTM, etc.) on the ternary tiles. |
| **Commit** | Finalizes the result buffer and signals the host handle. |

### Adaptive Pipeline Depth
The pipeline depth (`pipeline_depth`) dynamically scales:
- **Depth=3 (Full):** For high-density, compute-heavy workloads to maximize throughput.
- **Depth=1 (Short):** For sparse or low-latency workloads to minimize completion time.

## 6. Telemetry & Metrics
`economic_metrics.csv` has been extended to include:
- `fid`: The Fabric ID.
- `projected_cost`: The predicted cost used for scheduling.
- `efficiency`: The measured economic efficiency for that instance.

## 7. Configuration
The number of fabrics can be configured at runtime:
```bash
export TFMBS_NUM_FABRICS=4
LD_PRELOAD=./bin/libtfmbs_intercept.so ./my_app
```
(Default is 2 fabrics).
