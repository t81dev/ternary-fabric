# Phase 19: Data-Driven Adaptation (Cost-Aware Fabric)

Phase 19 focuses on transforming the Fabric from a passive executor into an active, economically-driven co-processor. It leverages the telemetry established in Phase 18 to make autonomous decisions about scheduling, residency, and optimization.

## 1. Cost-Aware Scheduler

The scheduler now selects tiles based on **projected cost** rather than fixed or round-robin assignment.

### projected_cost(tile, kernel, tensors)
- **Residency Hits:** Prefers tiles where weights or inputs are already resident.
- **Broadcast Reuse:** Accounts for the cost of moving data between tiles.
- **Memory Reads/Writes:** Estimates the I/O cost for the specific kernel.
- **Tile Local Reuse:** Favors tiles that have previously processed similar data.

## 2. Residency Policy Engine

Adaptive memory management replaces static LRU.

### Scoring Heuristic
Blocks are scored based on:
- **Recency:** Time since last access.
- **Frequency:** Total number of accesses.
- **Semantic Efficiency:** How much "meaning" (active ops) the block contributes per cost unit.

### Policies
- `keep_hot_state`: Prevents eviction of frequently used activation/state buffers.
- `evict_cold_weights`: Prioritizes eviction of weights with long reuse distances.
- `adaptive_pinning`: Automatically pins blocks that show high temporal locality.

## 3. Sparse-Regime Optimization

Reduces overhead for workloads where Zero-Skip is highly active.

- **Micro-kernel Fusion:** Fuses adjacent operations (e.g., GEMV + Bias + Activation) to reduce memory round-trips.
- **Control Flow Collapse:** Batches small asynchronous tasks to minimize worker thread wake-up overhead.

## 4. Temporal Pipelines

Optimizes recurrent and agentic workloads (LSTM, RNN).

- **Asynchronous Hydration:** Prefetches the next step's inputs/weights while the current step is computing.
- **State Persistence:** Keeps hidden states local to tiles to minimize fabric-wide broadcasts.

## 5. Metric-Driven Auto-Tuning

Rolling averages and baseline comparisons allow the fabric to "learn" the optimal configuration for a given workload over time.
