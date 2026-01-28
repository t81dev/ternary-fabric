# Phase 20: Learning & Self-Tuning Fabric

## Overview
Phase 20 introduces the **Learning Layer** to the Ternary Fabric. It moves beyond fixed heuristics and implements a self-tuning feedback loop that optimizes the co-processor's behavior based on real-time execution metrics.

## 1. Adaptive Cost Coefficients
The scheduler's decision-making process relies on a set of projection coefficients (`weight_cost`, `mem_read_cost`, etc.). In Phase 20, these are no longer static.

- **Mechanism:** Hill-climbing feedback loop.
- **Feedback Signal:** Error delta between `projected_cost` and actual measured `fabric_cost`.
- **Goal:** Minimize projection error to ensure the scheduler always selects the truly optimal tile.

## 2. Dynamic Scheduler Weighting
Each tile maintains a set of kernel-specific performance multipliers.

- **Learning:** After each kernel execution, the tile's performance multiplier is adjusted based on its **Economic Efficiency** relative to the moving average.
- **Decay:** Multipliers slowly decay back toward 1.0 to ensure the fabric remains responsive to changing workload patterns.
- **Result:** The fabric "learns" tile preferences automatically, favoring tiles that have historically performed best for specific operations.

## 3. Feedback-Driven Eviction
The eviction policy uses a composite score to determine which memory blocks to keep resident.

- **Scoring Formula:** `Score = W_freq * Frequency + W_age * Recency + W_success * Success_Rate`
- **Tuning:** If the fabric experiences high residency misses, it automatically increases the weight of success and frequency to protect critical weights.
- **Result:** Automatically protected hot weights that contribute most to system efficiency.

## 4. Temporal Pipeline Optimization (Auto-Batching)
The asynchronous execution engine dynamically tunes its batch size.

- **Objective Function:** `0.7 * Economic Efficiency + 0.3 * Throughput`
- **Exploration:** The engine occasionally (5% of the time) tries a slightly different batch size.
- **Adaptation:** If the exploration yields a better composite score, the "base" batch size is updated in that direction.

## 5. Economic Introspection
The `economic_metrics.csv` log has been expanded to track the evolution of these learning parameters over time.

| Field | Description |
| :--- | :--- |
| `projected_cost` | The cost predicted by the scheduler. |
| `batch_size` | The auto-tuned batch size used for the operation. |
| `weight_cost` | The current value of the learned weight cost coefficient. |
| `eviction_scores` | The scores assigned to blocks during the last eviction event. |

## 6. Interpreting the Learning Curve
When analyzing logs, a healthy "learning" phase should show:
1. `projected_cost` converging toward `fabric_cost`.
2. `economic_efficiency` steadily increasing as the scheduler favors better tiles.
3. `dynamic_batch_size` stabilizing at a value that balances latency and throughput.
