# Phase 20: Learning & Self-Tuning Fabric

Phase 20 transforms static heuristics into a dynamic, self-tuning system that optimizes its own parameters based on historical performance and workload characteristics.

## 🎯 Objectives

1.  **Adaptive Cost Coefficients:** Implement feedback loops to minimize the error in cost projection.
2.  **Dynamic Scheduler Weighting:** Optimize tile and kernel selection based on measured **Economic Efficiency**.
3.  **Eviction Policy Self-Tuning:** Automatically adjust scoring weights to maximize **Residency Hits**.
4.  **Temporal Pipeline Optimization:** Auto-tune asynchronous batch sizes for optimal throughput.

## 🔄 Adaptive Feedback Loops

The fabric utilizes a hill-climbing optimization strategy to refine its internal model:

- **Cost Delta Minimization:** The scheduler tracks the difference between `projected_cost` and actual **Fabric Cost**. It uses a small learning rate (0.05) to adjust coefficients for memory reads, writes, and operations.
- **Efficiency EMA:** A long-term Exponential Moving Average of **Economic Efficiency** is maintained per tile and kernel type.

## ⚖️ Dynamic Weighting

Tile selection is no longer just "locality first." The scheduler now considers:
- **Historical Efficiency:** Tiles that have historically delivered higher efficiency for a specific kernel (e.g., **T-GEMM**) receive a selection preference.
- **Success Weights:** Eviction scoring weights (frequency, recency, and reuse success) are adjusted in real-time to maximize the hit rate of the residency pool.

## 🚀 Temporal Optimization

The asynchronous worker thread now performs **Dynamic Batch Tuning**:
- **Composite Score:** A weighted metric of **Economic Efficiency** and throughput.
- **Batch Sizing:** The scheduler increases or decreases the number of tasks pulled from the queue per cycle to find the "sweet spot" where the three-stage pipeline (Pre-fetch -> Execute -> Commit) is fully saturated.

## 📊 Self-Tuning Metrics

Telemetry now includes "Learning State" metadata, allowing developers to see the current bias of the cost model and the current multiplier for each tile/kernel pair.
