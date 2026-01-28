# Phase 19: Data-Driven Adaptation (Cost-Aware Fabric)

Phase 19 leverages the metrics established in Phase 18 to drive autonomous fabric behavior and scheduling, ensuring optimal performance across varying workloads.

## 🎯 Objectives

1.  **Economic Introspection:** Expose internal decision metrics for system-level optimization.
2.  **Hysteresis Scheduling:** Stabilize tile selection and residency management.
3.  **Sparse-Regime Hardening:** Ensure robustness and efficiency in 95-99% sparse regimes.

## 💎 Economic Introspection

The fabric now logs its internal decision matrix to `economic_metrics.csv`. This provides transparency into why specific scheduling decisions were made.

- **Projected Cost:** The estimated **Fabric Cost** for a task before execution.
- **Rebates:** Reductions in cost due to **Residency Hits** or **Zero-Skip** potential.
- **Eviction Scores:** Metric-driven priorities for buffer replacement in the residency pool.

## ⚖️ Hysteresis Scheduling

To prevent "tile oscillation" (frequent switching of buffers between tiles), the scheduler implements hysteresis:
- **Sticky-Tile Affinity:** A 0.5 cost "rebate" is applied to tiles already holding the required weights.
- **Cost Smoothing:** Uses an Exponential Moving Average (EMA) to filter noise in measured **Economic Efficiency**.

## 🌵 Sparse-Regime Hardening

The **Zero-Skip** logic is stress-tested in extreme sparsity regimes (95%+).
- **Control Logic Robustness:** Ensuring the frame controller correctly handles cases where entire rows or tiles are skipped.
- **Throughput Scaling:** Verifying that effective GOPS scale linearly with sparsity as compute cycles are reclaimed.

## 📈 KPI Maturation

- **Semantic Efficiency:** Quantifies the "work done" per operation by accounting for **Zero-Skip** savings.
- **Economic Efficiency:** Quantifies the "value delivered" per unit of **Fabric Cost**, serving as the primary optimization target for the Phase 20 self-tuning loops.
