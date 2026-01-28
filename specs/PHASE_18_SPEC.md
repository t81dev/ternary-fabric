# Phase 18 — Ternary Workload Maturation & Measurement Plane

## Objectives
Transition Ternary Fabric from **measured correctness** to **semantic efficiency optimization**.
The goal is to provide a programmable substrate where cost is normalized by meaning, and data residency is a first-class architectural concern.

## Semantic Metrics Layer
We introduce derived KPIs that capture the economic value of fabric operations:
- **Semantic Efficiency**: `useful_ops / fabric_cost`
- **Fabric Cost**: A weighted sum of active operations, memory accesses, broadcasts, and residency misses.
- **Residency Hit Ratio**: Percentage of tensor accesses that find data already local to the tile fabric.

## Cost Model (P18-Standard)
The normalized cost model for all kernels is defined as:
```
cost = active_ops * 1.0 + mem_reads * 5.0 + mem_writes * 8.0 + broadcasts * 2.0 + residency_misses * 6.0
```
This forces developers to optimize for data reuse and sparsity.

## Data Residency & Persistence
Phase 18 promotes T-LSTM from a stateless kernel call to a **resident temporal system**.
- **Tile Binding**: Weights and hidden states can be pinned to specific tiles.
- **Persistence**: Once bound, recurrent state and gates stay in tile SRAM across steps.
- **Economic Impact**: Reduces memory read costs by up to 80% for long-sequence temporal workloads.

## Measurement Plane (Visualization)
A public API `tfmbs_dump_metrics_csv` is provided to capture time-series efficiency data.
The companion tool `tools/plot_metrics.py` generates perception-level plots for:
- Efficiency vs. Cost
- Residency Hit/Miss distribution
- Fabric Activity breakdowns

## Architectural Intent
Ternary Fabric is not just a math accelerator; it is a **measurable cognition engine** where architecture follows economic gravity.
