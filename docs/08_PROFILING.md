# 08: Profiling & Optimization

The Ternary Fabric provides built-in hardware counters to help developers optimize their workloads and quantify the benefits of ternary acceleration.

## 1. Zero-Skip Metrics

Zero-Skip is the primary mechanism for energy efficiency. You can measure its effectiveness using the `profile()` API.

### Calculating Efficiency
To calculate the sparsity exploitation for a specific lane:
$$Efficiency = \frac{Skip\_Count}{Active\_Cycles} \times 100\%$$

In highly sparse models (like pruned LLMs), you should see efficiencies exceeding 80-90%, which translates directly to dynamic power savings in the TPE lanes.

## 2. Lane Utilization

The `utilization` counter tracks how many SIMD lanes were actually doing work during the execution period.

*   **Under-utilization:** If `utilization` is significantly lower than `cycles * active_tiles * 15`, it suggests that your `lane_mask` or `lane_count` is too small, or your data is not aligned to the 15-lane boundary.
*   **Optimal Throughput:** Goal is to keep all lanes active by batching operations or using multi-tile broadcast.

## 3. Multi-Tile Scaling

When scaling workloads:
*   **Weight Broadcast:** Use the `0x9000` address to load weights once and use them across all tiles. This reduces AXI bus congestion.
*   **Lock-Step:** Tiles share a frame controller. If one tile is masked out, it still consumes "cycles" but zero "utilization" and zero power.

## 4. Phase 9 Telemetry (Interposer)

When using the `libtfmbs_intercept.so` interposer, high-level telemetry is automatically reported to `stderr`. This provides a non-intrusive way to monitor real-world application performance.

### Interpreting the Log
A typical telemetry output looks like:
```text
[TFMBS-Telemetry] GEMV Completed
  - Zero-Skips: 172,401 (65.8% reduction)
  - Pool Usage: 84.2 MB / 128 MB (65.8%)
  - Evictions:  2
  - Async Queue: 0 in-flight
```

*   **Zero-Skips:** Total trits skipped across all rows and columns of the GEMV.
*   **Pool Usage:** Memory occupancy of the 128MB Fabric emulation pool.
*   **Evictions:** Number of times an LRU block was freed to make room for new data. High eviction counts suggest the model is too large for the 128MB pool, potentially causing "thrashing."

## 5. Best Practices for Performance

1.  **Alignment:** Always try to process trits in multiples of 15. If your vector size is not a multiple of 15, the remaining lanes will be idle.
2.  **DMA Loading:** For large datasets, use `load_stream()` instead of multiple `load()` calls to leverage the high-bandwidth AXI-Stream path.
3.  **Kernel Fusing:** Use `exec_hints` to enable multiple optimizations (e.g., `ZERO_SKIP_EN | FREE_NEG_EN`) in a single pass.
4.  **Overflow Monitoring:** Regularly check `overflow_flags` in `profile_detailed()`. If a bit is set, the corresponding lane's 32-bit accumulator has wrapped around, which may affect accuracy.
