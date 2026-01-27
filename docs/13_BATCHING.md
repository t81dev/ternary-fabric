# 📦 Phase 13 — Large-Model Support & Multi-Layer Batching

Phase 13 introduces advanced prefetching and batching strategies to handle large models and hide DMA/compute latency.

## 🚀 Key Features

### 1. Asynchronous Execution
The `pytfmbs.Fabric` interface now supports non-blocking operation submission:
*   `submit(tfd)`: Start execution and return immediately.
*   `wait()`: Block until completion and return profiling metrics.
*   `is_done()`: Poll for completion status.

### 2. Multi-Layer Pipelining
The `TFMBSLinear` module supports weight prefetching. While layer $N$ is executing on the Fabric, the host can begin loading weights for layer $N+1$ into a different SRAM address.

```python
# Automatic pipelining using TFMBSSequential
model = TFMBSSequential(
    TFMBSLinear(4096, 4096, weight_addr=0x1000),
    TFMBSLinear(4096, 4096, weight_addr=0x20000), # Use different SRAM bank
    ...
)
output = model(input)
```

### 3. Batched Execution
To reduce Python-to-C transition overhead, `run_batch` allows submitting a sequence of TFDs in one call.

```python
fabric.run_batch([tfd1, tfd2, tfd3])
```

## 🧠 Large-Model Strategy

For models exceeding the physical Fabric SRAM:
1.  **Double-Buffering**: Partition SRAM into two banks. Load Bank B while Bank A is computing.
2.  **Streaming**: Use `load_stream` (AXI-Stream DMA) for high-speed weight ingestion.
3.  **Resident Weights**: Keep the most frequently used layers (e.g., embedding, small projection layers) resident.

## 📊 Verification

Verified using `tests/test_phase13.py`, demonstrating:
*   Asynchronous `submit`/`wait` cycles.
*   Multi-layer prefetching in a `TFMBSSequential` model.
*   Support for 1MB mock SRAM enabling larger weight blocks.
