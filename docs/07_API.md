# 07: Software API Guide (`pytfmbs`)

The `pytfmbs` module provides a high-level Python interface to the Ternary Fabric. It handles MMIO mapping, TFD submission, and data loading.

## 1. The `Fabric` Class

### `Fabric()`
Initializes the hardware interface.
*   **Linux/SoC:** Attempts to map `/dev/mem` at physical address `0x40000000`.
*   **Other/Fallback:** Initializes **Mock Mode**, simulating hardware behavior in software.

### `load(offset, data)`
Loads bytes into the fabric's address space.
*   `offset`: Integer (e.g., `0x1000`).
*   `data`: `bytes` object or filename string.

### `load_stream(tfd_dict, data)`
Loads data via the AXI-Stream DMA interface.
*   `tfd_dict`: Dictionary containing `base_addr`.
*   `data`: `bytes` object.

### `run(tfd_dict)`
Submits a task and waits for completion (Synchronous).
*   `tfd_dict`: Dictionary with keys like `frame_len`, `exec_hints`, `tile_mask`, etc.

### `submit(tfd_dict)`
Submits a task and returns immediately (Asynchronous). Returns a `task_id`.

### `wait(task_id=None)`
Blocks until the specified `task_id` (or the last submitted task) is complete.

### `is_done(task_id=None)`
Returns `True` if the specified `task_id` (or the last submitted task) has completed.

### `run_batch(tfd_list)`
Submits a list of TFDs for optimized batch execution.

### `results(tile_id=0)`
Returns a list of 15 integers representing the accumulated results for the specified tile. Use `tile_id=-1` to get results for all tiles.

## 2. Framework & GGUF Integration

### `pytfmbs.TFMBSLinear(in_features, out_features, ...)`
A PyTorch layer that offloads GEMV operations to the Ternary Fabric. Supports automatic quantization and prefetching.

### `pytfmbs.load_gguf_tensor(fabric, path, tensor_name, address)`
Utility to load a tensor from a GGUF file directly into Fabric SRAM, with automatic dequantization and PT-5 packing.

## 3. Profiling Methods

### `profile()`
Returns a dictionary with aggregated global metrics:
*   `cycles`: Total execution time.
*   `utilization`: Total lane activity.
*   `skips`: List of total Zero-Skip events per lane.

### `profile_tile(tile_id)`
Returns detailed metrics for a specific tile, including `skips`, `active_cycles`, and `overflow_flags`.

### `profile_detailed()`
Returns a superset of `profile()`, including DMA latency and per-lane active cycles.

## 3. Mock Mode Features
When running in Mock Mode:
*   Multi-tile execution is fully simulated.
*   DMA latency is modeled.
*   Register reads/writes are logged to the console.
*   Hardware results match the bit-accurate ternary arithmetic.
