# Phase 11: Multi-Tile & Multi-GPU Scaling

## 1. Overview
The Ternary Fabric is designed to scale horizontally. While a single **Tile** provides 15 parallel lanes of ternary execution, a physical **Fabric Device** typically contains multiple Tiles (e.g., 4 or 8) that can operate in lock-step or independently.

## 2. The `tile_mask` Mechanism
Execution requests (both via the high-level `fabric_exec_gemv` and the low-level `TFMBS_IOC_SUBMIT`) now support a `tile_mask`.

*   **Type:** `uint8_t` (Bit-field)
*   **Behavior:** Each bit corresponds to a physical Tile.
*   **Example:**
    *   `0x01`: Only Tile 0 is active (15 lanes).
    *   `0x03`: Tiles 0 and 1 are active (30 lanes).
    *   `0x0F`: All 4 Tiles are active (60 lanes).

## 3. Workload Partitioning
In Phase 11, the fabric controller automatically partitions GEMV workloads across the active tiles specified in the mask.

### 3.1 Row-Wise Distribution
For a matrix $W$ with $R$ rows, the controller distributes rows such that each tile $T$ handles approximately $R / N$ rows, where $N$ is the number of set bits in the `tile_mask`.

### 3.2 Performance Benefits
Scaling across tiles provides:
- **Increased Throughput:** Linear scaling of operations per cycle.
- **Power Efficiency:** Tiles not specified in the `tile_mask` can remain in a low-power clock-gated state.

## 4. Multi-GPU / Multi-Fabric (Future)
Phase 11 established the logic for intra-device scaling. Future updates will extend this to multi-device configurations where partial results are reduced across the PCIe or CXL interconnect.
