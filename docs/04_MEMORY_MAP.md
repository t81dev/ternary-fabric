# 04: Memory Map & Registers

The Ternary Fabric is controlled via an AXI4-Lite slave interface. The default base address in most system integrations is `0x40000000`. The canonical offsets are defined in `include/tfmbs_dma_regs.h`; run `tools/validate_register_map.py` to check the header against this table before making driver or RTL changes.

## 1. Control & Status Registers (Base + 0x00)

| Offset | Name | Description |
| --- | --- | --- |
| 0x00 | `CONTROL` | Bit 0: Start (Self-clearing).<br>Bits 15:8: **Tile Mask** (e.g., 0x1 for Tile 0, 0xF for all 4 tiles). |
| 0x04 | `STATUS` | Bit 0: Busy. Bit 1: Done. |
| 0x08 | `BASE_ADDR` | SRAM offset for the current frame. |
| 0x0C | `DEPTH` | Number of trits to process in the frame. |
| 0x10 | `STRIDE` | Trit distance between elements. |
| 0x14 | `EXEC_HINTS` | Kernel ID and optimization flags (See TFD Spec). |
| 0x18 | `LANE_COUNT`| Number of active SIMD lanes (Max 15). |
| 0x1C | `LANE_MASK` | Bitmask to enable/disable specific lanes. |

## 2. Multi-Tile SRAM Regions

Each tile has private address space for Weights and Inputs. Writing to these addresses loads data into the corresponding tile's local SRAM.

| Address Range | Target | Description |
| --- | --- | --- |
| `0x1000 - 0x1FFF` | Tile 0 Weights | 1024 words (24-bit effective). |
| `0x2000 - 0x2FFF` | Tile 0 Inputs | 1024 words. |
| `0x3000 - 0x3FFF` | Tile 1 Weights | |
| `0x4000 - 0x4FFF` | Tile 1 Inputs | |
| `0x5000 - 0x5FFF` | Tile 2 Weights | |
| `0x6000 - 0x6FFF` | Tile 2 Inputs | |
| `0x7000 - 0x7FFF` | Tile 3 Weights | |
| `0x8000 - 0x8FFF` | Tile 3 Inputs | |
| **`0x9000`** | **Broadcast** | Writes to this address go to ALL Tile Weight SRAMs. |

## 3. Results & Profiling (Read-Only)

### Results (0x100 - 0x1FF)
The accumulated 32-bit results for each lane and tile.
*   **Tile 0:** `0x100 + (lane * 4)`
*   **Tile 1:** `0x140 + (lane * 4)`
*   **Tile 2:** `0x180 + (lane * 4)`
*   **Tile 3:** `0x1C0 + (lane * 4)`

### Global Performance Counters
| Offset | Name | Description |
| --- | --- | --- |
| 0x20 | `CYCLES` | Total clock cycles spent executing the last frame. |
| 0x24 | `UTILIZATION`| Aggregated lane-cycles (Sum of active lanes per cycle). |
| 0x68 | `DMA_LATENCY`| Cycles spent waiting for AXI-Stream data. |

### Per-Tile Profiling
Each tile has a block of counters for **Zero-Skip** events, **Active Cycles**, and **Overflow Flags**.

| Tile | Skip Counts (15x32b) | Overflow Flags (1x32b) | Active Cycles (15x32b) |
| --- | --- | --- | --- |
| **Tile 0** | `0x28 - 0x64` | `0x6C` | `0x70 - 0xAC` |
| **Tile 1** | `0x228 - 0x264` | `0x26C` | `0x270 - 0x2AC` |
| **Tile 2** | `0x328 - 0x364` | `0x36C` | `0x370 - 0x3AC` |
| **Tile 3** | `0x428 - 0x464` | `0x46C` | `0x470 - 0x4AC` |
