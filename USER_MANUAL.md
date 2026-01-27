# Ternary Fabric User Manual (v0.1)

Welcome to the **Ternary Fabric** user manual. This documentation is designed to help hardware designers, software developers, and researchers understand and utilize the ternary-native acceleration substrate.

As of **Phase 9**, the Ternary Fabric provides a mature, transparent acceleration path for `llama.cpp` and other GEMV-heavy workloads, featuring LRU-based paging, asynchronous execution, and real-time telemetry.

---

## 📖 Table of Contents

1.  **[Project Overview](docs/00_OVERVIEW.md)** - Design philosophy, core innovation, and current status.
2.  **[Installation & Setup](docs/01_INSTALL.md)** - Dependencies, build instructions, and environment setup.
3.  **[Quick Start Guide](docs/02_QUICK_START.md)** - Run your first ternary operation in minutes using the Python API.
4.  **[Enabling Acceleration](#-enabling-acceleration)** - How to use the Fabric with existing applications like `llama.cpp`.
5.  **[Telemetry & Performance](#-telemetry--performance)** - Interpreting real-time metrics and Zero-Skip gains.
6.  **[Hardware Architecture](docs/03_HARDWARE.md)** - Multi-Tile Topology, TPE Lanes, and the hydration pipeline.
7.  **[Memory Map & Registers](docs/04_MEMORY_MAP.md)** - AXI address mapping and register definitions.
8.  **[Software API Guide](docs/07_API.md)** - Comprehensive guide to the `pytfmbs` Python library.
9.  **[Roadmap & Future Goals](docs/ROADMAP.md)** - Current progress (Phases 0-9) and future scaling plans.
10. **[Appendices](docs/10_APPENDICES.md)** - Acronyms, PT-5 details, and Validation Reports.

---

## 🚀 Enabling Acceleration

The Ternary Fabric achieves acceleration via a "Fabric Illusion"—the host application (e.g., `llama.cpp`) believes it is using standard system memory, but the Fabric interposer transparently redirects allocations and offloads compute.

### 1. Using `LD_PRELOAD`
The most common way to enable acceleration is by preloading the intercept library:

```bash
LD_PRELOAD=./libtfmbs_intercept.so ./main -m models/7B/ggml-model-q4_0.gguf -p "Hello"
```

### 2. Configuration Flags
You can tune the Fabric's behavior using environment variables:

| Flag | Default | Description |
| :--- | :--- | :--- |
| `FABRIC_SHORT_CIRCUIT` | `1` | Enable/Disable CPU short-circuiting (jumping over CPU loops). |
| `FABRIC_ASYNC` | `1` | Enable/Disable asynchronous GEMV execution. |
| `FABRIC_LOG_LEVEL` | `INFO` | Set logging verbosity (`DEBUG`, `INFO`, `WARN`, `ERROR`). |
| `FABRIC_POOL_SIZE` | `128MB` | Size of the Fabric-resident memory pool. |

---

## 📊 Telemetry & Performance

When acceleration is active, the Fabric provides a real-time telemetry dashboard in the terminal.

### Example Dashboard Output
```text
[TFMBS] Residency: 42 Blocks (PT-5 Packed)
[TFMBS] Offload GEMV [Queue: 2]
[TFMBS] Last Exec: 1.2ms | Total Gains: 68.4%
[TFMBS] Zero-Skip: 284,102 operations suppressed.
[TFMBS] Paging: 2 evictions, 0 re-loads this session.
```

### Key Metrics
*   **Zero-Skip Count:** The number of multiply-accumulate operations skipped because an operand was zero. Typical LLM workloads see **64-76% reduction**.
*   **Residency:** Indicates how much of the model weights are currently packed into the ternary-native PT-5 format in Fabric memory.
*   **Short-circuit:** Confirms the interposer successfully hijacked the CPU's instruction pointer to skip redundant calculations.

---

## 🧠 Technical Notes

### LRU Paging & Eviction (Phase 7)
The Fabric manages a finite memory pool (default 128MB). If a model's weights exceed this size, the Fabric uses a **Least Recently Used (LRU)** strategy to evict older PT-5 frames back to host RAM. They are transparently re-packed and re-loaded when next accessed.

### Asynchronous Pipelining (Phase 8)
GEMV operations are submitted to a hardware-backed command queue. This allows the CPU to continue with non-compute tasks (like token sampling or KV-cache management) while the Fabric executes the heavy matrix math in the background. Synchronization is handled automatically via `mprotect` traps.

---

## 🛠️ Key Examples
Check out the `examples/` directory for runnable scripts:
*   `quick_start.py`: Basic T-GEMM operation.
*   `multi_tile_tgemm.py`: Multi-tile and broadcast demonstration.
*   `tests/mock_llama.c`: Demonstrating `LD_PRELOAD` memory redirection and compute offload.

---
© 2026 Ternary Fabric Project. All rights reserved.
