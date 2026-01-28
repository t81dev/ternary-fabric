# Phase 16: Production Hardening

This phase focuses on making the TFMBS "Illusion" stable, reliable, and safe for production-like environments.

## 🛡️ Robust Signal Handling

The TFMBS interposer uses `SIGSEGV` to intercept memory accesses and redirect them to the Fabric. Previously, any unmanaged segfault would cause an immediate process exit.

- **Handler Delegation:** The interposer now saves the previous signal handler (`struct sigaction`) and delegates to it if a faulting address is not found in the TFMBS registry.
- **Proper Crashes:** Real segmentation faults in the host application now behave normally (causing a crash with a core dump) rather than being swallowed or causing a mysterious exit.

## 💾 Memory & Deadlock Safety

- **OOM Awareness:** The Fabric emulator (`emu_fabric_alloc`) now correctly handles out-of-memory situations where all blocks are "busy" (pinned by active asynchronous tasks).
- **Graceful Fallback:** If Fabric memory cannot be allocated, the interposer transparently falls back to host memory (via standard `malloc`).
- **Busy Pinning:** Tasks in the asynchronous pipeline pin their associated memory blocks, preventing the LRU evictor from removing data that is currently being used by a hardware kernel.

## 🤝 Cooperative Mode (Explicit API)

While the transparent interposer is the primary interface, some applications may prefer explicit control to reduce overhead or improve reliability.

- **`fabric_register_weight(ptr, size)`**: Explicitly marks a host memory buffer as a weight buffer for the Fabric. This bypasses the heuristic "First Scan" phase and immediately establishes residency.

## 🛠️ Error Propagation

- **Status Codes:** `fabric_wait` and other internal APIs have been updated to return integer status codes instead of `void`. This allows the interposer to detect kernel failures and potentially fall back to CPU compute if a hardware error occurs.
