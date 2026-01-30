# Fabric Illusion Contract

The Fabric Illusion is the promise that existing AI workloads execute unmodified while the interposer migrates the heavy work into the ternary fabric. This document spells out the observable contract we currently enforce and illustrates the safe boundaries that let reviewers verify the idea without having to dive into the RTL.

## 1. What the interposer is responsible for
- **Heavy allocations (> 1 KiB):** `malloc`/`mmap` calls that meet or exceed `FABRIC_THRESHOLD (1024 bytes)` are registered, trapped, and eventually packed into PT-5 memory. This is the gating heuristic defined in `src/libtfmbs_intercept.c` and documented in the instrumentation notes in `docs/16_PRODUCTION_HARDENING.md`.
- **Transparent residency:** The interposer signals residency by mprotecting weight buffers and only allowing CPU accesses once the background fabric task has completed. The `sigsegv` handler (same file) updates residency metadata and synchronizes via `fabric_wait()`, so host code never sees stale results.
- **GEMV detection:** The heuristics look for the two most recently touched large buffers (weights + activation) before submitting a ternary GEMV. This path is instrumented in `src/libtfmbs_intercept.c` and its behavior is captured in `docs/12_PYTORCH.md` and `docs/14_GGUF.md` where Python and GGUF workflows are described.
- **Interposition shim:** The `tools/tfmbs-run` wrapper exported in this repo sets `LD_PRELOAD`/`DYLD_INSERT_LIBRARIES` and loads `libtfmbs_intercept`, giving you a one-command reproduction of the Fabric Illusion. The smoke test `tools/reference_integration.py` exercises this path end-to-end (build → run mock GEMV → record logs + chart).

## 2. Guarantees exposed to the host
- **Data integrity:** Every offloaded kernel is followed by a `fabric_wait()` (or implicit wait via the residency heuristics) before the host reads the results. If validation mode is enabled (`TFMBS_VALIDATE=1`), a bit-exact CPU compare runs automatically and logs mismatches without altering the host flow.
- **Billing transparency:** Telemetry (zero skip count, fabric cost, residency hits) is available via `tools/capture_dma_telemetry.py` and the adaptive dashboard described in `docs/08_PROFILING.md`; nothing in the host interface is hidden or magical.
- **Static contract:** The interposer does not require recompiling benchmarks. All of the token-level workflows in `README.md` (and the `tools/benchmark_performance.sh` helper) keep the host binary untouched while `libtfmbs_intercept` short-circuits memory and compute.

## 3. Safe failure modes
- **Fallback to CPU:** If fabric allocation fails or an unsupported access pattern is detected, the interposer delegates the faulting signal handler to the previous handler and the host simply continues executing on the CPU, as described in Sections 7–10 of `docs/16_PRODUCTION_HARDENING.md`.
- **Graceful unregistration:** `free()` removes the buffer from the residency registry and releases the fabric allocation so subsequent workloads start clean.
- **Short-circuit control:** `FABRIC_SHORT_CIRCUIT` gates the optimization: set it to `0` to disable GPU-level overrides and observe the host doing the same work it always has.

## 4. Why this matters
By turning the Fabric Illusion into a documented contract, reviewers can assess brittleness (e.g., convolutional kernels that don’t trigger the heuristics) and tell when they have to add explicit registration via `fabric_register_weight()`. Recording this contract alongside `tools/reference_integration.py` and `tools/benchmark_performance.sh` makes the idea easy to verify without private hardware racks.
