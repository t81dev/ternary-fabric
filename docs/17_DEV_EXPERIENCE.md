# Phase 17: Developer Experience (DX)

Phase 17 improves the tools and visibility available to developers working with TFMBS.

## 🔍 Debugging & Visualization

- **`TFMBS_DEBUG=1`**: Setting this environment variable enables detailed real-time logging in the interposer. It logs:
    - Memory registry additions.
    - GEMV offload triggers and inferred dimensions.
    - Asynchronous wait events and short-circuit jumps.
- **`TFMBS_VALIDATE=1`**: Enabling validation mode causes the interposer to run a bit-exact reference CPU implementation of every GEMV task immediately after it completes on the Fabric. It compares the results and logs any mismatches.

## 🚀 CLI Tools

- **`tfmbs-run`**: A wrapper script located in `tools/` that automatically configures `LD_PRELOAD` and `LD_LIBRARY_PATH`.
    - Usage: `./tools/tfmbs-run ./my_app`
- **`benchmark_performance.sh`**: A tool to compare the wall-clock performance of the Fabric against a pure CPU baseline using the `mock_llama` benchmark.

## 📈 Performance Profiling

- **Wall-Clock Metrics:** We've shifted focus from "Peak GOPS" to "End-to-End Time", providing a more realistic view of the system's impact on inference latency.
