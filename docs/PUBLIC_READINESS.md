# Public Readiness Snapshot

This status note is designed to help anyone presenting Ternary Fabric publicly while the XC7Z020/XC7Z045 hardware remains unavailable. It bundles the public-facing story, compiler/telemetry workflow, measurable evidence, and outreach guidance so you can speak clearly about what’s ready today and what remains gated on silicon access.

## 1. Public Story
- **Architecture & Vision:** Reference the README/roadmap (Phases 0–26) for the semantic execution substrate, zero-skip, PT‑5 packing, and multi-fabric orchestration narrative. The README now covers how to build, run the `test_phase21` verification, and view benchmarks, while the roadmap documents Phases 0–27 plus the Track A/B/C splits. Currents status:
  - Software stack is in Phase 25 (distributed orchestration) and Phase 26 (adaptive runtime) operational mode.  
  - Compiler/MLIR stack (Phase 23) is fully built and regression-tested via shared LLVM/MLIR builds and `tests/mlir/run_tfmbs_to_linalg.py`.  
  - Hardware proof is blocked on XC7Z020/XC7Z045 access; see `docs/FPGA_VERIFICATION_CHECKLIST.md` and `docs/hardware_verification_report.md` for the outstanding synthesis/driver/execution checklist that will kick off once the boards return.
- **Toolkit readiness:** Documented Quick Start now includes the `tools/run_tfmbs_regression.sh` helper that configures Ninja, builds `tfmbs_plugin`, runs `mlir-opt`, and validates the lowering pipeline with the telemetry-rich MLIR fixtures. We also ship `tools/run_hw_dma_telemetry.sh` + `tools/capture_dma_telemetry.py` so the DMA/telemetry story can be exercised locally via Verilator and the adaptive dashboard before the physical FPGA is available.

## 2. Compiler & Telemetry Regression Workflow
- **TableGen & plugin:** `src/mlir/TfmbsOps.td` + generated headers define the dialect, and `TfmbsPasses.cpp` implements fusion + lowering. `CMakeLists.txt` produces `libtfmbs_dialect.*` / `libtfmbs_plugin.*`, and `tools/run_tfmbs_regression.sh` automatically discovers `mlir-opt` and runs `tests/mlir/run_tfmbs_to_linalg.py`.  
- **Regression gate:** `tests/mlir/run_tfmbs_to_linalg.py` ensures `linalg.matmul` appears while `tfmbs.gemv`/`tfmbs.fused_gemv` vanish after the pipeline. CI + local helpers now iterate over `tests/mlir/tfmbs_*fusion*.mlir` plus `tests/mlir/torch_tfmbs.mlir` so telemetry attributes (`fusion_order`, `fusion_sparsity`, `tile_mask`) stay aligned.  
- **Adaptive dashboard:** Runtime logs from `pytfmbs.AdaptiveRuntimeAgent.save_history(...)` are compared to compiler hints via `tools/adaptive_dashboard.py`. `tools/run_hw_dma_telemetry.sh` (which builds `hw_sim`, runs `bin/test_dma_driver`, captures metrics via `tools/capture_dma_telemetry.py`, and runs the dashboard) mirrors the telemetry contract expected from real hardware.

## 3. Measurable Evidence
- **Benchmarks:** `BENCHMARKS.md` summarizes phase-specific throughput (aggregated fabric GOPS, zero-skip reduction, PT-5 efficiency). When new simulation/driver logs are available, drop them into `logs/` (e.g., `logs/adaptive_history_dma.json`) and highlight throughput/latency columns from `benchmarks/` scripts.  
- **Telemetry logs:** The helper tools now produce `logs/adaptive_history_dma.json` (Verilator) to show fusion order + sparsity metrics; once the FPGA runs, place the real hardware logs in `logs/` with timestamps for future comparisons.  
- **Dashboard output:** After running `tools/adaptive_dashboard.py`, capture the console output and include it alongside the telemetry log to prove compiler/runtime agreement. Mention these artifacts when presenting the project to show you already validate the telemetry pipeline even before the hardware reconnects.

## 4. Outreach Notes (Ready vs Waiting on XC7Z020/XC7Z045)
- ✅ **Ready today**: Distributed multi-fabric orchestration (Phase 25) via simulation; compiler dialect/passes/regression tooling (Phase 23); adaptive runtime + telemetry dashboard (Phase 26); DMA mock driver + `libtfmbs_device.so`; tooling for capturing telemetry during Verilator runs.  
- ⏳ **Waiting on hardware**: RTL synthesis + bitstream programming for XC7Z020/XC7Z045; real `/dev/tfmbs` IOCTL/DMA validation; hardware telemetry vs emulator parity; >50× efficiency proof on physical boards. These steps remain blocked until the FCX FPGA racks return.  
- **Outreach story:** Frame the project as “software-ready with hardware verification queued” and point audiences to `docs/FPGA_VERIFICATION_CHECKLIST.md`/`docs/hardware_verification_report.md` for the precise checklist. Mention that the DMA/telemetry flow can already be exercised via `tools/run_hw_dma_telemetry.sh` while the physical testbeds remain offline.

## Links
- [README Quick Start](../README.md)  
- [Roadmap (Track A/B/C)](ROADMAP.md)  
- [FPGA Verification Checklist](FPGA_VERIFICATION_CHECKLIST.md)  
- [Hardware Verification Report](hardware_verification_report.md)  
- [Compiler Track Plan](compiler_track_plan.md)
