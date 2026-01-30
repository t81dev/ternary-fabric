# Compiler Track — MLIR & Lowering Roadmap

Track B (Compiler & Ecosystem) in `docs/ROADMAP.md:156-170` is positioned to deliver the MLIR dialect, lowering passes, and adaptive scheduling hooks needed to make TFMBS a first-class compiler target. Hardware verification remains blocked on the XC7Z020/XC7Z045 boards, but the compiler stack is now fully operational and documented so the telemetry contract is stable while we await silicon access.

## Current Status
- `src/mlir/TfmbsOps.td` drives a TableGen-based dialect; the generated headers (`src/mlir/TfmbsOps.h.inc`/`TfmbsOps.cpp.inc`) are included by `TfmbsDialect.{h,cpp}`, and the dialect registration is wired into `CMakeLists.txt` so `mlir-opt` can load it via `TfmbsPassPlugin`.
- `src/mlir/TfmbsPasses.{h,cpp}` implement `TfmbsFusionPass` (telmetry-aware fusion) and `TfmbsToLinalgPass`, and `TfmbsPassPlugin.cpp` exposes the `tfmbs-to-linalg` pipeline plus TableGen-defined ops. The shared MLIR/Ninja build now produces `build/libtfmbs_plugin.dylib` alongside `libtfmbs_dialect`, so the regression script can reuse the same artifacts.
- Front-end integration is exercised by `tools/torch_to_tfmbs.py`, which exports Torch/ONNX graphs into `tests/mlir/torch_tfmbs.mlir` with `telemetry`, `fusion_order`, `fusion_sparsity`, and `tile_mask` attributes that mirror the runtime hints consumed by `pytfmbs.TFMBSLinear.telemetry_hint`. This keeps the compiler/telemetry pipeline aligned even while the hardware path is under construction.
- `pytfmbs.AdaptiveRuntimeAgent` logs fusion telemetry (sparsity + fusion order) that `tools/adaptive_dashboard.py` compares back to the MLIR fixtures, so both compiler- and runtime-produced dictionaries are validated before we reconnect the FPGA environment.
- Existing emulator tests (`tests/mlir/TfmbsPassTest.cpp`, `tests/mlir/run_tfmbs_to_linalg.py`, `tests/mlir/tfmbs_*fusion*.mlir`) now cover the lowering pipeline and telemetry outputs; `ninja -C build check-tfmbs` runs the lit suite that exercises those fixtures and asserts `linalg.matmul` is present and `tfmbs.gemv` is consumed.

## Regression & Telemetry Guardrails
- `tests/mlir/run_tfmbs_to_linalg.py` wraps `mlir-opt --load-dialect-plugin=build/libtfmbs_plugin.dylib --pass-pipeline=builtin.module(tfmbs-to-linalg)` and fails if `linalg.matmul` disappears or `tfmbs.gemv`/`tfmbs.fused_gemv` survive the pass.
- CI clones `llvm/llvm-project`, builds the shared MLIR tools, regenerates the Torch/ONNX fixtures via `tools/torch_to_tfmbs.py`, and executes `tests/mlir/run_tfmbs_to_linalg.py` against `tests/mlir/tfmbs_to_linalg.mlir`, `tests/mlir/tfmbs_fusion.mlir`, `tests/mlir/tfmbs_multi_fusion.mlir`, and `tests/mlir/torch_tfmbs.mlir` so telemetry hints are always present before merges land.
- Runtime history is captured through `pytfmbs.AdaptiveRuntimeAgent.save_history("logs/adaptive_history.json")`, and `tools/adaptive_dashboard.py` compares the `fusion_order` / `fusion_sparsity` entries from the runtime log against the compiler-generated MLIR telemetry so both sides observe the same hints.

## Documentation & Tracking
- This document now doubles as the living compiler track status sheet; keep it updated whenever TableGen definitions change, new passes land, or the regression tooling evolves.
- The roadmap links to this file so other tracks can see the exact artifacts and tools that are ready while the hardware team is still waiting on the FPGA testbeds.
- After the FPGA verification report records telemetry findings, add notes here (and in `docs/hardware_verification_report.md`) describing which compiler hints were exercised so Track B and Track A stay aligned.

## Documentation/Tracking Updates
- Link `docs/compiler_track_plan.md` from `docs/ROADMAP.md` and any relevant HOWTOs (e.g., `docs/12_PYTORCH.md`, `docs/07_API.md`) so future contributors know where the compiler efforts are centered.  
- Keep a running TODO list inside this doc, noting which tasks have upstream owners and their dependencies (hardware telemetry, Phase 26 adaptive logic).
- After each major compiler milestone (dialect stabilized, pass landed), add entries to `docs/hardware_verification_report.md:6-39` under “Stakeholder alignment” so the hardware team knows what compiler metadata is available for telemetry comparison.

With the hardware team waiting on the XC7Z020/XC7Z045, finishing these compiler deliverables keeps Track B moving and ensures the software stack can fully exploit the fabric once Phase 22 hardware verification completes.

### Current Artifacts
- `src/mlir/TfmbsOps.td` defines the dialect and operations; `/Users/t81dev/Code/llvm-project/build-shared/bin/mlir-tblgen` now regenerates `src/mlir/TfmbsOps.h.inc`/`TfmbsOps.cpp.inc`, and the generated op classes are included from `TfmbsDialect.{h,cpp}` so the TableGen metadata stays synchronized with the hand-written scaffolding.
- `src/mlir/TfmbsPasses.{h,cpp}` implement `TfmbsToLinalgPass`, register the `tfmbs-to-linalg` pipeline, and now ensure both the `tfmbs` and `linalg` dialects are available to the MLIR context before running lowering so `mlir-opt --pass-pipeline=builtin.module(tfmbs-to-linalg)` can invoke the transformation.
- Tests under `tests/mlir/` cover both the lit-style `tfmbs_to_linalg.mlir` (for future mlir-opt runs) and the `TfmbsPassTest.cpp` executable that parses, runs the pass, and validates that it emits `linalg.matmul`.
- `tools/torch_to_tfmbs.py` exports simple Torch/ONNX graphs, scans `MatMul`/`Gemm` nodes, and writes `tests/mlir/torch_tfmbs.mlir` with computed `tile_mask` and telemetry dictionaries (layer name + sparsity). This sample MLIR keeps the front-end path exercised while the pass, regression script (`run_tfmbs_to_linalg.py`), and telemetry attributes remain in sync with the Adaptive Runtime Agent’s requirements.
- `pytfmbs.TFMBSLinear.telemetry_hint` mirrors the same metadata dictionary (layer name, sparsity, tile mask) so Torch/ONNX models can continuously emit compile-time hints that the runtime agent already expects. The telemetry dictionary now adds `fusion_order`/`fusion_sparsity` so the Adaptive Runtime Agent can consume the same fusion hints the compiler emits.
- **Operator fusion:** `TfmbsFusionPass` identifies sequential `tfmbs.gemv` kernels that share telemetry/tile-mask, collapses their weight/input/output triplets into a single `tfmbs.fused_gemv`, and carries the staged telemetry array forward before the lowering pass emits the corresponding `linalg.matmul` ops. `tests/mlir/tfmbs_fusion.mlir` exercises the fused path.
- **Multi-stage fusion:** Added `tests/mlir/tfmbs_multi_fusion.mlir` to cover chains of three GEMV stages so the fusion pass handles longer pipelines and CI exercises it alongside the previous fixtures.
- **Telemetry validation:** `tests/test_phase13.py` now records multi-stage telemetry from `TFMBSSequential` and verifies `AdaptiveRuntimeAgent` consumes the fused order/sparsity, so runtime dictionaries align with the compiler hints. Use `pytfmbs.AdaptiveRuntimeAgent.save_history()` plus `tools/adaptive_dashboard.py` to compare runtime logs against the compiler’s `tests/mlir/torch_tfmbs.mlir`.

### Runbook & Status
- **TableGen regeneration:** `/Users/t81dev/Code/llvm-project/build-shared/bin/mlir-tblgen -gen-op-decls/defs src/mlir/TfmbsOps.td` runs cleanly once the shared LLVM/MLIR build is ready.
- **Pass/test CMake wrapper:** `CMakeLists.txt` now links `src/mlir/TfmbsDialect.cpp`, `TfmbsPasses.cpp`, and `tests/mlir/TfmbsPassTest.cpp` against the shared MLIR targets; `cmake -G Ninja -DLLVM_DIR=/Users/t81dev/Code/llvm-project/build-shared/lib/cmake/llvm -DMLIR_DIR=/Users/t81dev/Code/llvm-project/build-shared/lib/cmake/mlir` followed by `ninja -C build` builds `TfmbsPassTest`, `libtfmbs_dialect.dylib`, and `libtfmbs_plugin.dylib`.
- **Live verification:** `./build/TfmbsPassTest` succeeds, and the lowering now runs via `mlir-opt` when the plugin is loaded as a dialect plugin:  
  `bash -lc '/Users/t81dev/Code/llvm-project/build-shared/bin/mlir-opt --load-dialect-plugin=/Users/t81dev/Code/ternary-fabric/build/libtfmbs_plugin.dylib --pass-pipeline="builtin.module(tfmbs-to-linalg)" tests/mlir/tfmbs_to_linalg.mlir'`  
  prints the lowered `linalg.matmul` IR without crashes because the decoder registers both dialects before the pass runs.
- **Regression helper:** `tests/mlir/run_tfmbs_to_linalg.py` wraps the mlir-opt invocation and asserts `linalg.matmul` appears while `tfmbs.gemv` disappears; run it with `--mlir-opt`/`--plugin` overrides in CI to ensure future commits keep the pass happy.
- **Lit guard:** `ninja -C build check-tfmbs` now runs the lit suite that exercises `tests/mlir/tfmbs_fusion.mlir` and `tests/mlir/tfmbs_multi_fusion.mlir`, verifying telemetry-capable fusion chains still lower cleanly after the pass; wire this target into CI so these regressions run immediately after the plugin build.
- **CI verification:** GitHub Actions now clones `llvm/llvm-project`, builds shared `mlir-opt`, configures the plugin against that build, runs `tools/torch_to_tfmbs.py`, and calls `python tests/mlir/run_tfmbs_to_linalg.py` against both `tests/mlir/tfmbs_to_linalg.mlir` and `tests/mlir/torch_tfmbs.mlir` so `linalg.matmul` is always present before merges land.
- **Operator fusion regression:** The same workflow now runs `tests/mlir/run_tfmbs_to_linalg.py --mlir=tests/mlir/tfmbs_fusion.mlir` so `tfmbs-fuse` is exercised and fusion outputs still lower to `linalg.matmul`.
- **Track status:** These compiler efforts are complete—TableGen, passes, regression scripts, and dashboard alignment all execute without FPGA hardware—and the remaining work is to resume the documented hardware verification steps once the XC7Z020/XC7Z045 boards return so Track A can graduate from the “Mock” driver state.
