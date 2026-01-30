# Compiler Track — MLIR & Lowering Roadmap

Track B (Compiler & Ecosystem) in `docs/ROADMAP.md:156-170` is positioned to deliver the MLIR dialect, lowering passes, and adaptive scheduling hooks needed to make TFMBS a first-class compiler target. With hardware verification pending, we can still press ahead on the software-facing front by clarifying the current status and outlining actionable work.

## Current State
- `src/mlir/TfmbsOps.cpp` contains a prototype `TfmbsDialect` plus comments sketching a `TfmbsToLinalgPass` and `TfmbsPackOp` verification; there is no TableGen, pass registration, or lowering logic yet.
- The Python/PyTorch bindings under `src/pytfmbs` and `docs/12_PYTORCH.md` already expose `TFMBSLinear`, but the pathway from TorchScript/ONNX backends to the low-level ternary kernels depends on compiler support that is only partially documented.
- The research paper (`docs/TFMBS_Research_Paper.md:297`) reinforces the need for compiler IR integration to unlock operator fusion and graph-level optimizations before the physical fabric is available.
- Existing tests (e.g., Stage 4 pyro) rely on software encoders (`src/fabric_emulator.c`) that already model PT-5 streaming; we can reuse these to validate the compiler backend even without hardware.

## Prioritized Tasks
1. **Dialect Definition & Registration:**  
   - Replace the prototype in `src/mlir/TfmbsOps.cpp` with a TableGen-driven dialect (`TfmbsOps.td`) that defines the core ops (`tfmbs.pack`, `tfmbs.gemv`, `tfmbs.transfer`, `tfmbs.dma_load`).  
   - Provide affinities for types (`TFMBSWeightType`, `TFMBSDataType`) and register the dialect with `mlir-opt`.

2. **Lowering Entries:**  
   - Implement `TfmbsToLinalgPass` (and optionally `TfmbsToAffinePass`) that lowers `tfmbs.gemv`/`tfmbs.transfer` into `linalg.matmul`/`linalg.generic` with explicit PT-5 unpacking helpers.  
   - Add a pass pipeline (TorchMLIR -> `tfmbs` -> `linalg`) and unit tests covering multi-tile GEMV sequences; capture PT-5 packing semantics in the pass so that the simulator and later hardware can re-use the same metadata.

3. **Front-End Lowering Hooks:**  
   - Build Torch-MLIR or ONNX lowering glue that emits `tfmbs.dma_load` for streaming weight ingestion and rewrites `linalg.matmul` patterns matching ternary sparsity to `tfmbs.gemv`.  
   - Document the heuristics for mapping `TFMBS_HINT_KERNEL` bits (see `docs/TFMBS_Research_Paper.md`) into the pass.

4. **Compiler Tests & Verification:**  
   - Add lit/cmake tests that run `mlir-opt` on sample Torch/ONNX graphs, verifying the TFMBS dialect usages, tensor shapes, and runtime attributes (PT-5 packing, tile masks).  
   - Reuse the emulator (via `tests/bench_top.cpp` or `bin/reference_tfmbs`) to execute the lowered IR and compare metrics to the software-only baseline.

5. **Telemetry & Adaptive Scheduling Hooks:**  
   - Surface sensitivity telemetry in the dialect (attributes for layer criticality, sparsity) so the `Adaptive Runtime Agent` (Phase 26) can consult compile-time hints.  
   - Update documents/README to describe how compiler passes emit telemetry metadata expected by `libtfmbs_device.so`.

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
