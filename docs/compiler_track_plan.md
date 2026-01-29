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
- `src/mlir/TfmbsOps.td` documents the dialect and operator grammar in TableGen, and `src/mlir/TfmbsDialect.{h,cpp}` leverage that schema to expose `tfmbs.pack`, `tfmbs.transfer`, `tfmbs.dma_load`, and `tfmbs.gemv`.
- `src/mlir/TfmbsPasses.{h,cpp}` implement `TfmbsToLinalgPass`, which rewrites `tfmbs.gemv` into `linalg.matmul`.
- Tests under `tests/mlir/` cover both the lit-style `tfmbs_to_linalg.mlir` (for future mlir-opt runs) and the `TfmbsPassTest.cpp` executable that parses, runs the pass, and asserts the expected transformation happens before linking into downstream toolchains.
