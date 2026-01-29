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
- `src/mlir/TfmbsOps.td` defines the dialect and operations; `mlir-tblgen` (run from `../llvm-project/build/bin/mlir-tblgen` with `-I` pointing at the MLIR/LLVM include tree) now regenerates `src/mlir/TfmbsOps.h.inc`/`TfmbsOps.cpp.inc`, which `TfmbsDialect.{h,cpp}` include so the generated op classes and `GET_OP_LIST` stay synchronized with the TableGen metadata.
- `src/mlir/TfmbsPasses.{h,cpp}` implement `TfmbsToLinalgPass`, and a `PassPipelineRegistration` hook registers the `tfmbs-to-linalg` pipeline so `mlir-opt --tfmbs-to-linalg` can be wired later.
- Tests under `tests/mlir/` cover both the lit-style `tfmbs_to_linalg.mlir` (for future mlir-opt runs) and the `TfmbsPassTest.cpp` executable that parses, runs the pass, and validates that it emits `linalg.matmul`.

-### Runbook & Status
- **TableGen regeneration:** `mlir-tblgen -gen-op-decls/defs src/mlir/TfmbsOps.td` succeeded once the LLVM/MLIR build under `/Users/t81dev/Code/llvm-project/build` was available.
- **Pass/test CMake wrapper:** Added `CMakeLists.txt` that links `src/mlir/TfmbsDialect.cpp`, `TfmbsPasses.cpp`, and `tests/mlir/TfmbsPassTest.cpp` against the MLIR build via `MLIR::` targets. Run `cmake -G Ninja -DLLVM_DIR=... -DMLIR_DIR=...` inside a `build` directory to generate `TfmbsPassTest` plus its plugin.
- **Next step:** Build with Ninja, run `./build/TfmbsPassTest`, and invoke `/Users/t81dev/Code/llvm-project/build/bin/mlir-opt --load-dialect-plugin=./build/libtfmbs_plugin.dylib --pass-pipeline=tfmbs-to-linalg tests/mlir/tfmbs_to_linalg.mlir` (or equivalent) so the lowering is verified. This currently crashes inside `StorageUniquerImpl::getOrCreate()` because the plugin pulls in its own copy of the static MLIR runtime, so we still need an MLIR build with shared libs (e.g., `-DBUILD_SHARED_LIBS=ON`) before the plugin can load cleanly.
- **Shared MLIR build:** Reconfiguring LLVM/MLIR with `-DBUILD_SHARED_LIBS=ON` to regain shared `libMLIRIR`/`libLLVMSupport` symbols takes a long time (the first `cmake --build ... mlir-opt` attempt ran for >8 minutes before the toolchain build timed out). If shared artifacts are available, the plugin crash should disappear because it will no longer embed a second set of `StorageUniquer` globals.
