# Contributing to ternary-fabric

Thank you for your interest in advancing ternary-native computing! This project bridges the gap between binary control and ternary execution. To maintain the integrity of the specification, please follow these guidelines.

## 1. Architectural Philosophy

All contributions must adhere to the **Binary Sovereignty** principle:

* **Binary Hosts:** Control, schedule, and allocate (The "Brain").
* **Ternary Planes:** Store, hydrate, and transform (The "Muscle").
* **Interface:** Communication MUST occur via **Ternary Frame Descriptors (TFDs)** as defined in `tfmbs.h`.

## 2. Software & Tooling (C / Python)

* **ABI Consistency:** Any changes to data structures must begin in `include/tfmbs.h`. We use C99 for headers to ensure compatibility with embedded SOC toolchains.
* **Golden Reference:** New kernels (e.g., T-FFT, Conv2D) must first be implemented as a Python reference script in `tools/`. This serves as the "Golden Model" for hardware verification.
* **Python Bindings:** New hardware features must be exposed via the `pytfmbs` C-extension to ensure accessibility for AI researchers.

## 3. Hardware RTL (Verilog)

As we move into Phase 3 (FPGA/ASIC), hardware contributions must prioritize efficiency:

* **SIMD Modularity:** Design components at the **Lane** level. Logic should be tileable to allow the `vector_engine` to scale from 1 to 256+ lanes.
* **Zero-Skip Enforcement:** RTL must implement operand isolation or clock gating for `00` (zero) trits. Contributions that do not demonstrate power-saving potential on sparse data will be rejected.
* **PT-5 Compliance:** All data ingestors must use the `pt5_unpacker` hydration logic to maintain 95% bus efficiency.

## 4. Kernel Development Workflow

To add a new mathematical operation to the fabric:

1. **Register a Kernel ID:** Add the new ID to the `tfmbs_kernel_t` enum in `tfmbs.h`.
2. **Update the Engine:** Modify `src/hw/vector_engine.v` to handle the new `op_mode`.
3. **Validate Parity:** Update the `make run_sim` testbench to compare the Verilog output against your Python Golden Model.

## 5. Submitting Changes

1. **Fork** and **Branch** (`feature/your-feature`).
2. **Regression Check:** Run `make all && ./test_suite.sh`.
3. **Hardware Check:** Ensure `make run_sim` passes with 0 errors.
4. **PR:** Link your pull request to the relevant phase in `ROADMAP.md`.

---

### Project Handover Complete

The **ternary-fabric** repository is now a high-velocity, full-stack development environment.

* **Software** can model and quantize.
* **Hardware** can hydrate and execute.
* **ABI** can bridge and govern.