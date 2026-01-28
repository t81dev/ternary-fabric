# Contributing to ternary-fabric

Thank you for your interest in advancing ternary-native computing! This project bridges the gap between binary control and ternary execution. To maintain the integrity of the specification, please follow these guidelines.

## 1. Architectural Philosophy

All contributions must adhere to the **Binary Sovereignty** principle:

* **Binary Hosts:** Control, schedule, and allocate (The "Brain").
* **Ternary Planes:** Store, hydrate, and transform (The "Muscle").
* **Interface:** Communication MUST occur via **Ternary Frame Descriptors (TFDs)** as defined in `tfmbs.h`.

## 2. Software & ML Researchers (Python / C)

* **ABI Consistency:** Any changes to data structures must begin in `include/tfmbs.h`. We use C99 for headers to ensure compatibility with embedded SOC toolchains.
* **Kernel Prototyping:** We prioritize **ML Research**. New kernels (e.g., T-LSTM, T-Attention) must first be implemented as a Python reference script in `tools/`.
* **Benchmarking:** All new features should include a corresponding benchmark in the measurement plane (see **[BENCHMARKS.md](BENCHMARKS.md)**).
* **Python Bindings:** New hardware features must be exposed via the `pytfmbs` C-extension to ensure accessibility for the AI research community.

## 3. RTL & Hardware Engineers (Verilog)

As we move toward high-density synthesis, hardware contributions must prioritize efficiency:

* **SIMD Modularity:** Design components at the **Ternary Lane** and **Tile** levels. Logic should be tileable to allow the fabric to scale multi-tile and multi-fabric.
* **Zero-Skip Enforcement:** RTL must implement operand isolation or clock gating for `00` (zero) trits. Contributions must demonstrate power-saving potential on sparse data via **Zero-Skip** logic.
* **PT-5 Compliance:** All data ingestors must use the `pt5_unpacker` hydration logic to maintain 95.1% bus efficiency.
* **Synthesis Aware:** Avoid vendor-specific primitives; use behavioral wrappers where possible to maintain ASIC/FPGA portability.

## 4. Kernel Development Workflow

To add a new mathematical operation to the fabric:

1. **Register a Kernel ID:** Add the new ID to the `tfmbs_kernel_t` enum in `tfmbs.h`.
2. **Update the Engine:** Modify the RTL or emulator to handle the new `exec_hints`.
3. **Validate Parity:** Compare the output against a Python Golden Model.

## 5. Submitting Changes

1. **Fork** and **Branch** (`feature/your-feature`).
2. **Regression Check:** Run `pytest tests/` and existing C testbenches (e.g., `bin/test_phase21`).
3. **PR:** Link your pull request to the relevant phase in **[ROADMAP.md](docs/ROADMAP.md)**.

---

© 2026 Ternary Fabric Project.
