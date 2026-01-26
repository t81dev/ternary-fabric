# Contributing to ternary-fabric

Thank you for your interest in advancing ternary-native computing! This project bridges the gap between binary control and ternary execution. To maintain the integrity of the specification, please follow these guidelines.

## 1. Architectural Philosophy

All contributions must adhere to the **Binary Sovereignty** principle:

* Binary hosts control and schedule (the "Brain").
* Ternary planes store and transform (the "Muscle").
* Communication MUST occur via **Ternary Frame Descriptors (TFDs)**.

## 2. Software Contributions (C / Python)

We are currently building the emulation and tooling suite.

* **Code Style:** Use C99 for headers and source to ensure maximum portability across embedded toolchains.
* **Header-First:** Any new feature (e.g., a new packing format) must be defined in `include/tfmbs.h` before implementation.
* **Testing:** Every new kernel or codec must include a Python reference script in `tools/` to verify mathematical correctness before it is optimized in C.

## 3. Hardware Contributions (Verilog / SystemVerilog)

We are moving toward FPGA-ready RTL.

* **Modularity:** Design components at the "Lane" level. A lane should be an independent unit that can be tiled to satisfy the `lane_count` in a TFD.
* **Balanced Logic:** Use symmetric representations. Ensure that your ALU treats `-1` and `+1` with equal latency and power profiles.
* **Interconnects:** Aim for AXI4-Lite for the control plane (TFD submission) and AXI4-Stream for the data plane (Trit streaming).

## 4. Proposing New Specs

If you wish to add a new normative specification (e.g., `NETWORK_FABRIC.md`):

1. Open an Issue titled `[PROPOSAL] <Spec Name>`.
2. Ensure the spec defines a **Normative** level and a **Fallback** path.
3. Once the community agrees, submit a PR adding the `.md` file to the `specs/` directory.

## 5. Development Workflow

1. **Fork** the repository.
2. **Create a Feature Branch** (`git checkout -b feature/ternary-alu`).
3. **Validate:** Run `./test_suite.sh` to ensure no regressions in the reference logic.
4. **Submit a Pull Request:** Link the PR to the relevant phase in `roadmap.md`.

---