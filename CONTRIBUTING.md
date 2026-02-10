# Contributing to Ternary Fabric

Welcome! Ternary Fabric is an open-source project aimed at redefining AI acceleration through balanced ternary hardware.

## How to Contribute
1. **Fork the repo** and create your branch from `main`.
2. **Implement your changes**. Ensure you follow the Phase-based roadmap.
3. **Add tests** for any new features or bug fixes.
4. **Submit a Pull Request** with a clear description of your work.

## Good First Issues
- **Add a new MLIR canonicalization pass**: Improve the `tfmbs-fuse` pass to handle more complex graph patterns.
- **Port synthesis to another FPGA family**: Adapt `tools/vivado_flow.tcl` and constraints for ECP5 (Lattice) or iCE40.
- **Implement a new TFMBS kernel**: Add a element-wise subtraction or pooling kernel to the Verilog source and emulator.
- **Improve Telemetry Dashboard**: Add more visualizations (e.g., using Matplotlib) to `tools/adaptive_dashboard.py`.

## Coding Standards
- C: Use K&R style with 4-space indentation.
- Python: Follow PEP 8; use type hints where possible.
- Verilog: Use 2-space indentation and descriptive signal names.

## License
By contributing, you agree that your contributions will be licensed under the Apache License, Version 2.0.
