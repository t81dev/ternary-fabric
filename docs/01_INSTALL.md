# 01: Installation & Setup

This guide covers the requirements and steps to set up the Ternary Fabric environment for both software simulation and hardware deployment.

## 1. Prerequisites

### Software Dependencies
*   **Python 3.8+**
*   **NumPy:** Used for data preparation and reference verification.
*   **PyTorch (1.10+):** Required for `TFMBSLinear` and framework integration.
*   **Setuptools:** Required to build the Python C-Extension.
*   **GCC:** Needed for compiling the `pytfmbs` extension.

### Hardware/Simulation Tools (Optional)
*   **Verilator:** For cycle-accurate hardware simulation.
*   **Icarus Verilog:** For RTL functional verification.
*   **Vivado 2023.1+:** If targeting Xilinx FPGAs (specifically Zynq-7000 series).

## 2. Software Setup

### Clone the Repository
```bash
git clone https://github.com/your-repo/ternary-fabric.git
cd ternary-fabric
```

### Install Python Dependencies
```bash
pip install numpy torch setuptools pytest
```

### Build the `pytfmbs` Extension
The `pytfmbs` library provides the bridge between Python and the hardware (or mock simulation).
```bash
make python_ext
```
This will create a `.so` file in `src/pytfmbs/`.

### Build the Interposer and Device Library
To use the Fabric with `llama.cpp` or other binary applications:
```bash
make all
```
This will produce `libtfmbs_device.so` and `libtfmbs_intercept.so`.

### Set Environment Variables
To ensure Python can find the extension, add it to your `PYTHONPATH`:
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src/pytfmbs
```

## 3. Hardware Setup

### ASIC/FPGA Path
The RTL is located in `src/hw/`. The top-level module is `ternary_fabric_top.v`.

*   **FPGA:** A sample synthesis script for Vivado is provided in `tools/synth.tcl`.
*   **SRAM:** The design uses `src/hw/ternary_sram_wrapper.v`, which provides behavioral models of the dual-bank SRAMs. For ASIC targets, these should be replaced with vendor-specific memory macros.

## 4. Verifying the Installation

Run the smoke test to ensure the Python extension and mock mode are working correctly:
```bash
python3 test_mock.py
```
Or run the full regression suite:
```bash
pytest tests/
```
To verify the Phase 9 interposer features:
```bash
make run_mock_llama
```
