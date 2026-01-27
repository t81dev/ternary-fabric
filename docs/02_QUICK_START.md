# 02: Quick Start Guide

This guide will walk you through running your first ternary operation using the Python API.

## 1. The "Hello Ternary" Operation
The simplest way to use the fabric is to perform a Dot Product (or T-GEMM) operation.

### Step 1: Initialize the Fabric
In a Python script, import the library and instantiate the `Fabric` class. By default, if the hardware device `/dev/mem` is not found, it will fall back to **Mock Mode**.

```python
import pytfmbs
fabric = pytfmbs.Fabric()
```

### Step 2: Prepare and Load Data
Ternary data must be packed into the **PT-5** format before being loaded. For this example, we'll use a helper to load random trits.

```python
import numpy as np
# (Assume pack_pt5 helper is defined)
weights = np.random.choice([-1, 0, 1], (10, 15))
packed = b"".join([pack_pt5(row) for row in weights])

# Load into Tile 0 Weight SRAM (0x1000)
fabric.load(0x1000, packed)
```

### Step 3: Define the Task (TFD)
The Ternary Frame Descriptor (TFD) tells the hardware what to do.

```python
tfd = {
    "frame_len": 10,
    "lane_count": 15,
    "exec_hints": 0x06, # Kernel ID for TGEMM
    "tile_mask": 0x1    # Enable Tile 0
}
```

### Step 4: Execute and Read Results
```python
fabric.run(tfd)
results = fabric.results(0)
print(f"Result for Lane 0: {results[0]}")
```

## 2. Running the Full Example
A complete, executable version of this walkthrough is available in the repository:

```bash
python3 examples/quick_start.py
```

## 3. Key CLI Tools
The repository includes several command-line tools for working with ternary data:

*   **Quantization:** Convert float weights to ternary.
    ```bash
    python3 tools/quantize.py my_weights.npy -o weights.txt
    ```
*   **Data Packing:** Convert text trits to `.tfrm` binary.
    ```bash
    python3 tools/ternary_cli.py weights.txt --kernel 6
    ```
*   **Visualization:** Inspect the spatial layout of a `.tfrm` file.
    ```bash
    python3 tools/txd.py weights.txt.tfrm
    ```
