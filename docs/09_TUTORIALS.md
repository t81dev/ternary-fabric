# 09: How-To Tutorials

This section provides step-by-step guides for common tasks.

## 1. Quantizing a Model for the Fabric

To run a standard binary model (e.g., in PyTorch or TensorFlow) on the fabric, you must first quantize the weights to $\{-1, 0, 1\}$.

1.  **Extract Weights:** Save your model weights as a NumPy `.npy` file.
2.  **Run Quantizer:** Use the provided tool to apply Ternary Weight Network (TWN) thresholding.
    ```bash
    python3 tools/quantize.py my_model_layer1.npy -o layer1_trits.txt
    ```
3.  **Pack for Fabric:**
    ```bash
    python3 tools/ternary_cli.py layer1_trits.txt --kernel 6
    ```
4.  **Load and Run:** Use `fabric.load("layer1_trits.txt.tfrm", 0x1000)` in your script.

## 2. Running a Multi-Tile T-GEMM

Multi-tile execution allows you to compute four dot products (or matrix rows) simultaneously.

1.  **Load weights to all tiles:**
    ```python
    fabric.load(0x9000, packed_weights)
    ```
2.  **Load unique inputs to each tile:**
    ```python
    fabric.load(0x2000, input0)
    fabric.load(0x4000, input1)
    # ... and so on
    ```
3.  **Run with full tile mask:**
    ```python
    fabric.run({"frame_len": 100, "tile_mask": 0xF, "exec_hints": 0x06})
    ```
4.  See `examples/multi_tile_tgemm.py` for a full implementation.

## 3. Using the AXI-Stream DMA Loader

The DMA loader is the fastest way to move data from host memory to the fabric.

1.  **Prepare a TFD-like header** to specify the destination.
2.  **Use `load_stream`** to push the raw bytes.
    ```python
    stream_config = {"base_addr": 0x1000}
    fabric.load_stream(stream_config, large_data_buffer)
    ```
3.  This bypasses the AXI-Lite register overhead and uses the 32-bit streaming bus.

## 4. Debugging Misaligned TFDs

If your results look like noise:
1.  **Check `lane_stride`:** If your data is 2D, ensure the stride matches the trit-distance between elements in the same lane.
2.  **Inspect with `txd.py`:** Use the Ternary Hex Dump tool to verify that your `.tfrm` file matches your expected spatial layout.
3.  **Verify PT-5 Packing:** Ensure your packing logic correctly maps $-1 \rightarrow 2, 0 \rightarrow 0, 1 \rightarrow 1$.
