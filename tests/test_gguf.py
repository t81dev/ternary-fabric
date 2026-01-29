import sys
import os
import numpy as np
import pytest

# Add src and tools to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
sys.path.append(os.path.join(os.getcwd(), 'tools'))

from gguf_mock import create_mock_gguf
import pytfmbs

def test_gguf_loading():
    """
    Tests loading a Q4_0 quantized tensor from a GGUF file into the Fabric.
    """
    # 1. Create a mock GGUF file with 128 elements (4 blocks of 32)
    # Using a ramp from -1 to 1 to ensure a good mix of ternary values
    weights = np.linspace(-1, 1, 128).astype(np.float32)
    filename = "test_weights.gguf"
    tensor_name = "blk.0.weight"

    try:
        create_mock_gguf(filename, tensor_name, weights)

        # 2. Initialize Fabric
        fabric = pytfmbs.Fabric()

        # 3. Load tensor from GGUF
        # The loader should:
        # - Read GGUF header
        # - Dequantize Q4_0 to float
        # - Quantize float to ternary
        # - Pack ternary to PT-5
        # - Load to Fabric SRAM
        shape = pytfmbs.load_gguf_tensor(fabric, filename, tensor_name, address=0x1000)

        # Verify shape (1D tensor is reshaped to 1xN)
        assert shape == (1, 128)

        # 4. Verify functional correctness via execution
        # Input: All 1s (packed for GEMV)
        inputs = np.ones(128, dtype=np.int8)
        packed_x = pytfmbs.torch_integration.pack_gemv_input(inputs)
        fabric.load(0x2000, packed_x)

        # Run T-GEMM kernel
        tfd = {
            "base_addr": 0x1000,
            "depth": 128,
            "lane_count": 1,
            "tile_mask": 1,
            "exec_hints": 1, # TFMBS_KERNEL_TGEMM / DOT
        }
        fabric.run(tfd)

        # Read results for Lane 0
        res = fabric.results(0)
        print(f"Result for Lane 0: {res[0]}")

        # The result should be the sum of ternary weights (since inputs are all 1)
        # We can't easily predict the exact sum because of the TWN quantization threshold,
        # but it shouldn't be zero for this ramp.
        # In mock mode, the C code actually performs the math.
        assert res[0] != 0

    finally:
        # Clean up
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == "__main__":
    test_gguf_loading()
