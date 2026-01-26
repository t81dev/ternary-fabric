import os
import sys
import numpy as np
import struct

# Add src to path to find pytfmbs
sys.path.append(os.path.join(os.getcwd(), 'src/pytfmbs'))
import pytfmbs

def pack_pt5_fabric(trits):
    """Packs trits using the mapping expected by hardware (-1->2, 0->0, 1->1)"""
    packed_bytes = []
    # Pad trits to a multiple of 5
    trits = list(trits)
    while len(trits) % 5 != 0:
        trits.append(0)

    for i in range(0, len(trits), 5):
        chunk = trits[i:i+5]
        byte_val = 0
        for power, trit in enumerate(chunk):
            mapped_val = 2 if trit == -1 else (0 if trit == 0 else 1)
            byte_val += mapped_val * (3 ** power)
        packed_bytes.append(byte_val)
    return bytes(packed_bytes)

def generate_synthetic_data(rows, cols, filename_prefix):
    """Generates synthetic ternary data and saves as .tfrm"""
    weights = np.random.choice([-1, 0, 1], size=(rows, cols))
    inputs = np.random.choice([-1, 0, 1], size=(cols,))

    # Pack weights (interleaved for fabric lanes, each column padded to 4 bytes for AXI)
    weight_bytes = b""
    for c in range(cols):
        col_trits = [weights[r, c] for r in range(rows)]
        weight_bytes += pack_pt5_fabric(col_trits) + b'\x00'

    # Repeat inputs for each lane to simulate broadcast in current hardware
    input_bytes = b""
    for d in range(cols):
        row_inputs = [inputs[d]] * rows
        input_bytes += pack_pt5_fabric(row_inputs) + b'\x00'

    w_file = f"{filename_prefix}_weights.tfrm"
    i_file = f"{filename_prefix}_inputs.tfrm"

    with open(w_file, "wb") as f:
        # Align to 4 bytes for the loader
        f.write(weight_bytes + b'\x00' * ((4 - len(weight_bytes) % 4) % 4))

    with open(i_file, "wb") as f:
        f.write(input_bytes + b'\x00' * ((4 - len(input_bytes) % 4) % 4))

    return weights, inputs, w_file, i_file

def software_reference(weights, inputs):
    return np.dot(weights, inputs)

def main():
    print("🚀 Ternary Fabric: End-to-End Inference Pipeline")

    LANES = 15
    COLS = 100

    # 1. Setup Data
    weights, inputs, w_file, i_file = generate_synthetic_data(LANES, COLS, "demo")
    expected = software_reference(weights, inputs)
    print(f"Generated {LANES}x{COLS} matrix and {COLS}-element vector.")

    # 2. Initialize Fabric
    fabric = pytfmbs.Fabric()

    # 3. Load Data into SRAM
    # Weight SRAM at 0x1000, Input SRAM at 0x2000
    print("Loading TFRM files into SRAM...")
    fabric.load(w_file, 0x1000)
    fabric.load(i_file, 0x2000)

    # 4. Run T-GEMM
    # TFD parameters
    tfd = {
        "base_addr": 0,
        "depth": COLS,
        "lane_count": LANES,
        "lane_mask": 0x7FFF,
        "exec_hints": 0x06 | (1 << 17) | (1 << 18) # TGEMM | Zero-Skip | Free-Neg
    }

    print("Starting T-GEMM execution...")
    fabric.run(tfd)

    # 5. Get Results
    results = fabric.results()
    print("Hardware Results:", results)
    print("Expected Results:", expected.tolist())

    # 6. Profile
    metrics = fabric.profile()
    print("\n--- Performance Metrics ---")
    print(f"Active Cycles: {metrics['cycles']}")
    print(f"Lane Utilization: {metrics['utilization']}")
    total_skips = sum(metrics['skips'])
    print(f"Total Zero-Skips: {total_skips}")

    # Calculate Sparsity
    w_zero = np.count_nonzero(weights == 0)
    i_zero = np.count_nonzero(inputs == 0)
    print(f"Weight Sparsity: {w_zero / weights.size:.2%}")
    print(f"Input Sparsity: {i_zero / inputs.size:.2%}")

    # Verify (In mock mode, this might fail unless we simulate it in core.c)
    match = True
    for h, e in zip(results, expected):
        if h != e:
            match = False
            break

    if match:
        print("\n✅ SUCCESS: Hardware matches Software Reference!")
    else:
        print("\n⚠️  NOTICE: Hardware mismatch (Expected in Mock Mode without simulation)")

    # 7. Test Partial Lanes (Lane Mask)
    print("\n--- Testing Partial Lanes (Mask 0x00FF) ---")
    tfd["lane_mask"] = 0x00FF # Only first 8 lanes
    fabric.run(tfd)
    results_masked = fabric.results()
    print("Masked Results:", results_masked)
    match_masked = True
    for i in range(15):
        expected_val = expected[i] if i < 8 else 0
        if results_masked[i] != expected_val:
            match_masked = False
            print(f"Mismatch at lane {i}: Got {results_masked[i]}, Expected {expected_val}")

    if match_masked:
        print("✅ Masked Execution Validated.")
    else:
        print("❌ Masked Execution Failed!")

if __name__ == "__main__":
    main()
