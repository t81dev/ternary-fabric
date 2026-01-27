import numpy as np
import pytfmbs
import os

def pack_pt5(trits):
    """
    Packs a list of trits (-1, 0, 1) into PT-5 format (5 trits per byte).
    Each byte is sum(t_i * 3^i) where t_i is mapped to (0, 1, 2).
    """
    packed = []
    # Pad to multiple of 5
    padding = (5 - (len(trits) % 5)) % 5
    trits_padded = np.append(trits, [0] * padding)

    for i in range(0, len(trits_padded), 5):
        chunk = trits_padded[i:i+5]
        val = 0
        for j, t in enumerate(chunk):
            # Map -1 -> 2, 0 -> 0, 1 -> 1
            ut = 1 if t == 1 else 2 if t == -1 else 0
            val += ut * (3**j)
        packed.append(val)

    # Pad to 4-byte boundary for AXI compatibility
    while len(packed) % 4 != 0:
        packed.append(0)
    return bytes(packed)

def main():
    print("--- Ternary Fabric Quick Start: T-GEMM ---")

    # 1. Initialize the Fabric (Mock mode by default if no hardware detected)
    fabric = pytfmbs.Fabric()

    # 2. Define workload (15 lanes x 10 depth)
    lanes = 15
    depth = 10
    weights = np.random.choice([-1, 0, 1], (depth, lanes))
    inputs = np.random.choice([-1, 0, 1], (depth, lanes))

    # 3. Pack data into PT-5 format
    w_packed = b"".join([pack_pt5(weights[d]) for d in range(depth)])
    i_packed = b"".join([pack_pt5(inputs[d]) for d in range(depth)])

    # 4. Load data into Fabric SRAM (Tile 0)
    # Weights go to 0x1000, Inputs to 0x2000
    fabric.load(0x1000, w_packed)
    fabric.load(0x2000, i_packed)

    # 5. Execute using a Ternary Frame Descriptor (TFD)
    tfd = {
        "base_addr": 0,    # Offset within SRAM
        "frame_len": depth,
        "lane_count": lanes,
        "exec_hints": 0x06, # TFMBS_KERNEL_TGEMM
        "tile_mask": 0x1    # Use Tile 0
    }

    print(f"Running T-GEMM kernel on {lanes} lanes, depth {depth}...")
    fabric.run(tfd)

    # 6. Read results
    results = fabric.results(0)

    # 7. Verify against NumPy reference
    expected = np.sum(weights * inputs, axis=0)

    print("\nResults (First 5 lanes):")
    print(f"  Hardware: {results[:5]}")
    print(f"  Expected: {expected[:5].tolist()}")

    match = np.array_equal(results, expected)
    if match:
        print("\n✅ SUCCESS: Hardware results match reference exactly.")
    else:
        print("\n❌ FAILURE: Mismatch detected.")

if __name__ == "__main__":
    main()
