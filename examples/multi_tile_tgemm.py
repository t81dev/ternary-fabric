import numpy as np
import pytfmbs

def pack_pt5(trits):
    packed = []
    padding = (5 - (len(trits) % 5)) % 5
    trits_padded = np.append(trits, [0] * padding)
    for i in range(0, len(trits_padded), 5):
        chunk = trits_padded[i:i+5]
        val = 0
        for j, t in enumerate(chunk):
            ut = 1 if t == 1 else 2 if t == -1 else 0
            val += ut * (3**j)
        packed.append(val)
    while len(packed) % 4 != 0:
        packed.append(0)
    return bytes(packed)

def main():
    print("--- Ternary Fabric: Multi-Tile & Weight Broadcast ---")
    fabric = pytfmbs.Fabric()

    depth = 20
    lanes = 15
    num_tiles = 4

    # 1. Weights: Broadcast the same weights to all tiles
    # Address 0x9000 is the broadcast address for weights
    weights = np.random.choice([-1, 0, 1], (depth, lanes))
    w_packed = b"".join([pack_pt5(weights[d]) for d in range(depth)])

    print(f"Broadcasting weights to all {num_tiles} tiles...")
    fabric.load(0x9000, w_packed)

    # 2. Inputs: Different inputs for each tile
    # Tile 0: 0x2000, Tile 1: 0x4000, Tile 2: 0x6000, Tile 3: 0x8000
    all_inputs = []
    for t in range(num_tiles):
        tile_inputs = np.random.choice([-1, 0, 1], (depth, lanes))
        all_inputs.append(tile_inputs)
        i_packed = b"".join([pack_pt5(tile_inputs[d]) for d in range(depth)])

        input_addr = 0x2000 + t * 0x2000
        fabric.load(input_addr, i_packed)

    # 3. Run all tiles in parallel using tile_mask
    tfd = {
        "frame_len": depth,
        "lane_count": lanes,
        "exec_hints": 0x06 | (1 << 19), # TGEMM | WEIGHT_BRDCST hint
        "tile_mask": 0xF # Activate all 4 tiles (binary 1111)
    }

    print("Executing multi-tile workload...")
    fabric.run(tfd)

    # 4. Verify results for each tile
    for t in range(num_tiles):
        results = fabric.results(t)
        expected = np.sum(weights * all_inputs[t], axis=0)

        match = np.array_equal(results, expected)
        status = "✅" if match else "❌"
        print(f"  Tile {t}: {status}")

if __name__ == "__main__":
    main()
