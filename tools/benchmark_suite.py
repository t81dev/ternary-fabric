import numpy as np
import pytfmbs
import time
import os

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

def run_benchmark(depth, lanes, num_tiles, sparsity=0.0):
    fabric = pytfmbs.Fabric()

    # Weights and Inputs
    weights = np.random.choice([-1, 0, 1], (depth, lanes), p=[(1-sparsity)/2, sparsity, (1-sparsity)/2])
    inputs = np.random.choice([-1, 0, 1], (depth, lanes))

    w_packed = b"".join([pack_pt5(weights[d]) for d in range(depth)])
    i_packed = b"".join([pack_pt5(inputs[d]) for d in range(depth)])

    # Load weights (broadcast)
    fabric.load(0x9000, w_packed)

    # Load inputs (per tile)
    for t in range(num_tiles):
        fabric.load(0x2000 + t * 0x2000, i_packed)

    tfd = {
        "frame_len": depth,
        "lane_count": lanes,
        "exec_hints": 0x06 | (1 << 17) | (1 << 19), # TGEMM | ZERO_SKIP | WEIGHT_BRDCST
        "tile_mask": (1 << num_tiles) - 1
    }

    start_wall = time.time()
    fabric.run(tfd)
    end_wall = time.time()

    profile = fabric.profile()
    detailed = fabric.profile_detailed()

    cycles = profile['cycles']
    total_macs = depth * lanes * num_tiles
    total_ops = 2 * total_macs

    # Simulated metrics (assuming 250MHz clock as per README)
    clock_freq = 250e6
    simulated_time = cycles / clock_freq
    gops = (total_ops / simulated_time) / 1e9 if simulated_time > 0 else 0

    # Bandwidth efficiency (approximation)
    # Total trits transferred: (depth * lanes * 2) [for one load of weights and inputs]
    # In multi-tile with broadcast: 1 weights load + num_tiles inputs load
    bytes_transferred = len(w_packed) + num_tiles * len(i_packed)
    bw_efficiency = bytes_transferred / (cycles * 4) # 4 bytes per cycle on 32-bit bus?

    total_skips = sum(profile['skips'])
    total_active = sum(detailed['active_cycles'])
    sparsity_ratio = total_skips / total_active if total_active > 0 else 0

    return {
        "size": f"{depth}x{lanes}x{num_tiles}",
        "cycles": cycles,
        "gops": gops,
        "sparsity_ratio": sparsity_ratio,
        "wall_time": end_wall - start_wall
    }

def main():
    print(f"{'Size':<15} | {'Cycles':<10} | {'GOPS':<10} | {'Sparsity %':<12} | {'Wall Time':<10}")
    print("-" * 65)

    scales = [
        (2, 2, 1), (3, 3, 1), (4, 4, 1), # Small edge cases
        (16, 15, 4), (32, 15, 4), (64, 15, 4), # Larger
        (100, 15, 4) # Full
    ]

    for depth, lanes, num_tiles in scales:
        res = run_benchmark(depth, lanes, num_tiles, sparsity=0.5)
        print(f"{res['size']:<15} | {res['cycles']:<10} | {res['gops']:<10.2f} | {res['sparsity_ratio']*100:<11.1f}% | {res['wall_time']:<10.4f}")

if __name__ == "__main__":
    main()
