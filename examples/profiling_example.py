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
    print("--- Ternary Fabric: Hardware Profiling & Zero-Skip ---")
    fabric = pytfmbs.Fabric()

    # Create a sparse workload (80% zeros)
    depth = 100
    lanes = 15
    weights = np.random.choice([-1, 0, 1], (depth, lanes), p=[0.1, 0.8, 0.1])
    inputs = np.ones((depth, lanes), dtype=int)

    w_packed = b"".join([pack_pt5(weights[d]) for d in range(depth)])
    i_packed = b"".join([pack_pt5(inputs[d]) for d in range(depth)])

    fabric.load(0x1000, w_packed)
    fabric.load(0x2000, i_packed)

    # Enable Zero-Skip in exec_hints (Bit 17)
    ZERO_SKIP_EN = (1 << 17)
    tfd = {
        "frame_len": depth,
        "lane_count": lanes,
        "exec_hints": 0x06 | ZERO_SKIP_EN,
        "tile_mask": 0x1
    }

    print(f"Executing sparse workload with Zero-Skip enabled...")
    fabric.run(tfd)

    # Read profiling counters
    profile = fabric.profile() # Aggregated
    detailed = fabric.profile_detailed()
    tile0 = fabric.profile_tile(0)

    print("\nGlobal Metrics:")
    print(f"  Total Cycles: {profile['cycles']}")
    print(f"  Total Active Lane-Cycles: {profile['utilization']}")

    print("\nZero-Skip Effectiveness (Lane 0):")
    skips = tile0['skips'][0]
    active = tile0['active_cycles'][0]
    print(f"  Lane 0 Skips: {skips}")
    print(f"  Lane 0 Active: {active}")
    print(f"  Sparsity Exploited: {(skips/active)*100:.1f}%")

    if 'overflow_flags' in detailed:
        print(f"\nOverflow Status (Bitmask): {hex(detailed['overflow_flags'])}")

if __name__ == "__main__":
    main()
