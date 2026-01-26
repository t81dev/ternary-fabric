import os
import sys
import numpy as np
import time

# Add src to path to find pytfmbs
sys.path.append(os.path.join(os.getcwd(), 'src/pytfmbs'))
import pytfmbs

def pack_pt5_fabric(trits):
    packed_bytes = []
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

def generate_sparse_data(rows, cols, sparsity):
    """Generates ternary data with specific sparsity (percentage of zeros)"""
    size = rows * cols
    data = np.random.choice([-1, 0, 1], size=size, p=[(1-sparsity)/2, sparsity, (1-sparsity)/2])
    return data.reshape(rows, cols)

def main():
    print("📊 Ternary Fabric: Zero-Skip Efficiency Profiling")
    print("-" * 60)
    print(f"{'Sparsity (%)':<15} | {'Cycles':<10} | {'Total Skips':<15} | {'Skip Ratio (%)':<15}")
    print("-" * 60)

    LANES = 15
    COLS = 1000
    fabric = pytfmbs.Fabric()

    sparsities = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

    results = []

    for s in sparsities:
        weights = generate_sparse_data(LANES, COLS, s)
        inputs = generate_sparse_data(1, COLS, s).flatten()

        # Pack
        weight_bytes = b""
        for c in range(COLS):
            col_trits = [weights[r, c] for r in range(LANES)]
            weight_bytes += pack_pt5_fabric(col_trits) + b'\x00'

        input_bytes = b""
        for d in range(COLS):
            row_inputs = [inputs[d]] * LANES
            input_bytes += pack_pt5_fabric(row_inputs) + b'\x00'

        fabric.load(0x1000, weight_bytes)
        fabric.load(0x2000, input_bytes)

        tfd = {
            "depth": COLS,
            "lane_count": LANES,
            "exec_hints": 0x06 | (1 << 17) # TGEMM | Zero-Skip
        }

        fabric.run(tfd)
        metrics = fabric.profile()

        total_skips = sum(metrics['skips'])
        total_ops = metrics['cycles'] * LANES
        skip_ratio = (total_skips / total_ops) * 100 if total_ops > 0 else 0

        print(f"{s*100:<15.1f} | {metrics['cycles']:<10} | {total_skips:<15} | {skip_ratio:<15.2f}")
        results.append((s, skip_ratio))

    # Verification of Theoretical Skip Ratio
    # Skip happens if W=0 OR I=0.
    # P(Skip) = P(W=0) + P(I=0) - P(W=0 AND I=0)
    # If both have sparsity 's', P(Skip) = s + s - s^2 = 2s - s^2
    print("-" * 60)
    print("\n🔍 Theoretical vs Measured Skip Effectiveness:")
    for s, measured in results:
        theoretical = (2*s - s*s) * 100
        print(f"Sparsity {s*100:>3.0f}%: Theoretical Skip {theoretical:>5.1f}%, Measured {measured:>5.1f}%")

if __name__ == "__main__":
    main()
