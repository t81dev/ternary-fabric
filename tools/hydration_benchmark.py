import numpy as np
import time

def hydrate_pt5(packed_bytes):
    trits = []
    for b in packed_bytes:
        val = b
        for _ in range(5):
            t = val % 3
            val //= 3
            trits.append(1 if t == 1 else -1 if t == 2 else 0)
    return trits

def hydrate_binary(packed_bytes):
    # Conventional binary: 2 bits per trit, 4 trits per byte
    trits = []
    for b in packed_bytes:
        val = b
        for _ in range(4):
            t_bits = val & 0x3
            val >>= 2
            # 00=0, 01=+1, 10=-1
            trits.append(1 if t_bits == 1 else -1 if t_bits == 2 else 0)
    return trits

def main():
    num_trits = 1_000_000

    # PT5: 1,000,000 trits / 5 trits per byte = 200,000 bytes
    pt5_data = np.random.bytes(num_trits // 5)

    # Binary: 1,000,000 trits / 4 trits per byte = 250,000 bytes
    binary_data = np.random.bytes(num_trits // 4)

    print(f"--- Hydration Micro-benchmark ({num_trits:,} trits) ---")

    start = time.time()
    _ = hydrate_pt5(pt5_data)
    pt5_time = time.time() - start
    print(f"PT5 Hydration:    {pt5_time:.4f}s ({len(pt5_data):,} bytes)")

    start = time.time()
    _ = hydrate_binary(binary_data)
    binary_time = time.time() - start
    print(f"Binary Hydration: {binary_time:.4f}s ({len(binary_data):,} bytes)")

    print("-" * 40)
    storage_saving = (1 - len(pt5_data) / len(binary_data)) * 100
    print(f"PT5 Storage Advantage: {storage_saving:.1f}%")

    # Speed comparison is tricky in Python because of loop overhead,
    # but it shows the complexity difference.
    speed_ratio = binary_time / pt5_time if pt5_time > 0 else 0
    print(f"Relative Speed Ratio: {speed_ratio:.2f}x")

if __name__ == "__main__":
    main()
