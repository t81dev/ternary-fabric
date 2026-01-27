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
    print("--- Ternary Fabric: AXI-Stream DMA Loader Demo ---")
    fabric = pytfmbs.Fabric()

    # The DMA loader (load_stream) allows streaming data directly into SRAM
    # using an AXI-Stream interface, which is faster than AXI-Lite register writes.

    depth = 100
    lanes = 15
    data = np.random.choice([-1, 0, 1], (depth, lanes))
    packed_data = b"".join([pack_pt5(data[d]) for d in range(depth)])

    # We specify the target address in a TFD-like dictionary
    stream_info = {
        "base_addr": 0x1000 # Loading into Tile 0 Weight SRAM
    }

    print(f"Streaming {len(packed_data)} bytes to 0x1000 via DMA...")
    fabric.load_stream(stream_info, packed_data)

    print("Data loaded successfully.")
    print("Note: In Mock Mode, this simulates the memory copy and increments")
    print("the 'burst_wait_cycles' counter to model DMA latency.")

    profile = fabric.profile_detailed()
    print(f"Measured DMA Burst Latency: {profile['burst_wait_cycles']} cycles")

if __name__ == "__main__":
    main()
