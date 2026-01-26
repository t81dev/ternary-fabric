import sys
import os
import struct
import numpy as np

# Add tools and pytfmbs to path
sys.path.append(os.path.join(os.getcwd(), 'tools'))
sys.path.append(os.path.join(os.getcwd(), 'src', 'pytfmbs'))

from pt5_codec import pack_pt5
import pytfmbs

def prepare_packed_buffer(trits):
    """
    Packs a list of trits into a buffer of 32-bit words for the Fabric loader.
    Each 32-bit word contains 15 trits (3 bytes of PT-5).
    """
    # Pad trits to multiple of 15
    while len(trits) % 15 != 0:
        trits.append(0)

    words = []
    for i in range(0, len(trits), 15):
        # Pack 15 trits into 3 bytes
        b0 = pack_pt5(trits[i:i+5])
        b1 = pack_pt5(trits[i+5:i+10])
        b2 = pack_pt5(trits[i+10:i+15])

        # Combine into a 24-bit value inside a 32-bit word
        word = b0 | (b1 << 8) | (b2 << 16)
        words.append(word)

    return struct.pack(f"<{len(words)}I", *words)

def main():
    print("🚀 Ternary Fabric: End-to-End Inference Demo")

    # 1. Initialize Fabric
    try:
        fabric = pytfmbs.Fabric()
    except Exception as e:
        print(f"❌ Failed to initialize Fabric: {e}")
        return

    # 2. Prepare Mock Data (150 trits)
    # Weights: Alternating [1, 0, -1]
    weights = ([1, 0, -1] * 50)
    # Inputs: All 1s
    inputs = ([1] * 150)

    # Expected result for Lane 0 (if all 150 trits are processed):
    # Sum of (w[i] * 1) = sum(w[i]) = 50 * (1 + 0 + -1) = 0
    # Let's change inputs to match weights to get a non-zero result
    inputs = weights.copy()
    # Expected result: 50 * (1*1 + 0*0 + -1*-1) = 50 * (1 + 0 + 1) = 100

    print("📦 Packing and Loading data...")
    w_buf = prepare_packed_buffer(weights)
    i_buf = prepare_packed_buffer(inputs)

    # Load into SRAM
    # Weights at 0x1000, Inputs at 0x2000
    fabric.load(0x1000, w_buf)
    fabric.load(0x2000, i_buf)

    # 3. Configure and Run
    # Depth = 150 trits / 15 trits-per-word = 10 words (but the engine works on trits)
    # In our TPE architecture, each lane processes one trit per cycle.
    # Total depth is number of cycles.
    # With 15 lanes, each word (24 bits) provides 5 trits per lane? No.
    # Let's check hardware: 3 bytes = 15 trits. LANES = 15.
    # So each word provides 1 trit for each of the 15 lanes.
    # Thus, depth = 150 / 15 = 10.

    depth = 10
    print(f"📡 Executing T-MAC kernel (Depth={depth})...")
    fabric.run(0x0, depth, 15, 1, 1) # base_addr=0 (ignored for SRAM demo), lanes=15, stride=1, kernel=1 (DOT)

    # 4. Read Results
    results = fabric.results()
    print(f"✅ Execution Finished.")
    print(f"📊 Results per Lane: {results}")

    # In mock mode, the results won't be calculated unless we implemented it in core.c
    # But we can see that the flow works.
    if results[0] != 0:
        print(f"🌟 Lane 0 Result: {results[0]}")
    else:
        print("ℹ️  Note: Results are 0 in basic Mock Mode (Hardware execution needed for math)")

if __name__ == "__main__":
    main()
