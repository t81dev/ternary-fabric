import numpy as np

def pack_pt5(trits):
    """Packs 5 trits into a 8-bit byte (approx 1.58 bits/trit)"""
    val = 0
    for i, t in enumerate(trits):
        val += (t + 1) * (3**i) # Shift -1,0,1 to 0,1,2
    return val

def generate_test_vectors(num_blocks=10):
    trits_out = []
    hex_out = []
    
    for _ in range(num_blocks):
        # Generate 5 trits (-1, 0, 1)
        block = np.random.randint(-1, 2, 5)
        trits_out.extend(block)
        hex_out.append(f"{pack_pt5(block):02x}")
    
    with open("tests/input_vectors.hex", "w") as f:
        f.write("\n".join(hex_out))
        
    with open("tests/expected_trits.txt", "w") as f:
        f.write("\n".join(map(str, trits_out)))

    print(f"✅ Generated {num_blocks*5} trits for verification.")

if __name__ == "__main__":
    generate_test_vectors()