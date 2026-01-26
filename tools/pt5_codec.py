def pack_pt5(trits):
    """
    Packs a list of 5 balanced trits into a single 8-bit byte.
    Input: List of trits like [-1, 0, 1, 1, -1]
    """
    if len(trits) != 5:
        raise ValueError("PT-5 packing requires exactly 5 trits.")

    byte_val = 0
    for i, trit in enumerate(trits):
        # Shift balanced (-1, 0, 1) to (0, 1, 2)
        unsigned_trit = trit + 1 
        # Apply power-of-3 weighting
        byte_val += unsigned_trit * (3 ** i)
        
    return byte_val

def unpack_pt5(byte_val):
    """
    Unpacks an 8-bit byte into 5 balanced trits.
    """
    trits = []
    temp = byte_val
    for _ in range(5):
        unsigned_trit = temp % 3
        trits.append(unsigned_trit - 1) # Shift back to balanced
        temp //= 3
    return trits

# Example Usage
sample_trits = [-1, 0, 1, 1, -1]
encoded = pack_pt5(sample_trits)
decoded = unpack_pt5(encoded)

print(f"Original: {sample_trits}")
print(f"Encoded (PT-5): {encoded} (0x{encoded:02X})")
print(f"Decoded:  {decoded}")