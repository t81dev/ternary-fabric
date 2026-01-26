import sys
import argparse
import struct

def pack_pt5(trits):
    """
    Packs a list of trits (-1, 0, 1) into PT-5 format.
    Formula: Byte = sum(t_i * 3^i) where t_i is mapped to (0, 1, 2)
    """
    packed_bytes = []
    # Pad trits to a multiple of 5
    while len(trits) % 5 != 0:
        trits.append(0)
    
    for i in range(0, len(trits), 5):
        chunk = trits[i:i+5]
        byte_val = 0
        for power, trit in enumerate(chunk):
            # Map -1 -> 2, 0 -> 0, 1 -> 1 (Balanced Ternary Mapping)
            mapped_val = 2 if trit == -1 else (0 if trit == 0 else 1)
            byte_val += mapped_val * (3 ** power)
        packed_bytes.append(byte_val)
    return bytes(packed_bytes)

def generate_tfd(args, data_len):
    """
    Creates a binary blob representing the tfmbs_tfd_t struct.
    """
    # packing_fmt = 1 (PT-5), version = 1
    # Struct format: Q (uint64), I (uint32), H (uint16), H (uint16), I (uint32), I (uint32), I (uint32), B (uint8), 7x (padding)
    # Fields: base_addr, frame_len, packing_fmt, lane_count, lane_stride, flags, exec_hints, version

    # If exec_hints not provided, use kernel as base
    hints = args.exec_hints if args.exec_hints != 0 else args.kernel

    return struct.pack("<QIHHI I I B 7x", 
        args.base_addr, data_len, 1, args.lanes, 
        args.stride, 0, hints, 1)

def main():
    parser = argparse.ArgumentParser(description="Ternary-CLI: Prepare data for the Fabric")
    parser.add_argument("input", help="Input file (.txt or .csv)")
    parser.add_argument("--base_addr", type=int, default=0x40000000)
    parser.add_argument("--lanes", type=int, default=15)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--kernel", type=int, default=1, help="Kernel ID (1: DOT, 3: MUL, 6: TGEMM)")
    parser.add_argument("--exec_hints", type=int, default=0, help="Full 32-bit exec_hints value")
    args = parser.parse_args()

    # Read trits (expects space-separated or comma-separated -1, 0, 1)
    with open(args.input, 'r') as f:
        content = f.read().replace(',', ' ')
        trits = [int(t) for t in content.split()]

    packed_data = pack_pt5(trits)
    tfd_header = generate_tfd(args, len(trits))

    with open(args.input + ".tfrm", "wb") as f:
        f.write(packed_data)
    
    with open(args.input + ".tfd", "wb") as f:
        f.write(tfd_header)

    print(f"Successfully generated {args.input}.tfrm ({len(packed_data)} bytes)")
    hints = args.exec_hints if args.exec_hints != 0 else args.kernel
    print(f"Header generated with exec_hints: {hex(hints)}")

if __name__ == "__main__":
    main()