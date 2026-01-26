#!/usr/bin/env python3
import sys
import argparse

def pack_pt5(trits):
    """Packs 5 balanced trits (-1, 0, 1) into one byte (0-242)."""
    byte_val = 0
    for i, trit in enumerate(trits):
        byte_val += (trit + 1) * (3 ** i)
    return byte_val

def unpack_pt5(byte_val):
    """Unpacks one byte into 5 balanced trits."""
    trits = []
    for _ in range(5):
        unsigned_trit = byte_val % 3
        trits.append(unsigned_trit - 1)
        byte_val //= 3
    return trits

def main():
    parser = argparse.ArgumentParser(description="Ternary Fabric CLI - PT-5 Codec")
    subparsers = parser.add_subparsers(dest="command")

    # Pack command: "tf-cli pack +0-0+"
    pack_parser = subparsers.add_parser("pack", help="Convert trit string to binary file")
    pack_parser.add_argument("trits", type=str, help="Trit string (e.g., '+0-++')")
    pack_parser.add_argument("-o", "--output", type=str, default="frame.bin", help="Output binary file")

    # Unpack command: "tf-cli unpack frame.bin"
    unpack_parser = subparsers.add_parser("unpack", help="Convert binary file to trit string")
    unpack_parser.add_argument("file", type=str, help="Binary file to read")

    args = parser.parse_args()

    if args.command == "pack":
        # Convert chars to ints: '+' -> 1, '0' -> 0, '-' -> -1
        mapping = {'+': 1, '0': 0, '-': -1}
        trit_list = [mapping[c] for c in args.trits if c in mapping]
        
        # Pad to multiple of 5
        while len(trit_list) % 5 != 0:
            trit_list.append(0)

        with open(args.output, "wb") as f:
            for i in range(0, len(trit_list), 5):
                f.write(bytes([pack_pt5(trit_list[i:i+5])]))
        print(f"Successfully packed {len(trit_list)} trits into {args.output}")

    elif args.command == "unpack":
        mapping_rev = {1: '+', 0: '0', -1: '-'}
        with open(args.file, "rb") as f:
            data = f.read()
            out_str = ""
            for byte in data:
                trits = unpack_pt5(byte)
                out_str += "".join([mapping_rev[t] for t in trits])
            print(out_str)

if __name__ == "__main__":
    main()