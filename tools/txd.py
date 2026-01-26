import sys
import argparse

def unpack_pt5_byte(b):
    """Decodes a single byte into its 5 constituent trits."""
    trits = []
    val = b
    for _ in range(5):
        trit_val = val % 3
        # Map 0 -> '0', 1 -> '+', 2 -> '-' (Internal PT-5 mapping)
        if trit_val == 0: trits.append('0')
        elif trit_val == 1: trits.append('+')
        else: trits.append('-')
        val //= 3
    return trits

def main():
    parser = argparse.ArgumentParser(description="TXD: Ternary Hex Dump")
    parser.add_argument("file", help="The .tfrm file to inspect")
    parser.add_argument("-w", "--width", type=int, default=10, help="Bytes per row")
    args = parser.parse_args()

    try:
        with open(args.file, "rb") as f:
            offset = 0
            while True:
                chunk = f.read(args.width)
                if not chunk:
                    break
                
                # Format the offset
                hex_offset = f"{offset:08x}"
                
                # Format the ternary representation
                all_trits = []
                for b in chunk:
                    all_trits.extend(unpack_pt5_byte(b))
                
                trit_str = "".join(all_trits)
                
                # Format the raw hex for comparison
                hex_str = " ".join(f"{b:02x}" for b in chunk)
                
                print(f"{hex_offset} | {trit_str:<{args.width*5}} | {hex_str}")
                offset += args.width
                
    except FileNotFoundError:
        print(f"Error: File {args.file} not found.")

if __name__ == "__main__":
    main()