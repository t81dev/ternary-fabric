import struct
import numpy as np
import os

def create_mock_gguf(filename, tensor_name, data):
    """
    Creates a minimal GGUF v3 file with one Q4_0 quantized tensor.
    """
    # Q4_0 block size = 32
    # Each block: 2 bytes (float16 delta) + 16 bytes (32 x 4-bit nibbles)

    n = data.size
    if n % 32 != 0:
        padding = 32 - (n % 32)
        data = np.concatenate([data, np.zeros(padding, dtype=np.float32)])
        n = data.size

    blocks = []
    for i in range(0, n, 32):
        chunk = data[i:i+32]
        amax = np.max(np.abs(chunk))
        delta = amax / 8.0
        if delta == 0: delta = 1.0

        # Quantize to -8 to 7
        # GGML_Q4_0: x = d * q, where q is in [-8, 7]
        # In GGUF/GGML implementation of Q4_0, they actually store (q + 8) to make it 0-15
        qs = np.round(chunk / delta).astype(int)
        qs = np.clip(qs, -8, 7)

        packed_qs = bytearray(16)
        for j in range(16):
            v0 = (qs[j]) & 0x0F
            v1 = (qs[j+16]) & 0x0F
            packed_qs[j] = v0 | (v1 << 4)

        d_bytes = struct.pack("<e", delta) # float16
        blocks.append(d_bytes + packed_qs)

    tensor_data = b"".join(blocks)

    name_bytes = tensor_name.encode('utf-8')

    # Calculate alignment padding
    # GGUF header + KV + Tensor Info
    header_size = 4 + 4 + 8 + 8
    tensor_info_size = 8 + len(name_bytes) + 4 + 8*1 + 4 + 8

    total_before_data = header_size + tensor_info_size
    padding_len = (32 - (total_before_data % 32)) % 32

    with open(filename, "wb") as f:
        # Header
        f.write(b"GGUF")
        f.write(struct.pack("<I", 3)) # Version
        f.write(struct.pack("<Q", 1)) # Tensor count
        f.write(struct.pack("<Q", 0)) # KV count

        # Tensor Info
        f.write(struct.pack("<Q", len(name_bytes)))
        f.write(name_bytes)
        f.write(struct.pack("<I", 1)) # n_dims
        f.write(struct.pack("<Q", n)) # dims[0]
        f.write(struct.pack("<I", 2)) # type = Q4_0 (GGML_TYPE_Q4_0)
        f.write(struct.pack("<Q", 0)) # offset (placeholder)

        f.write(b"\x00" * padding_len)

        data_offset = f.tell()
        # Go back and update offset if needed, but we used 0 for simplicity if it's relative to data start.
        # Actually in GGUF v3, offset is relative to the start of the data binary blob.

        f.write(tensor_data)

if __name__ == "__main__":
    import sys
    fname = sys.argv[1] if len(sys.argv) > 1 else "test.gguf"
    data = np.linspace(-1, 1, 64).astype(np.float32)
    create_mock_gguf(fname, "weights", data)
    print(f"✅ Created mock GGUF: {fname}")
