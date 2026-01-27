import struct
import numpy as np
import os
from .constants import SRAM_BANK_A_OFFSET, SRAM_TILE_STRIDE
from .torch import pack_gemv_weights

# GGML types
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q4_0 = 2

def dequantize_q4_0(data, n):
    """
    Dequantizes a GGML Q4_0 buffer.
    Q4_0 block: 2 bytes (float16 delta) + 16 bytes (32 x 4-bit signed nibbles)
    """
    blocks = n // 32
    weights = np.zeros(n, dtype=np.float32)

    for i in range(blocks):
        # Block structure: [delta (2 bytes)][qs (16 bytes)]
        block_start = i * 18
        if block_start + 18 > len(data):
            break

        block_data = data[block_start : block_start + 18]
        delta = struct.unpack("<e", block_data[0:2])[0]
        qs = block_data[2:18]

        for j in range(16):
            v0 = qs[j] & 0x0F
            v1 = (qs[j] >> 4) & 0x0F

            # Map 0-15 back to -8 to 7
            q0 = v0 - 8
            q1 = v1 - 8

            weights[i*32 + j] = q0 * delta
            weights[i*32 + j + 16] = q1 * delta

    return weights

class GGUFReader:
    """
    A minimal GGUF reader for extracting model weights.
    Supports GGUF v2 and v3.
    """
    def __init__(self, filename):
        self.filename = filename
        self.tensors = {}
        self.kv = {}
        self._read_header()

    def _read_header(self):
        with open(self.filename, "rb") as f:
            magic = f.read(4)
            if magic != b"GGUF":
                raise ValueError(f"Not a GGUF file: {magic}")

            version = struct.unpack("<I", f.read(4))[0]
            if version < 2:
                raise ValueError(f"Unsupported GGUF version: {version}")

            tensor_count = struct.unpack("<Q", f.read(8))[0]
            kv_count = struct.unpack("<Q", f.read(8))[0]

            # Read KV pairs
            for _ in range(kv_count):
                key_len = struct.unpack("<Q", f.read(8))[0]
                key = f.read(key_len).decode('utf-8')
                val_type = struct.unpack("<I", f.read(4))[0]
                val = self._read_kv_value(f, val_type)
                self.kv[key] = val

            # Read Tensor infos
            for _ in range(tensor_count):
                name_len = struct.unpack("<Q", f.read(8))[0]
                name = f.read(name_len).decode('utf-8')
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims = []
                for _ in range(n_dims):
                    dims.append(struct.unpack("<Q", f.read(8))[0])
                dtype = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]

                self.tensors[name] = {
                    "dims": dims,
                    "type": dtype,
                    "offset": offset
                }

            # Alignment for data start
            alignment = self.kv.get("general.alignment", 32)
            self.data_start = (f.tell() + alignment - 1) // alignment * alignment

    def _read_kv_value(self, f, val_type):
        """Reads a metadata value based on GGML_METADATA_VALUE_TYPE."""
        if val_type == 0:   # UINT8
            return struct.unpack("<B", f.read(1))[0]
        elif val_type == 1: # INT8
            return struct.unpack("<b", f.read(1))[0]
        elif val_type == 2: # UINT16
            return struct.unpack("<H", f.read(2))[0]
        elif val_type == 3: # INT16
            return struct.unpack("<h", f.read(2))[0]
        elif val_type == 4: # UINT32
            return struct.unpack("<I", f.read(4))[0]
        elif val_type == 5: # INT32
            return struct.unpack("<i", f.read(4))[0]
        elif val_type == 6: # FLOAT32
            return struct.unpack("<f", f.read(4))[0]
        elif val_type == 7: # BOOL
            return struct.unpack("<?", f.read(1))[0]
        elif val_type == 8: # STRING
            slen = struct.unpack("<Q", f.read(8))[0]
            return f.read(slen).decode('utf-8')
        elif val_type == 9: # ARRAY
            item_type = struct.unpack("<I", f.read(4))[0]
            item_count = struct.unpack("<Q", f.read(8))[0]
            return [self._read_kv_value(f, item_type) for _ in range(item_count)]
        elif val_type == 10: # UINT64
            return struct.unpack("<Q", f.read(8))[0]
        elif val_type == 11: # INT64
            return struct.unpack("<q", f.read(8))[0]
        elif val_type == 12: # FLOAT64
            return struct.unpack("<d", f.read(8))[0]
        else:
            raise ValueError(f"Unknown GGUF metadata type: {val_type}")

    def load_tensor(self, name):
        info = self.tensors.get(name)
        if not info:
            raise KeyError(f"Tensor {name} not found")

        with open(self.filename, "rb") as f:
            f.seek(self.data_start + info["offset"])

            n_elements = 1
            for d in info["dims"]:
                n_elements *= d

            if info["type"] == GGML_TYPE_Q4_0:
                data_size = (n_elements // 32) * 18
                data = f.read(data_size)
                return dequantize_q4_0(data, n_elements)
            elif info["type"] == GGML_TYPE_F32:
                data = f.read(n_elements * 4)
                return np.frombuffer(data, dtype=np.float32)
            else:
                raise NotImplementedError(f"GGML type {info['type']} not supported")

def load_gguf_tensor(fabric, filename, tensor_name, address=SRAM_BANK_A_OFFSET):
    """
    Loads a tensor from a GGUF file, quantizes it to ternary,
    and transfers it to the Fabric.
    """
    reader = GGUFReader(filename)
    weights = reader.load_tensor(tensor_name)

    # Quantize to ternary using thresholding
    delta = 0.7 * np.mean(np.abs(weights))
    ternary = np.zeros_like(weights)
    ternary[weights > delta] = 1
    ternary[weights < -delta] = -1

    # Determine dimensions
    dims = reader.tensors[tensor_name]["dims"]
    if len(dims) == 2:
        # GGUF dimensions are [width, height]
        in_f, out_f = dims[0], dims[1]
        ternary = ternary.reshape(out_f, in_f)
    else:
        out_f = 1
        in_f = ternary.size
        ternary = ternary.reshape(out_f, in_f)

    # Pack and load
    tile_data = pack_gemv_weights(ternary)
    for t, data in enumerate(tile_data):
        if data is not None:
            fabric.load(address + t * SRAM_TILE_STRIDE, data)

    print(f"[TFMBS] Loaded GGUF tensor '{tensor_name}' ({out_f}x{in_f}) to 0x{address:x}")
    return ternary.shape
