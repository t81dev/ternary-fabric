# 📄 Phase 14 — GGUF Model Optimizations

Phase 14 focuses on deep integration with the GGUF file format, commonly used by `llama.cpp` and the wider open-source LLM community. This allows TFMBS to act as a transparent acceleration layer for existing models without complex conversion pipelines.

## 🚀 Key Features

### 1. Native GGUF Parsing
The `pytfmbs` library now includes a minimal GGUF reader capable of extracting tensors and metadata from GGUF v2 and v3 files.

### 2. Dequantization Kernels
Support for standard GGML quantization types has been added. Currently supported types:
*   `GGML_TYPE_F32` (Float32)
*   `GGML_TYPE_Q4_0` (4-bit symmetric quantization)

These kernels allow the Fabric to ingest standard quantized weights and transparently convert them to the resident ternary format.

### 3. Direct Loading API
A new `load_gguf_tensor` utility simplifies the process of moving weights from a GGUF file directly into the Fabric's SRAM.

```python
import pytfmbs

fabric = pytfmbs.Fabric()

# Load a Q4_0 quantized tensor from a GGUF file
# This dequantizes to float, then quantizes to ternary, then packs to PT-5
pytfmbs.load_gguf_tensor(
    fabric,
    "llama-7b-q4_0.gguf",
    "blk.0.attn_q.weight",
    address=0x1000
)
```

## 🛠️ Implementation Details

### Dequantization to Ternary
When loading a quantized GGUF tensor (e.g., Q4_0), `pytfmbs` performs the following steps:
1.  **Block Dequantization**: Reconstructs floating-point values from the block-scaled 4-bit nibbles.
2.  **Ternary Quantization**: Uses a threshold-based Ternary Weight Network (TWN) approach to map floats to {-1, 0, 1}.
3.  **PT-5 Packing**: Packs the resulting trits into the hardware-native PT-5 format (5 trits per byte).
4.  **DMA Transfer**: Loads the packed data into the specified SRAM bank.

## 📊 Performance Benefits

*   **Memory Efficiency**: Weights remain in their compact GGUF format on disk and are only expanded to ternary in-memory during the loading phase.
*   **Ease of Use**: No need for separate quantization scripts; standard GGUF files "just work".
*   **Resident Speed**: Once loaded, the weights benefit from the Fabric's Zero-Skip and parallel SIMD execution.
