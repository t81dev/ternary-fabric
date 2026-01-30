# 12: PyTorch Framework Integration

The `pytfmbs` package includes a high-level integration with PyTorch, allowing Ternary Fabric acceleration to be used transparently within standard neural network models.

## 1. The `TFMBSLinear` Module

`TFMBSLinear` is a drop-in replacement for `torch.nn.Linear`. It manages the weight quantization, memory residency, and hardware offloading automatically.

### Usage
```python
import torch
from pytfmbs import TFMBSLinear

# Define a model using TFMBS-accelerated layers
model = torch.nn.Sequential(
    TFMBSLinear(784, 60), # 60 lanes = 4 tiles
    torch.nn.ReLU(),
    TFMBSLinear(60, 10)
)

# Inference (Weights are quantized and loaded on first pass)
input_data = torch.randn(1, 784)
output = model(input_data)
```

## 2. Residency Management

The `TFMBSLinear` layer keeps weights in standard PyTorch format for easy saving/loading but moves them to the Fabric SRAM upon the first forward pass.

*   **Weight Address:** Each layer can be assigned a specific `weight_addr` in the Fabric's SRAM Bank A to prevent overlaps.
*   **Automatic Quantization:** Weights are currently quantized using a simple symmetric sign function during the loading process.

## 3. Performance & Zero-Skip

The PyTorch integration fully supports the hardware's **Zero-Skip** feature. By using `TFMBSLinear`, the model automatically benefits from power and cycle savings when either weights or activations are zero.

### Profiling Integration
You can still use the standard `Fabric.profile()` methods to extract telemetry after running a PyTorch model.

```python
import pytfmbs
fabric = pytfmbs.Fabric()
# ... run model ...
stats = fabric.profile_detailed()
print(f"Cycles: {stats['cycles']}")
```

## 4. Advanced: `TFMBSLinearFunction`

For custom integration, you can use `TFMBSLinearFunction`, which is a `torch.autograd.Function`. This allows the Fabric to participate in the autograd graph, supporting potential future training scenarios or custom hybrid execution paths.

## 5. Technical Details: PT-5 Packing

The integration handles the complex PT-5 packing required for both weights and inputs.
- **Input Packing:** Activations are dynamically packed and broadcast to the active tiles' Bank B SRAM.
- **Weight Packing:** Weights are pre-packed into "Lane-Major" format to match the Vector Engine's parallel access patterns.

## 6. Torch/ONNX Frontend Samples

To keep the compiler-facing path exercised while Phase 23 hardware verification is pending, run `tools/torch_to_tfmbs.py` to emit MLIR that mirrors the converter pipeline.
`TFMBSLinear` now exposes `telemetry_hint`, which returns a dictionary (layer name, sparsity, tile mask) that matches what the helper script embeds in `tfmbs.gemv`. Supply `name` so the runtime and compiler agree on the layer identifier and telemetry attributes.
1. `torch_to_tfmbs.py` exports a small PyTorch sequential model to ONNX, scans for `MatMul`/`Gemm` nodes, computes tile masks, and writes `tests/mlir/torch_tfmbs.mlir` with telemetry metadata (e.g., `{telemetry = {layer = "...", sparsity = 0.0000}}`).
2. Feed the resulting MLIR into `mlir-opt` with `--load-dialect-plugin=build/libtfmbs_plugin.dylib --pass-pipeline="builtin.module(tfmbs-to-linalg)"` or the regression script `tests/mlir/run_tfmbs_to_linalg.py` to guarantee the `tfmbs` dialect lowers into `linalg.matmul` while telemetry attrs stay attached for the Adaptive Runtime Agent to consume later.
3. This flow lets the Torch/ONNX front end generate `tfmbs.gemv` ops with hardware hints (tile masks, telemetry) before we wire the same attributes back into the PyTorch stack (`Fabric.profile()`, `TFMBSLinear`) or ONNX exporters.
