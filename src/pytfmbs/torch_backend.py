import torch
import torch._dynamo
from typing import List
from .torch_integration import TFMBSLinear

def tfmbs_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    """
    TFMBS backend for torch.compile.
    Replaces nn.Linear modules with TFMBS-accelerated versions.
    """
    print("[TFMBS-Backend] Optimizing graph for Ternary Fabric...")

    # Iterate over modules and replace Linear with TFMBSLinear
    for name, module in gm.named_modules():
        if isinstance(module, torch.nn.Linear):
            print(f"[TFMBS-Backend] Replacing layer: {name}")
            # Create TFMBSLinear with same dimensions
            tfmbs_mod = TFMBSLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                name=name
            )
            # Copy weights (will be quantized during first forward/prefetch)
            tfmbs_mod.weight.data = module.weight.data
            if module.bias is not None:
                tfmbs_mod.bias.data = module.bias.data

            # Replace module in the GraphModule
            # Note: This is a simplified replacement for a GraphModule
            # In complex cases, you'd use a Transformer
            parent_name, last_name = name.rsplit('.', 1) if '.' in name else ('', name)
            if parent_name:
                setattr(gm.get_submodule(parent_name), last_name, tfmbs_mod)
            else:
                setattr(gm, last_name, tfmbs_mod)

    gm.recompile()
    return gm.forward

def register():
    try:
        torch._dynamo.register_backend("tfmbs", tfmbs_backend)
        print("[TFMBS-Backend] 'tfmbs' backend registered successfully.")
    except Exception as e:
        print(f"[TFMBS-Backend] Registration failed: {e}")
