import torch
import pytfmbs

def test_torch_not_shadowed():
    # Verify that 'torch' is the real PyTorch and not our integration module
    assert hasattr(torch, "Tensor"), "torch module is shadowed or not correctly installed"
    assert hasattr(torch, "nn"), "torch.nn is missing, likely shadowed"

    # Verify that we can still access our integration
    from pytfmbs import torch_integration
    assert hasattr(torch_integration, "TFMBSLinear"), "pytfmbs.torch_integration is broken"

def test_pytfmbs_package_structure():
    # Verify the package structure is correct
    assert hasattr(pytfmbs, "Fabric"), "pytfmbs.Fabric is missing"
    assert hasattr(pytfmbs, "TFMBSLinear"), "pytfmbs.TFMBSLinear is missing"
    assert hasattr(pytfmbs, "torch_integration"), "pytfmbs.torch_integration is missing"
