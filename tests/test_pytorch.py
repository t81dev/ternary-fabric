import torch
import numpy as np
import pytfmbs
from pytfmbs import TFMBSLinear, pack_pt5_numpy
import pytest

def test_pt5_packing():
    trits = np.array([1, 0, -1, 1, 0], dtype=np.int8)
    packed = pack_pt5_numpy(trits)
    assert len(packed) == 1
    # (1+1)*3^0 + (0+1)*3^1 + (-1+1)*3^2 + (1+1)*3^3 + (0+1)*3^4
    # 2*1 + 1*3 + 0*9 + 2*27 + 1*81 = 2 + 3 + 0 + 54 + 81 = 140
    assert packed[0] == 140

def test_tfmbs_linear_inference():
    fabric = pytfmbs.Fabric()
    # Use a small layer that fits in 1 tile
    in_features = 16
    out_features = 5
    layer = TFMBSLinear(in_features, out_features, fabric=fabric)

    # Random input
    x = torch.randn(1, in_features)

    # Run forward
    y = layer(x)

    assert y.shape == (1, out_features)
    assert layer.resident == True

    # Check that we can run it again
    y2 = layer(x)
    assert torch.equal(y, y2)

def test_tfmbs_linear_multi_tile():
    fabric = pytfmbs.Fabric()
    # 30 features should use 2 tiles
    layer = TFMBSLinear(10, 30, fabric=fabric)
    assert layer.tile_mask == 0x3 # 11 binary

    x = torch.randn(2, 10)
    y = layer(x)
    assert y.shape == (2, 30)

if __name__ == "__main__":
    pytest.main([__file__])
