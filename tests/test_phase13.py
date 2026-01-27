import torch
import numpy as np
import pytfmbs
from pytfmbs import TFMBSLinear, TFMBSSequential, Fabric
import pytest
import time

def test_async_fabric():
    fabric = Fabric()
    tfd = {
        "base_addr": 0x1000,
        "depth": 100,
        "exec_hints": 0x01, # DOT
    }

    # Test submit
    fabric.submit(tfd)

    # In mock, it might be done instantly or not, but we can check is_done
    # Since mock is currently sequential inside submit, it will be done.
    assert fabric.is_done()

    # Test wait
    profile = fabric.wait()
    assert "cycles" in profile
    assert profile["cycles"] >= 100

def test_prefetching():
    fabric = Fabric()
    layer = TFMBSLinear(16, 5, fabric=fabric)

    assert not layer.resident
    layer.prefetch()
    assert layer.resident

    # Check that forward doesn't crash after prefetch
    x = torch.randn(1, 16)
    y = layer(x)
    assert y.shape == (1, 5)

def test_sequential_pipelining():
    fabric = Fabric()
    # Create two layers at different addresses to allow prefetching without collision
    l1 = TFMBSLinear(16, 16, fabric=fabric, weight_addr=0x1000)
    l2 = TFMBSLinear(16, 5, fabric=fabric, weight_addr=0x20000)

    model = TFMBSSequential(l1, l2)

    x = torch.randn(2, 16)
    y = model(x)

    assert y.shape == (2, 5)
    assert l1.resident
    assert l2.resident

def test_run_batch():
    fabric = Fabric()
    tfds = [
        {"base_addr": 0x1000, "depth": 10},
        {"base_addr": 0x1000, "depth": 20},
        {"base_addr": 0x1000, "depth": 30},
    ]
    profile = fabric.run_batch(tfds)
    assert "cycles" in profile
    # Each run in batch should have added to total cycles in mock?
    # Actually, the mock resets regs[8] at the start of each submit.
    # So the last one wins.
    assert profile["cycles"] == 30

if __name__ == "__main__":
    pytest.main([__file__])
