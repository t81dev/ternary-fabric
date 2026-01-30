import torch
import numpy as np
import pytfmbs
from pytfmbs import TFMBSLinear, TFMBSSequential, Fabric
from pytfmbs.adaptive_agent import AdaptiveRuntimeAgent
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


def test_sequential_fusion_metadata():
    fabric = Fabric()
    l1 = TFMBSLinear(8, 8, fabric=fabric, name="fusion_a")
    l2 = TFMBSLinear(8, 4, fabric=fabric, name="fusion_b")
    seq = TFMBSSequential(l1, l2)
    hint = seq.telemetry_hint()
    assert hint["fusion_order"] == ["fusion_a", "fusion_b"]
    assert hint["fusion_sparsity"] >= 0.0
    assert "tile_mask" in hint


def test_adaptive_agent_consumes_fusion_hints():
    agent = AdaptiveRuntimeAgent()
    compile_hint = {
        "layer": "node_linear",
        "sparsity": 0.0,
        "tile_mask": 15,
        "fusion_order": ["node_linear", "node_linear_1"],
        "fusion_sparsity": 0.0,
    }
    fabric = Fabric()
    runtime_layer = TFMBSLinear(8, 8, fabric=fabric, name="runtime_layer")
    runtime_hint = runtime_layer.telemetry_hint
    agent_hint_compile = agent.consume(compile_hint)
    agent_hint_runtime = agent.consume(runtime_hint)
    assert agent_hint_compile["fusion_order"] == compile_hint["fusion_order"]
    assert agent_hint_runtime["fusion_order"] == runtime_hint["fusion_order"]


def test_linear_telemetry_hint():
    fabric = Fabric()
    layer = TFMBSLinear(4, 8, fabric=fabric, name="telemetry_test")
    hint = layer.telemetry_hint
    assert hint["layer"] == "telemetry_test"
    assert 0.0 <= hint["sparsity"] <= 1.0
    assert hint["tile_mask"] == layer.tile_mask

if __name__ == "__main__":
    pytest.main([__file__])
