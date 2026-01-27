import numpy as np
import pytfmbs
import pytest

def pack_pt5(trits):
    packed = []
    padding = (5 - (len(trits) % 5)) % 5
    trits_padded = np.append(trits, [0] * padding)
    for i in range(0, len(trits_padded), 5):
        chunk = trits_padded[i:i+5]
        val = 0
        for j, t in enumerate(chunk):
            ut = 1 if t == 1 else 2 if t == -1 else 0
            val += ut * (3**j)
        packed.append(val)
    while len(packed) % 4 != 0:
        packed.append(0)
    return bytes(packed)

@pytest.fixture
def fabric():
    return pytfmbs.Fabric()

def test_conv3d(fabric):
    depth = 5
    lanes = 15
    stride = 1
    conv_stride_val = 2 # (exec_hints >> 20) & 0x3 + 1 -> 1 + 1 = 2

    weights = np.random.choice([-1, 0, 1], (depth, lanes))
    inputs = np.random.choice([-1, 0, 1], (1024, lanes))

    w_packed = b"".join([pack_pt5(weights[d]) for d in range(depth)])
    i_packed = b"".join([pack_pt5(inputs[d]) for d in range(1024)])

    fabric.load(0x1000, w_packed)
    fabric.load(0x2000, i_packed)

    # TFMBS_KERNEL_CONV3D = 0x07
    # conv_stride bits (21:20) = 1 (means conv_stride=2)
    exec_hints = 0x07 | (1 << 20)

    tfd = {
        "base_addr": 0x1000,
        "frame_len": depth,
        "lane_count": lanes,
        "exec_hints": exec_hints,
        "tile_mask": 0x1
    }

    fabric.run(tfd)
    fabric_results = np.array(fabric.results(0))

    # Reference calculation
    expected = np.zeros(lanes, dtype=int)
    for d in range(depth):
        idx = d * stride * conv_stride_val * conv_stride_val
        expected += weights[d] * inputs[idx]

    assert np.array_equal(fabric_results, expected)

def test_lstm_with_bias_persistence(fabric):
    # Test that BIAS_EN prevents clearing the accumulator
    depth = 10
    lanes = 15

    weights1 = np.random.choice([-1, 0, 1], (depth, lanes))
    inputs1 = np.random.choice([-1, 0, 1], (depth, lanes))

    w1_packed = b"".join([pack_pt5(weights1[d]) for d in range(depth)])
    i1_packed = b"".join([pack_pt5(inputs1[d]) for d in range(depth)])

    fabric.load(0x1000, w1_packed)
    fabric.load(0x2000, i1_packed)

    # TFMBS_KERNEL_LSTM = 0x08
    tfd1 = {
        "base_addr": 0x1000,
        "frame_len": depth,
        "lane_count": lanes,
        "exec_hints": 0x08, # No BIAS_EN, should clear
        "tile_mask": 0x1
    }

    fabric.run(tfd1)
    res1 = np.array(fabric.results(0))

    expected1 = np.sum(weights1 * inputs1, axis=0)
    assert np.array_equal(res1, expected1)

    # Second run WITH BIAS_EN
    weights2 = np.random.choice([-1, 0, 1], (depth, lanes))
    inputs2 = np.random.choice([-1, 0, 1], (depth, lanes))

    w2_packed = b"".join([pack_pt5(weights2[d]) for d in range(depth)])
    i2_packed = b"".join([pack_pt5(inputs2[d]) for d in range(depth)])

    fabric.load(0x1000, w2_packed)
    fabric.load(0x2000, i2_packed)

    # TFMBS_HINT_BIAS_EN = 0x10000
    tfd2 = {
        "base_addr": 0x1000,
        "frame_len": depth,
        "lane_count": lanes,
        "exec_hints": 0x08 | 0x10000, # BIAS_EN set
        "tile_mask": 0x1
    }

    fabric.run(tfd2)
    res2 = np.array(fabric.results(0))

    expected2 = expected1 + np.sum(weights2 * inputs2, axis=0)
    assert np.array_equal(res2, expected2)

def test_attn_basic(fabric):
    # Basic check that ATTN kernel works (currently same as DOT in mock)
    depth = 8
    lanes = 15

    weights = np.random.choice([-1, 0, 1], (depth, lanes))
    inputs = np.random.choice([-1, 0, 1], (depth, lanes))

    w_packed = b"".join([pack_pt5(weights[d]) for d in range(depth)])
    i_packed = b"".join([pack_pt5(inputs[d]) for d in range(depth)])

    fabric.load(0x1000, w_packed)
    fabric.load(0x2000, i_packed)

    # TFMBS_KERNEL_ATTN = 0x09
    tfd = {
        "base_addr": 0x1000,
        "frame_len": depth,
        "lane_count": lanes,
        "exec_hints": 0x09,
        "tile_mask": 0x1
    }

    fabric.run(tfd)
    res = np.array(fabric.results(0))
    expected = np.sum(weights * inputs, axis=0)
    assert np.array_equal(res, expected)
