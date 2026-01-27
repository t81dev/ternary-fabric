import pytest
import numpy as np
import pytfmbs
import os

@pytest.fixture
def fabric():
    return pytfmbs.Fabric()

def pack_pt5(trits):
    """Simple PT-5 packer for testing, padded to 4 bytes."""
    packed = []
    for i in range(0, len(trits), 5):
        chunk = trits[i:i+5]
        val = 0
        for j, t in enumerate(chunk):
            ut = 1 if t == 1 else 2 if t == -1 else 0
            val += ut * (3**j)
        packed.append(val)
    while len(packed) < 4:
        packed.append(0)
    return bytes(packed)

def test_multi_tile_independent(fabric):
    # Load different data into 4 tiles
    all_weights = []
    all_inputs = []
    for t in range(4):
        weights = np.random.choice([-1, 0, 1], (10, 15))
        inputs = np.random.choice([-1, 0, 1], (10, 15))
        all_weights.append(weights)
        all_inputs.append(inputs)

        w_packed = b"".join([pack_pt5(weights[d]) for d in range(10)])
        i_packed = b"".join([pack_pt5(inputs[d]) for d in range(10)])

        fabric.load(0x1000 + t * 0x2000, w_packed)
        fabric.load(0x2000 + t * 0x2000, i_packed)

    # Run with all tiles active
    tfd = {
        "depth": 10,
        "lane_count": 15,
        "exec_hints": 0x06, # TGEMM
        "tile_mask": 0xF
    }
    fabric.run(tfd)

    # Check results per tile
    for t in range(4):
        results = fabric.results(t)
        expected = np.sum(all_weights[t] * all_inputs[t], axis=0)
        for l in range(15):
            assert results[l] == expected[l]

def test_multi_tile_broadcast(fabric):
    # Broadcast weights to all tiles
    weights = np.random.choice([-1, 0, 1], (10, 15))
    w_packed = b"".join([pack_pt5(weights[d]) for d in range(10)])
    fabric.load(0x9000, w_packed)

    # Different inputs for each tile
    all_inputs = []
    for t in range(4):
        inputs = np.random.choice([-1, 0, 1], (10, 15))
        all_inputs.append(inputs)
        i_packed = b"".join([pack_pt5(inputs[d]) for d in range(10)])
        fabric.load(0x2000 + t * 0x2000, i_packed)

    fabric.run({"depth": 10, "lane_count": 15, "exec_hints": 0x06, "tile_mask": 0xF})

    for t in range(4):
        results = fabric.results(t)
        expected = np.sum(weights * all_inputs[t], axis=0)
        for l in range(15):
            assert results[l] == expected[l]

def test_tile_masking(fabric):
    # Load same data into all tiles
    weights = np.ones((10, 15), dtype=int)
    inputs = np.ones((10, 15), dtype=int)
    w_packed = b"".join([pack_pt5(weights[d]) for d in range(10)])
    i_packed = b"".join([pack_pt5(inputs[d]) for d in range(10)])

    for t in range(4):
        fabric.load(0x1000 + t * 0x2000, w_packed)
        fabric.load(0x2000 + t * 0x2000, i_packed)

    # Run only tiles 0 and 2
    fabric.run({"depth": 10, "lane_count": 15, "exec_hints": 0x06, "tile_mask": 0x5})

    # Results for 0 and 2 should be 10, others 0
    assert all(r == 10 for r in fabric.results(0))
    assert all(r == 0 for r in fabric.results(1))
    assert all(r == 10 for r in fabric.results(2))
    assert all(r == 0 for r in fabric.results(3))

def test_multi_tile_profiling(fabric):
    # 50% sparsity in Tile 0, 100% in Tile 1
    w0 = np.random.choice([0, 1], (10, 15))
    i0 = np.ones((10, 15), dtype=int)

    w1 = np.zeros((10, 15), dtype=int)
    i1 = np.ones((10, 15), dtype=int)

    fabric.load(0x1000, b"".join([pack_pt5(w0[d]) for d in range(10)]))
    fabric.load(0x2000, b"".join([pack_pt5(i0[d]) for d in range(10)]))
    fabric.load(0x3000, b"".join([pack_pt5(w1[d]) for d in range(10)]))
    fabric.load(0x4000, b"".join([pack_pt5(i1[d]) for d in range(10)]))

    fabric.run({"depth": 10, "lane_count": 15, "exec_hints": 0x06 | (1 << 17), "tile_mask": 0x3})

    p0 = fabric.profile_tile(0)
    p1 = fabric.profile_tile(1)
    p_agg = fabric.profile()

    for l in range(15):
        expected_s0 = np.sum(w0[:, l] == 0)
        assert p0['skips'][l] == expected_s0
        assert p1['skips'][l] == 10 # All zeros
        assert p_agg['skips'][l] == expected_s0 + 10

    assert p_agg['cycles'] == 10
    # Total active lanes: 15 (Tile 0) + 15 (Tile 1) = 30 per cycle?
    # No, tile_mask=3 means 2 tiles active.
    # Actually mock: total_active_this_cycle includes all 15 lanes for both tiles.
    # 2 * 15 * 10 = 300?
    # Wait, my mock utilization_count_reg += total_active_this_cycle.
    # total_active_this_cycle counts (l < lane_count && (lane_mask & (1 << l))) for each active tile.
    # So 2 tiles * 15 lanes * 10 steps = 300.
    assert p_agg['utilization'] == 300
