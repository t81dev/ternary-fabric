import numpy as np
import pytfmbs

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

def conv3d_reference(weights, inputs, stride, conv_stride):
    depth, lanes = weights.shape
    results = np.zeros(lanes, dtype=int)

    for d in range(depth):
        idx = d * stride * conv_stride * conv_stride
        if idx >= 1024:
            continue
        results += weights[d] * inputs[idx]

    return results

def main():
    print("--- Ternary Fabric: T-Conv3D Validation ---")
    fabric = pytfmbs.Fabric()

    depth = 10
    lanes = 15
    stride = 1
    conv_stride_val = 2 # (exec_hints >> 20) & 0x3 + 1

    weights = np.random.choice([-1, 0, 1], (depth, lanes))
    # Inputs need to be large enough for the mock indexing
    inputs = np.random.choice([-1, 0, 1], (1024, lanes))

    w_packed = b"".join([pack_pt5(weights[d]) for d in range(depth)])
    i_packed = b"".join([pack_pt5(inputs[d]) for d in range(1024)])

    fabric.load(0x1000, w_packed)
    fabric.load(0x2000, i_packed)

    # TFMBS_KERNEL_CONV3D = 0x07
    # conv_stride bits (21:20) = 1 (means conv_stride=2)
    exec_hints = 0x07 | (1 << 20)

    tfd = {
        "frame_len": depth,
        "lane_count": lanes,
        "exec_hints": exec_hints,
        "tile_mask": 0x1
    }

    print("Running T-Conv3D on Fabric Mock...")
    status = fabric.run(tfd)

    print("Run returned profiling data:", list(status.keys()))

    fabric_results = fabric.results(0)
    expected = conv3d_reference(weights, inputs, stride, conv_stride_val)

    match = np.array_equal(fabric_results, expected)
    if match:
        print("✅ Validation PASSED: Mock matches reference!")
    else:
        print("❌ Validation FAILED: Mismatch detected.")
        print("Fabric:", fabric_results)
        print("Reference:", expected)

if __name__ == "__main__":
    main()
