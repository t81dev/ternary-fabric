from pt5_codec import pack_pt5, unpack_pt5

def simulate_mac(weights, inputs):
    return sum(w * i for w, i in zip(weights, inputs))

if __name__ == "__main__":
    # Test Data
    w_raw = [1, 0, -1, 1, 1]
    i_raw = [1, 1, 1, 0, -1]
    
    encoded_w = pack_pt5(w_raw)
    encoded_i = pack_pt5(i_raw)

    decoded_w = unpack_pt5(encoded_w)
    decoded_i = unpack_pt5(encoded_i)

    result = simulate_mac(decoded_w, decoded_i)

    print(f"Original W: {w_raw}")
    print(f"Original I: {i_raw}")
    print(f"Encoded W: {encoded_w}")
    print(f"Encoded I: {encoded_i}")
    print(f"Decoded W: {decoded_w}")
    print(f"Decoded I: {decoded_i}")
    print(f"MAC Result: {result}")

    if decoded_w == w_raw and decoded_i == i_raw:
        print("Python Logic Passed.")
    else:
        print("Python Logic Failed.")
