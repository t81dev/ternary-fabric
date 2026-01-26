import numpy as np
import argparse

def quantize_to_ternary(weights):
    """
    Implements the Ternary Weight Network (TWN) thresholding logic.
    W_ternary = +1 if W > delta
               -1 if W < -delta
                0 otherwise
    """
    # Heuristic for delta: 0.7 * mean(abs(weights))
    delta = 0.7 * np.mean(np.abs(weights))
    
    ternary = np.zeros_like(weights)
    ternary[weights > delta] = 1
    ternary[weights < -delta] = -1
    
    sparsity = (ternary == 0).sum() / ternary.size
    print(f"Quantization Complete. Sparsity: {sparsity:.2%} (Zero-Skip potential)")
    return ternary.astype(int)

def main():
    parser = argparse.ArgumentParser(description="Ternary Quantization Toolkit")
    parser.add_argument("input", help="Numpy (.npy) weight file")
    parser.add_argument("-o", "--output", default="ternary_weights.txt")
    args = parser.parse_args()

    # Load weights (assuming a flattened or 2D array)
    weights = np.load(args.input)
    t_weights = quantize_to_ternary(weights)

    # Save as space-separated integers for the Ternary-CLI
    np.savetxt(args.output, t_weights.flatten(), fmt='%d', delimiter=' ')
    print(f"Saved ternary weights to {args.output}")

if __name__ == "__main__":
    main()