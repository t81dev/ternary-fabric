import torch
import torch.nn as nn
import pytfmbs
from pytfmbs import TFMBSLinear
import numpy as np
import sys
import os

# Add src to path if not installed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

class TernaryMNIST(nn.Module):
    def __init__(self, fabric):
        super().__init__()
        self.flatten = nn.Flatten()

        # Layer 1: 784 -> 15 (Fits in 1 tile)
        # We start at 0x1000
        self.fc1 = TFMBSLinear(784, 15, fabric=fabric, weight_addr=0x1000)
        self.relu = nn.ReLU()

        # Layer 2: 15 -> 10
        # We start after fc1's weights: 0x1000 + (784 * 4) = 0x1C40
        self.fc2 = TFMBSLinear(15, 10, fabric=fabric, weight_addr=0x1C40)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def main():
    print("🚀 TFMBS PyTorch MNIST Demo (Mock Mode)")

    # Initialize Fabric (will fallback to Mock Mode)
    fabric = pytfmbs.Fabric()

    model = TernaryMNIST(fabric)

    # Create a random "image" (single batch)
    dummy_input = torch.randn(1, 1, 28, 28)

    print("\n--- Running Inference ---")
    # Weights are quantized and loaded upon first forward pass
    with torch.no_grad():
        output = model(dummy_input)

    print("\n✅ Inference Complete!")
    print(f"Output Shape: {output.shape}")
    print(f"Prediction: {torch.argmax(output, dim=1).item()}")
    print(f"Output Tensor: \n{output}")

    # Extract hardware telemetry
    print("\n--- Fabric Telemetry ---")
    stats = fabric.profile_detailed()
    print(f"Total Cycles:      {stats['cycles']}")
    print(f"Lane Utilization:  {stats['utilization']}")
    print(f"DMA Wait Cycles:   {stats['burst_wait_cycles']}")

    total_skips = sum(stats['skips'])
    skip_percentage = (total_skips / stats['utilization']) * 100 if stats['utilization'] > 0 else 0
    print(f"Zero-Skip Savings: {skip_percentage:.2f}%")

if __name__ == "__main__":
    main()
