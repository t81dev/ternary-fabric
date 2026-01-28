The initial Phase 18 benchmark results confirm that the multi-tile scaling and zero-skip logic are operating correctly within the new measurement plane:

Layer 1: Synthetic (Hardware Semantics)
Tile Saturation: As expected in the emulator, adding tiles reduced execution time for a fixed 1024x1024 workload, showing strong scaling from 1 tile to 4 tiles.
Mask 0x01 (1 Tile): 0.004089 s (69,905 Cycles)
Mask 0x0f (4 Tiles): 0.003014 s (17,476 Cycles)
Zero-Skip Density: Verified bit-exact skip ratios across the full density curve (0% to 100% sparsity).
Layer 2: Kernel (Primitive Performance)
GEMM Kernel (1024x1024): 10 iterations completed in 0.076 s, yielding ~0.27 GFLOPS effective throughput in the software emulator.
T-LSTM Kernel (H=512, I=512): 10 iterations completed in 0.072 s.
Sparsity Impact: At 90% weight sparsity, the T-LSTM step achieved a 90% cycle reduction in the execution core.
Phase 18 Metrics (Cost Model)
The synthetic cost model is now active. For the T-LSTM workload (H=512):

Cycles: 3,072
Fabric Cost: 2,373,276.0 (Calculated as: active_ops*1.0 + mem_reads*5.0 + mem_writes*8.0)
These results establish the baseline for Phase 18, proving that our "semantic work per cost unit" KPI can now be tracked and optimized.
