# 🔬 Ternary Fabric – Comprehensive Viability Audit Report

## 1. Executive Verdict

**Verdict: Research-Viable / Commercial-Skeptical**

The `t81dev/ternary-fabric` project is a technically coherent research prototype that successfully demonstrates the *potential* of ternary-native computing. The core mathematical premises (Balanced Ternary efficiency, PT-5 packing, Zero-Skip) are sound and implemented in both software emulation and RTL.

However, the current "Fabric Illusion" implementation relies on fragile `SIGSEGV` interception and manual `NOP` markers in application code to achieve CPU short-circuiting. This makes it a "white-box" accelerator requiring application modification, not a transparent "black-box" solution. The hardware implementation of the PT-5 unpacker uses expensive division operations, which may be a bottleneck in physical silicon without careful synthesis optimization.

---

## 2. Strengths

*   **Mathematical Rigor:** The **PT-5** packing format ($3^5 = 243 < 256$) effectively utilizes 95.1% of byte storage, validating the density claims.
*   **Multiplier-Less Microarchitecture:** The `ternary_lane_alu.v` correctly replaces complex multipliers with simple combinatorial gating logic, which would result in significant area and power savings on ASIC.
*   **Zero-Skip Implementation:** Both the emulator and RTL implement logic to skip operations where operands are zero, validating the sparsity exploitation claims.
*   **Software Stack:** The `libtfmbs_intercept.c` provides a clever (albeit fragile) mechanism to transparently offload workloads from existing binaries, and the `pytfmbs` integration shows a clear path for PyTorch adoption.

## 3. Critical Weaknesses

*   **Fragile Interception:** The `libtfmbs_intercept.c` relies on detecting specific memory access patterns and `NOP` sequences (`\x90` x 8) to safely skip CPU instructions. This is not robust for general-purpose binaries without recompilation or specific assembly markers.
*   **Unoptimized Hardware Unpacker:** The `pt5_unpacker.v` uses `/ 3` and `% 3` operators. In standard synthesis flow, this results in expensive logic. A lookup table (LUT) implementation is required for high-speed operation.
*   **Simulated Scalability:** Multi-node scaling is currently simulated via Unix domain sockets (`src/fabric_net.c`), which does not accurately model the latency or congestion of real RDMA/RoCE fabrics.
*   **Benchmark "Rigging":** The primary benchmark (`mock_llama.c`) compares a naive CPU loop against a thread-sleeping emulator. While this demonstrates the *mechanism* of offloading, the performance numbers are synthetic and depend entirely on the emulator's latency model.

## 4. Quantitative Estimates

*   **PT-5 Density:** 1.58 bits/trit (Theoretical max: $\log_2(3) \approx 1.58496$). Efficiency: $243/256 \approx 94.9\%$.
*   **Compute Density:** Removing 8-bit multipliers saves ~100-200 gates per MAC unit. The ternary ALU is estimated to be <50 gates.
*   **Sparsity Gain:** At 50% sparsity, Zero-Skip reduces dynamic switching activity by ~50% (linear scaling), directly translating to power savings.
*   **Latency Model:** The emulator uses `(rows * cols) / lanes`. This assumes infinite memory bandwidth and zero pipeline stalls, which is optimistic.

## 5. Microarchitectural Feasibility (Phase 3)

*   **RTL Validity:** `ternary_lane_alu.v` is solid.
*   **Clock Frequency:** The critical path likely lies in the `pt5_unpacker` (if not LUT-optimized) or the accumulator adder chain. 250 MHz on FPGA is achievable for the ALU, but the unpacker is the bottleneck.
*   **Area:** Extremely low. A single tile of 15 lanes is negligible on modern FPGAs.
*   **Power:** Static power is standard. Dynamic power will be excellent due to Zero-Skip gating.

## 6. Scaling Outlook (Phase 5)

*   **Multi-Tile:** Linear scaling is plausible within a single fabric due to independent lanes.
*   **Multi-Fabric:** Currently simulated. The software orchestration overhead (tracking residency, predicting kernels) may become a bottleneck before interconnect bandwidth does.
*   **Orchestration:** The "5-kernel lookahead" is a heuristic that works well for predictable transformers but may fail for dynamic control flow.

## 7. Commercial Outlook

*   **ASIC NRE:** >$5M for 28nm. Not justifiable yet.
*   **FPGA Deployment:** Highly feasible. The design fits easily on mid-range Zynq devices (XC7Z020).
*   **Differentiation:** Strong against traditional DSPs. Weaker against Bit-Serial (1-bit) accelerators which have even higher density but lower accuracy. Ternary offers a unique "middle ground" of accuracy vs. efficiency.

## 8. Required Proof to Continue

1.  **Physical LUT Implementation:** Replace `val / 3` in `pt5_unpacker.v` with a hardcoded LUT.
2.  **Real RDMA:** Replace Unix sockets with `libibverbs` or a real FPGA-based network interface.
3.  **Robust Intercept:** Remove reliance on `NOP` markers. Use binary instrumentation (e.g., Dyninst, FRIDA) or compiler plugins to inject reliable offload points.
4.  **Power Measurement:** Physical power measurement on the FPGA board to validate the Zero-Skip savings.

## 9. Kill Conditions

*   **Failure to Close Timing:** If the unpacker cannot run at >200 MHz on FPGA, throughput collapses.
*   **Sparsity Irrelevance:** If modern LLMs move to dense quantization (e.g., 4-bit dense), the Zero-Skip advantage vanishes.
*   **Compiler Complexity:** If the "Fabric Illusion" cannot reliably detect kernels in complex software stacks (like un-annotated PyTorch), usability is zero.

## 10. Strategic Recommendation

**Proceed with Research Phase.**
Do not tape out ASIC yet. Focus on:
1.  **FPGA Demo:** Demonstrate real power savings on the XC7Z020.
2.  **Compiler:** Move away from fragile `LD_PRELOAD` interception toward a proper MLIR/TVM backend that emits Ternary-native instructions.
3.  **Hybrid Quantization:** Prove that Ternary weights + FP16 activations (or similar) maintain accuracy for LLMs.

---

### 11. Three Radically Unexpected Applications

1.  **Genomic Sequence Alignment:** DNA sequences (A, C, G, T) can be mapped to ternary/quaternary representations. The "Zero-Skip" logic could be adapted to "Match-Skip" for rapid fuzzy matching of gene sequences.
2.  **High-Frequency Trading (HFT) Signal Processing:** The multiplier-less architecture offers extremely deterministic low latency. Ternary logic is naturally suited for "Buy / Hold / Sell" decision trees encoded directly in hardware.
3.  **Error-Correcting Code (ECC) decoding:** Ternary logic is used in some advanced ECC schemes. A native ternary fabric could accelerate decoding of specialized communication protocols (e.g., for deep space or quantum error correction).
