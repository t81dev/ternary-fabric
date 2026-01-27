# 10: Appendices

## A. Acronyms & Definitions

| Acronym | Definition |
| --- | --- |
| **AXI** | Advanced eXtensible Interface (ARM standard bus). |
| **MMIO** | Memory-Mapped I/O. |
| **PT-5** | Packed Ternary (5 trits per 8 bits). |
| **SIMD** | Single Instruction, Multiple Data. |
| **TFD** | Ternary Frame Descriptor. |
| **TFMBS**| Ternary Fabric Memory & Bus Specification. |
| **TPE** | Ternary Processing Element. |

## B. PT-5 Packing Format Details

The PT-5 format is a high-density encoding for balanced ternary data.
*   **Capacity:** $\lfloor 8 \times \log_3(2) \rfloor = 5$ trits per byte.
*   **Efficiency:** $3^5 = 243$ states used out of $2^8 = 256$ (95.1%).
*   **Mapping:**
    *   Trits are treated as digits in a base-3 number.
    *   Individual trits $\in \{-1, 0, 1\}$ are mapped to $\{2, 0, 1\}$ respectively.
    *   $Byte = \sum_{i=0}^{4} trit_i \times 3^i$.

## C. Verification & Validation Reports

The fabric has been validated against several benchmarks. Detailed logs and reports can be found in the `docs/validation/` directory:

*   **Phase 5 Report:** Verification of Zero-Skip and T-GEMM results (See `archive/PHASE_5_REPORT.md`).
*   **Phase 6b Report:** Multi-tile scaling and FPGA synthesis estimates (See `docs/validation/phase6b/report.md`).
*   **Regression Suite:** `pytest tests/` contains the latest functional validation tests.

## D. Architecture Diagrams (Full-Scale)

(Refer to `docs/03_HARDWARE.md` for logical and pipeline diagrams).
