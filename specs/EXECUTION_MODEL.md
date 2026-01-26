# Ternary Execution Model (TEM)

## 1. Overview

The Ternary Execution Model (TEM) defines the lifecycle of a Ternary Frame Descriptor (TFD) and the state transitions of the fabric logic. Unlike a traditional CPU that fetches individual instructions, the Ternary Fabric executes **Frame Ops**—atomic operations performed over a vectorized frame.

## 2. The Fabric Pipeline

The fabric processes data through a four-stage pipeline managed by a binary-to-ternary mediator.

1. **Submission:** Host writes a TFD to a memory-mapped command register.
2. **Hydration:** Fabric DMA fetches the packed trits from host memory based on the `base_addr` and `packing_fmt`.
3. **Transformation:** The fabric-attached accelerator (ALU, Signal Processor, or AI Engine) processes the trits in parallel across the defined `lanes`.
4. **Commit/Retire:** Results are packed back into binary format and written to the destination frame, and a completion signal (interrupt or status bit) is sent to the host.

---

## 3. Frame State Transitions

A frame within the fabric must exist in one of the following states:

| State | Description |
| --- | --- |
| **INVALID** | Descriptor is null or contains out-of-bounds addresses. |
| **PENDING** | TFD submitted but waiting for bus/DMA availability. |
| **HYDRATING** | Fabric is actively streaming packed trits from host memory. |
| **ACTIVE** | Accelerator is performing operations on the internal ternary lanes. |
| **COMMITTING** | Final results are being written back to host-visible memory. |
| **COMPLETE** | Operation finished; TFD memory is safe for host reuse. |

---

## 4. Execution Adjacency & Kerners

Rather than a complex Instruction Set Architecture (ISA), the TEM uses **Kernel Hints**.

* **Kernel ID:** A unique identifier for the hardware operation (e.g., `0x01` for Ternary Dot Product).
* **Adjacency:** The fabric optimizes for "Execution Adjacency," where the ALU is physically positioned near the trit-registers to minimize wire length and power consumption—critical for high-density ternary logic.

---

## 5. Error Handling

The execution model identifies three classes of fabric faults:

* **Alignment Fault:** `lane_stride` or `base_addr` does not match the hardware's minimum burst size.
* **Encoding Fault:** The fabric encountered a bit-pattern in a packed byte that does not map to a valid balanced trit sequence (e.g., an unused value in a PT-5 byte).
* **Resource Contention:** Multiple TFDs attempting to write to the same fabric-resident lane without proper fence flags.

---

### Implementation Note

This execution model shifts the burden of "thinking" to the Host (Binary) while the "doing" stays in the Fabric (Ternary). It prevents the need for a complex Ternary Program Counter or Branch Predictor.