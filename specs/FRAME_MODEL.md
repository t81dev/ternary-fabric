# Ternary Frame Model (TFM)

## 1. Overview

The Ternary Frame Model (TFM) defines the logical topology of ternary data. It abstracts physical storage into a structured multi-dimensional coordinate system, allowing accelerators to operate on trits without knowledge of the underlying binary packing.

## 2. Frame Geometry

A frame is not merely a flat array; it is a structured volume defined by three primary dimensions:

1. **Total Length ():** The total number of trits in the frame.
2. **Lane Count ():** The number of parallel execution paths (SIMD width).
3. **Depth ():** The number of trits per lane, where .

---

## 3. Logical Mapping

A trit's position within a frame is addressed by the tuple . The hardware maps this logical coordinate to a **Frame-Local Offset** () using the formula:

This mapping ensures that vector engines can pull "vertical" slices of data across lanes in a single clock cycle.

---

## 4. Packing & Boundary Semantics

Because trits are typically packed into 8-bit bytes (e.g., the PT-5 format), frames often have "logical tails."

* **Trit Alignment:** If a frame length is not a multiple of the packing factor, the remaining bits in the final byte of the buffer MUST be zero-padded and ignored by the fabric.
* **Lane Padding:** For high-performance SIMD execution, implementations MAY require lanes to be aligned to 64-bit boundaries. This is specified via the `alignment` flag in the TFD.

---

## 5. Atomicity and Granularity

The TFM enforces specific rules on how data is updated:

* **Trit Granularity:** The fabric provides the illusion of trit-level addressing.
* **Byte Atomicity:** On the binary bus, the smallest atomic operation is one byte (e.g., 5 trits in PT-5).
* **Lane Atomicity:** Within the ternary fabric, the smallest atomic operation is one **Lane**. A write to Lane 0 must not disturb the state of Lane 1.

---

## 6. Frame Metadata (Shadow State)

While the TFD (Descriptor) lives in binary memory, the fabric MAY maintain "Shadow State" for active frames:

* **Parity/Checksum:** Optional error detection trits.
* **Dirty Bits:** Tracking which lanes have been modified by an accelerator for optimized write-back.
* **Polarity:** A frame-wide flag indicating if the trits should be logically inverted () during read, used for signal processing optimizations.

---

## 7. Use Case Examples

### 7.1 Linear Signal (1D)

* **Lanes:** 1
* **Stride:** 1
* **Result:** A simple contiguous stream of trits for sequential DSP kernels.

### 7.2 Neural Weight Matrix (2D)

* **Lanes:** 64 (representing matrix columns)
* **Stride:** 1
* **Result:** Parallel dot-product engines can consume one "row" per cycle across all 64 lanes.

---

### Implementation Note

The Frame Model allows the hardware to be "geometry aware." By changing the `lane_stride` in the descriptor, the host can effectively "reshape" a matrix without moving a single byte of data in physical memory.
