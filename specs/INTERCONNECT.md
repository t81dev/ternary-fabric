## 1. Overview

The **Ternary Interconnect** defines the signaling protocols and physical/logical pathways that connect the Binary Host to the Ternary Fabric. It serves as the "nervous system" of the architecture, ensuring that Frame Descriptors (TFDs) and packed trit data move between planes with minimal latency.

The interconnect is designed as a **Request-Response** bus, optimized for high-throughput burst transfers rather than random-access word reads.

---

## 2. Bus Topology

The interconnect utilizes a **Mediated Bridge** topology. The Binary Host does not sit on the ternary bus directly; instead, it communicates with a **Fabric Controller (FC)**.

* **Control Plane:** Low-latency, narrow-width (32/64-bit) path for TFD submission and status polling.
* **Data Plane:** High-bandwidth, vectorized path for streaming packed trits between System RAM and Fabric RAM.
* **Accelerator Link:** A dedicated, ultra-low-latency internal bus connecting Fabric RAM to Execution Adjacency units (ALUs).

---

## 3. Signaling Protocol

The interconnect operates via four primary signal groups:

### 3.1 Descriptor Channel (D-CH)

Used by the host to "push" TFDs into the fabric's command queue.

* `DESC_VALID`: Asserted when the host has written a valid TFD to the aperture.
* `DESC_READY`: Asserted by the fabric when it has capacity to accept a new descriptor.
* `DESC_ID`: A unique tag for tracking the frame's lifecycle.

### 3.2 Data Streaming Channel (S-CH)

A burst-oriented channel for moving packed trit blocks.

* `STRM_DATA`: The physical lines carrying packed binary-encoded trits (e.g., 256-bit or 512-bit wide).
* `STRM_STRIDE`: Signal indicating the current lane-offset being transferred.
* `STRM_LAST`: End-of-frame marker.

### 3.3 Synchronization Channel (X-CH)

Manages fences and execution status.

* `FENCE_REQ`: Host request to ensure all prior writes are visible to the fabric.
* `EVENT_DONE`: Fabric-to-Host interrupt or signal indicating a TFD has reached the `COMPLETE` state.

---

## 4. Packing Efficiency & Alignment

Because the interconnect moves data in binary words, the transfer efficiency is governed by the `packing_fmt`.

For the standard **PT-5** format:

* A 64-bit binary word carries  groups of  trits ( trits total).
* The remaining  bits are utilized for **Internal Metadata** (e.g., parity or lane-routing tags) to ensure the bus is never "silent" or underutilized.

---

## 5. Arbitrated Access

When multiple accelerators (e.g., an AI Engine and a Signal Processor) are attached to the same fabric, the Interconnect provides **Fair-Share Arbitration**:

1. **Priority-Based:** AI inference frames may be flagged for higher priority than background compression tasks.
2. **Lane-Locking:** An accelerator can request an "Exclusive Lock" on specific lanes within the interconnect to prevent read/write hazards during complex multi-stage transformations.

---

## 6. Physical Implementation Hints

While the TFMBS is logically independent of the physical layer, implementations typically map these semantics to:

* **On-Chip:** AXI4-Stream or TileLink.
* **Off-Chip:** PCIe TLP (Transaction Layer Packets) or CXL (Compute Express Link) for cache-coherent attachments.

---