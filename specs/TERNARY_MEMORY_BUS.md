# Ternary Fabric Memory & Bus Specification (TFMBS)

Version: 0.1 (Draft)  
Status: Normative  

---

## 1. Scope

This document specifies a ternary-native memory and interconnect fabric embedded within binary-dominated systems. It defines how balanced-ternary data frames are represented, transported, addressed, and executed without replacing existing binary control, storage, or I/O planes.

The fabric targets acceleration of compression, signal processing, simulation, and AI workloads where ternary structure reduces semantic indirection.

This specification is normative for all implementations of the ternary-fabric repository.

---

## 2. Design Constraints

- Binary hosts remain authoritative.
- Ternary planes operate as attached fabrics.
- No global ternary address space is assumed.
- All fabric access is mediated by frame descriptors.
- Vectorization is first-class.
- Coexistence with binary cache and DMA is mandatory.

---

## 3. Terminology

**Trit**  
A ternary digit with values in balanced form: {-1, 0, +1}.

**Balanced Ternary**  
A radix-3 numeric representation using symmetric signed trits instead of {0,1,2}.

**Frame**  
A contiguous, logically-addressed region of ternary data managed as a single transport and execution unit within the fabric.

**Lane**  
A vectorized grouping of trits within a frame, enabling SIMD-style operations and parallel transport.

**Fabric**  
The ternary-native memory and interconnect plane attached to a binary host system.

**Host**  
The binary-dominant processor and memory system that controls fabric allocation, transport, and execution.

**Descriptor**  
A binary-resident control structure defining how a ternary frame is addressed, packed, transported, and executed.

**Plane**  
A logical domain of computation or storage (binary control plane vs ternary data plane).

**Execution Adjacency**  
The attachment point where accelerators, kernels, or coprocessors consume and produce fabric frames.

---

## 4. Fabric Architecture Overview

The ternary fabric is an attached data plane embedded within a conventional binary system. It does not replace the host CPU, memory, or I/O architecture, but augments them with a ternary-native transport and execution substrate.

Conceptually, the system consists of:

- A binary host CPU issuing control and scheduling commands.
- Binary system memory holding control structures and frame descriptors.
- Ternary fabric memory optimized for packed trit storage.
- An interconnect layer mediating binary ↔ ternary transfers.
- Optional accelerators attached to the fabric plane.

All access to ternary data is mediated by binary-resident Ternary Frame Descriptors (TFDs). The host never directly addresses ternary memory; instead, it submits descriptors to the fabric, which resolve transport, packing, and execution semantics.

The fabric supports vectorized lanes, burst transfers, and accelerator adjacency while remaining compatible with binary DMA, cache coherence, and paging models.

The architectural principle is separation of concerns:

- Binary planes control, schedule, and synchronize.
- Ternary planes store, move, and transform structured data.

This preserves binary sovereignty while allowing ternary-native semantics to operate efficiently as a specialized substrate.

---

## 5. Ternary Frame Descriptor (TFD)

### 5.1 Purpose

The Ternary Frame Descriptor (TFD) defines the ABI boundary between binary control logic and ternary fabric data planes.

A TFD specifies how a ternary frame is:

- located,
- packed,
- addressed,
- transported, and
- executed.

All fabric operations MUST be expressed in terms of TFDs. No implicit ternary addressing is permitted.

---

### 5.2 Descriptor Layout

The TFD is a binary-resident structure with a fixed header and extensible fields.

A canonical logical layout is:

| Field | Description |
|------|------------|
| `base_addr` | Fabric-local base address of the frame. |
| `frame_len` | Length of the frame in trits. |
| `packing_fmt` | Encoding and packing scheme for trits. |
| `lane_count` | Number of vector lanes. |
| `lane_stride` | Trit stride between lanes. |
| `flags` | Semantic and transport flags. |
| `exec_hints` | Optional execution adjacency hints. |
| `version` | Descriptor version for ABI evolution. |

The binary representation of the TFD is implementation-defined but MUST preserve semantic equivalence of the above fields.

---

### 5.3 Flags

The `flags` field MAY encode:

- read-only / write-back semantics  
- coherence requirements  
- compression enabled  
- alignment guarantees  
- execution ordering constraints  

Flags are advisory unless explicitly declared mandatory by a compliance level.

---

### 5.4 Execution Hints

The `exec_hints` field communicates optional information to accelerators, such as:

- preferred vector width  
- reduction operations  
- locality assumptions  
- fusion opportunities  

Hints MUST NOT alter correctness, only performance.

---

### 5.5 Lifetime and Ownership

TFDs are allocated and owned by the binary host. The fabric may cache or transform frames internally, but ownership and synchronization remain host-controlled.

Frames MAY be transient, persistent, or streaming, as indicated by descriptor flags.

---

## 6. Packing Model

### 6.1 Trit Encoding

Trits are represented in balanced form: {-1, 0, +1}. Physical storage MAY encode these values using binary bit patterns, but the logical semantics remain ternary.

Implementations MUST define a reversible mapping between binary storage units and trit sequences.

---

### 6.2 Byte Packing

Frames are stored as packed trit sequences inside binary-addressable memory units.

Packing strategies MAY include:

- fixed-width groups (e.g. 5 trits per byte),
- block-compressed layouts,
- vector-aligned lanes.

The selected packing format is identified by the `packing_fmt` field of the TFD.

---

### 6.3 Alignment

Frames MUST specify alignment constraints sufficient to support vectorized access. Lane-aligned frames SHOULD minimize unpacking overhead during execution adjacency.

---

### 6.4 Vector Grouping

Trits are grouped into lanes to support SIMD-style transport and execution. Lane width and grouping semantics are descriptor-defined and fabric-validated.

---

### 6.5 Endianness

Trit ordering within packed storage MUST be explicitly defined by the packing format. Binary host endianness does not implicitly apply to ternary ordering.

---

## 7. Addressing and Stride

This section defines how frames are addressed, how lanes are organized, and how the host interacts with ternary fabric memory via Ternary Frame Descriptors (TFDs).

---

### 7.1 Frame-Local Addressing

- Each ternary frame has a logical address space ranging from `0` to `frame_len - 1` trits.
- All internal lane and vector operations within a frame use **frame-local offsets**.
- Fabric implementations MUST support deterministic mapping of frame-local addresses to physical fabric locations.

---

### 7.2 Host-Visible Handles

- Binary hosts never access fabric memory directly; all access occurs via **TFDs**.
- Each TFD contains a unique `base_addr` field representing the fabric-resident frame.
- Hosts perform read, write, and execute operations using this handle only.
- Handles are immutable while the frame is live; updates require a new descriptor.

---

### 7.3 Lanes and Stride

- Frames are subdivided into **lanes** for SIMD-style transport and execution.
- Each lane contains a contiguous sequence of trits with a fixed **lane_stride** defined in the TFD.
- **Lane layout example:**

```

Frame Length: 16 trits
Lane Count: 4
Lane Stride: 1

Logical view:

Lane 0: trits 0,4,8,12
Lane 1: trits 1,5,9,13
Lane 2: trits 2,6,10,14
Lane 3: trits 3,7,11,15

```

- The **lane_stride** defines the distance between consecutive trits in a lane.
- Implementations may interleave, block-pack, or vectorize lanes, provided the mapping is reversible and respects the TFD.

---

### 7.4 Scatter / Gather Support

- Frames MAY support non-contiguous trit ranges.
- Scatter/gather patterns MUST be fully described in TFD flags or extended fields.
- Fabric MUST ensure logical ordering is preserved for all descriptor-driven operations.

---

### 7.5 DMA and Transfer Alignment

- Transfers between binary memory and the ternary fabric MUST honor **alignment constraints** specified in the TFD.
- Minimum transfer unit is one lane; burst transfers SHOULD align to full lane multiples.
- DMA engines MUST respect TFD-provided stride and packing formats to prevent semantic corruption.
- Fabric MAY support coalescing multiple small transfers into a single burst if consistent with lane alignment.

---

### 7.6 Example: Minimal Frame Descriptor

| Field         | Value Example       | Description                              |
|---------------|------------------|------------------------------------------|
| `base_addr`    | 0x1000           | Fabric-resident base address             |
| `frame_len`    | 16               | 16 trits total                            |
| `packing_fmt`  | PT-5             | 5 trits per byte packing format           |
| `lane_count`   | 4                | Four SIMD lanes                           |
| `lane_stride`  | 1                | Stride between trits in a lane            |
| `flags`        | 0x02             | Read/write, aligned                        |
| `exec_hints`   | 0x01             | Prefers vectorized execution               |
| `version`      | 0x01             | Descriptor version                         |

- In this configuration, the fabric maps four lanes over 16 trits. Each lane accesses trits spaced by `lane_stride`. DMA and accelerator operations follow these addresses to guarantee consistency.

---

### 7.7 Compliance Rules

- **Minimal level:** Single lane, contiguous frame, no scatter/gather required.  
- **Accelerated level:** Multi-lane vectorization with stride support.  
- **Hardware-native:** Full burst, scatter/gather, and descriptor-managed DMA.

---

### 7.8 Summary

- Frame-local addressing provides deterministic, reversible mapping of trits to fabric memory.
- Binary hosts interact exclusively through TFD handles.
- Lane layout and stride ensure vectorized access for accelerators.
- DMA and scatter/gather semantics preserve consistency across fabric operations.

---

## 8. Bus / Interconnect Semantics

This section defines how the ternary fabric communicates with the binary host, how data is moved between memory planes, and how multiple lanes and frames interact within the interconnect.

---

### 8.1 Transfer Units

- The smallest unit of fabric transfer is a **lane** as defined in the TFD.
- Each lane transfer MUST be atomic with respect to fabric operations.
- Partial-lane transfers MAY be allowed in hardware-native implementations but MUST preserve logical frame consistency.

---

### 8.2 Burst Behavior

- The fabric supports **burst transfers** to maximize throughput.
- Bursts operate over consecutive lanes or blocks of trits within a frame.
- Burst length MUST be aligned to lane boundaries.
- Fabric MAY coalesce multiple small TFD operations into a single burst if it does not violate descriptor semantics.

**Example:**  
For a 16-trit frame with 4 lanes, a single burst could transfer all lanes simultaneously, reducing interconnect overhead.

---

### 8.3 Coherence Model

- The fabric enforces **descriptor-driven coherence**:
  - Host writes to a frame are only visible after **TFD commit**.
  - Accelerator reads follow TFD-defined read/write flags.
  - Optional cache-invalidation or flush operations may be triggered via descriptor flags.
- Minimal compliance: no internal fabric caching; host responsible for consistency.
- Accelerated/hardware-native: fabric may maintain internal caches with coherent updates visible via fences.

---

### 8.4 Latency Assumptions

- Fabric implementations MUST define minimum and maximum expected latency per lane transfer.
- Latency MUST be bounded for predictable accelerator scheduling.
- For hardware-native levels, latency guarantees MAY differ for aligned burst vs scattered transfers; TFD MUST encode preferred access patterns.

---

### 8.5 Binary ↔ Ternary Mediation

- All fabric access originates from binary hosts via TFDs.
- Binary ↔ ternary mediation rules:
  1. Host issues a descriptor to read/write a frame.
  2. Fabric validates the descriptor (bounds, flags, alignment).
  3. Fabric executes transfer, respecting lane layout and packing format.
  4. Completion/fence signals returned to host.
- Binary hosts MUST NOT bypass TFDs to access fabric memory directly.
- Accelerator execution consumes frames via fabric endpoints; mediator ensures host-accelerator synchronization.

---

### 8.6 Fault and Error Handling

- Transfers violating alignment, stride, or lane boundaries MUST be rejected.
- Fabric MAY return error codes through TFD status fields or dedicated completion registers.
- Errors MUST NOT corrupt other frames or lanes.

---

### 8.7 Compliance Rules

| Level | Transfer Semantics |
|-------|------------------|
| Minimal | Single-lane atomic transfers; host manages coherence manually. |
| Accelerated | Multi-lane burst transfers; host-visible fences optional. |
| Hardware-native | Full burst, scatter/gather, internal caching, and accelerator-adjacent transfers with deterministic latency. |

---

### 8.8 Summary

- Lane is the atomic transfer unit.
- Bursts improve throughput while preserving frame integrity.
- Descriptor-driven coherence ensures host/accelerator consistency.
- Binary ↔ ternary mediation enforces TFD as the single source of truth.
- Compliance levels allow phased implementation from minimal to fully accelerated fabric.

---

## 9. Execution Adjacency

This section specifies how accelerators, kernels, or coprocessors attach to the ternary fabric, how they consume and produce frames, and how execution hazards are managed.

---

### 9.1 Accelerator Attachment

- Accelerators connect to the fabric via **lane-aligned endpoints**.
- Each endpoint is associated with one or more **frames** described by TFDs.
- Attachment MAY be static (fixed lanes) or dynamic (descriptor-driven allocation).
- Fabric MUST validate all descriptors before allowing accelerator access.

---

### 9.2 Frame Consumption and Production

- **Read operations:** Accelerators read frames using lane-aligned transfers defined by the TFD.
- **Write operations:** Accelerators write results back to frames respecting packing, stride, and lane layout.
- **In-place operations:** Allowed if `flags` indicate mutable frame ownership.
- **Multiple consumers:** Concurrent reads are allowed; writes must follow descriptor-defined hazard rules.

---

### 9.3 Hazard Management

- **Read-after-write (RAW):** Fabric MUST ensure that an accelerator does not read stale data; RAW hazards are resolved via TFD commit/fence semantics.
- **Write-after-read (WAR) and write-after-write (WAW):** Fabric MUST enforce order or provide advisory warnings; host MAY serialize operations.
- **Lane conflicts:** No two accelerators may write the same lane simultaneously unless explicitly flagged as synchronized.

---

### 9.4 Fence Semantics

- **Fences** are used to guarantee visibility and ordering of frame operations between host and accelerators.
- Fence types:
  1. **Acquire fence:** Ensures prior host writes are visible to accelerator.
  2. **Release fence:** Ensures accelerator writes are visible to host.
- Fences MAY be per-frame, per-lane, or global, depending on compliance level and fabric capabilities.
- Minimal compliance: host manually sequences operations without fences.
- Accelerated/hardware-native: fences are hardware-supported and integrated with burst transfers.

---

### 9.5 Descriptor-Driven Execution

- All accelerator operations MUST reference the TFD for frame layout, packing, and stride.
- Execution hints (`exec_hints`) in TFD are advisory; accelerators MAY optimize performance based on vector width, reduction patterns, or lane locality.
- Fabric MUST preserve semantic integrity regardless of hint usage.

---

### 9.6 Lifetime and Ownership Rules

- Accelerators MAY cache frames temporarily for processing.
- Ownership of frames and their memory regions always remains with the binary host unless explicitly transferred via descriptor flags.
- Transient frames used by accelerators MUST be invalidated or synchronized before host reuse.

---

### 9.7 Compliance Levels

| Level | Accelerator Semantics |
|-------|----------------------|
| Minimal | Single-lane execution; host-managed sequencing; no fences required. |
| Accelerated | Multi-lane burst execution; optional fences; in-place updates allowed. |
| Hardware-native | Full SIMD execution; scatter/gather; integrated fence support; hazard resolution enforced by fabric. |

---

### 9.8 Summary

- Execution adjacency defines how accelerators interact with the fabric using TFDs.
- Hazards, fences, and ownership rules ensure deterministic, safe operation.
- Compliance levels provide a roadmap from minimal host-managed execution to fully hardware-accelerated deterministic fabric operation.

---

## 10. Binary Coexistence

### 10.1 Cache Interaction

The ternary fabric is logically distinct from the binary CPU cache hierarchy.

* **Non-Coherent Access:** By default, fabric memory is treated as non-coherent. The Host is responsible for flushing binary caches before committing a TFD that points to shared physical memory.
* **Fabric-Side Caching:** If an implementation caches trits, it MUST provide a mechanism to invalidate these caches when a TFD is released or updated.

### 10.2 Page Behavior and Memory Mapping

* **Unified Address Space:** If the fabric shares physical DRAM with the host, the Host OS MUST map ternary frames into non-pageable (pinned) memory regions to prevent DMA faults during fabric execution.
* **Virtual Addressing:** While TFDs use `base_addr`, hardware implementations SHOULD support an IOMMU-style translation if the fabric performs direct memory access to host virtual addresses.

### 10.3 Fallback Paths

For systems lacking native ternary hardware, a **Software Emulation Layer (SEL)** MUST be provided.

* The SEL consumes TFDs and performs packing/unpacking via bit-manipulation on the binary CPU.
* The SEL MUST maintain semantic parity with hardware-native implementations, albeit at higher latency.

---

## 11. Compliance Levels

To allow for a variety of implementations (from FPGA prototypes to ASIC accelerators), three compliance levels are defined:

| Feature | Level 1: Minimal | Level 2: Accelerated | Level 3: Hardware-Native |
| --- | --- | --- | --- |
| **Transport** | Single-lane, Serial | Multi-lane, Burst | Full Crossbar / Mesh |
| **Packing** | PT-5 (Fixed) | Multiple Formats | Programmable Packing |
| **Coherence** | Manual (Software) | Hardware Fences | Full IOMMU/Coherence |
| **Execution** | CPU-Emulated | Fixed-Function Accel | Programmable Ternary ALU |
| **Addressing** | Contiguous Only | Stride Support | Scatter / Gather |

---

## 12. Versioning and Evolution

### 12.1 ABI Stability

The first 8 bytes of any TFD are reserved for the `version` and `flags` fields. This ensures that even as the fabric evolves, the host-mediator can identify and reject unsupported descriptor versions.

### 12.2 Extension Mechanism

Extensions to the TFMBS (such as custom compression or specific AI-op hints) MUST be stored in the extensible tail of the TFD.

* **Ignorable Extensions:** If a fabric does not recognize an extension bit in the `exec_hints`, it MUST ignore the hint and proceed with baseline execution.
* **Critical Extensions:** If a flag in the `flags` field is unknown and marked as "Critical," the fabric MUST return a `TFMBS_ERR_UNKNOWN_FEATURE` status.

### 12.3 Backwards Compatibility

Version 0.x descriptors are considered experimental. Starting from Version 1.0, all implementations MUST maintain backward compatibility with the PT-5 (5-trits-per-byte) packing format as the universal fallback.

---