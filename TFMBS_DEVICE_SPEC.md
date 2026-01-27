# 📄 TFMBS Device Specification (v0.1-draft)

---

## 1. Overview
This document defines the normative device contract for the Ternary Fabric Memory & Bus Specification (TFMBS). It specifies the interface between a binary host (e.g., llama.cpp) and the ternary-native execution substrate.

The Fabric acts as a specialized memory-compute engine that performs ternary operations (GEMV, Dot Product) directly on data residing in its own memory pool.

---

## 2. Memory Semantics

### 2.1 Addressing
- The Fabric operates on its own private address space (Fabric Memory).
- All host interactions are mediated by **Ternary Frame Descriptors (TFDs)**.
- Memory is managed in **Frames**, which are contiguous regions of ternary data.

### 2.2 Packing Format (PT-5)
- The primary packing format is **PT-5** (5 balanced trits per 8-bit byte).
- 1 byte stores values in the range [-121, 121] when interpreted as ternary.
- Alignment: All frame base addresses SHOULD be 64-bit aligned.

### 2.3 Residency Model
- **Resident Frames:** Weights that remain in Fabric memory across multiple inference passes.
- **Transient Frames:** Activations or temporary buffers that are loaded, processed, and discarded.
- **Eviction:** Managed by the host via `fabric_free` or explicit eviction commands.

---

## 3. Execution Interface

### 3.1 Kernel: Ternary GEMV
The primary workload for llama.cpp acceleration.
- **Inputs:**
  - Weight Frame (Resident)
  - Input Vector Frame (Transient)
- **Output:**
  - Result Vector Frame (Host-visible)
- **Semantics:** Performs $y = Wx$, where $W$ is ternary, $x$ is typically binary/FP (quantized to ternary by the fabric if necessary), and $y$ is binary (accumulated results).

### 3.2 Async vs Sync
- The interface supports both synchronous (blocking) and asynchronous (polling/interrupt) execution via IOCTL.
- `TFMBS_IOC_SUBMIT` initiates the operation using a TFD.
- `TFMBS_IOC_WAIT` blocks until completion.
- Completion can also be polled via the `DONE` bit in the control register.

---

## 4. Host API (UAPI / IOCTL)

The normative interface for the Fabric Device is mediated via the `/dev/tfmbs` character device using the following IOCTLs:

```c
/**
 * Submit a Ternary Frame Descriptor for execution.
 * Returns a unique task_id.
 */
#define TFMBS_IOC_SUBMIT _IOW(TFMBS_IOC_MAGIC, 1, tfmbs_tfd_t)

/**
 * Block until the specified task is complete.
 */
#define TFMBS_IOC_WAIT _IOW(TFMBS_IOC_MAGIC, 2, uint32_t)

/**
 * Submit a GEMV operation directly (Optimized path).
 */
#define TFMBS_IOC_SUBMIT_GEMV _IOW(TFMBS_IOC_MAGIC, 3, tfmbs_gemv_t)
```

For high-level languages, the `pytfmbs` library provides a Pythonic wrapper around these low-level calls.

---

## 5. Implementation Requirements

- **Software Emulation Layer (SEL):** Must provide a bit-exact simulation of ternary logic for development and testing.
- **Interconnect:** Minimum 10Gbps equivalent bandwidth for PT-5 frames.
- **Zero-Skip:** Hardware must natively skip zero-valued trits to save power and cycles.

---

## 6. References
- `specs/TERNARY_MEMORY_BUS.md`: Detailed bus and addressing specification.
- `specs/EXECUTION_MODEL.md`: Pipeline and state transition details.
- `specs/FRAME_MODEL.md`: Physical layout and lane organization.
