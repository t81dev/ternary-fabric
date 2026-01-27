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
- The interface supports both synchronous (blocking) and asynchronous (polling/interrupt) execution.
- `fabric_exec_gemv` initiates the operation.
- Completion is signalled via a status register or callback.

---

## 4. Host API (C ABI)

The following functions define the standard interface for the Fabric Device:

```c
/**
 * Allocate memory in the Fabric pool.
 * Returns a handle/pointer to Fabric-resident memory.
 */
void* fabric_alloc(size_t size);

/**
 * Free memory in the Fabric pool.
 */
void fabric_free(void* ptr);

/**
 * Copy data from Host RAM to Fabric Memory.
 * Handles automatic PT-5 packing if requested.
 */
int fabric_memcpy_to(void* dest_fabric, const void* src_host, size_t size, int pack_pt5);

/**
 * Copy data from Fabric Memory to Host RAM.
 * Handles automatic PT-5 unpacking if requested.
 */
int fabric_memcpy_from(void* dest_host, const void* src_fabric, size_t size, int unpack_pt5);

/**
 * Execute a Ternary General Matrix-Vector multiplication.
 */
int fabric_exec_gemv(void* weight_ptr, void* input_ptr, void* output_ptr, int rows, int cols);
```

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
