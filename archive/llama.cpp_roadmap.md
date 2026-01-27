Below is a **deep, systems-level roadmap for Strategy 4**:

> **Accelerate `llama.cpp` with Ternary-Fabric by exposing Fabric as a device-level memory/compute substrate (PCIe/CXL-like), so llama.cpp runs unmodified while Fabric transparently compresses, skips, and executes.**

This treats Ternary-Fabric not as a plugin, but as **a memory-centric accelerator that the OS maps into the process**.

No llama.cpp source changes.
No GGUF changes required (initially).
Acceleration happens below the application layer.

---

# 🧭 Strategy 4 Roadmap — Device-Level Fabric Acceleration

---

## 🧱 Core Architecture

Target illusion:

```
llama.cpp
   ↓
Virtual Memory (OS)
   ↓
Fabric Driver (kernel / userspace)
   ↓
TFMBS Device (PCIe / MMIO / DMA)
   ↓
Ternary Execution + Memory Fabric
```

llama.cpp believes it reads/writes RAM.
Fabric actually:

* compresses weights,
* keeps them resident,
* skips zeros,
* executes dot products internally.

This mirrors GPU Unified Memory / CXL.mem style systems.

---

## Phase 0 — Define the Fabric Device Contract

Before code, define what Fabric *is* to the OS.

### Decide:

* Is Fabric:

  * PCIe device?
  * CXL.mem-like?
  * userspace DMA engine?
* Addressing:

  * memory-mapped?
  * ioctl-driven?
* Operations:

  * load frame
  * execute GEMV
  * DMA in/out

### Minimal device API

Conceptual:

```c
FABRIC_ALLOC(size)
FABRIC_FREE(ptr)
FABRIC_DMA_TO(ptr, host_buf, size)
FABRIC_DMA_FROM(host_buf, ptr, size)
FABRIC_EXEC(opcode, args)
```

### Deliverable

* `TFMBS_DEVICE_SPEC.md`
* ABI for memory + execution.

---

## Phase 1 — Emulated Device (User-Space First)

Do **not** start in kernel space.

Build a user-space Fabric emulator:

* backed by malloc,
* logs accesses,
* simulates:

  * PT-5 frames,
  * skip logic,
  * SIMD execution.

Expose via:

* `libtfmbs_device.so`

### Implement:

```c
void *fabric_alloc(size);
void fabric_free(void*);
void fabric_memcpy_to(...);
void fabric_memcpy_from(...);
void fabric_exec_gemv(...);
```

This becomes your reference backend.

### Deliverable

* Userspace Fabric runtime.
* Test harness independent of llama.cpp.

---

## Phase 2 — Memory Interposition Layer

Now create the illusion.

You interpose memory so llama.cpp unknowingly uses Fabric memory.

Using:

* `LD_PRELOAD`

Intercept:

```c
malloc
free
mmap
munmap
memcpy
memmove
```

Logic:

* Large allocations → Fabric.
* Weight-like regions → Fabric resident.
* Small control buffers → normal RAM.

Example:

```c
if (size > FABRIC_THRESHOLD)
    return fabric_alloc(size);
else
    return real_malloc(size);
```

And:

```c
memcpy(dst, src, n):
  if (is_fabric(dst) || is_fabric(src))
      fabric_dma(...)
  else
      real_memcpy(...)
```

Now llama.cpp is unknowingly using Fabric-backed memory.

### Deliverable

* `libtfmbs_intercept.so`
* Allocation + DMA interception.

---

## Phase 3 — Pattern Recognition for Compute

Now Fabric needs to accelerate computation, not just memory.

Observe llama.cpp behavior:

* repeated reads of matrix rows,
* dot product loops,
* block quant unpacking.

Use heuristics:

* detect stride-1 vector access,
* detect matrix-vector reuse,
* detect repeated row scans.

When pattern matches GEMV:

Instead of letting CPU touch memory:

```
CPU reads W, x → CPU computes
```

You redirect:

```
fabric_exec_gemv(W_frame, x, y)
```

and short-circuit the CPU loop.

This is similar to how some DB engines offload scans.

You don’t need to understand llama.cpp semantically — just structurally.

### Deliverable

* Compute interception prototype.
* Logged “GEMV detected” events.

---

## Phase 4 — Weight Residency & Compression

Now activate Fabric’s real advantage.

When memory is identified as weights:

* convert to PT-5 ternary frames,
* compress,
* keep resident.

From then on:

* host never reloads weights,
* Fabric handles reuse.

Implement:

```c
on_first_touch(region):
    pack_to_pt5(region)
    mark_resident(region)
```

And future accesses hit Fabric memory, not CPU RAM.

### Deliverable

* Resident weight cache.
* Compression + hydration pipeline.

---

## Phase 5 — Execution Injection

Replace read-based compute with Fabric execution.

Instead of:

```
for i: y += W[i] * x[i]
```

Do:

```
fabric_exec(GEMV, W, x, y)
```

Return result to host buffer.

Host thinks memory changed.
Fabric actually computed it.

Now Ternary-Fabric is executing math transparently.

### Deliverable

* First end-to-end token path accelerated without llama.cpp changes.

---

## Phase 6 — Zero-Skip + SIMD Enablement

Activate native ternary advantages:

* zero digit skip,
* SIMD broadcast of activations,
* PT-5 dense packing.

Track metrics:

```
skip_rate
lanes_used
bytes_moved
fabric_cycles
```

Tune:

* threshold for ternary digitization,
* plane density.

### Deliverable

* Real bandwidth + compute reduction.

---

## Phase 7 — Paging & Eviction

Large models won’t all fit.

Add:

* LRU for Fabric memory,
* eviction to host RAM,
* prefetch next layers.

Pattern:

```
layer N used → keep
layer N-2 unused → evict
```

This mirrors GPU Unified Memory.

### Deliverable

* Stable execution on large GGUF models.

---

## Phase 8 — Asynchronous Pipelining

Hide latency.

Instead of blocking:

```
fabric_exec → wait → host
```

Use:

```
submit token N
host works on token N-1
fabric computes N+1
```

Add queues:

```c
fabric_submit(...)
fabric_poll(...)
```

### Deliverable

* Overlap host + Fabric execution.

---

## Phase 9 — Telemetry & Proof

Instrument:

* tokens/sec
* bytes/token
* fabric_time vs host_time
* energy proxy
* skip density

Build benchmark harness:

```
baseline llama.cpp
vs
fabric-accelerated llama.cpp
```

Without code changes.

### Deliverable

* Benchmark report.
* Performance plots.

---

## Phase 10 — Hardware Path (Optional, Real Device)

Once software works:

* expose Fabric as:

  * PCIe device,
  * CXL.mem region,
  * mmap’able BAR.

Kernel driver:

* handles page faults,
* routes DMA,
* triggers execution.

Userland remains unchanged.

Now Fabric becomes a real accelerator.

### Deliverable

* Kernel module + hardware interface.

---

# 🔑 What This Strategy Gives You

✅ Zero llama.cpp modifications
✅ Fabric as memory + compute substrate
✅ Transparent acceleration
✅ Works with existing GGUF models
✅ Matches Fabric’s identity as *memory fabric*

Instead of being a “backend,” Fabric becomes **part of the machine**.

---
