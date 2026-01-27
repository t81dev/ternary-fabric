# Device Driver Interface Specification (Phase 10)

This document specifies the kernel-space interface and error handling model for the Ternary Fabric co-processor.

## 1. IOCTL Interface

The driver exposes a character device (e.g., `/dev/tfmbs0`) which supports the following primary IOCTL commands:

| IOCTL Command | Data Structure | Description |
| :--- | :--- | :--- |
| `TFMBS_IOC_SUBMIT` | `tfmbs_tfd_t` | Submit a Ternary Frame Descriptor for execution. |
| `TFMBS_IOC_WAIT` | `uint32_t` (task_id) | Block until the specified task is complete. |
| `TFMBS_IOC_ALLOC` | `tfmbs_alloc_t` | Allocate a physically contiguous (CMA) buffer for DMA. |
| `TFMBS_IOC_FREE` | `uint64_t` (handle) | Free a previously allocated DMA buffer. |
| `TFMBS_IOC_GET_METRICS`| `tfmbs_metrics_t` | Retrieve hardware performance counters. |

## 2. Error Handling & Recovery

### Hardware Timeouts
To prevent system hangs, the driver implements a watchdog timer for every submitted task.
*   **Timeout Threshold:** Default is 500ms.
*   **Action:** If a task exceeds the threshold, the driver issues a hardware reset via the `SYS_RST` bit in the `CONTROL` register, clears the command queue, and returns `-ETIMEDOUT`.

### Memory Faults & Bounds Checking
The driver performs strict bounds checking on all TFD offsets before submission.
*   **Illegal Access:** If the hardware detects an out-of-bounds access to Tile SRAM, it raises a `FAULT` interrupt.
*   **Recovery:** The driver will capture the faulting state, reset the fabric, and signal the error to the calling process via `SIGBUS`.

## 3. Interrupt Model

The fabric utilizes a single active-high interrupt line:
1.  **DONE:** Signaled when a frame execution is complete.
2.  **ERROR:** Signaled on timeout, parity error (if enabled), or illegal memory access.

The driver's Interrupt Service Routine (ISR) utilizes a threaded IRQ handler to minimize latency and manage task wakeups efficiently.
