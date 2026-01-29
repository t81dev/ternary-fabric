# Hardware Verification Report

This living report captures the FPGA verification data required to graduate Phase 10 from the “Mock” driver state, proving the Silicon Reality milestone (Q3 2026) and enabling downstream Track updates.

## 1. Summary
- **Verification window:** YYYY-MM-DD to YYYY-MM-DD  
- **Hardware target:** XC7Z020 / XC7Z045 (specify board, bitstream hash)  
- **Driver path:** `/dev/tfmbs` via `libtfmbs_device.so` and `tfmbs_driver_mock.c`  
- **Key outcome:** [ ] DMA topology validated  [ ] Telemetry matches emulator  [ ] >50× efficiency proven

## 2. RTL Synthesis & Bitstream
- **Sources:** list `src/hw/` files used.  
- **Toolchain & version:** e.g., `Xilinx Vivado 2023.1` or `Yosys + NextPNR`.  
- **Implementation results:** timing (ns), power (W), LUT/FF/BRAM counts.  
- **Register map:** enumerate DMA register offsets/head/tail/irq fields; confirm against `include/tfmbs_dma_regs.h` and `docs/04_MEMORY_MAP.md` (use `tools/validate_register_map.py` to stay synced).
- **Bitstream hash & deploy steps:** store commit hash, checksum, and method for flashing board.

## 3. Driver Integration
- **IOCTL coverage:** `TFMBS_IOC_ALLOC`, `TFMBS_IOC_MEMCPY_*`, `TFMBS_IOC_SUBMIT`, `TFMBS_IOC_SUBMIT_DMA`, `TFMBS_IOC_WAIT`, `TFMBS_IOC_GET_*`.  
- **TLS ring buffer state transitions:** describe head/tail updates, interrupt handling, and error cases observed during tests.  
- **Mock driver vs. real registers:** any discrepancies between `src/tfmbs_driver_mock.c` simulation and hardware behavior.  
- **Scripts run:** list commands (e.g., `./bin/test_phase10`, `tests/test_dma_driver`). Include environment variables if set (`FABRIC_SHORT_CIRCUIT`, etc.).

## 4. Telemetry & Metrics
- **Workloads executed:** detail models (7B Llama layers, bench scripts) plus dataset info.  
- **Collected telemetry fields:** zero_skips, residency hits/misses, active_ops, fallback/offload counts, cycle counts.  
- **Comparison to emulator baselines:** table with percent delta for key fields; note any anomalies.  
- **Pass/fail thresholds:** e.g., ring throughput >X descriptors/s, telemetry within ±Y% of emulator.

## 5. Benchmark Results
- **Benchmarks run:** describe `benchmarks/` or `tests/bench_top.cpp` executions.  
- **Measured throughput:** ops/s, latency per layer, energy/power if available.  
- **Efficiency target:** confirm whether >50× CPU baseline metric is met; include raw numbers.  
- **Failures/Retests:** note any retries due to hardware issues and root cause.

## 6. Issues & Follow-up
- **Open hardware issues:** maintain list with severity, steps to reproduce, component owner, status.  
- **Next actions:** e.g., refine driver handshake, adjust telemetry scaling, rerun `benchmarks/sparse_stress.c`, update ROADMAP.  
- **Stakeholder alignment:** who needs the results (compiler/MLIR team, RDMA track leads) and how the findings will be shared.

## Attachments
- Reference docs: `docs/ROADMAP.md`, `docs/FPGA_VERIFICATION_CHECKLIST.md`, `include/uapi_tfmbs.h`, `src/tfmbs_driver_mock.c`.  
- Logs & transcripts: store zipped logs and trace files with timestamps.
