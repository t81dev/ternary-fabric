# Hardware Verification Report

This living report captures the FPGA verification data required to graduate Phase 10 from the “Mock” driver state, proving the Silicon Reality milestone (Q3 2026) and enabling downstream Track updates.

## 1. Summary
- **Verification window:** TBD (waiting for XC7Z020/XC7Z045 access)
- **Hardware target:** XC7Z020 / XC7Z045 (specify board, bitstream hash once built)
- **Driver path:** `/dev/tfmbs` via `libtfmbs_device.so` and `tfmbs_driver_mock.c`
- **Current status:** Hardware verification remains blocked because the physical FPGA testbeds are unavailable; the compiler/MLIR stack (dialect, fusion telemetry, regression scripts) is ready and will provide the telemetry dictionary once DMA/IOCTL validation resumes.
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

## 7. Preparation for Hardware Reconnection
- **Synthesis command:** rerun the platform flow (e.g., `make -C src/hw synth-fabric TARGET=xc7z020 PLATFORM=zc706`) to regenerate `ternary_fabric_top.bit`, capture LUT/FF/BRAM counts, and compute the checksum/hash for the artifact you will flash.
- **Flash & bring-up:** document the Vivado/SDK steps that load the bitstream (for example `vivado -mode batch -source scripts/flash_ternary.tcl -tclargs /path/to/ternary_fabric_top.bit`) along with board-specific boot sequences or PRC knobs required on the Zynq side.
- **Driver run plan:** once the board is powered, execute `tests/test_dma_driver` followed by the mock driver’s submit routine (e.g., `./bin/tfmbs_driver_mock --submit-dma`), observe `[TFMBS-Driver] DMA IRQ` logs, and verify descriptor head/tail updates; capture any env vars like `TFMBS_DMA_RING=512` and `TFMBS_VERBOSE=1` you used.
- **Telemetry re-sync:** rerun `python tools/torch_to_tfmbs.py && python tests/mlir/run_tfmbs_to_linalg.py --mlir=tests/mlir/tfmbs_to_linalg.mlir` with the shared MLIR build to regenerate telemetry dictionaries, then diff the CSV output (e.g., `diag/telemetry_YYYYMMDD.csv`) against emulator baselines using `tools/telemetry_diff.py` before and after the hardware run.

## Attachments
- Reference docs: `docs/ROADMAP.md`, `docs/FPGA_VERIFICATION_CHECKLIST.md`, `include/uapi_tfmbs.h`, `src/tfmbs_driver_mock.c`.  
- Logs & transcripts: store zipped logs and trace files with timestamps.
