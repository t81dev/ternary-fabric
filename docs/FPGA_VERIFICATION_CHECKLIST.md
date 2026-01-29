# FPGA Verification Checklist

Based on the ROADMAP Phase 22 callout in `docs/ROADMAP.md` (Hardware Sovereignty track), this checklist captures the immediate verification work needed to move beyond the mock driver and validate the bitstream + DMA/IOCTL path on the XC7Z020/XC7Z045 boards.

## 1. Source Inventory
- RTL files under `src/hw/` (`ternary_fabric_top.v`, `vector_engine.v`, `ternary_lane_alu.v`, `frame_controller.v`, `axi_interconnect_v1.v`, `ternary_sram_wrapper.v`, `ternary_sram_dense.v`, PT-5 support, and TBs) define the tile/lanes and DMA path that must map onto the PCIe/ioctl handshake.
- Mock driver (`src/tfmbs_driver_mock.c`) and UAPI (`include/uapi_tfmbs.h`) describe the current `/dev/tfmbs` behaviors, including `TFMBS_IOC_SUBMIT_DMA` descriptors that route payloads into `emu_*` helpers and the ring buffer logic.
- Telemetry expectations (sparsity, residency, cycle counts) are surfaced via `tfmbs_ioc_metrics_t` and have been implemented in `src/fabric_emulator.c` derivatives, which will need parity with hardware runs.
- The canonical DMA register map is defined in `include/tfmbs_dma_regs.h`; run `tools/validate_register_map.py` before hardware testing to ensure the header still matches `docs/04_MEMORY_MAP.md`.

## 2. Bitstream & DMA Integration
1. Confirm the RTL hierarchy in `src/hw/ternary_fabric_top.v` instantiates the DMA interface noted in the roadmaps (producer-consumer ring buffer, asynchronous interrupts on descriptor completion, support for `TFMBS_IOC_SUBMIT_DMA` flags).
2. Synthesize the current RTL (use existing `Makefile`/build scripts if available) targeting XC7Z020/XC7Z045 and capture timing/power reports. Document whether the generated bitstream matches the `ternary_fabric_top` IO interface used by `libtfmbs_device.so`.
3. Update or draft the DMA descriptor placement so the driver’s ring (`src/tfmbs_driver_mock.c`) matches the RTL-level register map (addresses for descriptor head/tail, control registers, IRQ lines).
4. Program the FPGA with the synthesized bitstream, and run the mock driver’s IOCTL flow against `/dev/tfmbs` to ensure that submitting DMA descriptors results in the expected host-device transfers (monitor the host log for `[TFMBS-Driver] DMA IRQ` and check descriptor counts).

## 3. Hardware-in-the-Loop Validation
- **Driver validation:** run `tests/test_dma_driver.c` (or similar) using the real kernel module or the mock driver wrapped around the hardware to ensure `TFMBS_IOC_SUBMIT_DMA` returns 0 and the ring does not overflow under high load.
- **Telemetry capture:** execute standard workloads once the FPGA is connected, capturing metrics that match `tfmbs_ioc_metrics_t` fields (zero_skips, residency hits/misses, fallback/offload counts). Compare these against emulator baselines to verify accuracy.
- **Benchmarks:** use existing benchmark scripts (e.g., anything under `benchmarks/` or `tests/`) to load 7B+ layers and run the DMA-enabled kernels, measuring throughput and energy to confirm the >50× efficiency target for Q3 2026 in the roadmap (`docs/ROADMAP.md:188-190`).

## 4. Acceptance Criteria & Documentation
- Define pass/fail thresholds: ring buffer throughput (descriptor/s), DMA latency, telemetry fidelity, default tile count (4 tiles × 15 lanes) as in the mock driver.
- Log results in a running report (e.g., `docs/hardware_verification_report.md`) with sections for synthesis, driver integration, telemetry comparison, and benchmark output.
- Update `docs/ROADMAP.md` to note Phase 10 moving past “Mock” once the hardware driver demonstrates correct IOCTL/DMA behavior; cite the verification report.

## 5. Next Steps (after verification)
1. Share telemetry findings with the compiler/MLIR track so they can refine sensitivity heuristics based on hardware behavior (`docs/ROADMAP.md:156-170`).
2. Feed DMA/performance numbers to the scale/RDMA track to ensure distributed orchestration simulations stay aligned with real hardware (`docs/ROADMAP.md:177-182`).
