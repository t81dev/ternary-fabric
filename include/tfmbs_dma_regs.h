#ifndef TFMBS_DMA_REGS_H
#define TFMBS_DMA_REGS_H

/*
 * Canonical register offsets for the Ternary Fabric AXI4-Lite interface.
 * These values mirror the control/status table in docs/04_MEMORY_MAP.md so
 * software and verification tooling can stay synchronized.
 */

#define TFMBS_REG_CONTROL     0x00
#define TFMBS_REG_STATUS      0x04
#define TFMBS_REG_BASE_ADDR   0x08
#define TFMBS_REG_DEPTH       0x0C
#define TFMBS_REG_STRIDE      0x10
#define TFMBS_REG_EXEC_HINTS  0x14
#define TFMBS_REG_LANE_COUNT  0x18
#define TFMBS_REG_LANE_MASK   0x1C

#define TFMBS_REG_CYCLES      0x20
#define TFMBS_REG_UTILIZATION 0x24
#define TFMBS_REG_DMA_LATENCY 0x68

/* Tile counters start at 0x28 with 15 skip counters per tile. */
#define TFMBS_TILE0_SKIP_START   0x28
#define TFMBS_TILE0_OVERFLOW     0x6C
#define TFMBS_TILE0_ACTIVE_START 0x70
#define TFMBS_TILE1_SKIP_START   0x228
#define TFMBS_TILE1_OVERFLOW     0x26C
#define TFMBS_TILE1_ACTIVE_START 0x270
#define TFMBS_TILE2_SKIP_START   0x328
#define TFMBS_TILE2_OVERFLOW     0x36C
#define TFMBS_TILE2_ACTIVE_START 0x370
#define TFMBS_TILE3_SKIP_START   0x428
#define TFMBS_TILE3_OVERFLOW     0x46C
#define TFMBS_TILE3_ACTIVE_START 0x470

#define TFMBS_BROADCAST_OFFSET  0x9000

#endif /* TFMBS_DMA_REGS_H */
