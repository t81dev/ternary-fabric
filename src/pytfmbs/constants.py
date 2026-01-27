# TFMBS Fabric Memory Map Constants
SRAM_BANK_A_OFFSET = 0x1000
SRAM_BANK_B_OFFSET = 0x2000
SRAM_TILE_STRIDE = 0x2000

# Kernel Selection & Execution Hints
KERNEL_T_GEMM = 0x01
HINT_ZERO_SKIP = (1 << 17)

# Multi-Tile Configuration
MAX_TILES = 4
LANES_PER_TILE = 15
