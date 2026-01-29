/**
 * @file ternary_sram_dense.v
 * @brief Behavioral model for Dense Ternary SRAM with PT-20 packing.
 *
 * This module demonstrates 1.58-bit storage efficiency by packing 20 trits
 * (ternary digits) into a single 32-bit binary word.
 *
 * Theoretical Efficiency: 20 * log2(3) = 31.699 bits.
 * Physical Utilization: 31.699 / 32 = 99.06%.
 */

module ternary_sram_dense #(
    parameter ADDR_WIDTH = 12, // 4096 lines
    parameter TRITS_PER_LINE = 20
)(
    input  wire                   clk,
    input  wire                   reset_n,

    // Dense Access Port
    input  wire [ADDR_WIDTH-1:0]  addr,
    input  wire                   we,
    input  wire [19:0] [1:0]      trits_in,  // 2-bit per trit: 00=0, 01=+1, 10=-1
    output reg  [19:0] [1:0]      trits_out
);

    // Internal 32-bit binary storage
    reg [31:0] mem [0:(2**ADDR_WIDTH)-1];

    // Functions to pack/unpack between base-3 (balanced) and binary
    function [31:0] pack_trits(input [19:0][1:0] t);
        integer i;
        reg [63:0] val; // Use 64-bit for intermediate math to avoid overflow
        reg [1:0] raw;
        begin
            val = 0;
            for (i = 19; i >= 0; i = i - 1) begin
                raw = t[i];
                // Map: 00 (0) -> 1, 01 (+1) -> 2, 10 (-1) -> 0
                if (raw == 2'b00) val = val * 3 + 1;
                else if (raw == 2'b01) val = val * 3 + 2;
                else val = val * 3 + 0;
            end
            pack_trits = val[31:0];
        end
    endfunction

    function [19:0][1:0] unpack_trits(input [31:0] b);
        integer i;
        reg [63:0] val;
        reg [1:0] rem;
        begin
            val = b;
            for (i = 0; i < 20; i = i + 1) begin
                rem = val % 3;
                val = val / 3;
                // Map: 1 -> 00, 2 -> 01, 0 -> 10
                if (rem == 1) unpack_trits[i] = 2'b00;
                else if (rem == 2) unpack_trits[i] = 2'b01;
                else unpack_trits[i] = 2'b10;
            end
        end
    endfunction

    always @(posedge clk) begin
        if (!reset_n) begin
            // Reset logic (optional for SRAM)
        end else begin
            if (we) begin
                mem[addr] <= pack_trits(trits_in);
            end
            trits_out <= unpack_trits(mem[addr]);
        end
    end

    // Efficiency Annotations:
    // - Area: ~20% reduction compared to 2-bit-per-trit naive storage (40 bits -> 32 bits).
    // - Power: Reduced toggle rate on wide binary buses due to dense packing.
    // - Bandwidth: 1.25x increase in "Trit Bandwidth" for the same memory bus width.

endmodule
