/*
 * TPE: Ternary Processing Element (Lane ALU)
 * Performs: Accumulator = Accumulator + (Weight * Input)
 * Logic: Ternary {-1, 0, 1} Weights/Inputs, 32-bit Binary Accumulator.
 * Encoding: 2-bit 'Simple' encoding (00: 0, 01: +1, 10: -1)
 */

module ternary_lane_alu (
    input  wire        clk,
    input  wire        reset,
    input  wire [1:0]  weight,
    input  wire [1:0]  trit_in,
    input  wire [2:0]  op_mode,     // Map to tfmbs_kernel_t
    input  wire        enable,
    output reg  [31:0] accumulator
);

    // Combinatorial "Intermediate" Result
    wire signed [1:0] product; 
    // Logic: 01*01=01, 10*10=01, 01*10=10, 10*01=10, else 00
    assign product = (weight == 2'b00 || trit_in == 2'b00) ? 2'sb0 :
                     (weight == trit_in) ? 2'sb01 : 2'sb11; // 11 is -1 in 2's complement

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            accumulator <= 32'd0;
        end else if (enable) begin
            case (op_mode)
                3'h1: // TFMBS_KERNEL_DOT: Accumulate
                    accumulator <= accumulator + {{30{product[1]}}, product};
                3'h3: // TFMBS_KERNEL_MUL: Direct Output (No accumulation)
                    accumulator <= {{30{product[1]}}, product};
                default: accumulator <= accumulator;
            endcase
        end
    end
endmodule