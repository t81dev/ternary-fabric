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
    input  wire [31:0] exec_hints,
    input  wire        enable,
    output reg  [31:0] accumulator
);

    wire [7:0] op_mode      = exec_hints[7:0];
    wire       zero_skip_en = exec_hints[17];

    // Combinatorial "Intermediate" Result (Free Negation Logic)
    // 00=0, 01=+1, 10=-1
    wire [1:0] out_trit;
    assign out_trit = (weight == 2'b10) ? (trit_in == 2'b01 ? 2'b10 :
                                          trit_in == 2'b10 ? 2'b01 : 2'b00) :
                      (weight == 2'b01) ? trit_in :
                      2'b00;

    wire signed [1:0] product; 
    assign product = (out_trit == 2'b10) ? 2'sb11 : {1'b0, out_trit[0]};

    // Zero-Skip Logic
    wire skip_cycle;
    assign skip_cycle = zero_skip_en && (weight == 2'b00 || trit_in == 2'b00);

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            accumulator <= 32'd0;
        end else if (enable) begin
            case (op_mode)
                8'h01, 8'h06: begin // DOT, TGEMM
                    if (!skip_cycle)
                        accumulator <= accumulator + {{30{product[1]}}, product};
                end
                8'h03: begin // MUL
                    accumulator <= {{30{product[1]}}, product};
                end
                default: accumulator <= accumulator;
            endcase
        end
    end
endmodule