/*
 * PT-5 Unpacker
 * Logic: Decodes an 8-bit PT-5 byte into 5 independent 2-bit ternary signals.
 * Encoding: 00 (0), 01 (+1), 10 (-1)
 * Formula: Byte = t0*1 + t1*3 + t2*9 + t3*27 + t4*81
 */
module pt5_unpacker (
    input  wire [7:0]  packed_byte,
    output wire [1:0]  trit0,
    output wire [1:0]  trit1,
    output wire [1:0]  trit2,
    output wire [1:0]  trit3,
    output wire [1:0]  trit4
);

    // Internal function to map 0, 1, 2 (unsigned) to our 2-bit signed encoding
    function [1:0] map_trit(input [7:0] val);
        case (val % 3)
            0: map_trit = 2'b00; // 0
            1: map_trit = 2'b01; // +1
            2: map_trit = 2'b10; // -1
            default: map_trit = 2'b00;
        endcase
    endfunction

    // Note: In high-performance ASIC, this would be a lookup table (LUT)
    // or optimized combinatorial logic rather than div/mod operators.
    assign trit0 = map_trit(packed_byte);
    assign trit1 = map_trit(packed_byte / 3);
    assign trit2 = map_trit(packed_byte / 9);
    assign trit3 = map_trit(packed_byte / 27);
    assign trit4 = map_trit(packed_byte / 81);

endmodule