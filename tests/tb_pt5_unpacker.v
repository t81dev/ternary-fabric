`timescale 1ns/1ps

module tb_pt5_unpacker;
    reg [7:0] packed_in;
    wire [1:0] trit0, trit1, trit2, trit3, trit4;
    
    reg [7:0] mem [0:9];
    integer i;

    // Fixed: Using the correct port names from your grep output
    pt5_unpacker dut (
        .packed_byte(packed_in),
        .trit0(trit0), 
        .trit1(trit1), 
        .trit2(trit2), 
        .trit3(trit3), 
        .trit4(trit4)
    );

    initial begin
        // System tasks must stand alone
        $readmemh("tests/input_vectors.hex", mem);
        
        #1; // Brief delay to ensure memory is loaded
        
        for (i = 0; i < 10; i = i + 1) begin
            packed_in = mem[i];
            #10;
            $display("%0d", $signed({1'b0, trit0}) - 1);
            $display("%0d", $signed({1'b0, trit1}) - 1);
            $display("%0d", $signed({1'b0, trit2}) - 1);
            $display("%0d", $signed({1'b0, trit3}) - 1);
            $display("%0d", $signed({1'b0, trit4}) - 1);
        end
        $finish;
    end
endmodule