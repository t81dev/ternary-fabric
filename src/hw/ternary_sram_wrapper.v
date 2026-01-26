/*
 * Ternary SRAM Wrapper (Dual-Bank)
 * Provides simultaneous access to Weight and Input frames.
 * Compliant with TFMBS PT-5 packing.
 */
module ternary_sram_wrapper #(
    parameter ADDR_WIDTH = 12, // 4KB of Ternary Space
    parameter DATA_WIDTH = 8   // 8-bit (PT-5) bytes
)(
    input  wire                   clk,
    
    // Bank A: Weights
    input  wire [ADDR_WIDTH-1:0]  addr_a,
    input  wire                   we_a,
    input  wire [DATA_WIDTH-1:0]  din_a,
    output reg  [DATA_WIDTH-1:0]  dout_a,
    
    // Bank B: Inputs/Activations
    input  wire [ADDR_WIDTH-1:0]  addr_b,
    input  wire                   we_b,
    input  wire [DATA_WIDTH-1:0]  din_b,
    output reg  [DATA_WIDTH-1:0]  dout_b
);

    // Physical Memory Arrays
    reg [DATA_WIDTH-1:0] bank_weight [0:(2**ADDR_WIDTH)-1];
    reg [DATA_WIDTH-1:0] bank_input  [0:(2**ADDR_WIDTH)-1];

    always @(posedge clk) begin
        // Port A (Weights)
        if (we_a) bank_weight[addr_a] <= din_a;
        dout_a <= bank_weight[addr_a];
        
        // Port B (Inputs)
        if (we_b) bank_input[addr_b] <= din_b;
        dout_b <= bank_input[addr_b];
    end

endmodule