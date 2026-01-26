module ternary_sram_wrapper #(
    parameter ADDR_WIDTH = 12,
    parameter DATA_WIDTH = 24
)(
    input  wire                   clk,
    // Port A (Weights)
    input  wire [ADDR_WIDTH-1:0]  addr_a,
    input  wire                   we_a,
    input  wire [DATA_WIDTH-1:0]  din_a,
    output reg  [DATA_WIDTH-1:0]  dout_a,
    
    // Port B (Inputs)
    input  wire [ADDR_WIDTH-1:0]  addr_b,
    input  wire                   we_b,
    input  wire [DATA_WIDTH-1:0]  din_b,
    output reg  [DATA_WIDTH-1:0]  dout_b
);

    // Internal Memory Arrays
    reg [DATA_WIDTH-1:0] bank_weight [0:(2**ADDR_WIDTH)-1];
    reg [DATA_WIDTH-1:0] bank_input  [0:(2**ADDR_WIDTH)-1];

    // Initialize memory with pattern data
    initial begin
        $display("💾 Attempting to load SRAM banks...");
        $readmemh("weights.mem", bank_weight);
        $readmemh("inputs.mem",  bank_input);
        
        // Debug: Print the first address to see if it's still 0 or loaded with 0x55
        $display("🔍 SRAM Check: Addr 0 | Weight: %h | Input: %h", 
                  bank_weight[0], bank_input[0]);
    end

    always @(posedge clk) begin
        // Weight Port (A)
        if (we_a) bank_weight[addr_a] <= din_a;
        dout_a <= bank_weight[addr_a];

        // Input Port (B)
        if (we_b) bank_input[addr_b] <= din_b;
        dout_b <= bank_input[addr_b];
    end

endmodule