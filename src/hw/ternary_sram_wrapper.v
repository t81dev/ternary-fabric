module ternary_sram_wrapper #(
    parameter ADDR_WIDTH = 12,
    parameter DATA_WIDTH = 24
)(
    input  wire                   clk,
    // Port A (Weights)
    input  wire                   en_a,
    input  wire [ADDR_WIDTH-1:0]  addr_a,
    input  wire                   we_a,
    input  wire [DATA_WIDTH-1:0]  din_a,
    output reg  [DATA_WIDTH-1:0]  dout_a,
    
    // Port B (Inputs)
    input  wire                   en_b,
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
        // Use full paths or ensure these files are available during synthesis/sim
        $readmemh("weights.mem", bank_weight);
        $readmemh("inputs.mem",  bank_input);
        
        // Debug: Print the first address to see if it's still 0 or loaded with 0x55
        $display("🔍 SRAM Check: Addr 0 | Weight: %h | Input: %h", 
                  bank_weight[0], bank_input[0]);
    end

    // ASIC-ready behavioral model with enable hooks for power gating simulation
    always @(posedge clk) begin
        // Weight Port (A)
        if (en_a) begin
            if (we_a) bank_weight[addr_a] <= din_a;
            dout_a <= bank_weight[addr_a];
        end

        // Input Port (B)
        if (en_b) begin
            if (we_b) bank_input[addr_b] <= din_b;
            dout_b <= bank_input[addr_b];
        end
    end

endmodule