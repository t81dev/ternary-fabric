`timescale 1ns/1ps

module tb_vector_system();
    // Clock/Reset
    reg clk;
    reg reset;
    
    // Test Parameters
    localparam LANES = 15;
    localparam ADDR_W = 32;
    
    // Signals
    reg  start_trigger;
    wire engine_enable;
    wire frame_done;
    wire [ADDR_W-1:0] mem_addr;
    reg  [7:0] mock_mem [0:255]; // 256 bytes of mock SRAM
    wire [119:0] bus_w_trits, bus_i_trits; // Unpacked data wires
    wire [(LANES*32)-1:0] vector_out;

    // 1. Instantiate Controller
    frame_controller #(ADDR_W, LANES) dut_ctrl (
        .clk(clk), .reset(reset),
        .base_addr(32'd0), .frame_depth(16'd10), .lane_stride(8'd1),
        .start_trigger(start_trigger), .engine_enable(engine_enable),
        .frame_done(frame_done), .mem_addr(mem_addr), .mem_ready(1'b1)
    );

    // 2. Mock Memory Feed (Simplified Unpacker Integration)
    // In a full design, the unpacker sits here. 
    // For this TB, we simulate the output of the unpackers.
    assign bus_w_trits = {LANES{2'b01}}; // Weights = All +1
    assign bus_i_trits = {LANES{2'b01}}; // Inputs = All +1 (Expected: Accumulate +1 per cycle)

    // 3. Instantiate Vector Engine
    vector_engine #(LANES, 32) dut_engine (
        .clk(clk), .reset(reset),
        .enable(engine_enable), .op_mode(2'b00),
        .bus_weights(bus_w_trits), .bus_inputs(bus_i_trits),
        .vector_out(vector_out)
    );

    // Clock Gen
    always #5 clk = ~clk;

    initial begin
        // Initialize
        clk = 0; reset = 1; start_trigger = 0;
        #20 reset = 0;
        
        // Trigger Frame Processing
        #10 start_trigger = 1;
        #10 start_trigger = 0;
        
        // Wait for completion
        wait(frame_done);
        
        // Check Lane 0 result: 1 * 1 * 10 cycles = 10
        $display("Lane 0 Result: %d", vector_out[31:0]);
        if (vector_out[31:0] == 32'd10) $display("TEST PASSED: Accumulation Correct");
        else $display("TEST FAILED: Expected 10");
        
        $finish;
    end
endmodule