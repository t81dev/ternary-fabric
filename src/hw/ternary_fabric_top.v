/*
 * Ternary Fabric Top-Level Module
 * Integrates TFMBS-compliant AXI interface, Dual-Bank SRAM, 
 * PT-5 Unpackers, and the Vector Engine.
 */
module ternary_fabric_top #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32,
    parameter LANES = 15 // TFM-compliant 15-lane SIMD
)(
    // Global Clock and Reset
    input  wire                   clk,
    input  wire                   reset_n,

    // AXI4-Lite Slave Interface (Control/Status)
    input  wire [ADDR_WIDTH-1:0]  s_axi_awaddr,
    input  wire                   s_axi_awvalid,
    output wire                   s_axi_awready,
    input  wire [DATA_WIDTH-1:0]  s_axi_wdata,
    input  wire                   s_axi_wvalid,
    output wire                   s_axi_wready,
    output wire                   s_axi_bvalid,
    input  wire                   s_axi_bready,

    // Final Accumulated Result Output (for Host Readback)
    output wire [(LANES*32)-1:0]  vector_results
);

    // Internal Interconnect Wires
    wire [ADDR_WIDTH-1:0] f_base_addr;
    wire [15:0]           f_depth;
    wire [7:0]            f_stride;
    wire                  f_start;
    wire                  f_done;
    wire [31:0]           f_mem_addr;
    wire                  f_engine_en;

    // Memory Wires
    wire [7:0] weight_byte, input_byte;
    wire [(LANES*2)-1:0] weight_trits, input_trits;

    // 1. AXI Control Plane
    axi_interconnect_v1 #(ADDR_WIDTH, DATA_WIDTH) axi_slave (
        .s_axi_aclk(clk),
        .s_axi_aresetn(reset_n),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .fabric_base_addr(f_base_addr),
        .fabric_depth(f_depth),
        .fabric_stride(f_stride),
        .fabric_start(f_start),
        .fabric_done(f_done)
    );

    // 2. Frame Controller (The "Brain")
    frame_controller #(ADDR_WIDTH, LANES) controller (
        .clk(clk),
        .reset(~reset_n),
        .base_addr(f_base_addr),
        .frame_depth(f_depth),
        .lane_stride(f_stride),
        .start_trigger(f_start),
        .engine_enable(f_engine_en),
        .frame_done(f_done),
        .mem_addr(f_mem_addr),
        .mem_ready(1'b1) // SRAM is always ready
    );

    // 3. Dual-Bank SRAM (The "Locker")
    ternary_sram_wrapper #(12, 8) sram (
        .clk(clk),
        .addr_a(f_mem_addr[11:0]), .we_a(1'b0), .din_a(8'b0), .dout_a(weight_byte),
        .addr_b(f_mem_addr[11:0]), .we_b(1'b0), .din_b(8'b0), .dout_b(input_byte)
    );

    // 4. PT-5 Unpackers (Hydration Logic)
    // Using 3 unpackers per bank to feed 15 lanes (5 trits per byte)
    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : unpack_lanes
            pt5_unpacker u_weight (.packed_byte(weight_byte), .trit0(weight_trits[i*10+0+:2]), .trit1(weight_trits[i*10+2+:2]), .trit2(weight_trits[i*10+4+:2]), .trit3(weight_trits[i*10+6+:2]), .trit4(weight_trits[i*10+8+:2]));
            pt5_unpacker u_input  (.packed_byte(input_byte),  .trit0(input_trits[i*10+0+:2]),  .trit1(input_trits[i*10+2+:2]),  .trit2(input_trits[i*10+4+:2]),  .trit3(input_trits[i*10+6+:2]),  .trit4(input_trits[i*10+8+:2]));
        end
    endgenerate

    // 5. Vector Engine (The "Muscle")
    vector_engine #(LANES, 32) engine (
        .clk(clk),
        .reset(~reset_n),
        .enable(f_engine_en),
        .op_mode(2'b01), // Default to DOT product
        .bus_weights(weight_trits),
        .bus_inputs(input_trits),
        .vector_out(vector_results)
    );

endmodule