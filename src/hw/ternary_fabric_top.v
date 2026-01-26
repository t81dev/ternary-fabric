/*
 * Ternary Fabric Top-Level Module
 * Integrates AXI control, Dual-Bank SRAM (24-bit), 
 * PT-5 Unpackers, and the Vector Engine.
 */
module ternary_fabric_top #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32,
    parameter LANES = 15 
)(
    // Global Clock and Reset
    input  wire                   clk,
    input  wire                   reset_n,

    // AXI4-Lite Slave Interface (Write Channel)
    input  wire [ADDR_WIDTH-1:0]  s_axi_awaddr,
    input  wire                   s_axi_awvalid,
    output wire                   s_axi_awready,
    input  wire [DATA_WIDTH-1:0]  s_axi_wdata,
    input  wire                   s_axi_wvalid,
    output wire                   s_axi_wready,
    output wire                   s_axi_bvalid,
    input  wire                   s_axi_bready,

    // AXI4-Lite Slave Interface (Read Channel)
    input  wire [ADDR_WIDTH-1:0]  s_axi_araddr,
    input  wire                   s_axi_arvalid,
    output wire                   s_axi_arready,
    output wire [DATA_WIDTH-1:0]  s_axi_rdata,
    output wire [1:0]             s_axi_rresp,
    output wire                   s_axi_rvalid,
    input  wire                   s_axi_rready,

    // Final Accumulated Result Output
    output wire [(LANES*32)-1:0]  vector_results
);

    // Internal Interconnect Wires
    wire [ADDR_WIDTH-1:0] f_base_addr;
    wire [15:0]           f_depth;
    wire [7:0]            f_stride;
    wire [31:0]           f_exec_hints;
    wire [15:0]           f_lane_count;
    wire                  f_start;
    wire                  f_done;
    wire [31:0]           f_mem_addr;
    wire                  f_engine_en;

    // SRAM Loader Wires
    wire [11:0]           axi_sram_waddr;
    wire [23:0]           axi_sram_wdata;
    wire                  axi_sram_we_weight;
    wire                  axi_sram_we_input;
    
    // AXI Response Wire
    wire [1:0]            s_axi_bresp;

    // Memory Wires
    wire [23:0]           weight_bus;
    wire [23:0]           input_bus;
    wire [(LANES*2)-1:0]  weight_trits;
    wire [(LANES*2)-1:0]  input_trits;

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
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        // Read Channel Connections
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        // Fabric Wires
        .fabric_base_addr(f_base_addr),
        .fabric_depth(f_depth),
        .fabric_stride(f_stride),
        .fabric_exec_hints(f_exec_hints),
        .fabric_lane_count(f_lane_count),
        .fabric_start(f_start),
        .fabric_done(f_done),
        .vector_results(vector_results),
        // SRAM Write Interface
        .sram_waddr(axi_sram_waddr),
        .sram_wdata(axi_sram_wdata),
        .sram_we_weight(axi_sram_we_weight),
        .sram_we_input(axi_sram_we_input)
    );

    // 2. Frame Controller
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
        .mem_ready(1'b1) 
    );

    // 3. Dual-Bank SRAM
    // Multiplex AXI loader and Frame Controller
    wire [11:0] weight_addr = f_start ? f_mem_addr[11:0] : axi_sram_waddr;
    wire [11:0] input_addr  = f_start ? f_mem_addr[11:0] : axi_sram_waddr;

    ternary_sram_wrapper #(12, 24) sram (
        .clk    (clk),
        .addr_a (weight_addr),
        .we_a   (axi_sram_we_weight),
        .din_a  (axi_sram_wdata),
        .dout_a (weight_bus),
        .addr_b (input_addr),
        .we_b   (axi_sram_we_input),
        .din_b  (axi_sram_wdata),
        .dout_b (input_bus)
    );

    // 4. PT-5 Unpackers
    genvar i;
    generate
        for (i = 0; i < 3; i = i + 1) begin : unpack_lanes
            pt5_unpacker u_weight (
                .packed_byte(weight_bus[i*8 +: 8]), 
                .trit0(weight_trits[(i*5+0)*2 +: 2]), 
                .trit1(weight_trits[(i*5+1)*2 +: 2]), 
                .trit2(weight_trits[(i*5+2)*2 +: 2]), 
                .trit3(weight_trits[(i*5+3)*2 +: 2]), 
                .trit4(weight_trits[(i*5+4)*2 +: 2])
            );
            pt5_unpacker u_input (
                .packed_byte(input_bus[i*8 +: 8]),  
                .trit0(input_trits[(i*5+0)*2 +: 2]), 
                .trit1(input_trits[(i*5+1)*2 +: 2]), 
                .trit2(input_trits[(i*5+2)*2 +: 2]), 
                .trit3(input_trits[(i*5+3)*2 +: 2]), 
                .trit4(input_trits[(i*5+4)*2 +: 2])
            );
        end
    endgenerate

    // 5. Vector Engine
    vector_engine #(LANES, 32) engine (
        .clk(clk),
        .reset(~reset_n),
        .enable(f_engine_en),
        .exec_hints(f_exec_hints),
        .lane_count(f_lane_count),
        .bus_weights(weight_trits),
        .bus_inputs(input_trits),
        .vector_out(vector_results)
    );

endmodule