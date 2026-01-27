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

    // AXI4-Stream Slave Interface (DMA Loader)
    input  wire [31:0]            s_axis_tdata,
    input  wire                   s_axis_tvalid,
    output reg                    s_axis_tready,
    input  wire                   s_axis_tlast,

    // Final Accumulated Result Output
    output wire [(LANES*32)-1:0]  vector_results
);

    // Internal Interconnect Wires
    wire [ADDR_WIDTH-1:0] f_base_addr;
    wire [15:0]           f_depth;
    wire [7:0]            f_stride;
    wire [31:0]           f_exec_hints;
    wire [15:0]           f_lane_count;
    wire [LANES-1:0]      f_lane_mask;
    wire                  f_start;
    wire                  f_done;
    wire [31:0]           f_mem_addr;
    wire                  f_engine_en;

    // SRAM Loader Wires
    wire [11:0]           axi_sram_waddr;
    wire [23:0]           axi_sram_wdata;
    wire                  axi_sram_we_weight;
    wire                  axi_sram_we_input;

    reg  [11:0]           stream_sram_waddr;
    reg  [23:0]           stream_sram_wdata;
    reg                   stream_sram_we_weight;
    reg                   stream_sram_we_input;
    
    // AXI Response Wire
    wire [1:0]            s_axi_bresp;

    // Profiling Wires
    wire [(LANES*32)-1:0] f_skip_counts;
    wire [(LANES*32)-1:0] f_active_cycles;
    wire [LANES-1:0]      f_overflow_flags;
    wire [31:0]           f_cycle_count;
    wire [31:0]           f_utilization_count;
    reg  [31:0]           f_burst_wait_cycles;

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
        .fabric_lane_mask(f_lane_mask),
        .fabric_start(f_start),
        .fabric_done(f_done),
        .vector_results(vector_results),
        .skip_counts(f_skip_counts),
        .active_cycles(f_active_cycles),
        .overflow_flags(f_overflow_flags),
        .cycle_count(f_cycle_count),
        .utilization_count(f_utilization_count),
        .burst_wait_cycles(f_burst_wait_cycles),
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
        .exec_hints(f_exec_hints),
        .start_trigger(f_start), 
        .engine_enable(f_engine_en),
        .frame_done(f_done),
        .mem_addr(f_mem_addr),
        .mem_ready(1'b1) 
    );

    // 3. Dual-Bank SRAM
    // Multiplex AXI loader, Stream loader, and Frame Controller
    wire [11:0] weight_addr = f_start ? f_mem_addr[11:0] :
                             (s_axis_tvalid ? stream_sram_waddr : axi_sram_waddr);
    wire [11:0] input_addr  = f_start ? f_mem_addr[11:0] :
                             (s_axis_tvalid ? stream_sram_waddr : axi_sram_waddr);

    wire weight_we = f_start ? 1'b0 : (s_axis_tvalid ? stream_sram_we_weight : axi_sram_we_weight);
    wire input_we  = f_start ? 1'b0 : (s_axis_tvalid ? stream_sram_we_input : axi_sram_we_input);

    wire [23:0] sram_din = s_axis_tvalid ? stream_sram_wdata : axi_sram_wdata;

    ternary_sram_wrapper #(12, 24) sram (
        .clk    (clk),
        .addr_a (weight_addr),
        .we_a   (weight_we),
        .din_a  (sram_din),
        .dout_a (weight_bus),
        .addr_b (input_addr),
        .we_b   (input_we),
        .din_b  (sram_din),
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
        .lane_mask(f_lane_mask),
        .bus_weights(weight_trits),
        .bus_inputs(input_trits),
        .vector_out(vector_results),
        .skip_counts(f_skip_counts),
        .active_cycles(f_active_cycles),
        .overflow_flags(f_overflow_flags),
        .cycle_count(f_cycle_count),
        .utilization_count(f_utilization_count)
    );

    // AXI-Stream DMA State Machine
    reg [2:0] stream_state;
    localparam S_IDLE    = 3'd0;
    localparam S_HEADER  = 3'd1;
    localparam S_DATA    = 3'd2;
    localparam S_DONE    = 3'd3;

    reg [31:0] stream_base_addr;
    reg [31:0] stream_frame_len;
    reg [3:0]  header_count;

    always @(posedge clk or negedge reset_n) begin
        if (!reset_n) begin
            stream_state <= S_IDLE;
            s_axis_tready <= 1'b0;
            stream_sram_we_weight <= 1'b0;
            stream_sram_we_input <= 1'b0;
            f_burst_wait_cycles <= 0;
        end else begin
            stream_sram_we_weight <= 1'b0;
            stream_sram_we_input <= 1'b0;
            s_axis_tready <= 1'b1;

            case (stream_state)
                S_IDLE: begin
                    if (s_axis_tvalid) begin
                        stream_state <= S_HEADER;
                        header_count <= 0;
                    end
                end
                S_HEADER: begin
                    if (s_axis_tvalid) begin
                        case (header_count)
                            4'd0: stream_base_addr <= s_axis_tdata;
                            4'd1: stream_frame_len <= s_axis_tdata;
                            // Add more header fields as needed
                        endcase
                        header_count <= header_count + 1;
                        if (header_count == 4'd3) begin
                            stream_state <= S_DATA;
                            stream_sram_waddr <= stream_base_addr[11:0];
                        end
                    end
                end
                S_DATA: begin
                    if (s_axis_tvalid) begin
                        stream_sram_wdata <= s_axis_tdata[23:0];
                        // Simplified: write to both for now or use base_addr to distinguish
                        if (stream_base_addr[12])
                            stream_sram_we_input <= 1'b1;
                        else
                            stream_sram_we_weight <= 1'b1;

                        stream_sram_waddr <= stream_sram_waddr + 1;
                        if (s_axis_tlast) stream_state <= S_IDLE;
                    end else begin
                        f_burst_wait_cycles <= f_burst_wait_cycles + 1;
                    end
                end
            endcase
        end
    end

endmodule