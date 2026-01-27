/*
 * Ternary Fabric Top-Level Module
 * Integrates AXI control, Dual-Bank SRAM (24-bit), 
 * PT-5 Unpackers, and the Vector Engine.
 */
module ternary_fabric_top #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32,
    parameter LANES = 15,
    parameter NUM_TILES = 4
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

    // Final Accumulated Result Output (Wide vector for all tiles)
    output wire [(NUM_TILES*LANES*32)-1:0]  vector_results
);

    // Internal Interconnect Wires
    wire [ADDR_WIDTH-1:0] f_base_addr;
    wire [15:0]           f_depth;
    wire [7:0]            f_stride;
    wire [31:0]           f_exec_hints;
    wire [15:0]           f_lane_count;
    wire [LANES-1:0]      f_lane_mask;
    wire [NUM_TILES-1:0]  f_tile_mask;
    wire                  f_start;
    wire                  f_done;
    wire [31:0]           f_mem_addr;
    wire                  f_engine_en;

    // SRAM Loader Wires
    wire [11:0]           axi_sram_waddr;
    wire [23:0]           axi_sram_wdata;
    wire [NUM_TILES-1:0]  axi_sram_we_weight;
    wire [NUM_TILES-1:0]  axi_sram_we_input;
    wire                  axi_sram_we_broadcast;

    reg  [11:0]           stream_sram_waddr;
    reg  [23:0]           stream_sram_wdata;
    reg  [NUM_TILES-1:0]  stream_sram_we_weight;
    reg  [NUM_TILES-1:0]  stream_sram_we_input;
    reg                   stream_sram_we_broadcast;
    
    // AXI Response Wire
    wire [1:0]            s_axi_bresp;

    // Profiling Wires
    wire [(NUM_TILES*LANES*32)-1:0] f_skip_counts;
    wire [(NUM_TILES*LANES*32)-1:0] f_active_cycles;
    wire [(NUM_TILES*LANES)-1:0]    f_overflow_flags;
    wire [31:0]                     f_cycle_count;
    wire [31:0]                     f_utilization_count;
    reg  [31:0]                     f_burst_wait_cycles;

    // 1. AXI Control Plane
    axi_interconnect_v1 #(ADDR_WIDTH, DATA_WIDTH, NUM_TILES) axi_slave (
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
        .fabric_tile_mask(f_tile_mask),
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
        .sram_we_input(axi_sram_we_input),
        .sram_we_broadcast(axi_sram_we_broadcast)
    );

    // 2. Frame Controller (Shared for Lock-Step)
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

    // Common SRAM Data
    wire [23:0] sram_din = s_axis_tvalid ? stream_sram_wdata : axi_sram_wdata;
    wire [31:0] tile_cycle_counts [NUM_TILES-1:0];
    wire [31:0] tile_util_counts [NUM_TILES-1:0];

    // 3. Multi-Tile Engine Matrix
    genvar t;
    generate
        for (t = 0; t < NUM_TILES; t = t + 1) begin : tile_gen
            wire [23:0] t_weight_bus;
            wire [23:0] t_input_bus;
            wire [(LANES*2)-1:0] t_weight_trits;
            wire [(LANES*2)-1:0] t_input_trits;

            wire [11:0] t_addr = f_start ? f_mem_addr[11:0] :
                                (s_axis_tvalid ? stream_sram_waddr : axi_sram_waddr);

            wire t_we_weight = f_start ? 1'b0 :
                              (s_axis_tvalid ? (stream_sram_we_weight[t] || stream_sram_we_broadcast) :
                                               (axi_sram_we_weight[t] || axi_sram_we_broadcast));
            wire t_we_input  = f_start ? 1'b0 :
                              (s_axis_tvalid ? stream_sram_we_input[t] : axi_sram_we_input[t]);

            // Private SRAM per Tile
            ternary_sram_wrapper #(12, 24) sram_inst (
                .clk    (clk),
                .en_a   (t_we_weight || f_engine_en),
                .addr_a (t_addr),
                .we_a   (t_we_weight),
                .din_a  (sram_din),
                .dout_a (t_weight_bus),
                .en_b   (t_we_input || f_engine_en),
                .addr_b (t_addr),
                .we_b   (t_we_input),
                .din_b  (sram_din),
                .dout_b (t_input_bus)
            );

            // PT-5 Unpackers per Tile
            genvar i;
            for (i = 0; i < 3; i = i + 1) begin : unpack_lanes
                pt5_unpacker u_weight (
                    .packed_byte(t_weight_bus[i*8 +: 8]),
                    .trit0(t_weight_trits[(i*5+0)*2 +: 2]),
                    .trit1(t_weight_trits[(i*5+1)*2 +: 2]),
                    .trit2(t_weight_trits[(i*5+2)*2 +: 2]),
                    .trit3(t_weight_trits[(i*5+3)*2 +: 2]),
                    .trit4(t_weight_trits[(i*5+4)*2 +: 2])
                );
                pt5_unpacker u_input (
                    .packed_byte(t_input_bus[i*8 +: 8]),
                    .trit0(t_input_trits[(i*5+0)*2 +: 2]),
                    .trit1(t_input_trits[(i*5+1)*2 +: 2]),
                    .trit2(t_input_trits[(i*5+2)*2 +: 2]),
                    .trit3(t_input_trits[(i*5+3)*2 +: 2]),
                    .trit4(t_input_trits[(i*5+4)*2 +: 2])
                );
            end

            // Vector Engine per Tile
            vector_engine #(LANES, 32) engine_inst (
                .clk(clk),
                .reset(~reset_n),
                .enable(f_engine_en && f_tile_mask[t]),
                .exec_hints(f_exec_hints),
                .lane_count(f_lane_count),
                .lane_mask(f_lane_mask),
                .bus_weights(t_weight_trits),
                .bus_inputs(t_input_trits),
                .vector_out(vector_results[t*LANES*32 +: LANES*32]),
                .skip_counts(f_skip_counts[t*LANES*32 +: LANES*32]),
                .active_cycles(f_active_cycles[t*LANES*32 +: LANES*32]),
                .overflow_flags(f_overflow_flags[t*LANES +: LANES]),
                .cycle_count(tile_cycle_counts[t]),
                .utilization_count(tile_util_counts[t])
            );
        end
    endgenerate

    // Profiling Aggregation
    assign f_cycle_count = (NUM_TILES > 0) ? tile_cycle_counts[0] : 32'd0;
    reg [31:0] total_util;
    integer ut;
    always @(*) begin
        total_util = 0;
        for (ut = 0; ut < NUM_TILES; ut = ut + 1) begin
            total_util = total_util + tile_util_counts[ut];
        end
    end
    assign f_utilization_count = total_util;

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

                        stream_sram_we_weight <= 0;
                        stream_sram_we_input <= 0;
                        stream_sram_we_broadcast <= 0;

                        if (stream_base_addr[15:12] == 4'h9) begin
                            stream_sram_we_broadcast <= 1'b1;
                        end else if (stream_base_addr[15:13] < 4 && stream_base_addr[12]) begin
                            stream_sram_we_weight[stream_base_addr[15:13]] <= 1'b1;
                        end else if (stream_base_addr[15:13] > 0 && stream_base_addr[15:13] <= 4 && !stream_base_addr[12]) begin
                            stream_sram_we_input[stream_base_addr[15:13]-1] <= 1'b1;
                        end

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