module axi_interconnect_v1 #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32,
    parameter NUM_TILES = 4
)(
    input  wire                   s_axi_aclk,
    input  wire                   s_axi_aresetn,

    // Write Address Channel
    input  wire [ADDR_WIDTH-1:0]  s_axi_awaddr,
    input  wire                   s_axi_awvalid,
    output wire                   s_axi_awready,
    
    // Write Data Channel
    input  wire [DATA_WIDTH-1:0]  s_axi_wdata,
    input  wire                   s_axi_wvalid,
    output wire                   s_axi_wready,
    
    // Write Response Channel
    output wire [1:0]             s_axi_bresp,
    output wire                   s_axi_bvalid,
    input  wire                   s_axi_bready,

    // Read Address Channel
    input  wire [ADDR_WIDTH-1:0]  s_axi_araddr,
    input  wire                   s_axi_arvalid,
    output wire                   s_axi_arready,

    // Read Data Channel
    output reg [DATA_WIDTH-1:0]   s_axi_rdata,
    output wire [1:0]             s_axi_rresp,
    output reg                    s_axi_rvalid,
    input  wire                   s_axi_rready,

    // Fabric Signals
    output reg [ADDR_WIDTH-1:0]   fabric_base_addr,
    output reg [15:0]             fabric_depth,
    output reg [7:0]              fabric_stride,
    output reg [31:0]             fabric_exec_hints,
    output reg [15:0]             fabric_lane_count,
    output reg [14:0]             fabric_lane_mask,
    output reg [NUM_TILES-1:0]    fabric_tile_mask,
    output reg                    fabric_start,
    input  wire                   fabric_done,

    // Vector Results & Profiling Input (Multi-tile)
    input  wire [(NUM_TILES*15*32)-1:0] vector_results,
    input  wire [(NUM_TILES*15*32)-1:0] skip_counts,
    input  wire [(NUM_TILES*15*32)-1:0] active_cycles,
    input  wire [(NUM_TILES*15)-1:0]    overflow_flags,
    input  wire [31:0]            cycle_count,
    input  wire [31:0]            utilization_count,
    input  wire [31:0]            burst_wait_cycles,

    // SRAM Write Interface
    output reg [11:0]             sram_waddr,
    output reg [23:0]             sram_wdata,
    output reg [NUM_TILES-1:0]    sram_we_weight,
    output reg [NUM_TILES-1:0]    sram_we_input,
    output reg                    sram_we_broadcast
);

    assign s_axi_awready = 1'b1;
    assign s_axi_wready  = 1'b1;
    assign s_axi_arready = 1'b1;
    assign s_axi_bresp   = 2'b00;
    assign s_axi_rresp   = 2'b00;

    reg bvalid_reg;
    assign s_axi_bvalid = bvalid_reg;

    // Register Write Logic
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            fabric_start      <= 1'b0;
            fabric_base_addr  <= 0;
            fabric_depth      <= 0;
            fabric_stride     <= 0;
            fabric_exec_hints <= 0;
            fabric_lane_count <= 15;
            fabric_lane_mask  <= 15'h7FFF;
            fabric_tile_mask  <= {NUM_TILES{1'b1}};
            bvalid_reg        <= 1'b0;
            sram_we_weight    <= {NUM_TILES{1'b0}};
            sram_we_input     <= {NUM_TILES{1'b0}};
            sram_we_broadcast <= 1'b0;
            sram_waddr        <= 12'b0;
            sram_wdata        <= 24'b0;
        end else begin
            sram_we_weight    <= {NUM_TILES{1'b0}};
            sram_we_input     <= {NUM_TILES{1'b0}};
            sram_we_broadcast <= 1'b0;

            if (fabric_done) fabric_start <= 1'b0;

            if (s_axi_awvalid && s_axi_wvalid) begin
                if (s_axi_awaddr[15:12] == 4'h9) begin
                    // Broadcast Weight (0x9000)
                    sram_we_broadcast <= 1'b1;
                    sram_waddr        <= s_axi_awaddr[11:2];
                    sram_wdata        <= s_axi_wdata[23:0];
                end else if (s_axi_awaddr[15:13] < 4 && s_axi_awaddr[12]) begin
                    // Weight SRAM Range (0x1000, 0x3000, 0x5000, 0x7000)
                    sram_we_weight[s_axi_awaddr[15:13]] <= 1'b1;
                    sram_waddr                          <= s_axi_awaddr[11:2];
                    sram_wdata                          <= s_axi_wdata[23:0];
                end else if (s_axi_awaddr[15:13] > 0 && s_axi_awaddr[15:13] <= 4 && !s_axi_awaddr[12]) begin
                    // Input SRAM Range (0x2000, 0x4000, 0x6000, 0x8000)
                    sram_we_input[s_axi_awaddr[15:13]-1] <= 1'b1;
                    sram_waddr                           <= s_axi_awaddr[11:2];
                    sram_wdata                           <= s_axi_wdata[23:0];
                end else begin
                    case (s_axi_awaddr[6:0])
                        7'h00: begin
                            fabric_start     <= s_axi_wdata[0];
                            fabric_tile_mask <= s_axi_wdata[15:8];
                        end
                        7'h08: fabric_base_addr  <= s_axi_wdata;
                        7'h0C: fabric_depth      <= s_axi_wdata[15:0];
                        7'h10: fabric_stride     <= s_axi_wdata[7:0];
                        7'h14: fabric_exec_hints <= s_axi_wdata;
                        7'h18: fabric_lane_count <= s_axi_wdata[15:0];
                        7'h1C: fabric_lane_mask  <= s_axi_wdata[14:0];
                    endcase
                end
                bvalid_reg <= 1'b1;
            end else if (s_axi_bready) begin
                bvalid_reg <= 1'b0;
            end
        end
    end

    // Register Read Logic (For Debugging)
    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            s_axi_rvalid <= 1'b0;
            s_axi_rdata  <= 0;
        end else if (s_axi_arvalid && !s_axi_rvalid) begin
            s_axi_rvalid <= 1'b1;
            if (s_axi_araddr[8] && s_axi_araddr[11:9] == 0) begin
                // Results Range (0x100 - 0x1FF)
                s_axi_rdata <= vector_results[(s_axi_araddr[7:6]*15*32 + s_axi_araddr[5:2]*32) +: 32];
            end else if (s_axi_araddr[11:8] >= 4'h2 && s_axi_araddr[11:8] <= 4'h4) begin
                // Profiling Range for Tiles 1-3 (0x200, 0x300, 0x400)
                case (s_axi_araddr[7:0])
                    8'h6C: s_axi_rdata <= {17'b0, overflow_flags[(s_axi_araddr[11:8]-1)*15 +: 15]};
                    default: begin
                        if (s_axi_araddr[7:0] >= 8'h28 && s_axi_araddr[7:0] <= 8'h64) begin
                            s_axi_rdata <= skip_counts[((s_axi_araddr[11:8]-1)*15*32 + ((s_axi_araddr[7:0]-8'h28)>>2)*32) +: 32];
                        end else if (s_axi_araddr[7:0] >= 8'h70 && s_axi_araddr[7:0] <= 8'hAC) begin
                            s_axi_rdata <= active_cycles[((s_axi_araddr[11:8]-1)*15*32 + ((s_axi_araddr[7:0]-8'h70)>>2)*32) +: 32];
                        end else begin
                            s_axi_rdata <= 32'hDEADBEEF;
                        end
                    end
                endcase
            end else begin
                case (s_axi_araddr[7:0])
                    8'h00: s_axi_rdata <= {16'b0, fabric_tile_mask, 7'b0, fabric_start};
                    8'h04: s_axi_rdata <= {30'b0, fabric_done, fabric_start};
                    8'h08: s_axi_rdata <= fabric_base_addr;
                    8'h0C: s_axi_rdata <= {16'b0, fabric_depth};
                    8'h10: s_axi_rdata <= {24'b0, fabric_stride};
                    8'h14: s_axi_rdata <= fabric_exec_hints;
                    8'h18: s_axi_rdata <= {16'b0, fabric_lane_count};
                    8'h1C: s_axi_rdata <= {17'b0, fabric_lane_mask};
                    8'h20: s_axi_rdata <= cycle_count;
                    8'h24: s_axi_rdata <= utilization_count;
                    8'h68: s_axi_rdata <= burst_wait_cycles;
                    8'h6C: s_axi_rdata <= {17'b0, overflow_flags[14:0]};
                    default: begin
                        if (s_axi_araddr[7:0] >= 8'h28 && s_axi_araddr[7:0] <= 8'h64) begin
                            s_axi_rdata <= skip_counts[(((s_axi_araddr[7:0]-8'h28)>>2)*32) +: 32];
                        end else if (s_axi_araddr[7:0] >= 8'h70 && s_axi_araddr[7:0] <= 8'hAC) begin
                            s_axi_rdata <= active_cycles[(((s_axi_araddr[7:0]-8'h70)>>2)*32) +: 32];
                        end else begin
                            s_axi_rdata <= 32'hDEADBEEF;
                        end
                    end
                endcase
            end
        end else if (s_axi_rready) begin
            s_axi_rvalid <= 1'b0;
        end
    end
endmodule