/*
 * AXI4-Lite Wrapper for Ternary Frame Controller
 * Provides CPU access to TFD (Ternary Frame Descriptor) registers.
 * Register Map:
 * 0x00: Control (Bit 0: Start, Bit 1: Reset)
 * 0x04: Status  (Bit 0: Busy, Bit 1: Done)
 * 0x08: Base Address (Source Pointer)
 * 0x0C: Frame Depth
 * 0x10: Lane Stride
 */
module axi_interconnect_v1 #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32
)(
    // AXI4-Lite Interface
    input  wire                   s_axi_aclk,
    input  wire                   s_axi_aresetn,
    input  wire [ADDR_WIDTH-1:0]  s_axi_awaddr,
    input  wire                   s_axi_awvalid,
    output wire                   s_axi_awready,
    input  wire [DATA_WIDTH-1:0]  s_axi_wdata,
    input  wire                   s_axi_wvalid,
    output wire                   s_axi_wready,
    output wire [1:0]             s_axi_bresp,
    output wire                   s_axi_bvalid,
    input  wire                   s_axi_bready,

    // Internal Fabric Signals
    output reg [ADDR_WIDTH-1:0]   fabric_base_addr,
    output reg [15:0]             fabric_depth,
    output reg [7:0]              fabric_stride,
    output reg                    fabric_start,
    input  wire                   fabric_done
);

    // Simple Register Logic (Simplified AXI handshake)
    assign s_axi_awready = 1'b1;
    assign s_axi_wready  = 1'b1;
    assign s_axi_bresp   = 2'b00; // OKAY

    always @(posedge s_axi_aclk) begin
        if (!s_axi_aresetn) begin
            fabric_start <= 1'b0;
        end else if (s_axi_awvalid && s_axi_wvalid) begin
            case (s_axi_awaddr[4:0])
                5'h00: fabric_start <= s_axi_wdata[0];
                5'h08: fabric_base_addr <= s_axi_wdata;
                5'h0C: fabric_depth <= s_axi_wdata[15:0];
                5'h10: fabric_stride <= s_axi_wdata[7:0];
            endcase
        end else begin
            fabric_start <= 1'b0; // Pulse start
        end
    end

    // Status logic (B-channel)
    assign s_axi_bvalid = s_axi_awvalid && s_axi_wvalid;

endmodule