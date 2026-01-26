module axi_interconnect_v1 #(
    parameter ADDR_WIDTH = 32,
    parameter DATA_WIDTH = 32
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
    output reg                    fabric_start,
    input  wire                   fabric_done,

    // Vector Results Input
    input  wire [(15*32)-1:0]     vector_results,

    // SRAM Write Interface
    output reg [11:0]             sram_waddr,
    output reg [23:0]             sram_wdata,
    output reg                    sram_we_weight,
    output reg                    sram_we_input
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
            fabric_start     <= 1'b0;
            fabric_base_addr <= 0;
            fabric_depth     <= 0;
            fabric_stride    <= 0;
            bvalid_reg       <= 1'b0;
            sram_we_weight   <= 1'b0;
            sram_we_input    <= 1'b0;
            sram_waddr       <= 12'b0;
            sram_wdata       <= 24'b0;
        end else begin
            sram_we_weight <= 1'b0;
            sram_we_input  <= 1'b0;

            if (fabric_done) fabric_start <= 1'b0;

            if (s_axi_awvalid && s_axi_wvalid) begin
                if (s_axi_awaddr[15:12] == 4'h1) begin
                    // Weight SRAM Range (Base 0x1000, 1024 words)
                    sram_we_weight <= 1'b1;
                    sram_waddr     <= s_axi_awaddr[11:2];
                    sram_wdata     <= s_axi_wdata[23:0];
                end else if (s_axi_awaddr[15:12] == 4'h2) begin
                    // Input SRAM Range (Base 0x2000, 1024 words)
                    sram_we_input <= 1'b1;
                    sram_waddr    <= s_axi_awaddr[11:2];
                    sram_wdata    <= s_axi_wdata[23:0];
                end else begin
                    case (s_axi_awaddr[6:0])
                        7'h00: fabric_start     <= s_axi_wdata[0];
                        7'h08: fabric_base_addr <= s_axi_wdata;
                        7'h0C: fabric_depth     <= s_axi_wdata[15:0];
                        7'h10: fabric_stride    <= s_axi_wdata[7:0];
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
            if (s_axi_araddr[8]) begin
                // Results Range (Base 0x100)
                // Use s_axi_araddr[7:2] to select lane
                case (s_axi_araddr[7:2])
                    6'd0: s_axi_rdata <= vector_results[0*32 +: 32];
                    6'd1: s_axi_rdata <= vector_results[1*32 +: 32];
                    6'd2: s_axi_rdata <= vector_results[2*32 +: 32];
                    6'd3: s_axi_rdata <= vector_results[3*32 +: 32];
                    6'd4: s_axi_rdata <= vector_results[4*32 +: 32];
                    6'd5: s_axi_rdata <= vector_results[5*32 +: 32];
                    6'd6: s_axi_rdata <= vector_results[6*32 +: 32];
                    6'd7: s_axi_rdata <= vector_results[7*32 +: 32];
                    6'd8: s_axi_rdata <= vector_results[8*32 +: 32];
                    6'd9: s_axi_rdata <= vector_results[9*32 +: 32];
                    6'd10: s_axi_rdata <= vector_results[10*32 +: 32];
                    6'd11: s_axi_rdata <= vector_results[11*32 +: 32];
                    6'd12: s_axi_rdata <= vector_results[12*32 +: 32];
                    6'd13: s_axi_rdata <= vector_results[13*32 +: 32];
                    6'd14: s_axi_rdata <= vector_results[14*32 +: 32];
                    default: s_axi_rdata <= 32'h0;
                endcase
            end else begin
                case (s_axi_araddr[6:0])
                    7'h00: s_axi_rdata <= {31'b0, fabric_start};
                    7'h04: s_axi_rdata <= {30'b0, fabric_done, fabric_start};
                    7'h08: s_axi_rdata <= fabric_base_addr;
                    7'h0C: s_axi_rdata <= {16'b0, fabric_depth};
                    7'h10: s_axi_rdata <= {24'b0, fabric_stride};
                    default: s_axi_rdata <= 32'hDEADBEEF;
                endcase
            end
        end else if (s_axi_rready) begin
            s_axi_rvalid <= 1'b0;
        end
    end
endmodule