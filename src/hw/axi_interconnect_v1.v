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
    input  wire                   fabric_done
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
        end else begin
            if (fabric_done) fabric_start <= 1'b0;

            if (s_axi_awvalid && s_axi_wvalid) begin
                case (s_axi_awaddr[4:0])
                    5'h00: fabric_start     <= s_axi_wdata[0];
                    5'h08: fabric_base_addr <= s_axi_wdata;
                    5'h0C: fabric_depth     <= s_axi_wdata[15:0];
                    5'h10: fabric_stride    <= s_axi_wdata[7:0];
                endcase
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
            case (s_axi_araddr[4:0])
                5'h00: s_axi_rdata <= {31'b0, fabric_start};
                5'h08: s_axi_rdata <= fabric_base_addr;
                5'h0C: s_axi_rdata <= {16'b0, fabric_depth};
                5'h10: s_axi_rdata <= {24'b0, fabric_stride};
                default: s_axi_rdata <= 32'hDEADBEEF;
            endcase
        end else if (s_axi_rready) begin
            s_axi_rvalid <= 1'b0;
        end
    end
endmodule