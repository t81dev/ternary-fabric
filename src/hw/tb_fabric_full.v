`timescale 1ns/1ps

module tb_fabric_full();
    reg clk, reset_n;
    reg [31:0] awaddr;
    reg awvalid, wvalid;
    reg [31:0] wdata;
    wire [(15*32)-1:0] results;

    // Instantiate the Top-Level Fabric
    ternary_fabric_top #(32, 32, 15) uut (
        .clk(clk), .reset_n(reset_n),
        .s_axi_awaddr(awaddr), .s_axi_awvalid(awvalid), .s_axi_awready(),
        .s_axi_wdata(wdata), .s_axi_wvalid(wvalid), .s_axi_wready(),
        .s_axi_bvalid(), .s_axi_bready(1'b1),
        .vector_results(results)
    );

    always #5 clk = ~clk;

    initial begin
        // 1. System Reset
        clk = 0; reset_n = 0; awvalid = 0; wvalid = 0;
        #20 reset_n = 1;

        // 2. Load Frame Geometry (AXI Writes)
        // Write Base Address (0x08)
        #10 awaddr = 32'h08; wdata = 32'h0000_0000; awvalid = 1; wvalid = 1;
        #10 awvalid = 0; wvalid = 0;
        
        // Write Frame Depth (0x0C)
        #10 awaddr = 32'h0C; wdata = 32'h0000_0005; awvalid = 1; wvalid = 1;
        #10 awvalid = 0; wvalid = 0;

        // 3. Trigger Start (0x00)
        #10 awaddr = 32'h00; wdata = 32'h0000_0001; awvalid = 1; wvalid = 1;
        #10 awvalid = 0; wvalid = 0;

        // 4. Observe Hydration
        #100;
        $display("Hardware Lane 0 Accumulator: %d", results[31:0]);
        $finish;
    end
endmodule