`timescale 1ns/1ps

module tb_tgemm();
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
        .s_axi_araddr(32'h0), .s_axi_arvalid(1'b0), .s_axi_arready(),
        .s_axi_rdata(), .s_axi_rresp(), .s_axi_rvalid(), .s_axi_rready(1'b1),
        .vector_results(results)
    );

    always #5 clk = ~clk;

    task axi_write(input [31:0] addr, input [31:0] data);
    begin
        awaddr = addr; wdata = data; awvalid = 1; wvalid = 1;
        #10 awvalid = 0; wvalid = 0;
    end
    endtask

    initial begin
        $display("Starting T-GEMM Hardware Testbench...");

        // 1. System Reset
        clk = 0; reset_n = 0; awvalid = 0; wvalid = 0;
        #20 reset_n = 1;

        // 2. Load SRAM with test data
        // Weight SRAM (0x1000): Byte 0 = t0:1, t1:-1, t2:0, t3:1, t4:-1
        // PT-5 mapping: 0->0, 1->1, -1->2.
        // Byte 0 = 1 + 2*3 + 0*9 + 1*27 + 2*81 = 1 + 6 + 0 + 27 + 162 = 196 (0xC4)
        axi_write(32'h1000, 32'h0000C4);

        // Input SRAM (0x2000): Byte 0 = all 1s (t0-t4 = 1) -> 1+3+9+27+81 = 121 (0x79)
        axi_write(32'h2000, 32'h000079);

        // 3. Configure Frame & Hints
        axi_write(32'h08, 32'h0); // base_addr
        axi_write(32'h0C, 32'h1); // depth = 1
        axi_write(32'h10, 32'h1); // stride = 1
        axi_write(32'h18, 32'h000F); // lane_count = 15

        // Test 1: DOT (0x01) with Zero-Skip (bit 17) and Free-Neg (bit 18)
        // exec_hints = 0x01 | (1 << 17) | (1 << 18) = 0x60001
        axi_write(32'h14, 32'h00060001);

        // Trigger Start
        axi_write(32'h00, 32'h1);
        #100;

        $display("--- Test 1 Results (Free-Neg & Zero-Skip) ---");
        $display("Lane 0 (W: 1, I: 1) -> Acc: %d (Expected:  1)", $signed(results[0*32 +: 32]));
        $display("Lane 1 (W:-1, I: 1) -> Acc: %d (Expected: -1)", $signed(results[1*32 +: 32]));
        $display("Lane 2 (W: 0, I: 1) -> Acc: %d (Expected:  0)", $signed(results[2*32 +: 32]));
        $display("Lane 3 (W: 1, I: 1) -> Acc: %d (Expected:  1)", $signed(results[3*32 +: 32]));
        $display("Lane 4 (W:-1, I: 1) -> Acc: %d (Expected: -1)", $signed(results[4*32 +: 32]));

        // Test 2: Weight Broadcast (bit 19)
        #10 reset_n = 0; #20 reset_n = 1; // Full reset

        // Weight SRAM (0x1000): t0 = -1 (2), others 0. Byte = 2.
        axi_write(32'h1000, 32'h2);
        // Input SRAM (0x2000): all 1s.
        axi_write(32'h2000, 32'h79);

        // exec_hints = TGEMM (0x06) + ZeroSkip (1<<17) + FreeNeg (1<<18) + Brdcst (1<<19) = 0xE0006
        axi_write(32'h14, 32'h000E0006);
        axi_write(32'h0C, 32'h1);
        axi_write(32'h18, 32'h000F);
        axi_write(32'h00, 32'h1);
        #100;

        $display("--- Test 2 Results (Weight Broadcast W=-1) ---");
        $display("Lane 0  (I:1, W_brdcst:-1) -> Acc: %d (Expected: -1)", $signed(results[0*32 +: 32]));
        $display("Lane 1  (I:1, W_brdcst:-1) -> Acc: %d (Expected: -1)", $signed(results[1*32 +: 32]));
        $display("Lane 14 (I:1, W_brdcst:-1) -> Acc: %d (Expected: -1)", $signed(results[14*32 +: 32]));

        // Test 3: Lane Masking
        #10 reset_n = 0; #20 reset_n = 1;
        axi_write(32'h1000, 32'h000079); // All W=1
        axi_write(32'h2000, 32'h000079); // All I=1
        axi_write(32'h18, 32'h0002);     // Only 2 lanes active
        axi_write(32'h14, 32'h00060006); // TGEMM + ZeroSkip + FreeNeg
        axi_write(32'h0C, 32'h1);
        axi_write(32'h00, 32'h1);
        #100;

        $display("--- Test 3 Results (Lane Masking: 2 lanes) ---");
        $display("Lane 0 (Active) -> Acc: %d (Expected: 1)", $signed(results[0*32 +: 32]));
        $display("Lane 1 (Active) -> Acc: %d (Expected: 1)", $signed(results[1*32 +: 32]));
        $display("Lane 2 (Masked) -> Acc: %d (Expected: 0)", $signed(results[2*32 +: 32]));

        $finish;
    end
endmodule
