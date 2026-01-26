/*
 * Frame Controller
 * Purpose: Generates addresses and control signals for the Vector Engine.
 * Implements: Offset = (depth * Lane_Count) + lane_id
 * Handles the "Logical-to-Physical" translation for PT-5 packed memory.
 */
module frame_controller #(
    parameter ADDR_WIDTH = 32,
    parameter LANE_COUNT = 15 // Must be multiple of 5 for PT-5 alignment
)(
    input  wire                   clk,
    input  wire                   reset,
    
    // TFM Configuration (from Shadow State / Descriptor)
    input  wire [ADDR_WIDTH-1:0]  base_addr,
    input  wire [15:0]            frame_depth,
    input  wire [7:0]             lane_stride, // Usually 1 for contiguous
    
    // Command Interface
    input  wire                   start_trigger,
    output reg                    engine_enable,
    output reg                    frame_done,

    // Memory Bus Interface
    output reg [ADDR_WIDTH-1:0]   mem_addr,
    input  wire                   mem_ready
);

    reg [15:0] current_depth;
    
    // State Machine for Frame Traversal
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            current_depth <= 16'd0;
            mem_addr      <= 0;
            engine_enable <= 1'b0;
            frame_done    <= 1'b0;
        end else if (start_trigger) begin
            current_depth <= 16'd0;
            mem_addr      <= base_addr;
            engine_enable <= 1'b1;
            frame_done    <= 1'b0;
        end else if (engine_enable && mem_ready) begin
            if (current_depth < frame_depth - 1) begin
                current_depth <= current_depth + 1;
                
                // Address increment logic:
                // Since each "depth" step pulls L lanes, and we have 5 trits/byte,
                // we move the pointer by (LANE_COUNT / 5) * stride.
                mem_addr <= mem_addr + ((LANE_COUNT / 5) * lane_stride);
            end else begin
                engine_enable <= 1'b0;
                frame_done    <= 1'b1;
            end
        end
    end

endmodule