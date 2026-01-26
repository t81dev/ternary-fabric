/*
 * Vector Engine: TFM-Compliant Parallel Ternary Processor
 * Tiles 'N' Ternary Processing Elements (TPEs) for SIMD operations.
 * Implements Logical Mapping: Offset = (depth * LANES) + lane_id
 */
module vector_engine #(
    parameter LANES = 16,           // Configurable SIMD width
    parameter ACCUM_WIDTH = 32      // Binary precision per lane
)(
    input  wire                     clk,
    input  wire                     reset,
    
    // Control Interface
    input  wire                     enable,
    input  wire [31:0]              exec_hints,
    input  wire [15:0]              lane_count,
    input  wire [LANES-1:0]         lane_mask,
    
    // Data Interface (Unpacked from PT-5 by upstream Bus Controller)
    input  wire [(LANES*2)-1:0]     bus_weights,  // 2-bits per lane
    input  wire [(LANES*2)-1:0]     bus_inputs,   // 2-bits per lane
    
    // Output: Parallel Accumulator State
    output wire [(LANES*ACCUM_WIDTH)-1:0] vector_out,
    output wire [(LANES*32)-1:0]    skip_counts,
    output reg  [31:0]              cycle_count,
    output reg  [31:0]              utilization_count
);

    wire weight_brdcst = exec_hints[19];
    wire [(LANES*2)-1:0] effective_weights;

    // Broadcast lane 0 weight if hint is set
    assign effective_weights = weight_brdcst ? {(LANES){bus_weights[1:0]}} : bus_weights;

    genvar i;
    generate
        for (i = 0; i < LANES; i = i + 1) begin : gen_lanes
            wire lane_active = enable && (i < lane_count) && lane_mask[i];

            // Instantiate the TPE for each lane
            ternary_lane_alu lane_inst (
                .clk(clk),
                .reset(reset),
                .enable(lane_active),
                .exec_hints(exec_hints),
                .weight(effective_weights[i*2 +: 2]),
                .trit_in(bus_inputs[i*2 +: 2]),
                .accumulator(vector_out[i*ACCUM_WIDTH +: ACCUM_WIDTH]),
                .skip_count(skip_counts[i*32 +: 32])
            );
        end
    endgenerate

    // Performance Profiling Logic
    integer j;
    reg [7:0] active_lanes_this_cycle;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            cycle_count <= 32'd0;
            utilization_count <= 32'd0;
        end else if (enable) begin
            cycle_count <= cycle_count + 1;

            // Calculate active lanes this cycle for utilization metric
            active_lanes_this_cycle = 0;
            for (j = 0; j < LANES; j = j + 1) begin
                if ((j < lane_count) && lane_mask[j])
                    active_lanes_this_cycle = active_lanes_this_cycle + 1;
            end
            utilization_count <= utilization_count + active_lanes_this_cycle;
        end
    end

endmodule