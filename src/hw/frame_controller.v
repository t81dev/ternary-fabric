module frame_controller #(
    parameter ADDR_WIDTH = 32,
    parameter LANE_COUNT = 15 
)(
    input  wire                   clk,
    input  wire                   reset,
    
    input  wire [ADDR_WIDTH-1:0]  base_addr,
    input  wire [15:0]            frame_depth,
    input  wire [7:0]             lane_stride, 
    
    input  wire                   start_trigger,
    output reg                    engine_enable,
    output reg                    frame_done,

    output reg [ADDR_WIDTH-1:0]   mem_addr,
    input  wire                   mem_ready
);

    reg [15:0] current_depth;
    
    // State definitions
    localparam IDLE = 2'b00;
    localparam RUN  = 2'b01;
    localparam DONE = 2'b10;
    reg [1:0] state;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            state         <= IDLE;
            current_depth <= 0;
            mem_addr      <= 0;
            engine_enable <= 1'b0;
            frame_done    <= 1'b0;
        end else begin
            case (state)
                IDLE: begin
                    frame_done <= 1'b0; // Force low while waiting
                    if (start_trigger) begin
                        state         <= RUN;
                        current_depth <= 0;
                        mem_addr      <= base_addr;
                        engine_enable <= 1'b1;
                    end
                end

                RUN: begin
                    if (mem_ready) begin
                        if (current_depth < frame_depth - 1) begin
                            current_depth <= current_depth + 1;
                            // For 15-lanes, each word (3 bytes) holds one step.
                            // Increment address by lane_stride words.
                            mem_addr      <= mem_addr + ((LANE_COUNT / 15) * lane_stride);
                        end else begin
                            state         <= DONE;
                            engine_enable <= 1'b0;
                        end
                    end
                end

                DONE: begin
                    frame_done <= 1'b1;
                    state      <= IDLE; // Return to IDLE to clear done next cycle
                end

                default: state <= IDLE;
            endcase
        end
    end

endmodule