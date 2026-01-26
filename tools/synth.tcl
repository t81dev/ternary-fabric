# Read the Verilog files
read_verilog src/hw/ternary_lane_alu.v
read_verilog src/hw/pt5_unpacker.v
read_verilog src/hw/vector_engine.v
read_verilog src/hw/ternary_fabric_top.v

# Define the top module
hierarchy -top ternary_fabric_top

# Synthesis for generic FPGA (or target iCE40/ECP5)
synth -top ternary_fabric_top

# Display resource utilization (LUTs, Flip-Flops)
stat