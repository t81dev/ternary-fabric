# Vivado Automated Flow for Ternary Fabric (Zynq-7000)
# Target: XC7Z020-clg484-1 (ZedBoard)

set outputDir ./vivado_output
file mkdir $outputDir

# 1. Setup - Create Project
create_project -force tfmbs_fpga ./vivado_project -part xc7z020clg484-1

# 2. Add Sources
add_files [ glob src/hw/*.v ]
add_files -fileset constrs_1 tools/zedboard.xdc

# 3. Create Block Design with Zynq PS
create_bd_design "tfmbs_system"
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 -config {make_external "FIXED_IO, DDR" apply_board_preset "1" Master "Disable" Slave "Disable" } [get_bd_cells ps7]

# 4. Instantiate Ternary Fabric as a Module
create_bd_cell -type module -reference ternary_fabric_top fabric_0

# 5. Connect AXI Interface
# Assuming ternary_fabric_top has an AXI Slave interface (S_AXI)
# apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config { Master "/ps7/M_AXI_GP0" intc_ip "New AXI Interconnect" Clk_xbar "Auto" Clk_master "Auto" Clk_slave "Auto" }  [get_bd_intf_pins fabric_0/S_AXI]

# 6. Mark Signals for Debug (ILA)
set_property MARK_DEBUG true [get_nets -hierarchical *zero_skip*]
set_property MARK_DEBUG true [get_nets -hierarchical *trit_mem*]

# 7. Synthesis & ILA Insertion
launch_runs synth_1 -jobs 4
wait_on_run synth_1

open_run synth_1
# Automated ILA insertion
set debug_nets [get_nets -hier -filter {MARK_DEBUG == 1}]
if {[llength $debug_nets] > 0} {
    create_debug_core u_ila_0 ila
    set_property port_width 1 [get_debug_ports u_ila_0/clk]
    connect_debug_port u_ila_0/clk [get_nets [get_clocks]]
    set_property port_width [llength $debug_nets] [get_debug_ports u_ila_0/probe0]
    connect_debug_port u_ila_0/probe0 $debug_nets
}

# 8. Implementation & Bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# 9. Reports
report_timing_summary -file $outputDir/post_route_timing_summary.rpt
report_utilization -file $outputDir/post_route_util.rpt

# 10. Export Hardware
write_hw_platform -fixed -force -file $outputDir/tfmbs_hw.xsa

puts "Vivado flow complete. Bitstream and XSA generated in $outputDir"
