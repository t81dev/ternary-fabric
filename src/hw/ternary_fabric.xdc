## 1. Primary System Clock (Assumed 100MHz)
create_clock -add -name sys_clk_pin -period 10.00 -waveform {0 5} [get_ports clk]

## 2. Reset (Active Low)
set_property -dict { PACKAGE_PIN "PIN_HERE" IOSTANDARD LVCMOS33 } [get_ports { reset_n }];

## 3. AXI Interface Timing (Standard setup for 100MHz Interconnect)
# These constraints ensure the AXI handshake meets timing across the fabric
set_input_delay -clock [get_clocks sys_clk_pin] -max 2.000 [get_ports {s_axi_awaddr[*] s_axi_awvalid s_axi_wdata[*] s_axi_wvalid}]
set_output_delay -clock [get_clocks sys_clk_pin] -max 2.000 [get_ports {s_axi_awready s_axi_wready s_axi_bvalid}]

## 4. Debug LEDs (Optional - Mapping Vector Results[3:0] for visual heart-beat)
set_property -dict { PACKAGE_PIN "LED1_PIN" IOSTANDARD LVCMOS33 } [get_ports { vector_results[0] }];
set_property -dict { PACKAGE_PIN "LED2_PIN" IOSTANDARD LVCMOS33 } [get_ports { vector_results[1] }];