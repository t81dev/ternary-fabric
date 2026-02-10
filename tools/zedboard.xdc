# ZedBoard Constraints for Ternary Fabric
# Clock signal (100MHz)
set_property PACKAGE_PIN Y9 [get_ports {clk}]
set_property IOSTANDARD LVCMOS33 [get_ports {clk}]
create_clock -period 10.000 -name clk -waveform {0.000 5.000} [get_ports {clk}]

# Reset
set_property PACKAGE_PIN P16 [get_ports {reset_n}]
set_property IOSTANDARD LVCMOS33 [get_ports {reset_n}]

# UART (example mapping to ZedBoard USB-UART)
# set_property PACKAGE_PIN W24 [get_ports {uart_tx}]
# set_property IOSTANDARD LVCMOS33 [get_ports {uart_tx}]
