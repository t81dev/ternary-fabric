#include <iostream>
#include <iomanip>
#include <vector>
#include "Vternary_fabric_top.h"
#include "Vternary_fabric_top___024root.h" // Needed for internal signal access
#include "verilated.h"
#include "verilated_vcd_c.h"

#define REG_START  0x00
#define REG_BASE   0x08
#define REG_DEPTH  0x0C
#define REG_STRIDE 0x10

void tick(Vternary_fabric_top* top, VerilatedVcdC* tfp, int& main_time) {
    top->clk = 0;
    top->eval();
    if (tfp) tfp->dump(main_time++);
    top->clk = 1;
    top->eval();
    if (tfp) tfp->dump(main_time++);
}

void axi_write(Vternary_fabric_top* top, uint32_t addr, uint32_t data) {
    top->s_axi_awaddr = addr;
    top->s_axi_wdata = data;
    top->s_axi_awvalid = 1;
    top->s_axi_wvalid = 1;
    top->s_axi_bready = 1;
}

uint32_t axi_read(Vternary_fabric_top* top, VerilatedVcdC* tfp, int& main_time, uint32_t addr) {
    top->s_axi_araddr = addr;
    top->s_axi_arvalid = 1;
    top->s_axi_rready = 1;
    
    int timeout = 0;
    while(!top->s_axi_rvalid && timeout++ < 20) {
        tick(top, tfp, main_time);
    }
    
    uint32_t result = top->s_axi_rdata;
    top->s_axi_arvalid = 0;
    tick(top, tfp, main_time);
    return result;
}

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vternary_fabric_top* top = new Vternary_fabric_top;
    
    Verilated::traceEverOn(true);
    VerilatedVcdC* tfp = new VerilatedVcdC;
    top->trace(tfp, 99);
    tfp->open("bin/trace.vcd");
    int main_time = 0;

    printf("🚀 Verilator PT-5 Fabric: Advanced AXI Handshake Test\n");

    top->reset_n = 0;
    for(int i=0; i<5; i++) tick(top, tfp, main_time);
    top->reset_n = 1;
    tick(top, tfp, main_time);

    // Write Config
    axi_write(top, REG_BASE, 0x00000000);
    tick(top, tfp, main_time);
    axi_write(top, REG_DEPTH, 100);
    tick(top, tfp, main_time);
    axi_write(top, REG_STRIDE, 1);
    tick(top, tfp, main_time);
    
    top->s_axi_awvalid = 0;
    top->s_axi_wvalid = 0;
    tick(top, tfp, main_time);

    // Verify
    uint32_t read_depth = axi_read(top, tfp, main_time, REG_DEPTH);
    printf("🔍 Hardware Readback: Depth = %d\n", read_depth);

    // Start
    axi_write(top, REG_START, 0x1);
    tick(top, tfp, main_time);
    top->s_axi_awvalid = 0;
    top->s_axi_wvalid = 0;

    printf("----------------------------------------------------------------------------------------------------\n");
    printf(" Cyc | ADDR | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10| L11| L12| L13| L14|\n");
    printf("----------------------------------------------------------------------------------------------------\n");

    for (int i = 0; i < 300; i++) {
        tick(top, tfp, main_time);

        if (i % 2 == 0) {
            // Correct way to access internal wires in Verilator 5.0
            uint32_t current_addr = top->rootp->ternary_fabric_top__DOT__f_mem_addr;
            printf("%4d | %4d |", i, current_addr);
            
            for(int l=0; l<15; l++) {
                // Accessing wide signal array
                uint32_t lane_val = top->vector_results[l]; 
                printf("%3d|", lane_val & 0xFF); 
            }
            printf("\n");
        }

        if (top->rootp->ternary_fabric_top__DOT__f_done && i > 50) {
            printf("----------------------------------------------------------------------------------------------------\n");
            printf("✅ Hardware Signaled DONE at cycle %d\n", i);
            break;
        }
    }

    tfp->close();
    delete top;
    return 0;
}