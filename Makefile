# Ternary Fabric Master Makefile
CC = gcc
CXX = g++
PYTHON = python3
IVERILOG = iverilog
VVP = vvp
YOSYS = yosys
VERILATOR = verilator

# Paths
CFLAGS = -I./include -Wall -Wextra -O2
SRC_DIR = src
HW_DIR = src/hw
PY_DIR = src/pytfmbs
EX_DIR = examples
BIN_DIR = bin
OBJ_DIR = obj_dir

# Verilator Flags
# Add -DHEADLESS=1 to CFLAGS if we want to skip printfs in simulation
VERILATOR_FLAGS = --cc $(HW_DIR)/ternary_fabric_top.v -I$(HW_DIR) --Mdir $(OBJ_DIR) -Wno-fatal

# Targets
ALL_C_BINS = $(BIN_DIR)/mediator_mock $(BIN_DIR)/pt5_example
ALL_HW_SIM = $(BIN_DIR)/fabric_tb.vvp

all: directories $(ALL_C_BINS) python_ext hw_sim

directories:
	mkdir -p $(BIN_DIR)

# --- C Tools & Examples ---
$(BIN_DIR)/mediator_mock: $(SRC_DIR)/mediator_mock.c
	$(CC) $(CFLAGS) -o $@ $<

$(BIN_DIR)/pt5_example: $(EX_DIR)/pt5_pack_example.c
	$(CC) $(CFLAGS) -o $@ $<

# --- Python Bindings (Phase 4) ---
python_ext:
	cd $(PY_DIR) && $(PYTHON) setup.py build_ext --inplace

# --- Hardware Simulation ---
$(BIN_DIR)/fabric_tb.vvp: $(HW_DIR)/*.v
	$(IVERILOG) -o $@ -I $(HW_DIR) $(HW_DIR)/tb_fabric_full.v $(HW_DIR)/ternary_fabric_top.v $(HW_DIR)/axi_interconnect_v1.v $(HW_DIR)/frame_controller.v $(HW_DIR)/pt5_unpacker.v $(HW_DIR)/ternary_sram_wrapper.v $(HW_DIR)/vector_engine.v $(HW_DIR)/ternary_lane_alu.v

run_sim: $(BIN_DIR)/fabric_tb.vvp
	$(VVP) $<

# --- Verilator Benchmark Target ---
benchmark_hw:
	mkdir -p $(BIN_DIR)
	$(VERILATOR) $(VERILATOR_FLAGS) --trace --exe tests/bench_top.cpp --build -j `sysctl -n hw.ncpu` -o v_bench_binary
	cp $(OBJ_DIR)/v_bench_binary $(BIN_DIR)/v_bench
	@echo "🔥 Hardware Compiled. Running Benchmark..."
	./$(BIN_DIR)/v_bench

# --- Sustained Throughput Profiling ---
# Compiles without tracing or printfs for maximum simulation speed
profile:
	mkdir -p $(BIN_DIR)
	@echo "🚀 Compiling Optimized Headless Model (No Traces)..."
	$(VERILATOR) $(VERILATOR_FLAGS) --exe tests/bench_top.cpp \
		-CFLAGS "-DHEADLESS=1" --build -j `sysctl -n hw.ncpu` -o v_profile_binary
	cp $(OBJ_DIR)/v_profile_binary $(BIN_DIR)/v_profile
	@$(PYTHON) tests/profile_throughput.py

# --- Open-Source Synthesis ---
synth:
	$(YOSYS) -p "read_verilog $(HW_DIR)/ternary_fabric_top.v \
                             $(HW_DIR)/axi_interconnect_v1.v \
                             $(HW_DIR)/frame_controller.v \
                             $(HW_DIR)/pt5_unpacker.v \
                             $(HW_DIR)/ternary_sram_wrapper.v \
                             $(HW_DIR)/vector_engine.v \
                             $(HW_DIR)/ternary_lane_alu.v; \
                 hierarchy -top ternary_fabric_top; \
                 synth; stat"

# --- Bit-Exact Verification ---
verify_pt5:
	$(PYTHON) tests/gen_vectors.py
	$(IVERILOG) -o bin/verify_pt5.vvp tests/tb_pt5_unpacker.v src/hw/pt5_unpacker.v
	$(VVP) bin/verify_pt5.vvp | grep -E "^-?[0-1]$$" > tests/rtl_clean.txt
	@echo "Comparing RTL output to Expected Trits..."
	@diff -u -b -B tests/expected_trits.txt tests/rtl_clean.txt && echo "⭐ MATCH: Hardware matches Software!" || echo "❌ ERROR: Mismatch detected!"

# --- Utilities ---
test_bridge: python_ext
	$(PYTHON) tests/hardware_bridge.py

clean:
	rm -rf $(BIN_DIR)
	rm -rf $(OBJ_DIR)
	rm -f $(PY_DIR)/*.so
	rm -rf $(PY_DIR)/build
	rm -f tests/*.hex tests/*.txt tests/*.vvp tests/*.vcd

.PHONY: all directories clean python_ext hw_sim run_sim verilate synth verify_pt5 test_bridge benchmark_hw profile