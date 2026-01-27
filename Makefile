# Ternary Fabric Master Makefile
CC = gcc
CXX = g++
PYTHON = python3
IVERILOG = iverilog
VVP = vvp
YOSYS = yosys
VERILATOR = verilator

# OS Detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	SHLIB_EXT = .dylib
	LDFLAGS_SHARED = -dynamiclib
	PRELOAD_ENV = DYLD_INSERT_LIBRARIES
	LIBPATH_ENV = DYLD_LIBRARY_PATH
	LDLIBS_DL =
	NPROC = $(shell sysctl -n hw.ncpu)
else
	SHLIB_EXT = .so
	LDFLAGS_SHARED = -shared
	PRELOAD_ENV = LD_PRELOAD
	LIBPATH_ENV = LD_LIBRARY_PATH
	LDLIBS_DL = -ldl
	NPROC = $(shell nproc 2>/dev/null || echo 1)
endif

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
ALL_C_BINS = $(BIN_DIR)/mediator_mock $(BIN_DIR)/pt5_example $(BIN_DIR)/reference_tfmbs $(BIN_DIR)/test_device $(BIN_DIR)/mock_llama $(BIN_DIR)/test_dynamic_detection $(BIN_DIR)/test_phase10
ALL_LIBS = $(BIN_DIR)/libtfmbs_device$(SHLIB_EXT) $(BIN_DIR)/libtfmbs_intercept$(SHLIB_EXT)
ALL_HW_SIM = $(BIN_DIR)/fabric_tb.vvp

all: directories $(ALL_LIBS) $(ALL_C_BINS) python_ext hw_sim

directories:
	mkdir -p $(BIN_DIR)

# --- Shared Libraries ---
$(BIN_DIR)/libtfmbs_device$(SHLIB_EXT): $(SRC_DIR)/libtfmbs_device.c $(SRC_DIR)/fabric_emulator.c $(SRC_DIR)/tfmbs_driver_mock.c
	$(CC) $(CFLAGS) -fPIC $(LDFLAGS_SHARED) -o $@ $^

$(BIN_DIR)/libtfmbs_intercept$(SHLIB_EXT): $(SRC_DIR)/libtfmbs_intercept.c $(BIN_DIR)/libtfmbs_device$(SHLIB_EXT)
	$(CC) $(CFLAGS) -fPIC $(LDFLAGS_SHARED) -o $@ $< -L$(BIN_DIR) -ltfmbs_device $(LDLIBS_DL)

# --- C Tools & Examples ---
$(BIN_DIR)/mediator_mock: $(SRC_DIR)/mediator_mock.c
	$(CC) $(CFLAGS) -o $@ $<

$(BIN_DIR)/pt5_example: $(EX_DIR)/pt5_pack_example.c
	$(CC) $(CFLAGS) -o $@ $<

$(BIN_DIR)/reference_tfmbs: $(SRC_DIR)/reference_tfmbs.c
	$(CC) $(CFLAGS) -o $@ $<

$(BIN_DIR)/test_device: tests/test_device.c $(BIN_DIR)/libtfmbs_device$(SHLIB_EXT)
	$(CC) $(CFLAGS) -o $@ $< -L$(BIN_DIR) -ltfmbs_device

$(BIN_DIR)/mock_llama: tests/mock_llama.c
	$(CC) $(CFLAGS) -o $@ $<

$(BIN_DIR)/test_dynamic_detection: tests/test_dynamic_detection.c
	$(CC) $(CFLAGS) -o $@ $<

$(BIN_DIR)/test_phase10: tests/test_phase10.c $(SRC_DIR)/tfmbs_driver_mock.c $(SRC_DIR)/fabric_emulator.c
	$(CC) $(CFLAGS) -I$(SRC_DIR) -o $@ $^

run_mock_llama: $(BIN_DIR)/mock_llama $(ALL_LIBS)
	export FABRIC_SHORT_CIRCUIT=1; \
	$(LIBPATH_ENV)=$(BIN_DIR) $(PRELOAD_ENV)=$(BIN_DIR)/libtfmbs_intercept$(SHLIB_EXT) ./$(BIN_DIR)/mock_llama

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
	$(VERILATOR) $(VERILATOR_FLAGS) --trace --exe tests/bench_top.cpp --build -j $(NPROC) -o v_bench_binary
	cp $(OBJ_DIR)/v_bench_binary $(BIN_DIR)/v_bench
	@echo "🔥 Hardware Compiled. Running Benchmark..."
	./$(BIN_DIR)/v_bench

# --- Sustained Throughput Profiling ---
# Compiles without tracing or printfs for maximum simulation speed
profile:
	mkdir -p $(BIN_DIR)
	@echo "🚀 Compiling Optimized Headless Model (No Traces)..."
	$(VERILATOR) $(VERILATOR_FLAGS) --exe tests/bench_top.cpp \
		-CFLAGS "-DHEADLESS=1" --build -j $(NPROC) -o v_profile_binary
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