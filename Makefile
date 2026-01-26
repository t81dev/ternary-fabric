# Ternary Fabric Master Makefile
CC = gcc
PYTHON = python3
IVERILOG = iverilog
VVP = vvp

# Paths
CFLAGS = -I./include -Wall -Wextra -O2
SRC_DIR = src
HW_DIR = src/hw
PY_DIR = src/pytfmbs
EX_DIR = examples
BIN_DIR = bin

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

# --- Hardware Simulation (Phase 3) ---
hw_sim: $(ALL_HW_SIM)

$(BIN_DIR)/fabric_tb.vvp: $(HW_DIR)/tb_vector_system.v $(HW_DIR)/ternary_fabric_top.v
	$(IVERILOG) -o $@ -I $(HW_DIR) $^

run_sim: $(BIN_DIR)/fabric_tb.vvp
	$(VVP) $<

# --- Utilities ---
clean:
	rm -rf $(BIN_DIR)
	rm -f $(PY_DIR)/*.so
	rm -rf $(PY_DIR)/build

.PHONY: all directories clean python_ext hw_sim run_sim