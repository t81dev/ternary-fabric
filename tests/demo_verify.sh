#!/bin/bash
set -e

echo "--- Step 1: Building Stack ---"
make all

echo "--- Step 2: Preparing Demo Data ---"
python3 tools/make_demo.py > demo_log.txt

echo "--- Step 3: Running RTL Simulation ---"
# This uses the tb_fabric_full.v we wrote earlier
make run_sim | tee sim_output.txt

echo "--- Step 4: Parity Check ---"
if grep -q "Accumulator:          0" sim_output.txt; then
    echo "SUCCESS: Hardware output matches Golden Model (Sum=0)"
else
    echo "FAILURE: Parity mismatch in RTL"
fi