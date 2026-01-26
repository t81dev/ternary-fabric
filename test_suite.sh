#!/bin/bash

# Ternary Fabric Test Suite - Phase 1 Validation
echo "--- Starting TFMBS Phase 1 Tests ---"

# 1. Validate Python Codec & Kernel logic
echo "[Step 1] Testing Python PT-5 Codec and AI Kernel..."
python3 tools/pt5_codec.py
python3 tools/ternary_kernel_sim.py

if [ $? -eq 0 ]; then
    echo "Result: Python Logic Validated."
else
    echo "Result: Python Logic Failed."
    exit 1
fi

# 2. Compile C Components
echo -e "\n[Step 2] Compiling C Mediator and Examples..."
make clean && make all

if [ $? -eq 0 ]; then
    echo "Result: C Components Compiled Successfully."
else
    echo "Result: Compilation Failed."
    exit 1
fi

# 3. Run Mock Handshake
echo -e "\n[Step 3] Running Mock Mediator Handshake..."
./bin/mediator_mock

echo -e "\n--- Phase 1 Tests Complete ---"