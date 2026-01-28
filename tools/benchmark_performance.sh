#!/bin/bash
# TFMBS Performance Benchmark Tool

# Find the absolute path to the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
BIN_DIR="$ROOT_DIR/bin"
MOCK_LLAMA="$BIN_DIR/mock_llama"

if [ ! -f "$MOCK_LLAMA" ]; then
    echo "Compiling mock_llama..."
    make -C "$ROOT_DIR" bin/mock_llama
fi

echo "===================================================="
echo "🚀 TFMBS Performance Benchmark"
echo "Comparing Pure CPU vs. Fabric-Accelerated Execution"
echo "===================================================="

# 1. Pure CPU Baseline
echo -e "\n[1/2] Running Pure CPU Baseline..."
if date +%s%N | grep -q N; then
    # BSD date (macOS) doesn't support %N
    START_CPU=$(date +%s)
    MULTIPLIER=1000
else
    # GNU date
    START_CPU=$(date +%s%N)
    MULTIPLIER=0.000001
fi
"$MOCK_LLAMA" > /dev/null
if [ "$MULTIPLIER" == "1000" ]; then
    END_CPU=$(date +%s)
    DIFF_CPU=$(( (END_CPU - START_CPU) * 1000 ))
else
    END_CPU=$(date +%s%N)
    DIFF_CPU=$(( (END_CPU - START_CPU) / 1000000 ))
fi

# 2. Fabric Accelerated
echo -e "[2/2] Running Fabric Accelerated (Interposer)..."
if [ "$MULTIPLIER" == "1000" ]; then
    START_FAB=$(date +%s)
else
    START_FAB=$(date +%s%N)
fi
"$ROOT_DIR/tools/tfmbs-run" "$MOCK_LLAMA" > /dev/null 2>&1
if [ "$MULTIPLIER" == "1000" ]; then
    END_FAB=$(date +%s)
    DIFF_FAB=$(( (END_FAB - START_FAB) * 1000 ))
else
    END_FAB=$(date +%s%N)
    DIFF_FAB=$(( (END_FAB - START_FAB) / 1000000 ))
fi

# Calculate Results
echo -e "\n===================================================="
echo "📊 Results (Wall-clock time)"
echo "----------------------------------------------------"
printf "Pure CPU:   %d ms\n" "$DIFF_CPU"
printf "TFMBS:      %d ms\n" "$DIFF_FAB"

if [ "$DIFF_FAB" -lt "$DIFF_CPU" ]; then
    SPEEDUP=$(echo "scale=2; $DIFF_CPU / $DIFF_FAB" | bc)
    printf "Speedup:    %sx faster\n" "$SPEEDUP"
else
    printf "Note: Fabric overhead exceeds compute savings for this tiny model.\n"
fi
echo "===================================================="
