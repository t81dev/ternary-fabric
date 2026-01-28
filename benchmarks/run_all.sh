#!/bin/bash
set -e

BIN_DIR=$(realpath ../bin)
LOG_DIR="logs"
mkdir -p $LOG_DIR

export TFMBS_FIXED_SEED=42

run_bench() {
    local name=$1
    local cmd=$2
    local fabrics=$3
    local sc=$4
    local dla=$5
    local preload=$6

    local p_suffix="none"
    if [ ! -z "$preload" ]; then p_suffix="intercept"; fi
    local log_file="$LOG_DIR/${name}_f${fabrics}_sc${sc}_la${dla}_p${p_suffix}.log"
    echo "Running $name (Fabrics: $fabrics, SC: $sc, DisableLA: $dla, Preload: $preload)..."

    (
        export TFMBS_NUM_FABRICS=$fabrics
        export FABRIC_SHORT_CIRCUIT=$sc
        export TFMBS_DISABLE_LOOKAHEAD=$dla
        export LD_LIBRARY_PATH=$BIN_DIR
        if [ ! -z "$preload" ]; then
            export "$preload"
        fi
        $cmd
    ) > "$log_file" 2>&1 || echo "Failed $name"
}

echo "Building benchmarks..."
# make -C .. all # Skip root build as python part fails
# make all

# Configurations to test
FABRIC_COUNTS=(1 2 4)
SC_MODES=(0 1)
LA_MODES=(0 1) # 0: Enabled (Phase 21), 1: Disabled (Phase 20)

# 1. Synthetic Benchmarks
for f in "${FABRIC_COUNTS[@]}"; do
    run_bench "saturation" "./synthetic/saturation" $f 0 0 ""
    run_bench "density" "./synthetic/density" $f 0 0 ""
done

# 2. Kernel Benchmarks
KERNELS=("./gemm/gemm_bench" "./lstm/lstm_bench" "./gemm/attn_bench" "./gemm/conv3d_bench")
for k_path in "${KERNELS[@]}"; do
    k_name=$(basename $k_path)
    for f in "${FABRIC_COUNTS[@]}"; do
        for la in "${LA_MODES[@]}"; do
            run_bench "$k_name" "$k_path" $f 0 $la ""
        done
    done
done

# 3. Application Benchmarks (mock_llama)
# Baseline: CPU (No LD_PRELOAD, no Short Circuit)
echo "Running mock_llama CPU Baseline..."
$BIN_DIR/mock_llama > "$LOG_DIR/mock_llama_cpu.log" 2>&1

# Fabric Accelerated
for f in "${FABRIC_COUNTS[@]}"; do
    for sc in "${SC_MODES[@]}"; do
        for la in "${LA_MODES[@]}"; do
            run_bench "mock_llama" "timeout 30 $BIN_DIR/mock_llama" $f $sc $la "LD_PRELOAD=$BIN_DIR/libtfmbs_intercept.so"
        done
    done
done

# 4. CPU Kernel Baseline (using reference_tfmbs)
echo "Running Kernel CPU Baselines..."
$BIN_DIR/reference_tfmbs 1000000 0 > "$LOG_DIR/kernel_cpu_dot.log" 2>&1
$BIN_DIR/reference_tfmbs 1000000 1 > "$LOG_DIR/kernel_cpu_gemm.log" 2>&1

echo "All benchmarks completed. Logs are in $LOG_DIR/"
