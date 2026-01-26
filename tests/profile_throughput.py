import time
import os
import subprocess

# --- Configuration ---
LANES = 15
TOTAL_CYCLES = 1_000_000
CLOCK_SPEED_MHZ = 250 # Targeted FPGA clock for math

def run_profile():
    print(f"🚀 Starting Sustained Throughput Test ({TOTAL_CYCLES:,} cycles)...")
    
    # 1. Hardware Simulation Profile
    start_time = time.perf_counter()
    process = subprocess.run(['./bin/v_bench'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    duration_hw = time.perf_counter() - start_time
    
    total_ops = TOTAL_CYCLES * LANES * 2
    gops_hw = (total_ops / duration_hw) / 1e9
    
    # 2. Software Reference Profile
    # depth = cycles * lanes
    depth = TOTAL_CYCLES * LANES
    start_time = time.perf_counter()
    process = subprocess.run(['./bin/reference_tfmbs', str(depth)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    duration_sw = time.perf_counter() - start_time
    gops_sw = (total_ops / duration_sw) / 1e9

    print("-" * 50)
    print(f"{'Metric':<20} | {'Hardware (Sim)':<15} | {'Software (Ref)':<15}")
    print("-" * 50)
    print(f"{'Execution Time (s)':<20} | {duration_hw:<15.4f} | {duration_sw:<15.4f}")
    print(f"{'Throughput (GOPS)':<20} | {gops_hw:<15.4f} | {gops_sw:<15.4f}")
    print("-" * 50)
    print(f"⚡ HW is {duration_sw/duration_hw:.2f}x faster than SW (in simulation context)")
    print(f"⚡ Efficiency: {(gops_hw / (LANES * 2 * CLOCK_SPEED_MHZ / 1000)) * 100:.2f}% of Theoretical Peak")
    print("-" * 50)

if __name__ == "__main__":
    if not os.path.exists("./bin/v_bench"):
        print("❌ Error: v_bench binary not found. Run 'make benchmark_hw' first.")
    else:
        run_profile()