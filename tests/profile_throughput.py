import time
import os
import subprocess

# --- Configuration ---
LANES = 15
TOTAL_CYCLES = 1_000_000
CLOCK_SPEED_MHZ = 250 # Targeted FPGA clock for math

def run_profile():
    print(f"🚀 Starting Sustained Throughput Test ({TOTAL_CYCLES:,} cycles)...")
    
    # We use the compiled verilator binary in 'benchmark' mode
    # If you haven't added a 'quiet' or 'profile' mode to the C++, 
    # we'll capture the time it takes for the binary to execute.
    
    start_time = time.perf_counter()
    
    # Run the hardware benchmark
    # Note: We redirect output to dev/null to ensure we're measuring 
    # hardware speed, not terminal printing speed.
    process = subprocess.run(['./bin/v_bench'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    # --- Performance Math ---
    # Total Operations = Cycles * Lanes * 2 (1 MAC = 1 Mul + 1 Add)
    total_ops = TOTAL_CYCLES * LANES * 2
    ops_per_sec = total_ops / duration
    gops = ops_per_sec / 1e9
    
    print("-" * 40)
    print(f"⏱️  Execution Time: {duration:.4f} seconds")
    print(f"📊 Sustained Throughput: {gops:.4f} GOPS")
    print(f"⚡ Efficiency: {(gops / (LANES * 2 * CLOCK_SPEED_MHZ / 1000)) * 100:.2f}% of Theoretical Peak")
    print("-" * 40)

if __name__ == "__main__":
    if not os.path.exists("./bin/v_bench"):
        print("❌ Error: v_bench binary not found. Run 'make benchmark_hw' first.")
    else:
        run_profile()