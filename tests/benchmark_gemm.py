import time
import numpy as np
import pytfmbs

def run_benchmark(matrix_size=64):
    print(f"🚀 Starting GEMM Benchmark: {matrix_size}x{matrix_size}")
    
    # 1. Setup Data (Ternary weights -1, 0, 1)
    weights = np.random.randint(-1, 2, (matrix_size, matrix_size))
    vec_in = np.random.randint(-1, 2, matrix_size)
    
    # 2. Initialize the Bridge
    fabric = pytfmbs.Fabric()
    
    # 3. Execution & Timing
    start_time = time.perf_counter()
    
    # Map the TFD (Ternary Fabric Descriptor)
    # Stride 4 for 32-bit alignment, 15 lanes based on your previous logs
    # Using TGEMM kernel (0x06) with Zero-Skip (bit 17) and Free-Neg (bit 18)
    tfd = {
        'base_addr': 0xDEADBEEF,
        'depth': matrix_size,
        'lane_count': 15,
        'stride': 4,
        'exec_hints': 0x06 | (1 << 17) | (1 << 18)
    }
    
    fabric.run(tfd)
    end_time = time.perf_counter()
    
    # 4. Calculate Metrics
    elapsed = end_time - start_time
    ops = matrix_size * matrix_size * 2 # Multiply + Add counts as 2 ops
    tops = (ops / elapsed) / 1e12
    
    print(f"✅ Completed in {elapsed:.6f} seconds")
    print(f"📊 Effective Performance: {tops:.6f} TOPS (Simulated)")

if __name__ == "__main__":
    run_benchmark()