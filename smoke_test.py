import sys
sys.path.append('src/pytfmbs')
import pytfmbs

try:
    print("--- Initializing Fabric ---")
    fab = pytfmbs.Fabric()
    
    print("--- Dispatching Test Frame ---")
    # args: base_addr, depth, lanes, stride, kernel_id
    fab.run(0x1000, 64, 15, 1, 0)
    
    print("--- SUCCESS: Smoke test passed! ---")
except Exception as e:
    print(f"--- FAILURE: {e} ---")