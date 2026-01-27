import pytfmbs
import numpy as np

f = pytfmbs.Fabric()

# Load some dummy data
data = np.zeros(1024, dtype=np.uint32)
f.load(0x1000, data.tobytes())
f.load(0x2000, data.tobytes())

# Run a DOT kernel
tfd = {
    "base_addr": 0,
    "depth": 100,
    "lane_count": 15,
    "exec_hints": 0x01 | (1 << 17), # DOT + Zero-Skip
    "lane_mask": 0x7FFF
}
f.run(tfd)

prof = f.profile_detailed()
print(f"Cycles: {prof['cycles']}")
print(f"Utilization: {prof['utilization']}")
print(f"Burst Wait: {prof['burst_wait_cycles']}")
print(f"Overflow Flags: {prof['overflow_flags']}")
print(f"Active Cycles (Lane 0): {prof['active_cycles'][0]}")
print(f"Skips (Lane 0): {prof['skips'][0]}")

# Run a MAXPOOL kernel
tfd_pool = {
    "depth": 50,
    "lane_count": 15,
    "exec_hints": 0x05 | (0 << 29), # MAXPOOL + MAX
    "lane_mask": 0x0001 # Only lane 0
}
f.run(tfd_pool)
results = f.results()
print(f"Pool result (Lane 0): {results[0]}")
