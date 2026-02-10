#!/usr/bin/env python3
import subprocess
import json
import os
import argparse
from pathlib import Path

def run_workload(target):
    env = os.environ.copy()
    env["TFMBS_TARGET"] = target
    log_file = f"logs/telemetry_{target}.json"
    print(f"Running workload on {target}...")
    subprocess.run(["./tools/run_hw_dma_telemetry.sh"], env=env, check=True)
    # Move the output to a target-specific file
    if os.path.exists("logs/adaptive_history_dma.json"):
        os.rename("logs/adaptive_history_dma.json", log_file)
    return log_file

def main():
    parser = argparse.ArgumentParser(description="Compare Ternary Fabric Sim vs HW performance.")
    parser.add_argument("--skip-run", action="store_true", help="Skip running and just compare existing logs")
    args = parser.parse_args()

    if not args.skip_run:
        sim_log = run_workload("sim")
        # Note: running 'fpga' will fail if no real hardware is connected,
        # but the script structure is ready for it.
        try:
            hw_log = run_workload("fpga")
        except subprocess.CalledProcessError:
            print("Warning: HW run failed (likely no FPGA connected). Using mock/existing HW log if available.")
            hw_log = "logs/telemetry_fpga.json"
    else:
        sim_log = "logs/telemetry_sim.json"
        hw_log = "logs/telemetry_fpga.json"

    if not os.path.exists(sim_log) or not os.path.exists(hw_log):
        print("Error: Missing log files for comparison.")
        return

    with open(sim_log, 'r') as f:
        sim_data = json.load(f)
    with open(hw_log, 'r') as f:
        hw_data = json.load(f)

    print("\n=== Performance Comparison: Simulation vs. Hardware ===")
    print(f"{'Metric':<25} | {'Simulation':<15} | {'Hardware':<15} | {'Delta':<10}")
    print("-" * 75)

    # Simplified comparison for the hello-world/DMA test
    # In a real scenario, we'd iterate over all layers in the history.
    s_metrics = sim_data["history"][-1] if sim_data["history"] else {}
    h_metrics = hw_data["history"][-1] if hw_data["history"] else {}

    for key in ["sparsity", "fusion_sparsity"]:
        sv = s_metrics.get(key, 0)
        hv = h_metrics.get(key, 0)
        delta = hv - sv
        print(f"{key:<25} | {sv:<15.4f} | {hv:<15.4f} | {delta:<10.4f}")

    print("\nNote: Cycle counts and energy proxies are extracted from raw logs if available.")

if __name__ == "__main__":
    main()
