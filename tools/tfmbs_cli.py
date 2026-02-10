#!/usr/bin/env python3
import argparse
import sys
import os
import json
from pathlib import Path

# Ensure we can import pytfmbs
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def main():
    parser = argparse.ArgumentParser(prog="tfmbs", description="TFMBS CLI Tool")
    subparsers = parser.add_subparsers(dest="command")

    # 'run' command
    run_parser = subparsers.add_parser("run", help="Run a model on TFMBS")
    run_parser.add_argument("--model", type=str, required=True, help="Path to model file or torchvision model name")
    run_parser.add_argument("--quant", choices=["ternary"], default="ternary", help="Quantization type")
    run_parser.add_argument("--target", choices=["sim", "fpga"], default="sim", help="Execution target")
    run_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")

    args = parser.parse_args()

    if args.command == "run":
        run_model(args)
    else:
        parser.print_help()

def run_model(args):
    print(f"[*] Initializing TFMBS on {args.target}...")

    if args.target == "fpga":
        os.environ["FABRIC_HARDWARE_PATH"] = "1"
    else:
        os.environ.pop("FABRIC_HARDWARE_PATH", None)

    try:
        import torch
        import torchvision
        from pytfmbs.torch_backend import register
        from pytfmbs.adaptive_agent import AdaptiveRuntimeAgent

        register()

        # Load model
        if hasattr(torchvision.models, args.model):
            print(f"[*] Loading torchvision model: {args.model}")
            model = getattr(torchvision.models, args.model)(weights=None)
        elif os.path.exists(args.model):
            print(f"[*] Loading model file: {args.model}")
            model = torch.load(args.model)
        else:
            print(f"Error: Model {args.model} not found.")
            return

        model.eval()
        compiled_model = torch.compile(model, backend="tfmbs")

        # Run dummy inference to trigger offload
        print("[*] Running inference...")
        dummy_input = torch.randn(args.batch_size, 3, 224, 224) if "resnet" in args.model.lower() else torch.randn(args.batch_size, 784)

        with torch.no_grad():
            output = compiled_model(dummy_input)

        # Report via AdaptiveRuntimeAgent
        agent = AdaptiveRuntimeAgent()
        # In a real run, the interposer or device lib would have filled the history
        # For the CLI, we can fetch metrics from the device lib
        import ctypes
        try:
            from pytfmbs.torch_integration import default_shlib_suffix
            lib_path = ROOT / "bin" / f"libtfmbs_device{default_shlib_suffix()}"
            if lib_path.exists():
                from tools.capture_dma_telemetry import load_metrics
                metrics = load_metrics(lib_path)

                print("\n" + "="*40)
                print("TFMBS EXECUTION SUMMARY")
                print("="*40)
                print(f"Total Operations:   {metrics.total_ops}")
                print(f"Zero-Skips:         {metrics.zero_skips}")
                print(f"Sparsity Reduction: {100.0 * metrics.zero_skips / metrics.total_ops if metrics.total_ops else 0:.2f}%")
                print(f"Active Operations:  {metrics.active_ops}")
                print(f"Memory Reads:       {metrics.mem_reads}")
                print(f"Memory Writes:      {metrics.mem_writes}")
                print(f"Economic Efficiency: {metrics.economic_efficiency:.4f}")
                print("="*40)
        except Exception as e:
            print(f"[*] Results captured. (Metrics retrieval failed: {e})")

    except ImportError as e:
        print(f"Error: Dependencies missing ({e}). Ensure torch and torchvision are installed.")

if __name__ == "__main__":
    main()
