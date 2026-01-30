#!/usr/bin/env python3
"""Run the tfmbs-to-linalg mlir-opt regression and verify the lowered IR."""
import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def find_mlir_opt(arg_path: Optional[str]) -> Path:
    if arg_path:
        candidate = Path(arg_path)
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Specified mlir-opt binary not found: {candidate}")
    env_path = os.environ.get("MLIR_OPT")
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate
    fallback = REPO_ROOT.parent / "llvm-project" / "build-shared" / "bin" / "mlir-opt"
    if fallback.is_file():
        return fallback
    raise FileNotFoundError("mlir-opt binary not found. Pass --mlir-opt or set MLIR_OPT.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Regression runner for tfmbs-to-linalg pass")
    parser.add_argument("--mlir-opt", help="Path to the mlir-opt binary")
    parser.add_argument("--plugin", default=REPO_ROOT / "build" / "libtfmbs_plugin.dylib",
                        help="Path to the tfmbs plugin")
    parser.add_argument("--pipeline", default="builtin.module(tfmbs-to-linalg)",
                        help="Pass pipeline string (include the anchor op)")
    parser.add_argument("--mlir", default=REPO_ROOT / "tests" / "mlir" / "tfmbs_to_linalg.mlir",
                        help="MLIR test file")
    args = parser.parse_args()

    mlir_opt = find_mlir_opt(args.mlir_opt)
    plugin = Path(args.plugin)
    if not plugin.is_file():
        raise FileNotFoundError(f"TFMBS plugin not built yet: {plugin}")
    mlir_file = Path(args.mlir)
    if not mlir_file.is_file():
        raise FileNotFoundError(f"MLIR test file not found: {mlir_file}")

    cmd = [
        str(mlir_opt),
        f"--load-dialect-plugin={plugin}",
        f"--pass-pipeline={args.pipeline}",
        str(mlir_file),
    ]
    print("Running:", " ".join(cmd))
    process = subprocess.run(cmd, capture_output=True, text=True)
    if process.returncode != 0:
        print(process.stdout, process.stderr, sep="", end="")
        raise SystemExit(process.returncode)
    if "linalg.matmul" not in process.stdout:
        print(process.stdout)
        raise SystemExit("Lowered IR did not include linalg.matmul.")
    if "tfmbs.gemv" in process.stdout:
        print(process.stdout)
        raise SystemExit("TFMBS ops still present after running the pipeline.")
    print(process.stdout)
    print("tfmbs-to-linalg pipeline passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
