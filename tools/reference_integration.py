#!/usr/bin/env python3
"""Produce a reproducible mock GEMV integration demonstrating the Fabric interposer."""

from __future__ import annotations

import csv
import os
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "bin"
TOOLS_DIR = ROOT / "tools"
LOG_DIR = ROOT / "logs"
PLOT_DIR = ROOT / "plots"
CSV_PATH = LOG_DIR / "reference_integration.csv"
PLOT_PATH = PLOT_DIR / "reference_integration_summary.png"


def shlib_ext() -> str:
    return ".dylib" if platform.system() == "Darwin" else ".so"


def build_targets() -> None:
    targets = [
        "bin/mock_llama",
        f"bin/libtfmbs_device{shlib_ext()}",
        f"bin/libtfmbs_intercept{shlib_ext()}",
    ]
    subprocess.run(["make", *targets], check=True)


def run_process(command: list[str], env: dict[str, str] | None, log_path: Path) -> float:
    start = time.perf_counter()
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    duration_ms = (time.perf_counter() - start) * 1000
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path.write_text(result.stdout + result.stderr)
    result.check_returncode()
    return duration_ms


def try_plot(durations: list[tuple[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        print("📉 matplotlib not installed; skipping chart.")
        return

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    labels, values = zip(*durations)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=["#2f7bbf", "#1f9d4d"])
    ax.set_ylabel("Milliseconds")
    ax.set_title("Mock Llama: CPU vs. Fabric-Accelerated")
    for idx, value in enumerate(values):
        ax.text(idx, value + max(values) * 0.01, f"{value:.1f} ms", ha="center")
    fig.tight_layout()
    fig.savefig(PLOT_PATH)
    print(f"📊 Saved chart to {PLOT_PATH.relative_to(ROOT)}")


def main() -> int:
    build_targets()

    baseline_duration = run_process(
        [str(BIN_DIR / "mock_llama")],
        env=None,
        log_path=LOG_DIR / "reference_integration_cpu.log",
    )

    fabric_env = os.environ.copy()
    fabric_env["FABRIC_SHORT_CIRCUIT"] = "1"
    fabric_env["TFMBS_DEBUG"] = "1"

    fabric_duration = run_process(
        [str(TOOLS_DIR / "tfmbs-run"), str(BIN_DIR / "mock_llama")],
        env=fabric_env,
        log_path=LOG_DIR / "reference_integration_fabric.log",
    )

    durations = [
        ("CPU Baseline", baseline_duration),
        ("Fabric Interposer", fabric_duration),
    ]

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["mode", "duration_ms", "notes"])
        writer.writeheader()
        writer.writerow(
            {
                "mode": "cpu_baseline",
                "duration_ms": f"{baseline_duration:.2f}",
                "notes": "Pure CPU mock GEMV run",
            }
        )
        writer.writerow(
            {
                "mode": "fabric_interposer",
                "duration_ms": f"{fabric_duration:.2f}",
                "notes": "LD_PRELOAD/libtfmbs_intercept run",
            }
        )

    try_plot(durations)

    speedup = baseline_duration / fabric_duration if fabric_duration > 0 else float("inf")

    print("✅ Reference integration complete.")
    print(f"- Baseline: {baseline_duration:.2f} ms")
    print(f"- Fabric:   {fabric_duration:.2f} ms ({speedup:.2f}x faster)")
    print(f"- Logs:     {LOG_DIR.relative_to(ROOT)}")
    print(f"- CSV:      {CSV_PATH.relative_to(ROOT)}")
    if PLOT_PATH.exists():
        print(f"- Chart:    {PLOT_PATH.relative_to(ROOT)}")
    else:
        print("- Chart:    (matplotlib missing; install it to generate the bar chart)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
