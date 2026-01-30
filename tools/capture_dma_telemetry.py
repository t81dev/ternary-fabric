#!/usr/bin/env python3
"""Query fabric metrics and save telemetry entries for adaptive_dashboard.py."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pytfmbs.adaptive_agent import AdaptiveRuntimeAgent

try:
    import ctypes
except ImportError:  # pragma: no cover
    raise


def default_shlib_suffix() -> str:
    return ".dylib" if platform.system() == "Darwin" else ".so"


class FabricMetrics(ctypes.Structure):  # type: ignore[misc]
    _fields_ = [
        ("zero_skips", ctypes.c_long),
        ("total_ops", ctypes.c_long),
        ("lanes_used", ctypes.c_int),
        ("sim_cycle_reduction", ctypes.c_double),
        ("pool_used", ctypes.c_size_t),
        ("pool_total", ctypes.c_size_t),
        ("eviction_count", ctypes.c_int),
        ("active_ops", ctypes.c_uint64),
        ("mem_reads", ctypes.c_uint64),
        ("mem_writes", ctypes.c_uint64),
        ("broadcasts", ctypes.c_uint64),
        ("residency_hits", ctypes.c_uint64),
        ("residency_misses", ctypes.c_uint64),
        ("tile_local_reuse", ctypes.c_uint64),
        ("cycles", ctypes.c_long),
        ("fabric_cost", ctypes.c_double),
        ("semantic_efficiency", ctypes.c_double),
        ("economic_efficiency", ctypes.c_double),
        ("fallback_count", ctypes.c_uint64),
        ("offload_count", ctypes.c_uint64),
    ]


def load_metrics(lib_path: Path) -> FabricMetrics:
    lib = ctypes.CDLL(str(lib_path))
    lib.fabric_get_metrics.argtypes = [ctypes.POINTER(FabricMetrics)]
    lib.fabric_get_metrics.restype = None
    metrics = FabricMetrics()
    lib.fabric_get_metrics(ctypes.byref(metrics))
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="Emit telemetry JSON from fabric metrics.")
    parser.add_argument("--libtfmbs", type=Path, help="Path to libtfmbs_device shared library")
    parser.add_argument("--output", type=Path, default=Path("logs/adaptive_history_dma.json"))
    parser.add_argument("--layer", default="dma_driver_hw_sim")
    args = parser.parse_args()

    shlib = args.libtfmbs or (Path("bin") / f"libtfmbs_device{default_shlib_suffix()}")
    if not shlib.exists():
        raise FileNotFoundError(f"Shared lib not found: {shlib}")

    metrics = load_metrics(shlib)
    delta = metrics.offload_count + metrics.fallback_count
    fusion_sparsity = float(metrics.offload_count) / delta if delta else 0.0
    runtime_entry = {
        "layer": args.layer,
        "tile_mask": 0,
        "fusion_order": [args.layer],
        "fusion_sparsity": fusion_sparsity,
        "sparsity": 1.0 - fusion_sparsity,
    }

    agent = AdaptiveRuntimeAgent()
    agent.consume(runtime_entry)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    agent.save_history(args.output)
    print(f"Wrote telemetry runtime history to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
