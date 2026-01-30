"""Adaptive Runtime Agent helpers for telemetry consumption."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


class AdaptiveRuntimeAgent:
    """Simple runtime agent that consumes telemetry dictionaries."""

    def __init__(self) -> None:
        self.telemetry_history: List[Dict[str, Any]] = []

    def consume(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fusion metadata and record it for later analysis."""
        fusion_order = list(telemetry.get("fusion_order", []))
        fusion_sparsity = telemetry.get("fusion_sparsity", telemetry.get("sparsity"))
        entry = {
            "layer": telemetry.get("layer"),
            "tile_mask": telemetry.get("tile_mask"),
            "fusion_order": fusion_order,
            "fusion_sparsity": fusion_sparsity,
        }
        self.telemetry_history.append(entry)
        return entry

    def save_history(self, path: str | Path) -> None:
        """Persist the telemetry history for dashboards."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as handle:
            json.dump(self.telemetry_history, handle, indent=2)

    @classmethod
    def load_history(cls, path: str | Path) -> List[Dict[str, Any]]:
        """Read a previously saved telemetry history."""
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def extend_history(self, entries: Iterable[Dict[str, Any]]) -> None:
        """Merge other telemetry entries into the agent history."""
        self.telemetry_history.extend(entries)
