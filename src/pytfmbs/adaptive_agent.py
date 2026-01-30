"""Adaptive Runtime Agent helpers for telemetry consumption."""

from __future__ import annotations

from typing import Any, Dict, List


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
