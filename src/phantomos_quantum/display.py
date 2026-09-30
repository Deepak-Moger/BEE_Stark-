"""Multi-monitor desktop coordinate mapping and edge-warp policy."""

from __future__ import annotations

from dataclasses import dataclass

from .models import SpatialPoint


@dataclass(frozen=True, slots=True)
class Monitor:
    x: int
    y: int
    width: int
    height: int
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.scale <= 0:
            raise ValueError("monitor dimensions and scale must be positive")


class MultiMonitorLayout:
    """Map normalized spatial coordinates into a virtual desktop rectangle."""

    def __init__(self, monitors: tuple[Monitor, ...]) -> None:
        if not monitors:
            raise ValueError("at least one monitor is required")
        self.monitors = monitors
        self.left = min(m.x for m in monitors)
        self.top = min(m.y for m in monitors)
        self.right = max(m.x + m.width for m in monitors)
        self.bottom = max(m.y + m.height for m in monitors)

    def map(self, point: SpatialPoint) -> tuple[int, int]:
        x = self.left + max(0.0, min(1.0, point.x)) * (self.right - self.left - 1)
        y = self.top + max(0.0, min(1.0, point.y)) * (self.bottom - self.top - 1)
        return round(x), round(y)

    def warp(self, point: SpatialPoint, velocity_x: float, edge_threshold: float = 0.03) -> tuple[int, int]:
        """Map a point and preserve fast boundary flicks across the virtual desktop."""
        normalized_x = point.x
        if velocity_x > 0 and normalized_x >= 1.0 - edge_threshold:
            normalized_x = 1.0
        elif velocity_x < 0 and normalized_x <= edge_threshold:
            normalized_x = 0.0
        return self.map(SpatialPoint(normalized_x, point.y, point.z, point.confidence))
