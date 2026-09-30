"""Window/workspace intent contracts without unsafe platform-specific shell calls."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .models import Intent, IntentKind, SpatialPoint


class SnapDirection(Enum):
    LEFT = "left"
    RIGHT = "right"
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


@dataclass(frozen=True, slots=True)
class WindowTarget:
    title: str
    bounds: tuple[int, int, int, int]
    monitor: int = 0


class WorkspaceManager:
    """Produces auditable workspace intents for a platform compositor adapter."""

    def snap(self, direction: SnapDirection, confidence: float) -> Intent:
        return Intent(IntentKind.WINDOW, f"window_snap_{direction.value}", confidence=confidence,
                      metadata={"direction": direction.value})

    def move_to_monitor(self, monitor: int, confidence: float, target: SpatialPoint | None = None) -> Intent:
        if monitor < 0:
            raise ValueError("monitor index must be nonnegative")
        return Intent(IntentKind.WINDOW, "move_monitor", confidence=confidence, point=target,
                      metadata={"monitor": monitor})

    def app_switcher(self, confidence: float, target: SpatialPoint | None = None) -> Intent:
        return Intent(IntentKind.WINDOW, "app_switcher", confidence=confidence, point=target)
