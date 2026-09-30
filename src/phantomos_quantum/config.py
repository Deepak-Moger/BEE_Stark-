"""Validated, environment-friendly engine configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from typing import FrozenSet


DEFAULT_ALLOWED_ACTIONS = frozenset({
    "pointer_move", "left_click", "right_click", "middle_click", "scroll", "key", "text",
})


@dataclass(frozen=True, slots=True)
class EngineConfig:
    camera_index: int = 0
    target_fps: int = 60
    prediction_horizon_ms: float = 15.0
    min_hand_confidence: float = 0.65
    min_intent_confidence: float = 0.72
    queue_size: int = 3
    dry_run: bool = True
    require_unlock: bool = True
    allow_system_actions: bool = False
    enable_hud: bool = False
    screen_width: int = 1920
    screen_height: int = 1080
    native_backend: str = "auto"
    allowed_actions: FrozenSet[str] = field(default_factory=lambda: DEFAULT_ALLOWED_ACTIONS)

    def __post_init__(self) -> None:
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        if self.prediction_horizon_ms < 0 or self.prediction_horizon_ms > 100:
            raise ValueError("prediction horizon must be between 0 and 100ms")
        if not 0.0 <= self.min_hand_confidence <= 1.0:
            raise ValueError("min_hand_confidence must be in [0, 1]")
        if not 0.0 <= self.min_intent_confidence <= 1.0:
            raise ValueError("min_intent_confidence must be in [0, 1]")
        if self.queue_size < 1:
            raise ValueError("queue_size must be at least 1")
        if self.screen_width < 1 or self.screen_height < 1:
            raise ValueError("screen dimensions must be positive")

    @classmethod
    def from_env(cls) -> "EngineConfig":
        """Build configuration from PHANTOMOS_* variables without unsafe defaults."""
        truthy = {"1", "true", "yes", "on"}
        actions = os.getenv("PHANTOMOS_ALLOWED_ACTIONS")
        return cls(
            camera_index=int(os.getenv("PHANTOMOS_CAMERA_INDEX", "0")),
            target_fps=int(os.getenv("PHANTOMOS_TARGET_FPS", "60")),
            prediction_horizon_ms=float(os.getenv("PHANTOMOS_PREDICTION_MS", "15")),
            min_hand_confidence=float(os.getenv("PHANTOMOS_HAND_CONFIDENCE", "0.65")),
            min_intent_confidence=float(os.getenv("PHANTOMOS_INTENT_CONFIDENCE", "0.72")),
            queue_size=int(os.getenv("PHANTOMOS_QUEUE_SIZE", "3")),
            dry_run=os.getenv("PHANTOMOS_DRY_RUN", "true").lower() in truthy,
            require_unlock=os.getenv("PHANTOMOS_REQUIRE_UNLOCK", "true").lower() in truthy,
            allow_system_actions=os.getenv("PHANTOMOS_ALLOW_SYSTEM_ACTIONS", "false").lower() in truthy,
            enable_hud=os.getenv("PHANTOMOS_HUD", "false").lower() in truthy,
            screen_width=int(os.getenv("PHANTOMOS_SCREEN_WIDTH", "1920")),
            screen_height=int(os.getenv("PHANTOMOS_SCREEN_HEIGHT", "1080")),
            native_backend=os.getenv("PHANTOMOS_NATIVE_BACKEND", "auto"),
            allowed_actions=frozenset(a.strip() for a in actions.split(",") if a.strip())
            if actions
            else DEFAULT_ALLOWED_ACTIONS,
        )
