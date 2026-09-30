"""Core immutable data contracts shared by all PhantomOS stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from time import monotonic
from typing import Any, Mapping, Sequence


class GestureKind(Enum):
    NONE = auto()
    MOVE = auto()
    LEFT_CLICK = auto()
    DRAG_START = auto()
    DRAG_END = auto()
    RIGHT_CLICK = auto()
    MIDDLE_CLICK = auto()
    SCROLL = auto()
    DICTATION = auto()
    BACKSPACE = auto()
    SPACE = auto()
    ENTER = auto()
    ESCAPE = auto()
    WINDOW_SNAP_LEFT = auto()
    WINDOW_SNAP_RIGHT = auto()
    WINDOW_MAXIMIZE = auto()
    WINDOW_MINIMIZE = auto()
    APP_SWITCHER = auto()
    COPY = auto()
    PASTE = auto()
    LOCK = auto()
    VOLUME_DIAL = auto()
    BRIGHTNESS_DIAL = auto()


class IntentKind(Enum):
    POINTER_MOVE = auto()
    BUTTON = auto()
    SCROLL = auto()
    KEY = auto()
    TEXT = auto()
    WINDOW = auto()
    SYSTEM = auto()
    VOICE = auto()
    NOOP = auto()


@dataclass(frozen=True, slots=True)
class Landmark3D:
    """A normalized hand/face landmark. x/y are [0, 1], z is camera-relative."""

    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0

    def __post_init__(self) -> None:
        for value in (self.x, self.y, self.z, self.visibility):
            if not isinstance(value, (int, float)):
                raise TypeError("landmark coordinates must be numeric")


@dataclass(frozen=True, slots=True)
class HandObservation:
    landmarks: tuple[Landmark3D, ...]
    handedness: str = "unknown"
    confidence: float = 0.0
    timestamp: float = field(default_factory=monotonic)
    frame_id: int = 0

    def __post_init__(self) -> None:
        if len(self.landmarks) not in (0, 21):
            raise ValueError("MediaPipe hand observations contain 21 landmarks")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    @property
    def index_tip(self) -> Landmark3D | None:
        return self.landmarks[8] if len(self.landmarks) == 21 else None

    @property
    def thumb_tip(self) -> Landmark3D | None:
        return self.landmarks[4] if len(self.landmarks) == 21 else None

    @property
    def wrist(self) -> Landmark3D | None:
        return self.landmarks[0] if len(self.landmarks) == 21 else None


@dataclass(frozen=True, slots=True)
class FaceObservation:
    gaze_x: float
    gaze_y: float
    head_yaw: float = 0.0
    head_pitch: float = 0.0
    confidence: float = 0.0
    timestamp: float = field(default_factory=monotonic)


@dataclass(frozen=True, slots=True)
class VoiceCommand:
    text: str
    confidence: float = 0.0
    timestamp: float = field(default_factory=monotonic)
    is_final: bool = True


@dataclass(frozen=True, slots=True)
class SensorFrame:
    """A frame crossing a pipeline boundary."""

    frame_id: int
    timestamp: float
    width: int
    height: int
    payload: Any = None


@dataclass(frozen=True, slots=True)
class SpatialPoint:
    x: float
    y: float
    z: float = 0.0
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class Intent:
    kind: IntentKind
    action: str
    timestamp: float = field(default_factory=monotonic)
    point: SpatialPoint | None = None
    value: float = 0.0
    text: str = ""
    confidence: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DispatchResult:
    accepted: bool
    dry_run: bool
    action: str
    reason: str = ""
    timestamp: float = field(default_factory=monotonic)


@dataclass(frozen=True, slots=True)
class PhaseSample:
    name: str
    duration_ms: float
    timestamp: float = field(default_factory=monotonic)


@dataclass(frozen=True, slots=True)
class TelemetrySnapshot:
    phases_ms: Mapping[str, float]
    total_ms: float
    fps: float
    queue_depths: Mapping[str, int]
    dropped_frames: int
    timestamp: float = field(default_factory=monotonic)
