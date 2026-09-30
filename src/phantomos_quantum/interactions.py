"""Higher-level spatial actions built on top of landmark geometry."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .gestures import GestureEvent
from .models import GestureKind, HandObservation, SpatialPoint


@dataclass(slots=True)
class SpatialActionInterpreter:
    """Classify short motion primitives with cooldowns to prevent event storms."""

    sweep_threshold: float = 0.22
    cooldown_s: float = 0.35
    _previous: SpatialPoint | None = None
    _previous_time: float | None = None
    _previous_angle: float | None = None
    _last_emit: dict[GestureKind, float] | None = None

    def __post_init__(self) -> None:
        self._last_emit = {}

    def _emit(self, kind: GestureKind, point: SpatialPoint, confidence: float, now: float,
              value: float = 0.0) -> GestureEvent | None:
        assert self._last_emit is not None
        if now - self._last_emit.get(kind, -math.inf) < self.cooldown_s:
            return None
        self._last_emit[kind] = now
        return GestureEvent(kind, point, confidence, value)

    @staticmethod
    def _open_palm(hand: HandObservation) -> bool:
        if len(hand.landmarks) != 21:
            return False
        wrist = hand.landmarks[0]
        return all(math.hypot(hand.landmarks[t].x - wrist.x, hand.landmarks[t].y - wrist.y) >
                   math.hypot(hand.landmarks[p].x - wrist.x, hand.landmarks[p].y - wrist.y) * 1.08
                   for t, p in ((8, 6), (12, 10), (16, 14), (20, 18)))

    def update(self, hand: HandObservation, point: SpatialPoint) -> tuple[GestureEvent, ...]:
        now = hand.timestamp
        output: list[GestureEvent] = []
        if self._previous is not None and self._previous_time is not None:
            dt = max(now - self._previous_time, 1e-4)
            dx, dy, dz = point.x - self._previous.x, point.y - self._previous.y, point.z - self._previous.z
            open_palm = self._open_palm(hand)
            speed = math.hypot(dx, dy) / dt
            # Short lateral swipes are keyboard shortcuts; an open-palm sweep
            # instead controls the active window/workspace.
            if not open_palm and speed > self.sweep_threshold:
                shortcut = GestureKind.BACKSPACE if dx < -self.sweep_threshold * dt else GestureKind.SPACE
                event = self._emit(shortcut, point, hand.confidence, now, speed)
                if event:
                    output.append(event)
            # Open-palm sweep selects a window snap direction.
            if open_palm and speed > self.sweep_threshold:
                direction = "right" if abs(dx) > abs(dy) and dx > 0 else "left" if abs(dx) > abs(dy) else "down" if dy > 0 else "up"
                kind = {"left": GestureKind.WINDOW_SNAP_LEFT, "right": GestureKind.WINDOW_SNAP_RIGHT,
                        "up": GestureKind.WINDOW_MAXIMIZE, "down": GestureKind.WINDOW_MINIMIZE}[direction]
                event = self._emit(kind, point, hand.confidence, now, math.hypot(dx, dy) / dt)
                if event:
                    output.append(GestureEvent(kind, point, event.confidence, event.value, {"direction": direction}))
            # Two raised fingers closing together is a copy/scissors action.
            index_middle = math.hypot(hand.landmarks[8].x - hand.landmarks[12].x,
                                      hand.landmarks[8].y - hand.landmarks[12].y)
            if index_middle < 0.07 and self._open_palm(hand) is False:
                event = self._emit(GestureKind.COPY, point, hand.confidence, now)
                if event:
                    output.append(event)
            # A negative relative z is a push towards the camera in MediaPipe space.
            if dz < -0.08:
                kind = GestureKind.PASTE if open_palm else GestureKind.ENTER
                event = self._emit(kind, point, hand.confidence, now, abs(dz))
                if event:
                    output.append(event)

            # A changing thumb-to-index angle is a normalized radial dial delta.
            thumb, index = hand.landmarks[4], hand.landmarks[8]
            angle = math.atan2(index.y - thumb.y, index.x - thumb.x)
            if self._previous_angle is not None:
                delta = (angle - self._previous_angle + math.pi) % (2 * math.pi) - math.pi
                if abs(delta) > 0.08:
                    event = self._emit(GestureKind.VOLUME_DIAL, point, hand.confidence, now, delta)
                    if event:
                        output.append(event)
            self._previous_angle = angle
        self._previous, self._previous_time = point, now
        return tuple(output)

    def reset(self) -> None:
        self._previous = self._previous_time = self._previous_angle = None
        if self._last_emit is not None:
            self._last_emit.clear()
