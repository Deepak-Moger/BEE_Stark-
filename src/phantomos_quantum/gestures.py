"""Debounced hand gesture recognizer and spatial shortcut state machine."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Iterable

from .models import GestureKind, HandObservation, SpatialPoint
from .spatial import DepthEngine, distance


def _point(hand: HandObservation, index: int) -> SpatialPoint:
    item = hand.landmarks[index]
    return SpatialPoint(item.x, item.y, item.z, item.visibility)


def _dist(hand: HandObservation, a: int, b: int) -> float:
    return distance(_point(hand, a), _point(hand, b))


@dataclass(frozen=True, slots=True)
class GestureEvent:
    kind: GestureKind
    point: SpatialPoint | None
    confidence: float
    value: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class _PinchState:
    active: bool = False
    started_at: float = 0.0
    dragged: bool = False


class GestureRecognizer:
    """Recognizer with hysteresis, dwell timing and one-shot transitions."""

    def __init__(self, depth: DepthEngine | None = None, pinch_on: float = 0.065,
                 pinch_off: float = 0.09, drag_hold_s: float = 0.2) -> None:
        self.depth = depth or DepthEngine()
        self.pinch_on, self.pinch_off, self.drag_hold_s = pinch_on, pinch_off, drag_hold_s
        self._pinches = {"left": _PinchState(), "right": _PinchState(), "middle": _PinchState(), "dictation": _PinchState()}
        self._last_point: SpatialPoint | None = None
        self._last_time: float | None = None
        self._fist_since: float | None = None
        self._last_lock_emit = 0.0
        self._last_switcher_emit = 0.0

    @staticmethod
    def _extended(hand: HandObservation, tip: int, pip: int, mcp: int) -> bool:
        # In normalized camera coordinates a fingertip farther from the wrist than
        # the PIP is extended; this is tolerant of horizontal hand rotation.
        wrist = _point(hand, 0)
        return distance(_point(hand, tip), wrist) > distance(_point(hand, pip), wrist) * 1.08 and \
            distance(_point(hand, tip), _point(hand, mcp)) > distance(_point(hand, pip), _point(hand, mcp))

    def _fingers(self, hand: HandObservation) -> tuple[bool, bool, bool, bool, bool]:
        if len(hand.landmarks) != 21:
            return (False,) * 5
        # Thumb uses lateral distance; the remaining four use extension tests.
        thumb = _dist(hand, 4, 5) > _dist(hand, 3, 5) * 1.05
        return (
            thumb,
            self._extended(hand, 8, 6, 5),
            self._extended(hand, 12, 10, 9),
            self._extended(hand, 16, 14, 13),
            self._extended(hand, 20, 18, 17),
        )

    def _pinch(self, hand: HandObservation, finger_tip: int, key: str, now: float) -> list[GestureEvent]:
        state = self._pinches[key]
        metric = _dist(hand, finger_tip, 4)
        if state.active:
            active = metric < self.pinch_off
        else:
            active = metric < self.pinch_on
        events: list[GestureEvent] = []
        point = self.depth.pointer(hand)
        if active and not state.active:
            state.active = True
            state.started_at = now
            state.dragged = False
            if key == "left":
                events.append(GestureEvent(GestureKind.LEFT_CLICK, point, hand.confidence, metadata={"phase": "down"}))
            elif key == "right":
                events.append(GestureEvent(GestureKind.RIGHT_CLICK, point, hand.confidence))
            elif key == "middle":
                events.append(GestureEvent(GestureKind.MIDDLE_CLICK, point, hand.confidence))
            elif key == "dictation":
                events.append(GestureEvent(GestureKind.DICTATION, point, hand.confidence, metadata={"phase": "down"}))
        elif active and key == "left" and not state.dragged and now - state.started_at >= self.drag_hold_s:
            state.dragged = True
            events.append(GestureEvent(GestureKind.DRAG_START, point, hand.confidence, metadata={"phase": "down"}))
        elif not active and state.active:
            if key == "left":
                if state.dragged:
                    events.append(GestureEvent(GestureKind.DRAG_END, point, hand.confidence, metadata={"phase": "up"}))
                else:
                    events.append(GestureEvent(GestureKind.LEFT_CLICK, point, hand.confidence, metadata={"phase": "up"}))
            elif key == "right":
                events.append(GestureEvent(GestureKind.RIGHT_CLICK, point, hand.confidence, metadata={"phase": "up"}))
            elif key == "middle":
                events.append(GestureEvent(GestureKind.MIDDLE_CLICK, point, hand.confidence, metadata={"phase": "up"}))
            elif key == "dictation":
                events.append(GestureEvent(GestureKind.DICTATION, point, hand.confidence, metadata={"phase": "up"}))
            state.active = False
            state.dragged = False
        return events

    def update(self, hand: HandObservation) -> tuple[GestureEvent, ...]:
        if len(hand.landmarks) != 21 or hand.confidence <= 0.0:
            return ()
        now = hand.timestamp
        point = self.depth.pointer(hand)
        events: list[GestureEvent] = []
        # Index is pointer motion regardless of whether a click is active.
        if point is not None:
            events.append(GestureEvent(GestureKind.MOVE, point, hand.confidence))
        events.extend(self._pinch(hand, 8, "left", now))
        events.extend(self._pinch(hand, 12, "right", now))
        events.extend(self._pinch(hand, 16, "middle", now))
        events.extend(self._pinch(hand, 20, "dictation", now))

        fingers = self._fingers(hand)
        index, middle, ring, pinky = fingers[1:]
        if index and middle and not ring and not pinky and point is not None:
            dy = 0.0
            if self._last_point is not None and self._last_time is not None:
                dt = max(now - self._last_time, 1e-4)
                dy = (point.y - self._last_point.y) / dt
            events.append(GestureEvent(GestureKind.SCROLL, point, hand.confidence, value=dy,
                                       metadata={"dx": 0, "dy": -dy * 0.08}))

        # Five-finger open palm is the basis for workspace actions; the action layer
        # may add directional sweep classification from the point history.
        if all(fingers) and now - self._last_switcher_emit >= 0.75:
            self._last_switcher_emit = now
            events.append(GestureEvent(GestureKind.APP_SWITCHER, point, hand.confidence))
        if not any(fingers):
            if self._fist_since is None:
                self._fist_since = now
            if now - self._fist_since >= 2.0 and now - self._last_lock_emit > 2.0:
                self._last_lock_emit = now
                events.append(GestureEvent(GestureKind.LOCK, point, hand.confidence, value=2.0))
        else:
            self._fist_since = None

        self._last_point, self._last_time = point, now
        return tuple(events)

    def reset(self) -> None:
        for state in self._pinches.values():
            state.active = state.dragged = False
            state.started_at = 0.0
        self._last_point = None
        self._last_time = None
        self._fist_since = None
        self._last_switcher_emit = 0.0
