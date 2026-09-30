"""Gaze, hand, and voice fusion into explicit spatial intents."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .gestures import GestureEvent
from .models import FaceObservation, GestureKind, HandObservation, Intent, IntentKind, SpatialPoint, VoiceCommand


@dataclass(frozen=True, slots=True)
class GazeTarget:
    point: SpatialPoint
    confidence: float


class GazeMapper:
    """Map head/gaze coordinates to a broad cursor target with confidence decay."""

    def __init__(self, smoothing: float = 0.35) -> None:
        self.smoothing = max(0.0, min(1.0, smoothing))
        self._point: SpatialPoint | None = None

    def update(self, face: FaceObservation) -> GazeTarget:
        point = SpatialPoint(max(0.0, min(1.0, face.gaze_x)), max(0.0, min(1.0, face.gaze_y)), 0.0, face.confidence)
        if self._point is not None:
            point = SpatialPoint(
                self._point.x + self.smoothing * (point.x - self._point.x),
                self._point.y + self.smoothing * (point.y - self._point.y),
                0.0,
                face.confidence,
            )
        self._point = point
        return GazeTarget(point, face.confidence)


class VoiceIntentResolver:
    """Small deterministic command parser for local speech transcripts."""

    _actions = (
        (re.compile(r"\b(close|dismiss)\s+(this|it|window)\b", re.I), "close_window"),
        (re.compile(r"\b(send|move)\s+(this|it)\s+to\s+(the\s+)?second\s+monitor\b", re.I), "move_second_monitor"),
        (re.compile(r"\b(open|edit)\s+(this|it)\s+in\s+(a\s+)?code editor\b", re.I), "open_code_editor"),
        (re.compile(r"\b(summarize|sum up)\s+(this|it|selection)\b", re.I), "summarize_selection"),
        (re.compile(r"\b(copy)\b", re.I), "copy"),
        (re.compile(r"\b(paste)\b", re.I), "paste"),
    )

    def resolve(self, command: VoiceCommand, target: SpatialPoint | None) -> Intent | None:
        if not command.is_final or command.confidence < 0.55:
            return None
        for pattern, action in self._actions:
            if pattern.search(command.text):
                kind = IntentKind.SYSTEM if action in {"copy", "paste", "close_window"} else IntentKind.VOICE
                return Intent(kind, action, command.timestamp, target, confidence=command.confidence,
                              text=command.text, metadata={"requires_target": True})
        return None


class MultimodalFusion:
    """Fuse precise hand events with optional gaze and voice context."""

    def __init__(self, gaze: GazeMapper | None = None, voice: VoiceIntentResolver | None = None) -> None:
        self.gaze = gaze or GazeMapper()
        self.voice = voice or VoiceIntentResolver()
        self._gaze_target: GazeTarget | None = None
        self._hand_target: SpatialPoint | None = None

    def set_gaze(self, face: FaceObservation | None) -> None:
        if face and face.confidence > 0.45:
            self._gaze_target = self.gaze.update(face)

    def set_hand_target(self, point: SpatialPoint | None) -> None:
        if point is not None:
            self._hand_target = point

    def hand_intent(self, event: GestureEvent) -> Intent | None:
        point = event.point or self._hand_target or (self._gaze_target.point if self._gaze_target else None)
        confidence = event.confidence
        mapping = {
            GestureKind.MOVE: (IntentKind.POINTER_MOVE, "pointer_move"),
            GestureKind.LEFT_CLICK: (IntentKind.BUTTON, "left_click"),
            GestureKind.DRAG_START: (IntentKind.BUTTON, "left_click"),
            GestureKind.DRAG_END: (IntentKind.BUTTON, "left_click"),
            GestureKind.RIGHT_CLICK: (IntentKind.BUTTON, "right_click"),
            GestureKind.MIDDLE_CLICK: (IntentKind.BUTTON, "middle_click"),
            GestureKind.SCROLL: (IntentKind.SCROLL, "scroll"),
            GestureKind.BACKSPACE: (IntentKind.KEY, "backspace"),
            GestureKind.SPACE: (IntentKind.KEY, "space"),
            GestureKind.ENTER: (IntentKind.KEY, "enter"),
            GestureKind.ESCAPE: (IntentKind.KEY, "escape"),
            GestureKind.COPY: (IntentKind.KEY, "c"),
            GestureKind.PASTE: (IntentKind.KEY, "v"),
            GestureKind.WINDOW_SNAP_LEFT: (IntentKind.WINDOW, "window_snap_left"),
            GestureKind.WINDOW_SNAP_RIGHT: (IntentKind.WINDOW, "window_snap_right"),
            GestureKind.WINDOW_MAXIMIZE: (IntentKind.WINDOW, "window_maximize"),
            GestureKind.WINDOW_MINIMIZE: (IntentKind.WINDOW, "window_minimize"),
            GestureKind.APP_SWITCHER: (IntentKind.WINDOW, "app_switcher"),
            GestureKind.LOCK: (IntentKind.SYSTEM, "lock"),
            GestureKind.VOLUME_DIAL: (IntentKind.SYSTEM, "volume_dial"),
            GestureKind.BRIGHTNESS_DIAL: (IntentKind.SYSTEM, "brightness_dial"),
        }
        mapped = mapping.get(event.kind)
        if mapped is None:
            return None
        kind, action = mapped
        # The initial left-pinch event already pressed the button; drag-start is
        # a state transition for UI locking, not a second native mouse-down.
        if event.kind is GestureKind.DRAG_START:
            return None
        metadata = dict(event.metadata)
        if event.kind in {GestureKind.LEFT_CLICK, GestureKind.DRAG_START, GestureKind.DRAG_END}:
            metadata.setdefault("button", "left")
            metadata.setdefault("pressed", event.metadata.get("phase") != "up")
        elif event.kind is GestureKind.RIGHT_CLICK:
            metadata.update(button="right", pressed=event.metadata.get("phase") != "up")
        elif event.kind is GestureKind.MIDDLE_CLICK:
            metadata.update(button="middle", pressed=event.metadata.get("phase") != "up")
        elif event.kind is GestureKind.SCROLL:
            metadata.setdefault("dy", event.value)
        elif event.kind is GestureKind.COPY:
            metadata.update(shortcut=("ctrl",), pressed=True)
        elif event.kind is GestureKind.PASTE:
            metadata.update(shortcut=("ctrl",), pressed=True)
        elif event.kind is GestureKind.DICTATION:
            return None
        return Intent(kind, action, point=point, value=event.value, confidence=confidence, metadata=metadata)

    def voice_intent(self, command: VoiceCommand) -> Intent | None:
        target = self._hand_target or (self._gaze_target.point if self._gaze_target else None)
        return self.voice.resolve(command, target)
