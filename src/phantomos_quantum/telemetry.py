"""Low-overhead phase telemetry and rolling FPS calculations."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import threading
import time
from typing import Mapping

from .models import TelemetrySnapshot


@dataclass(slots=True)
class _FrameTiming:
    phases: dict[str, float]
    started: float
    finished: float


class Telemetry:
    def __init__(self, window: int = 120) -> None:
        self._window: deque[_FrameTiming] = deque(maxlen=window)
        self._active: dict[int, _FrameTiming] = {}
        self._lock = threading.Lock()
        self.dropped_frames = 0
        self.queue_depths: dict[str, int] = {}

    def begin(self, frame_id: int) -> None:
        with self._lock:
            # A downstream stage may drop a frame before dispatch. Bound active
            # bookkeeping so prolonged overload cannot become a memory leak.
            if len(self._active) >= 512:
                stale_id = next(iter(self._active))
                self._active.pop(stale_id, None)
                self.dropped_frames += 1
            self._active[frame_id] = _FrameTiming({}, time.perf_counter(), 0.0)

    def phase(self, frame_id: int, name: str, duration_ms: float) -> None:
        with self._lock:
            timing = self._active.get(frame_id)
            if timing:
                timing.phases[name] = max(0.0, duration_ms)

    def finish(self, frame_id: int) -> None:
        with self._lock:
            timing = self._active.pop(frame_id, None)
            if timing:
                timing.finished = time.perf_counter()
                self._window.append(timing)

    def set_queue_depths(self, depths: Mapping[str, int]) -> None:
        with self._lock:
            self.queue_depths = dict(depths)

    def snapshot(self) -> TelemetrySnapshot:
        with self._lock:
            if not self._window:
                return TelemetrySnapshot({}, 0.0, 0.0, dict(self.queue_depths), self.dropped_frames)
            count = len(self._window)
            phases: dict[str, float] = {}
            for item in self._window:
                for name, value in item.phases.items():
                    phases[name] = phases.get(name, 0.0) + value / count
            total = sum(phases.values())
            elapsed = self._window[-1].finished - self._window[0].started
            fps = (count - 1) / elapsed if elapsed > 0 and count > 1 else 0.0
            return TelemetrySnapshot(phases, total, fps, dict(self.queue_depths), self.dropped_frames)

    def format_bar(self) -> str:
        snapshot = self.snapshot()
        labels = [
            ("Camera", "camera"), ("Neural Inference", "inference"),
            ("Prediction", "physics"), ("Filter", "filter"), ("OS Dispatch", "dispatch"),
        ]
        parts = [f"[{label}: {snapshot.phases_ms.get(key, 0.0):.1f}ms]" for label, key in labels]
        fps = f"{snapshot.fps:.0f} FPS" if snapshot.fps else "-- FPS"
        return " -> ".join(parts) + f" | Total: {snapshot.total_ms:.1f}ms ({fps})"
