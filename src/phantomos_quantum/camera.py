"""Camera sources with optional OpenCV acceleration and deterministic test source."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterator, Protocol

from .errors import DependencyUnavailable
from .models import SensorFrame


class FrameSource(Protocol):
    def frames(self, stop_event: object) -> Iterator[SensorFrame]: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class SyntheticFrameSource:
    """A clock-paced source for CI, demos, and pipeline profiling."""

    width: int = 640
    height: int = 480
    fps: int = 60
    _closed: bool = False

    def frames(self, stop_event: object) -> Iterator[SensorFrame]:
        interval = 1.0 / max(1, self.fps)
        frame_id = 0
        next_frame = time.perf_counter()
        while not self._closed and not getattr(stop_event, "is_set", lambda: False)():
            now = time.perf_counter()
            if now < next_frame:
                time.sleep(next_frame - now)
            now = time.perf_counter()
            yield SensorFrame(frame_id, now, self.width, self.height, payload=None)
            frame_id += 1
            next_frame += interval
            if next_frame < now - interval:
                next_frame = now + interval

    def close(self) -> None:
        self._closed = True


class OpenCVFrameSource:
    """OpenCV source configured for low-latency capture.

    The backend is selected explicitly where possible. Camera ownership remains in
    this object so it can be closed deterministically on every platform.
    """

    def __init__(self, index: int = 0, fps: int = 60, width: int = 1280, height: int = 720) -> None:
        self.index, self.fps, self.width, self.height = index, fps, width, height
        self._capture = None
        self._closed = False

    def _open(self) -> object:
        try:
            import cv2
        except ImportError as exc:
            raise DependencyUnavailable("OpenCV is required for camera capture; install .[vision]") from exc
        backend = getattr(cv2, "CAP_ANY", 0)
        if hasattr(cv2, "CAP_DSHOW") and __import__("sys").platform.startswith("win"):
            backend = cv2.CAP_DSHOW
        elif hasattr(cv2, "CAP_AVFOUNDATION") and __import__("sys").platform == "darwin":
            backend = cv2.CAP_AVFOUNDATION
        elif hasattr(cv2, "CAP_V4L2") and __import__("sys").platform.startswith("linux"):
            backend = cv2.CAP_V4L2
        capture = cv2.VideoCapture(self.index, backend)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"unable to open camera {self.index}")
        self._capture = capture
        return capture

    def frames(self, stop_event: object) -> Iterator[SensorFrame]:
        capture = self._open()
        frame_id = 0
        try:
            while not self._closed and not getattr(stop_event, "is_set", lambda: False)():
                ok, image = capture.read()
                if not ok:
                    continue
                height, width = image.shape[:2]
                yield SensorFrame(frame_id, time.perf_counter(), width, height, payload=image)
                frame_id += 1
        finally:
            self.close()

    def close(self) -> None:
        self._closed = True
        if self._capture is not None:
            self._capture.release()
            self._capture = None
