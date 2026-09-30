"""Bounded five-stage threaded hardware pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import queue
import threading
import time
from typing import Callable, Generic, TypeVar

from .camera import FrameSource
from .filters import TrajectoryPredictor
from .fusion import MultimodalFusion
from .gestures import GestureRecognizer
from .interactions import SpatialActionInterpreter
from .inference import InferenceBackend, InferenceResult
from .models import HandObservation, Intent, SensorFrame, SpatialPoint, VoiceCommand
from .native import NativeDispatcher
from .spatial import DepthEngine
from .telemetry import Telemetry

T = TypeVar("T")


class LatestQueue(Generic[T]):
    """Bounded queue that drops the oldest item under camera back-pressure."""

    def __init__(self, maxsize: int) -> None:
        self._queue: queue.Queue[T] = queue.Queue(maxsize=maxsize)
        self.dropped = 0

    def put_latest(self, item: T) -> None:
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                pass
            self.dropped += 1
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                self.dropped += 1

    def get(self, timeout: float = 0.1) -> T:
        return self._queue.get(timeout=timeout)

    def task_done(self) -> None:
        self._queue.task_done()

    def qsize(self) -> int:
        return self._queue.qsize()


@dataclass(frozen=True, slots=True)
class InferencePacket:
    frame: SensorFrame
    result: InferenceResult


@dataclass(frozen=True, slots=True)
class PhysicsPacket:
    frame: SensorFrame
    hands: tuple[HandObservation, ...]
    face: object | None


@dataclass(frozen=True, slots=True)
class IntentPacket:
    frame_id: int
    intents: tuple[Intent, ...]


class PhysicsStage:
    def __init__(self, horizon_ms: float = 15.0, min_confidence: float = 0.65) -> None:
        self.depth = DepthEngine()
        self.min_confidence = min_confidence
        self._predictors: dict[str, list[TrajectoryPredictor]] = {}
        self._horizon_ms = horizon_ms

    def _predict_hand(self, hand: HandObservation, track_key: str | None = None) -> HandObservation:
        key = track_key or hand.handedness
        predictors = self._predictors.setdefault(key, [TrajectoryPredictor(self._horizon_ms) for _ in range(21)])
        projected = tuple(
            predictor.update(SpatialPoint(point.x, point.y, point.z, point.visibility), hand.timestamp)
            for predictor, point in zip(predictors, hand.landmarks)
        )
        from .models import Landmark3D
        landmarks = tuple(Landmark3D(p.x, p.y, p.z, point.visibility) for p, point in zip(projected, hand.landmarks))
        return HandObservation(landmarks, hand.handedness, hand.confidence, hand.timestamp, hand.frame_id)

    def process(self, packet: InferencePacket) -> PhysicsPacket:
        hands = tuple(self._predict_hand(hand, f"{hand.handedness}:{index}")
                      for index, hand in enumerate(packet.result.hands)
                      if hand.confidence >= self.min_confidence)
        return PhysicsPacket(packet.frame, hands, packet.result.face)


class Pipeline:
    """Capture -> inference -> physics -> fusion -> dispatch.

    Threads are daemonized for crash containment, but `stop()` always closes the
    source and joins them so native devices are not leaked.
    """

    def __init__(self, source: FrameSource, inference: InferenceBackend,
                 dispatcher: NativeDispatcher, queue_size: int = 3,
                 horizon_ms: float = 15.0, min_hand_confidence: float = 0.65,
                 on_telemetry: Callable[[object], None] | None = None,
                 on_lock: Callable[[], None] | None = None) -> None:
        self.source, self.inference, self.dispatcher = source, inference, dispatcher
        self.capture_q: LatestQueue[SensorFrame] = LatestQueue(queue_size)
        self.inference_q: LatestQueue[InferencePacket] = LatestQueue(queue_size)
        self.physics_q: LatestQueue[PhysicsPacket] = LatestQueue(queue_size)
        self.intent_q: LatestQueue[IntentPacket] = LatestQueue(queue_size)
        self.telemetry = Telemetry()
        self.physics = PhysicsStage(horizon_ms, min_hand_confidence)
        self.gestures = GestureRecognizer(self.physics.depth)
        self.advanced = SpatialActionInterpreter()
        self.fusion = MultimodalFusion()
        self.on_telemetry = on_telemetry
        self.on_lock = on_lock
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._voice_q: LatestQueue[VoiceCommand] = LatestQueue(2)
        self.errors: list[BaseException] = []

    def submit_voice(self, command: VoiceCommand) -> None:
        self._voice_q.put_latest(command)

    def _run(self, name: str, target: Callable[[], None]) -> None:
        try:
            target()
        except (queue.Empty, StopIteration):
            pass
        except BaseException as exc:
            self.errors.append(exc)
            self._stop.set()

    def _capture(self) -> None:
        for frame in self.source.frames(self._stop):
            if self._stop.is_set():
                break
            self.telemetry.begin(frame.frame_id)
            self.capture_q.put_latest(frame)
            self.telemetry.phase(frame.frame_id, "camera", 1000.0 / max(1, 60))
            self.telemetry.dropped_frames = max(self.telemetry.dropped_frames, self.capture_q.dropped,
                                                self.inference_q.dropped, self.physics_q.dropped,
                                                self.intent_q.dropped)

    def _infer(self) -> None:
        while not self._stop.is_set():
            try:
                frame = self.capture_q.get()
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                result = self.inference.infer(frame)
                self.telemetry.phase(frame.frame_id, "inference", result.inference_ms or (time.perf_counter() - started) * 1000)
                self.inference_q.put_latest(InferencePacket(frame, result))
            finally:
                self.capture_q.task_done()

    def _physics(self) -> None:
        while not self._stop.is_set():
            try:
                packet = self.inference_q.get()
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                result = self.physics.process(packet)
                self.telemetry.phase(packet.frame.frame_id, "physics", (time.perf_counter() - started) * 1000)
                self.telemetry.phase(packet.frame.frame_id, "filter", 0.0)
                self.physics_q.put_latest(result)
            finally:
                self.inference_q.task_done()

    def _fusion(self) -> None:
        while not self._stop.is_set():
            try:
                packet = self.physics_q.get()
            except queue.Empty:
                continue
            try:
                self.fusion.set_gaze(packet.face)  # type: ignore[arg-type]
                intents: list[Intent] = []
                for hand in packet.hands:
                    target = self.physics.depth.pointer(hand)
                    self.fusion.set_hand_target(target)
                    events = list(self.gestures.update(hand))
                    if target is not None:
                        events.extend(self.advanced.update(hand, target))
                    for event in events:
                        if event.kind.name == "LOCK" and self.on_lock:
                            self.on_lock()
                        intent = self.fusion.hand_intent(event)
                        if intent:
                            intents.append(intent)
                while True:
                    try:
                        voice = self._voice_q.get(timeout=0.0)
                    except queue.Empty:
                        break
                    voice_intent = self.fusion.voice_intent(voice)
                    if voice_intent:
                        intents.append(voice_intent)
                    self._voice_q.task_done()
                self.intent_q.put_latest(IntentPacket(packet.frame.frame_id, tuple(intents)))
            finally:
                self.physics_q.task_done()

    def _dispatch(self) -> None:
        while not self._stop.is_set():
            try:
                packet = self.intent_q.get()
            except queue.Empty:
                continue
            started = time.perf_counter()
            try:
                for intent in packet.intents:
                    self.dispatcher.dispatch(intent)
                self.telemetry.phase(packet.frame_id, "dispatch", (time.perf_counter() - started) * 1000)
                self.telemetry.finish(packet.frame_id)
                self.telemetry.set_queue_depths({
                    "capture": self.capture_q.qsize(), "inference": self.inference_q.qsize(),
                    "physics": self.physics_q.qsize(), "intent": self.intent_q.qsize(),
                })
                if self.on_telemetry:
                    self.on_telemetry(self.telemetry.snapshot())
            finally:
                self.intent_q.task_done()

    def start(self) -> None:
        if self._threads:
            return
        stages = [("capture", self._capture), ("inference", self._infer), ("physics", self._physics),
                  ("fusion", self._fusion), ("dispatch", self._dispatch)]
        self._stop.clear()
        self._threads = [threading.Thread(
            target=lambda n=name, fn=fn: self._run(n, fn),
            name=f"phantomos-{name}", daemon=True,
        ) for name, fn in stages]
        for thread in self._threads:
            thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        self.source.close()
        for thread in self._threads:
            thread.join(timeout)
        self._threads.clear()
        self.inference.close()
        self.dispatcher.close()
