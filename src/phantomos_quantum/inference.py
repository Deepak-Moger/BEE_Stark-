"""Inference adapters: normalized hand observations and optional MediaPipe backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .errors import DependencyUnavailable
from .models import FaceObservation, HandObservation, Landmark3D, SensorFrame


@dataclass(frozen=True, slots=True)
class InferenceResult:
    hands: tuple[HandObservation, ...] = ()
    face: FaceObservation | None = None
    inference_ms: float = 0.0


class InferenceBackend(Protocol):
    def infer(self, frame: SensorFrame) -> InferenceResult: ...

    def close(self) -> None: ...


class NullInference:
    """No-op backend for headless operation and camera pipeline smoke tests."""

    def infer(self, frame: SensorFrame) -> InferenceResult:
        return InferenceResult()

    def close(self) -> None:
        pass


class OnnxInferenceBackend:
    """Generic ONNX Runtime adapter with hardware-provider negotiation.

    `decoder` owns model-specific preprocessing/postprocessing. This keeps the core
    independent of a particular exported landmark network while preferring TensorRT,
    DirectML, CUDA, CoreML, WebGPU, then CPU when providers are available.
    """

    def __init__(self, model_path: str, decoder: Any) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DependencyUnavailable("ONNX Runtime is required; install .[vision]") from exc
        available = set(ort.get_available_providers())
        preference = ("TensorrtExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider",
                      "CoreMLExecutionProvider", "WebGPUExecutionProvider", "CPUExecutionProvider")
        providers = [provider for provider in preference if provider in available]
        self.session = ort.InferenceSession(model_path, providers=providers or ["CPUExecutionProvider"])
        self.decoder = decoder
        self.providers = tuple(self.session.get_providers())

    def infer(self, frame: SensorFrame) -> InferenceResult:
        if frame.payload is None:
            return InferenceResult()
        import time
        started = time.perf_counter()
        inputs = self.decoder.preprocess(frame.payload)
        output = self.session.run(None, inputs)
        result = self.decoder.decode(output, frame.timestamp, frame.frame_id)
        return InferenceResult(result.hands, result.face, (time.perf_counter() - started) * 1000.0)

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]


class MediaPipeHandsBackend:
    """MediaPipe Hands + lightweight face landmark adapter.

    MediaPipe is imported lazily and is not required for the core predictor or tests.
    The adapter keeps model setup out of the capture thread.
    """

    def __init__(self, max_hands: int = 2, min_confidence: float = 0.65,
                 enable_face: bool = True) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise DependencyUnavailable("MediaPipe is required; install .[vision]") from exc
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._face = None
        if enable_face:
            try:
                self._face = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                    min_detection_confidence=min_confidence, min_tracking_confidence=min_confidence,
                )
            except AttributeError:
                self._face = None

    def infer(self, frame: SensorFrame) -> InferenceResult:
        if frame.payload is None:
            return InferenceResult()
        import time
        started = time.perf_counter()
        # MediaPipe expects RGB and owns no reference to the source image.
        rgb = frame.payload[:, :, ::-1]
        result = self._hands.process(rgb)
        face_observation = None
        if self._face is not None:
            face_result = self._face.process(rgb)
            if face_result.multi_face_landmarks:
                face = face_result.multi_face_landmarks[0].landmark
                nose, left_eye, right_eye = face[1], face[33], face[263]
                eye_mid_x = (left_eye.x + right_eye.x) / 2.0
                eye_mid_y = (left_eye.y + right_eye.y) / 2.0
                face_observation = FaceObservation(
                    gaze_x=float(nose.x), gaze_y=float(nose.y),
                    head_yaw=float(nose.x - eye_mid_x), head_pitch=float(nose.y - eye_mid_y),
                    confidence=0.75, timestamp=frame.timestamp,
                )
        observations: list[HandObservation] = []
        if result.multi_hand_landmarks:
            handed = result.multi_handedness or []
            for index, landmarks in enumerate(result.multi_hand_landmarks):
                points = tuple(Landmark3D(p.x, p.y, p.z, 1.0) for p in landmarks.landmark)
                label = "unknown"
                confidence = 0.0
                if index < len(handed):
                    classification = handed[index].classification[0]
                    label = classification.label.lower()
                    confidence = float(classification.score)
                observations.append(HandObservation(points, label, confidence, frame.timestamp, frame.frame_id))
        return InferenceResult(tuple(observations), face_observation,
                               (time.perf_counter() - started) * 1000.0)

    def close(self) -> None:
        self._hands.close()
        if self._face is not None:
            self._face.close()
