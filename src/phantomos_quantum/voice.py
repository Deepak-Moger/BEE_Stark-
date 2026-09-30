"""Optional local voice adapters; no network speech service is required."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .errors import DependencyUnavailable
from .models import VoiceCommand


class VoiceRecognizer(Protocol):
    def transcribe(self, audio: object, timestamp: float) -> VoiceCommand: ...

    def close(self) -> None: ...


@dataclass(slots=True)
class NullVoiceRecognizer:
    def transcribe(self, audio: object, timestamp: float) -> VoiceCommand:
        return VoiceCommand("", 0.0, timestamp, False)

    def close(self) -> None:
        pass


class WhisperOnnxRecognizer:
    """Adapter seam for a bundled Whisper-tiny ONNX model.

    Model/tokenizer details differ between Whisper exports, so preprocessing and
    decoding are deliberately injected. The class validates local execution and
    keeps the engine independent of a cloud service.
    """

    def __init__(self, model_path: str, decoder: object) -> None:
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise DependencyUnavailable("ONNX Runtime is required; install .[voice]") from exc
        self.session = ort.InferenceSession(model_path, providers=self._providers(ort))
        self.decoder = decoder

    @staticmethod
    def _providers(ort: object) -> list[str]:
        available = set(getattr(ort, "get_available_providers")())
        preferred = ["CUDAExecutionProvider", "DmlExecutionProvider", "CoreMLExecutionProvider", "CPUExecutionProvider"]
        return [item for item in preferred if item in available] or ["CPUExecutionProvider"]

    def transcribe(self, audio: object, timestamp: float) -> VoiceCommand:
        # Decoder is responsible for mel spectrograms and tokenizer compatibility.
        feeds = self.decoder.prepare_audio(audio)
        output = self.session.run(None, feeds)
        text, confidence = self.decoder.decode(output)
        return VoiceCommand(text, float(confidence), timestamp, True)

    def close(self) -> None:
        self.session = None  # type: ignore[assignment]
