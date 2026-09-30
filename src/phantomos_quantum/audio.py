"""Optional local audio feedback for spatial key/button activation."""

from __future__ import annotations

import math
import struct
from array import array

from .errors import DependencyUnavailable


class AudioFeedback:
    def __init__(self, sample_rate: int = 48_000, enabled: bool = True) -> None:
        self.sample_rate = sample_rate
        self.enabled = enabled
        self._output = None
        if enabled:
            try:
                import sounddevice as sd
                self._output = sd
            except ImportError:
                self.enabled = False

    def click(self, frequency: float = 1300.0, duration_ms: float = 24.0) -> None:
        if not self.enabled or self._output is None:
            return
        count = max(1, int(self.sample_rate * duration_ms / 1000.0))
        samples = array("f")
        for index in range(count):
            envelope = math.exp(-index / max(1, count * 0.22))
            samples.append(0.10 * envelope * math.sin(2.0 * math.pi * frequency * index / self.sample_rate))
        self._output.play(samples, self.sample_rate, blocking=False)

    def close(self) -> None:
        if self._output is not None:
            try:
                self._output.stop()
            except Exception:
                pass
            self._output = None
