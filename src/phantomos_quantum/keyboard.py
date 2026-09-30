"""Screen-independent air keyboard layout and dwell activation."""

from __future__ import annotations

from dataclasses import dataclass
import time


@dataclass(frozen=True, slots=True)
class KeyRect:
    label: str
    x: float
    y: float
    width: float
    height: float

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x <= self.x + self.width and self.y <= y <= self.y + self.height


class AirKeyboard:
    """A normalized QWERTY layout; rendering and text dispatch stay separate."""

    _rows = ("QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM")

    def __init__(self, dwell_ms: int = 100) -> None:
        if dwell_ms < 50:
            raise ValueError("dwell must be at least 50ms")
        self.dwell_s = dwell_ms / 1000.0
        self.keys = self._layout()
        self._hover: str | None = None
        self._hover_started = 0.0

    @classmethod
    def _layout(cls) -> tuple[KeyRect, ...]:
        keys: list[KeyRect] = []
        for row, letters in enumerate(cls._rows):
            width = 0.075
            start = 0.05 + row * 0.035
            for column, letter in enumerate(letters):
                keys.append(KeyRect(letter, start + column * (width + 0.008), 0.62 + row * 0.09, width, 0.075))
        keys.extend((KeyRect("SPACE", 0.28, 0.90, 0.42, 0.075), KeyRect("ENTER", 0.72, 0.90, 0.12, 0.075),
                     KeyRect("BACKSPACE", 0.70, 0.62, 0.22, 0.075)))
        return tuple(keys)

    def hover(self, x: float, y: float, now: float | None = None) -> str | None:
        now = time.monotonic() if now is None else now
        label = next((key.label for key in self.keys if key.contains(x, y)), None)
        if label != self._hover:
            self._hover, self._hover_started = label, now
            return None
        if label and now - self._hover_started >= self.dwell_s:
            self._hover_started = now
            return label
        return None

    @staticmethod
    def text_for(label: str) -> str:
        return {"SPACE": " ", "ENTER": "\n", "BACKSPACE": "\b"}.get(label, label.lower())
