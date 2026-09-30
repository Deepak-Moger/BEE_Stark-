"""Optional PyQt6 glassmorphic telemetry and tracking overlay."""

from __future__ import annotations

from typing import Iterable

from .errors import DependencyUnavailable
from .models import Landmark3D, TelemetrySnapshot


class PhantomHud:
    """Always-on-top, click-through overlay. Importing this module stays headless-safe."""

    def __init__(self, width: int, height: int) -> None:
        try:
            from PyQt6.QtCore import Qt, QTimer
            from PyQt6.QtGui import QColor, QFont, QPainter, QPen
            from PyQt6.QtWidgets import QApplication
        except ImportError as exc:
            raise DependencyUnavailable("PyQt6 is required for the HUD; install .[ui]") from exc
        self._app = QApplication.instance() or QApplication([])
        self._Qt, self._QTimer = Qt, QTimer
        self._QColor, self._QFont, self._QPainter, self._QPen = QColor, QFont, QPainter, QPen
        self._widget = _HudWidget(self, width, height, Qt, QPainter, QPen, QColor, QFont)
        self._telemetry: TelemetrySnapshot | None = None
        self._landmarks: tuple[Landmark3D, ...] = ()
        self._timer = QTimer(self._widget)
        self._timer.timeout.connect(self._widget.update)
        self._timer.start(1000 // 120)

    @property
    def widget(self) -> object:
        return self._widget

    def update_telemetry(self, snapshot: TelemetrySnapshot) -> None:
        self._telemetry = snapshot
        self._widget.update()

    def update_landmarks(self, landmarks: Iterable[Landmark3D]) -> None:
        self._landmarks = tuple(landmarks)
        self._widget.update()

    def show(self) -> None:
        self._widget.show()

    def close(self) -> None:
        self._timer.stop()
        self._widget.close()


class _HudWidget:
    """Private QWidget kept separate so PyQt remains an optional import."""

    def __new__(cls, hud: PhantomHud, width: int, height: int, Qt: object, QPainter: object,
                QPen: object, QColor: object, QFont: object) -> object:
        # Class construction cannot inherit QWidget without importing it at module load.
        from PyQt6.QtWidgets import QWidget

        class Widget(QWidget):
            def __init__(self) -> None:
                super().__init__()
                self.hud = hud
                self.setFixedSize(width, height)
                self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                                    Qt.WindowType.Tool)
                self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
                self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

            def paintEvent(self, event: object) -> None:
                painter = QPainter(self)
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setBrush(QColor(5, 8, 20, 125))
                painter.setPen(QPen(QColor(70, 220, 255, 180), 1))
                painter.drawRoundedRect(18, 18, 600, 42, 14, 14)
                snapshot = self.hud._telemetry
                if snapshot:
                    painter.setPen(QColor(150, 240, 255, 230))
                    painter.setFont(QFont("Consolas", 9))
                    text = (f"CAM {snapshot.phases_ms.get('camera', 0):.1f}ms  "
                            f"AI {snapshot.phases_ms.get('inference', 0):.1f}ms  "
                            f"PRED {snapshot.phases_ms.get('physics', 0):.1f}ms  "
                            f"OUT {snapshot.phases_ms.get('dispatch', 0):.1f}ms  "
                            f"TOTAL {snapshot.total_ms:.1f}ms  {snapshot.fps:.0f} FPS")
                    painter.drawText(32, 44, text)
                points = self.hud._landmarks
                if len(points) == 21:
                    painter.setPen(QPen(QColor(85, 255, 210, 220), 2))
                    for point in points:
                        painter.drawEllipse(int(point.x * width) - 3, int(point.y * height) - 3, 6, 6)
                painter.end()

        return Widget()
