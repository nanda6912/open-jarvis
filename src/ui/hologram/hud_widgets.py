"""HUD panel widgets: glass panels, status banner, live waveform.

Pure Qt widgets — no voice-engine coupling. All text updates arrive via
plain setter methods invoked from the HUD bridge on the GUI thread.
"""

from __future__ import annotations

import logging
from collections import deque

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stylesheet fragments (applied by HologramHUD)
# ---------------------------------------------------------------------------

HUD_STYLESHEET = """
QWidget#hudRoot { background: transparent; }
QFrame#hudPanel {
    background-color: rgba(6, 18, 32, 0.62);
    border: 1px solid rgba(0, 200, 255, 0.28);
    border-radius: 6px;
}
QFrame#hudPanelTitle {
    background: transparent;
    border: none;
}
QLabel#hudPanelTitle {
    color: #57d7ff;
    font-size: 12px;
    font-weight: bold;
    letter-spacing: 2px;
}
QLabel#hudRowKey {
    color: #6fa8c9;
    font-size: 11px;
}
QLabel#hudRowValue {
    color: #d8f4ff;
    font-size: 12px;
    font-weight: bold;
}
QLabel#hudBigStatus {
    color: #7fe9ff;
    font-size: 26px;
    font-weight: bold;
    letter-spacing: 4px;
}
QLabel#hudSubStatus {
    color: #4d8aa8;
    font-size: 12px;
    letter-spacing: 1px;
}
QLabel#hudFieldLabel {
    color: #57d7ff;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
}
QLabel#hudFieldValue {
    color: #e6f7ff;
    font-size: 13px;
}
QLabel#hudHistory {
    color: #9fd8ef;
    font-size: 11px;
}
QPushButton#hudBtn {
    background-color: rgba(10, 40, 60, 0.7);
    color: #7fe9ff;
    border: 1px solid rgba(0, 200, 255, 0.4);
    border-radius: 4px;
    font-size: 11px;
    padding: 3px 10px;
}
QPushButton#hudBtn:hover { background-color: rgba(0, 120, 180, 0.5); }
QPushButton#hudBtn:checked {
    background-color: rgba(0, 160, 230, 0.45);
    color: #ffffff;
}
"""


class HudPanel(QFrame):
    """A titled glass panel."""

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("hudPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 10)
        layout.setSpacing(6)
        if title:
            head = QHBoxLayout()
            head.setContentsMargins(0, 0, 0, 0)
            label = QLabel(title)
            label.setObjectName("hudPanelTitle")
            head.addWidget(label)
            head.addStretch(1)
            layout.addLayout(head)
        self._body = QVBoxLayout()
        self._body.setSpacing(4)
        layout.addLayout(self._body)
        layout.addStretch(1)

    def body(self) -> QVBoxLayout:
        return self._body


class _Row(QWidget):
    """key/value row inside a panel."""

    def __init__(self, key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        k = QLabel(key)
        k.setObjectName("hudRowKey")
        self.value = QLabel("—")
        self.value.setObjectName("hudRowValue")
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight
                                | Qt.AlignmentFlag.AlignVCenter)
        lay.addWidget(k)
        lay.addStretch(1)
        lay.addWidget(self.value)

    def set_value(self, text: str) -> None:
        self.value.setText(text or "—")


class WaveformWidget(QWidget):
    """Scrolling live audio-level strip (painted, GPU-free)."""

    def __init__(self, parent: QWidget | None = None,
                 capacity: int = 140) -> None:
        super().__init__(parent)
        self.setMinimumHeight(46)
        self._samples = deque(maxlen=capacity)
        self._level = 0.0
        self._intensity = 1.0

    def set_intensity(self, value: float) -> None:
        """Appearance control: scale waveform amplitude/alpha (0..1+)."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return
        v = max(0.0, min(1.5, v))
        if v != self._intensity:
            self._intensity = v
            self.update()

    def push(self, level: float) -> None:
        self._samples.append(max(0.0, min(1.0, float(level))))
        self.update()

    def clear(self) -> None:
        self._samples.clear()
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        painter.fillRect(self.rect(), QColor(4, 14, 24, 160))
        mid = h * 0.62
        pen = QPen(QColor(0, 190, 255, 90))
        pen.setWidthF(1.0)
        painter.setPen(pen)
        painter.drawLine(0, int(mid), w, int(mid))

        if self._samples:
            from PySide6.QtCore import QPointF
            from PySide6.QtGui import QPolygonF

            inten = self._intensity
            pen = QPen(QColor(60, 220, 255,
                              int(220 * min(1.0, inten))))
            pen.setWidthF(1.6)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            step = w / max(1, len(self._samples) - 1)
            amp = h * 0.42 * inten
            top = [QPointF(i * step, mid - v * amp)
                   for i, v in enumerate(self._samples)]
            bottom = [QPointF(i * step, mid + v * amp)
                      for i, v in enumerate(self._samples)]
            painter.setBrush(QColor(0, 170, 240,
                                     int(60 * min(1.0, inten))))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPolygon(QPolygonF(top + list(reversed(bottom))))
            painter.setPen(pen)
            painter.drawPolyline(top)
        painter.end()


class HudStatusBar(QWidget):
    """Top-center big status text ('AWAITING WAKE WORD' etc.)."""

    _TEXT = {
        "standby": "AWAITING WAKE WORD",
        "wake": "WAKE WORD DETECTED",
        "activated": "JARVIS ACTIVATED",
        "listening": "LISTENING",
        "processing": "ANALYSING",
        "speaking": "RESPONDING",
        "error": "SYSTEM ERROR",
        "goodbye": "STANDBY",
    }
    _COLOR = {
        "standby": QColor(90, 200, 255),
        "wake": QColor(140, 240, 255),
        "activated": QColor(140, 240, 255),
        "listening": QColor(120, 255, 190),
        "processing": QColor(255, 210, 110),
        "speaking": QColor(190, 150, 255),
        "error": QColor(255, 120, 90),
        "goodbye": QColor(90, 200, 255),
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        self._main = QLabel(self._TEXT["standby"])
        self._main.setObjectName("hudBigStatus")
        self._main.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._sub = QLabel("")
        self._sub.setObjectName("hudSubStatus")
        self._sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self._main)
        lay.addWidget(self._sub)

    def set_state(self, state: str, detail: str = "") -> None:
        key = state if state in self._TEXT else "standby"
        self._main.setText(self._TEXT[key])
        self._main.setStyleSheet(
            f"color: {self._COLOR[key].name()}; font-size: 26px;"
            " font-weight: bold; letter-spacing: 4px;"
        )
        self._sub.setText(detail)
