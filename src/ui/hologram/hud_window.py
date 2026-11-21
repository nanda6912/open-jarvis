"""HologramHUD — the V10.7 full-screen 3D interface window.

Layout: the 3D orb fills the window; glass panels float around it
(SYSTEM left, DESKTOP right, ACTIVITY bottom) with a status banner on
top and a live waveform strip. A slim control bar offers hide /
always-on-top / fullscreen-windowed / exit. Closing the window only
hides it — the voice engine is untouched.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.ui.hologram.hud_widgets import (
    HUD_STYLESHEET,
    HudPanel,
    HudStatusBar,
    WaveformWidget,
    _Row,
)
from src.ui.hologram.orb_renderer import OrbGLWidget

logger = logging.getLogger(__name__)


class HologramHUD(QMainWindow):
    """Full-screen holographic HUD.

    Signals:
        hideRequested: user asked to hide the HUD (voice keeps running).
        topmostChanged(bool): always-on-top preference changed.
        fullscreenChanged(bool): fullscreen preference changed.
    """

    hideRequested = Signal()
    topmostChanged = Signal(bool)
    fullscreenChanged = Signal(bool)

    def __init__(self, parent: QWidget | None = None,
                 target_fps: int = 30) -> None:
        super().__init__(parent)
        self.setWindowTitle("VoiceOS HUD")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet(HUD_STYLESHEET)

        central = QWidget(self)
        central.setObjectName("hudRoot")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(28, 20, 28, 24)
        root.setSpacing(10)

        # --- top bar: status + controls ---------------------------------
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        title = QLabel("J.A.R.V.I.S — HOLOGRAPHIC INTERFACE")
        title.setStyleSheet(
            "color: rgba(87,215,255,180); font-size: 12px;"
            " letter-spacing: 3px; font-weight: bold;")
        top.addWidget(title)
        top.addStretch(1)

        self._topmost = QCheckBox("TOPMOST")
        self._topmost.setObjectName("hudBtn")
        self._topmost.setChecked(True)
        self._topmost.toggled.connect(self._on_topmost)
        top.addWidget(self._topmost)

        self._fullscreen = QCheckBox("FULLSCREEN")
        self._fullscreen.setObjectName("hudBtn")
        self._fullscreen.setChecked(True)
        self._fullscreen.toggled.connect(self._on_fullscreen)
        top.addWidget(self._fullscreen)

        hide_btn = QPushButton("HIDE")
        hide_btn.setObjectName("hudBtn")
        hide_btn.clicked.connect(self.hideRequested.emit)
        top.addWidget(hide_btn)

        exit_btn = QPushButton("EXIT HUD")
        exit_btn.setObjectName("hudBtn")
        exit_btn.clicked.connect(self.hideRequested.emit)
        top.addWidget(exit_btn)
        root.addLayout(top)

        # --- main area ----------------------------------------------------
        body = QHBoxLayout()
        body.setSpacing(16)

        # Left panel: SYSTEM
        self.system_panel = HudPanel("SYSTEM")
        p = self.system_panel.body()
        self.row_mic = _Row("MICROPHONE")
        self.row_kws = _Row("WAKE WORD")
        self.row_stt = _Row("SPEECH REC")
        self.row_tts = _Row("TTS ENGINE")
        self.row_cpu = _Row("CPU")
        self.row_ram = _Row("RAM")
        for r in (self.row_mic, self.row_kws, self.row_stt, self.row_tts,
                  self.row_cpu, self.row_ram):
            p.addWidget(r)
        p.addStretch(1)
        self.system_panel.setFixedWidth(250)
        body.addWidget(self.system_panel)

        # Center: status + orb
        center = QVBoxLayout()
        center.setSpacing(4)
        self.status_bar = HudStatusBar()
        center.addWidget(self.status_bar)

        self.orb = OrbGLWidget(self, target_fps=target_fps)
        self.orb.setMinimumSize(420, 420)
        center.addWidget(self.orb, 1)

        self.waveform = WaveformWidget()
        center.addWidget(self.waveform)
        body.addLayout(center, 1)

        # Right panel: DESKTOP
        self.desktop_panel = HudPanel("DESKTOP")
        p = self.desktop_panel.body()
        self.row_app = _Row("ACTIVE APP")
        self.row_windows = _Row("OPEN WINS")
        self.row_mon1 = _Row("MONITOR 1")
        self.row_mon2 = _Row("MONITOR 2")
        self.row_state = _Row("SYSTEM STATE")
        for r in (self.row_app, self.row_windows, self.row_mon1,
                  self.row_mon2, self.row_state):
            p.addWidget(r)
        p.addStretch(1)
        self.desktop_panel.setFixedWidth(250)
        body.addWidget(self.desktop_panel)

        root.addLayout(body, 1)

        # --- bottom panel: ACTIVITY --------------------------------------
        self.activity_panel = HudPanel("ACTIVITY")
        p = self.activity_panel.body()
        grid = QGridLayout()
        grid.setSpacing(6)
        self.lbl_transcript = self._field(grid, 0, 0, "TRANSCRIPT")
        self.lbl_intent = self._field(grid, 0, 1, "INTENT")
        self.lbl_app = self._field(grid, 1, 0, "APPLICATION")
        self.lbl_dest = self._field(grid, 1, 1, "DESTINATION")
        self.lbl_status = self._field(grid, 2, 0, "STATUS")
        self.lbl_result = self._field(grid, 2, 1, "RESULT")
        p.addLayout(grid)
        hist_label = QLabel("RECENT")
        hist_label.setObjectName("hudFieldLabel")
        p.addWidget(hist_label)
        self.lbl_history = QLabel("—")
        self.lbl_history.setObjectName("hudHistory")
        self.lbl_history.setWordWrap(True)
        p.addWidget(self.lbl_history)
        self.activity_panel.setMaximumHeight(190)
        root.addWidget(self.activity_panel)

        # Frameless drag region along the top bar.
        for w in (title,):
            w.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    # ------------------------------------------------------------------
    @staticmethod
    def _field(grid: QGridLayout, row: int, col: int,
               caption: str) -> QLabel:
        cap = QLabel(caption)
        cap.setObjectName("hudFieldLabel")
        val = QLabel("—")
        val.setObjectName("hudFieldValue")
        val.setWordWrap(True)
        grid.addWidget(cap, row * 2, col)
        grid.addWidget(val, row * 2 + 1, col)
        return val

    # ------------------------------------------------------------------
    # Panel update API (called by the bridge on the GUI thread)
    # ------------------------------------------------------------------
    def set_status(self, state: str, detail: str = "") -> None:
        self.status_bar.set_state(state, detail)

    def set_orb_mode(self, mode: str) -> None:
        self.orb.set_mode(mode)

    def pulse_orb(self) -> None:
        self.orb.trigger_pulse()

    def set_orb_level(self, level: float) -> None:
        self.orb.set_level(level)
        self.waveform.push(level)

    def set_orb_speaking(self, speaking: bool) -> None:
        self.orb.set_speaking(speaking)

    def set_system_rows(self, mic: str, kws: str, stt: str, tts: str,
                        cpu: str, ram: str) -> None:
        self.row_mic.set_value(mic)
        self.row_kws.set_value(kws)
        self.row_stt.set_value(stt)
        self.row_tts.set_value(tts)
        self.row_cpu.set_value(cpu)
        self.row_ram.set_value(ram)

    def set_desktop_rows(self, app: str, wins: str, mon1: str, mon2: str,
                         state: str) -> None:
        self.row_app.set_value(app)
        self.row_windows.set_value(wins)
        self.row_mon1.set_value(mon1)
        self.row_mon2.set_value(mon2)
        self.row_state.set_value(state)

    def set_activity(self, transcript: str, intent: str, app: str,
                     dest: str, status: str, result: str) -> None:
        self.lbl_transcript.setText(transcript or "—")
        self.lbl_intent.setText(intent or "—")
        self.lbl_app.setText(app or "—")
        self.lbl_dest.setText(dest or "—")
        self.lbl_status.setText(status or "—")
        self.lbl_result.setText(result or "—")

    def set_history(self, lines: list[str]) -> None:
        self.lbl_history.setText("\n".join(lines) if lines else "—")

    def set_topmost(self, on: bool, block: bool = True) -> None:
        if block:
            self._topmost.blockSignals(True)
        self._topmost.setChecked(on)
        if block:
            self._topmost.blockSignals(False)
        flags = self.windowFlags()
        if on:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        was_visible = self.isVisible()
        self.setWindowFlags(flags)
        if was_visible:
            self.show()

    def set_fullscreen(self, on: bool, block: bool = True) -> None:
        if block:
            self._fullscreen.blockSignals(True)
        self._fullscreen.setChecked(on)
        if block:
            self._fullscreen.blockSignals(False)
        # Geometry is applied by the controller; here we only record it.

    # ------------------------------------------------------------------
    def _on_topmost(self, checked: bool) -> None:
        self.set_topmost(checked, block=False)
        self.topmostChanged.emit(checked)

    def _on_fullscreen(self, checked: bool) -> None:
        self.fullscreenChanged.emit(checked)

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self.orb.set_rendering_active(True)

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        # Stop burning GPU while hidden; voice engine is unaffected.
        self.orb.set_rendering_active(False)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Closing the HUD never shuts down VoiceOS — hide instead."""
        logger.info("HUD close requested; hiding (voice engine unaffected)")
        self.hide()
        event.accept()
