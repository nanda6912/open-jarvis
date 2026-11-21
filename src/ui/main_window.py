"""Main window for the VoiceOS desktop interface."""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from src.core.application import VoiceOSApplication
from src.core.state_manager import AppState
from src.voice.audio_device_manager import AUTO_DEVICE_ID

logger = logging.getLogger(__name__)

_DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", sans-serif;
    font-size: 13px;
}
QFrame#card {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 8px;
}
QLabel#title {
    font-size: 22px;
    font-weight: bold;
    color: #cba6f7;
}
QLabel#subtitle {
    font-size: 14px;
    color: #a6adc8;
}
QLabel#value {
    font-size: 15px;
    font-weight: bold;
    color: #89b4fa;
}
QLabel#caption {
    font-size: 11px;
    color: #6c7086;
}
QLabel#banner {
    font-size: 16px;
    font-weight: bold;
    color: #a6e3a1;
    background-color: #313244;
    border: 1px solid #a6e3a1;
    border-radius: 8px;
    padding: 10px;
}
QLabel#activity {
    font-size: 14px;
    font-weight: bold;
    border-radius: 8px;
    padding: 8px;
    background-color: #45475a;
    color: #cdd6f4;
}
QLabel#heard, QLabel#response {
    font-size: 13px;
    color: #cdd6f4;
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 6px;
}
QComboBox {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 4px 8px;
    color: #cdd6f4;
}
"""


class _StatusRow(QWidget):
    """A labeled status row used inside the main window."""

    def __init__(self, caption: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        caption_label = QLabel(caption)
        caption_label.setObjectName("caption")
        self.value_label = QLabel("—")
        self.value_label.setObjectName("value")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(caption_label)
        layout.addStretch(1)
        layout.addWidget(self.value_label)

    def set_value(self, text: str) -> None:
        """Update the displayed value."""
        self.value_label.setText(text)


class MainWindow(QMainWindow):
    """Primary VoiceOS window showing application status."""

    _state_refresh = Signal()

    # Activity indicator colors per assistant activity.
    _ACTIVITY_STYLE = {
        "standby": ("Standby", "#45475a", "#cdd6f4"),
        "listening": ("Listening", "#40a02b", "#1e1e2e"),
        "speaking": ("Speaking", "#8839ef", "#1e1e2e"),
        "processing": ("Processing", "#df8e1d", "#1e1e2e"),
    }

    def __init__(
        self,
        app: VoiceOSApplication,
        parent: QWidget | None = None,
        close_to_tray: bool = True,
        microphone_controller=None,
    ) -> None:
        """Create the main window bound to an application instance.

        Args:
            app: The core application facade.
            parent: Optional parent widget.
            close_to_tray: When True, closing the window hides it to the
                system tray instead of shutting down. Set False when no
                tray is available so closing exits the application.
            microphone_controller: Optional microphone controller used to
                populate the device selector and status rows.
        """
        super().__init__(parent)
        self._app = app
        self._mic = microphone_controller
        self._close_to_tray = close_to_tray
        self._banner_timer: QTimer | None = None
        self.setWindowTitle(f"{app.config.app_name} — {app.config.assistant_name}")
        self.resize(440, 600)
        self.setStyleSheet(_DARK_STYLESHEET)

        central = QWidget(self)
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(16)

        title = QLabel(app.config.app_name)
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(title)

        subtitle = QLabel(f"Assistant: {app.config.assistant_name}")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(subtitle)

        # Activation banner: shown briefly when the wake word fires.
        self._banner = QLabel("")
        self._banner.setObjectName("banner")
        self._banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._banner.hide()
        outer.addWidget(self._banner)

        # Activity indicator: clear visual cue for listening/speaking.
        self._activity_label = QLabel("Standby")
        self._activity_label.setObjectName("activity")
        self._activity_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._activity_label)

        # Speech exchange panels.
        heard_label = QLabel("Recognized speech")
        heard_label.setObjectName("caption")
        outer.addWidget(heard_label)
        self._heard_label = QLabel("—")
        self._heard_label.setObjectName("heard")
        self._heard_label.setWordWrap(True)
        outer.addWidget(self._heard_label)

        response_label = QLabel("Jarvis response")
        response_label.setObjectName("caption")
        outer.addWidget(response_label)
        self._response_label = QLabel("—")
        self._response_label.setObjectName("response")
        self._response_label.setWordWrap(True)
        outer.addWidget(self._response_label)

        card = QFrame()
        card.setObjectName("card")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(16, 16, 16, 16)
        card_layout.setSpacing(12)

        self._state_row = _StatusRow("Application state")
        self._mic_row = _StatusRow("Microphone")
        self._mic_device_row = _StatusRow("Input device")
        self._wake_row = _StatusRow("Wake word")
        self._stt_row = _StatusRow("STT engine")
        self._tts_row = _StatusRow("TTS engine")
        self._engine_row = _StatusRow("KWS engine")
        self._audio_row = _StatusRow("Audio input")
        self._detection_row = _StatusRow("Detection")
        self._session_row = _StatusRow("Session timer")
        self._monitors_row = _StatusRow("Displays")
        self._version_row = _StatusRow("Version")

        card_layout.addWidget(self._state_row)
        card_layout.addWidget(self._mic_row)
        card_layout.addWidget(self._mic_device_row)
        card_layout.addWidget(self._wake_row)
        card_layout.addWidget(self._stt_row)
        card_layout.addWidget(self._tts_row)
        card_layout.addWidget(self._engine_row)
        card_layout.addWidget(self._audio_row)
        card_layout.addWidget(self._detection_row)
        card_layout.addWidget(self._session_row)
        card_layout.addWidget(self._monitors_row)
        card_layout.addWidget(self._version_row)

        if self._mic is not None:
            mic_label = QLabel("Select microphone")
            mic_label.setObjectName("caption")
            card_layout.addWidget(mic_label)
            self._mic_combo = QComboBox()
            self._mic_combo.currentIndexChanged.connect(self._on_device_selected)
            card_layout.addWidget(self._mic_combo)
            self.refresh_devices()
            # Periodically refresh so hot-plugged microphones appear and
            # the active-device row tracks automatic switching.
            self._mic_refresh_timer = QTimer(self)
            self._mic_refresh_timer.setInterval(3000)
            self._mic_refresh_timer.timeout.connect(self._poll_devices)
            self._mic_refresh_timer.start()

        outer.addWidget(card)
        outer.addStretch(1)

        # Register as state listener so the UI stays in sync. State
        # transitions may originate on worker threads, so the listener
        # only emits a queued signal; widgets are touched on the GUI
        # thread inside _refresh().
        app.state_manager.set_listener(self._on_state_changed)
        self._state_refresh.connect(self._refresh)
        self._refresh()

    # ------------------------------------------------------------------
    def _on_state_changed(self, old: AppState, new: AppState) -> None:
        """Handle application state changes (any thread)."""
        self._state_refresh.emit()

    def update_audio_status(self, mic: str, wake: str) -> None:
        """Update microphone and wake-word status rows."""
        self._mic_row.set_value(mic)
        self._wake_row.set_value(wake)
        self._refresh_device_row()

    def update_audio_stats(self, stats: dict) -> None:
        """Update the audio-input and detection diagnostic rows."""
        if not stats:
            return
        rms = stats.get("rms") or 0.0
        peak = stats.get("peak") or 0.0
        age = stats.get("frame_age_seconds")
        activity = "active" if age is not None and age < 1.0 else "silent"
        self._audio_row.set_value(
            f"{activity} · rms {rms:.3f} · peak {peak:.2f}"
        )
        processed = stats.get("frames_processed")
        detections = stats.get("detections")
        last_kw = stats.get("last_keyword")
        det = f"{processed or 0} frames"
        if last_kw:
            det += f" · last: {last_kw}"
        self._detection_row.set_value(det)

    def notify_wake_word(self, keyword: str) -> None:
        """Show a clear visual confirmation that the wake word fired."""
        label = (keyword or "hello jarvis").strip().title() or "Hello Jarvis"
        self._banner.setText(f"Wake word detected: {label} — Jarvis activated")
        self._banner.show()
        if self._banner_timer is None:
            self._banner_timer = QTimer(self)
            self._banner_timer.setSingleShot(True)
            self._banner_timer.timeout.connect(self._banner.hide)
        self._banner_timer.start(5000)
        logger.info("Wake-word activation shown in UI: %r", keyword)

    # ------------------------------------------------------------------
    # Voice session UI updates (invoked via queued Qt signals)
    # ------------------------------------------------------------------
    def update_transcript(self, text: str) -> None:
        """Show the latest recognized user speech."""
        self._heard_label.setText(text or "—")

    def update_jarvis_response(self, text: str) -> None:
        """Show Jarvis's latest spoken response."""
        self._response_label.setText(text or "—")

    def set_activity(self, activity: str) -> None:
        """Update the listening/speaking indicator."""
        label, bg, fg = self._ACTIVITY_STYLE.get(
            activity, self._ACTIVITY_STYLE["standby"]
        )
        self._activity_label.setText(label)
        self._activity_label.setStyleSheet(
            f"background-color: {bg}; color: {fg};"
        )

    def set_session_countdown(self, seconds: int | None) -> None:
        """Show remaining session seconds, or '—' when inactive."""
        if seconds is None or seconds < 0:
            self._session_row.set_value("—")
        else:
            self._session_row.set_value(f"{seconds} s")

    def set_monitor_summary(self, text: str) -> None:
        """Show the connected-display summary (count/positions)."""
        self._monitors_row.set_value(text or "—")

    def set_engine_status(self, stt: str, tts: str) -> None:
        """Show STT and TTS engine identifiers."""
        self._stt_row.set_value(stt)
        self._tts_row.set_value(tts)

    def notify_no_speech(self, reason: str) -> None:
        """Briefly indicate that nothing intelligible was heard."""
        self._heard_label.setText("(no speech detected)")

    # ------------------------------------------------------------------
    def _device_signature(self) -> tuple:
        """Return a cheap signature of the selectable device list."""
        try:
            devices = self._mic.devices
        except Exception:  # noqa: BLE001
            return ()
        return tuple(
            (d.device_id, d.name, d.hostapi) for d in devices
        )

    def _poll_devices(self) -> None:
        """Timer slot: refresh the combo only when the device set changed."""
        sig = self._device_signature()
        if sig != getattr(self, "_last_device_signature", None):
            self.refresh_devices()
        else:
            self._refresh_device_row()

    def refresh_devices(self) -> None:
        """Repopulate the microphone selector from available devices."""
        if self._mic is None or not hasattr(self, "_mic_combo"):
            return
        try:
            devices = self._mic.devices
        except Exception:  # noqa: BLE001 - device query may fail
            logger.debug("Cannot list microphone devices", exc_info=True)
            return
        self._last_device_signature = tuple(
            (d.device_id, d.name, d.hostapi) for d in devices
        )
        current = getattr(
            self._mic, "selection_key", self._mic.selected_device_id
        )
        self._mic_combo.blockSignals(True)
        self._mic_combo.clear()
        self._mic_combo.addItem("Auto (Windows Default)", AUTO_DEVICE_ID)
        for device in devices:
            # Disambiguate identical names across host APIs. PortAudio
            # indexes are unstable, so they are not shown.
            label = f"{device.name}  [{device.hostapi}]"
            self._mic_combo.addItem(label, device.device_id)
        idx = self._mic_combo.findData(current)
        if idx >= 0:
            self._mic_combo.setCurrentIndex(idx)
        self._mic_combo.blockSignals(False)
        self._refresh_device_row()

    def _refresh_device_row(self) -> None:
        """Show the actual active microphone and its connection state."""
        if self._mic is None:
            return
        diagnostics = (
            self._mic.diagnostics()
            if hasattr(self._mic, "diagnostics")
            else None
        )
        if diagnostics is not None:
            active = diagnostics.get("active_device")
            connected = diagnostics.get("connected")
            fallback = diagnostics.get("fallback")
            if active:
                state = "connected" if connected else "disconnected"
                text = f"{active} — {state}"
                mode = diagnostics.get("mode")
                if mode == "auto":
                    text += " (auto)"
                if fallback:
                    text += f" — {fallback}"
                self._mic_device_row.set_value(text)
                return
            if fallback:
                self._mic_device_row.set_value(f"no device — {fallback}")
                return
        capture_device = getattr(self._mic, "active_device", None)
        if capture_device is not None:
            self._mic_device_row.set_value(capture_device.name)
        elif getattr(self._mic, "selection_mode", "device") == "auto":
            self._mic_device_row.set_value("auto (waiting for device)")
        elif self._mic.selected_device_id:
            self._mic_device_row.set_value("configured (not active)")
        else:
            self._mic_device_row.set_value("system default")

    def _on_device_selected(self, index: int) -> None:
        """Persist a new microphone selection."""
        if self._mic is None or index < 0:
            return
        device_id = self._mic_combo.itemData(index)
        if not device_id:
            return
        try:
            ok = self._mic.select_device(str(device_id))
        except Exception:  # noqa: BLE001
            logger.exception("Microphone selection failed")
            ok = False
        if not ok:
            logger.warning("Microphone selection rejected: %s", device_id)
            self.refresh_devices()

    def _refresh_engine_row(self) -> None:
        """Show wake-word engine/model status."""
        if self._mic is None:
            return
        name = getattr(self._mic, "engine_name", "unknown")
        model_dir = getattr(self._mic, "engine_model_dir", None)
        if name == "null":
            self._engine_row.set_value("unavailable (model missing)")
        elif model_dir:
            self._engine_row.set_value(f"{name} · {Path(model_dir).name}")
        else:
            self._engine_row.set_value(name)

    def _refresh(self) -> None:
        """Refresh all displayed values."""
        self._state_row.set_value(self._app.state_manager.state.name)
        self._version_row.set_value(self._app.config.version)
        self._refresh_engine_row()
        # Reflect the current activity from the app state as a fallback;
        # explicit set_activity() calls keep it precise during sessions.
        state = self._app.state_manager.state
        if state is AppState.STANDBY:
            self.set_activity("standby")
        elif state in (AppState.ACTIVATED, AppState.LISTENING):
            self.set_activity("listening")
        elif state is AppState.EXECUTING:
            self.set_activity("speaking")
        elif state is AppState.PROCESSING:
            self.set_activity("processing")

    def show_window(self) -> None:
        """Show, raise and activate the window (restore from tray)."""
        self.refresh_devices()
        self._refresh()
        self.show()
        self.raise_()
        self.activateWindow()
        logger.debug("Main window shown")

    def hide_window(self) -> None:
        """Hide the window without stopping the application."""
        self.hide()
        logger.debug("Main window hidden")

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Hide to the tray on close, or shut down when no tray exists."""
        if self._close_to_tray:
            logger.info("Main window close requested; hiding to tray")
            self.hide_window()
            event.accept()
            return
        self._app.shutdown()
        super().closeEvent(event)
