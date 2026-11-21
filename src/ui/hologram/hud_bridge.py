"""HUD event bridge: voice-engine signals -> hologram presentation.

This is the only component that translates VoiceOS events into HUD
state. It never blocks audio callbacks: every voice signal arrives via
the existing queued ``VoiceSignals`` bridge, and heavy-ish reads
(service ``stats()``, recognition log, Win32 probes) run on throttled
GUI-thread timers at HUD-appropriate rates.

Level pipeline:
  * TTS: real measured amplitude envelope emitted by
    ``SpeechOutputService`` (``ttsEnvelope`` signal).
  * LISTENING: per-frame RMS from ``SpeechRecognitionService.stats()``.
  * STANDBY: capture/KWS RMS from ``MicrophoneController.audio_stats()``.

The bridge is fully optional — every dependency may be ``None`` and
every call is guarded, so HUD failure can never affect the voice engine.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from PySide6.QtCore import QObject, QTimer, Signal

from src.ui.hologram.hud_probe import DesktopProbe, SystemProbe, normalize_level

logger = logging.getLogger(__name__)

# How often the bridge samples audio level / pushes to the orb.
LEVEL_INTERVAL_MS = 33          # ~30 Hz, matches orb frame rate
PANEL_INTERVAL_MS = 500         # recognition log + status rows
DESKTOP_INTERVAL_MS = 2000      # foreground app / window count
SYSTEM_INTERVAL_MS = 1000       # CPU / RAM

# RMS ceilings per source (float audio, -1..1 scale).
_STT_LEVEL_CEILING = 0.14
_MIC_LEVEL_CEILING = 0.06


class HudBridge(QObject):
    """Translates voice events into HUD updates.

    Args:
        hud: The :class:`HologramHUD` window (or any object exposing the
            same setter API — keeps tests free of OpenGL).
        voice_signals: Existing ``VoiceSignals`` QObject (optional).
        state_manager: ``StateManager`` for AppState polling (optional).
        session_controller: For ``recognition_log``/``last_intent``.
        microphone_controller: For standby audio stats + mic status.
        stt_service: For live per-frame RMS while listening.
        tts_service: For ``is_speaking`` fallback state.
        desktop_probe / system_probe: injectable probes (tests).
    """

    # Internal signal so tests can observe emitted HUD states without a
    # real window.
    hudStateChanged = Signal(str, str)   # (orb_mode, status_text)

    def __init__(
        self,
        hud,
        voice_signals=None,
        state_manager=None,
        session_controller=None,
        microphone_controller=None,
        stt_service=None,
        tts_service=None,
        desktop_probe: Optional[DesktopProbe] = None,
        system_probe: Optional[SystemProbe] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._hud = hud
        self._signals = voice_signals
        self._state = state_manager
        self._session = session_controller
        self._mic = microphone_controller
        self._stt = stt_service
        self._tts = tts_service
        self._desktop_probe = desktop_probe or DesktopProbe()
        self._system_probe = system_probe or SystemProbe()

        # Smoothed level + envelope state.
        self._level = 0.0
        self._raw_level = 0.0
        self._tts_envelope: Optional[tuple[list[float], float, float]] = None
        self._speaking_state = False
        self._mic_status = "—"
        self._wake_status = "—"
        self._wake_flash_until = 0.0
        self._wake_keyword = ""
        self._last_activity = "standby"
        self._session_active = False
        self._error_active = False
        self._transcript = ""
        self._last_recognition = None
        self._activity_lines: list[str] = []

        self._connect_signals()
        self._start_timers()
        self._refresh_static_rows()

    # ------------------------------------------------------------------
    def _connect_signals(self) -> None:
        sig = self._signals
        if sig is None:
            return
        try:
            sig.wakeWordDetected.connect(self._on_wake_word)
            sig.transcriptReceived.connect(self._on_transcript)
            sig.jarvisResponse.connect(self._on_jarvis_response)
            sig.assistantActivity.connect(self._on_activity)
            sig.sessionStateChanged.connect(self._on_session_state)
            sig.micStatusChanged.connect(self._on_mic_status)
            sig.micError.connect(self._on_mic_error)
            sig.ttsEnvelope.connect(self._on_tts_envelope)
        except Exception:  # noqa: BLE001
            logger.exception("HUD bridge: signal connection failed")

    def _start_timers(self) -> None:
        self._level_timer = QTimer(self)
        self._level_timer.setInterval(LEVEL_INTERVAL_MS)
        self._level_timer.timeout.connect(self._level_tick)
        self._level_timer.start()

        self._panel_timer = QTimer(self)
        self._panel_timer.setInterval(PANEL_INTERVAL_MS)
        self._panel_timer.timeout.connect(self._panel_tick)
        self._panel_timer.start()

        self._desktop_timer = QTimer(self)
        self._desktop_timer.setInterval(DESKTOP_INTERVAL_MS)
        self._desktop_timer.timeout.connect(self._desktop_tick)
        self._desktop_timer.start()

        self._system_timer = QTimer(self)
        self._system_timer.setInterval(SYSTEM_INTERVAL_MS)
        self._system_timer.timeout.connect(self._system_tick)
        self._system_timer.start()

    # ------------------------------------------------------------------
    # Voice signal slots (GUI thread via queued connection)
    # ------------------------------------------------------------------
    def _on_wake_word(self, keyword: str = "") -> None:
        # A real keyword string came back from Sherpa-ONNX — this is
        # the only event allowed to claim a wake-word match.
        self._wake_keyword = keyword or "wake word"
        self._wake_status = f"matched: {self._wake_keyword}"
        self._wake_flash_until = time.monotonic() + 1.5
        try:
            self._hud.pulse_orb()
            self._hud.set_status("wake", self._wake_keyword)
        except Exception:  # noqa: BLE001
            logger.debug("HUD wake update failed", exc_info=True)
        self.hudStateChanged.emit("activated", "wake")

    def _on_transcript(self, text: str) -> None:
        self._transcript = text or ""

    def _on_jarvis_response(self, text: str) -> None:
        # Jarvis finished speaking this line -> back to listening state.
        self._speaking_state = False
        self._tts_envelope = None

    def _on_activity(self, activity: str) -> None:
        self._last_activity = activity or "standby"
        if activity == "speaking":
            self._speaking_state = True
        elif activity in ("listening", "standby", "processing"):
            if activity != "processing":
                self._speaking_state = False
                self._tts_envelope = None

    def _on_session_state(self, active: bool) -> None:
        self._session_active = bool(active)
        if not active:
            self._speaking_state = False
            self._tts_envelope = None
            self._transcript = ""
            self._last_activity = "standby"

    def _on_mic_status(self, mic: str, detail: str) -> None:
        self._mic_status = f"{mic}: {detail}" if detail else mic

    def _on_mic_error(self, message: str) -> None:
        self._error_active = True
        try:
            self._hud.set_status("error", message[:60])
        except Exception:  # noqa: BLE001
            pass

    def _on_tts_envelope(self, levels, chunk_seconds: float) -> None:
        """Real measured TTS amplitude (levels normalized 0..1)."""
        try:
            self._tts_envelope = (
                list(levels), float(chunk_seconds), time.monotonic()
            )
        except Exception:  # noqa: BLE001
            logger.debug("Bad TTS envelope", exc_info=True)

    # ------------------------------------------------------------------
    # Audio level pipeline
    # ------------------------------------------------------------------
    def _raw_audio_level(self) -> float:
        """Best available 0..1 level for the current voice state."""
        # 1. TTS envelope — real measured playback amplitude.
        env = self._tts_envelope
        if env is not None and self._speaking_state:
            levels, chunk, started = env
            idx = int((time.monotonic() - started) / max(1e-3, chunk))
            if 0 <= idx < len(levels):
                return float(levels[idx])
            # Envelope exhausted but still 'speaking': gentle state pulse.
            return 0.35 + 0.25 * abs(
                __import__("math").sin(time.monotonic() * 5.0))
        # 2. Live STT per-frame RMS while a session listens.
        if self._session_active and self._stt is not None:
            try:
                rms = float(self._stt.stats().get("last_rms") or 0.0)
                return normalize_level(rms, _STT_LEVEL_CEILING)
            except Exception:  # noqa: BLE001
                pass
        # 3. Standby: shared capture/KWS stats.
        if self._mic is not None:
            try:
                stats = self._mic.audio_stats() or {}
                raw = (
                    stats.get("rms")
                    or stats.get("indata_rms")
                    or stats.get("kws_input_rms")
                    or 0.0
                )
                return normalize_level(float(raw), _MIC_LEVEL_CEILING)
            except Exception:  # noqa: BLE001
                pass
        return 0.0

    def _level_tick(self) -> None:
        """~30 Hz: smooth level, drive orb + waveform, resolve mode."""
        try:
            self._raw_level = self._raw_audio_level()
            # Attack fast, release slow — natural speech envelope.
            target = self._raw_level
            if target > self._level:
                self._level += (target - self._level) * 0.55
            else:
                self._level += (target - self._level) * 0.12
            self._hud.set_orb_level(self._level)
        except Exception:  # noqa: BLE001
            logger.debug("HUD level tick failed", exc_info=True)
            return

        mode, status = self._resolve_mode()
        try:
            self._hud.set_orb_mode(mode)
            self._hud.set_orb_speaking(self._speaking_state)
            self._hud.set_status(status[0], status[1])
        except Exception:  # noqa: BLE001
            logger.debug("HUD mode update failed", exc_info=True)
        self.hudStateChanged.emit(mode, status[0])

    def _resolve_mode(self) -> tuple[str, tuple[str, str]]:
        """Map voice state -> (orb_mode, (status_key, detail))."""
        if time.monotonic() < self._wake_flash_until:
            return "activated", ("wake", self._wake_keyword)
        if self._error_active:
            return "error", ("error", "subsystem fault")
        if self._speaking_state or self._last_activity == "speaking":
            return "speaking", ("speaking", "jarvis output")
        if self._last_activity == "processing":
            return "processing", ("processing", "analysing command")
        # AppState as fallback/authoritative for session phases.
        state_name = None
        if self._state is not None:
            try:
                state_name = self._state.state.name
            except Exception:  # noqa: BLE001
                pass
        if state_name == "PROCESSING":
            return "processing", ("processing", "analysing command")
        if state_name in ("ACTIVATED", "LISTENING") or (
            self._session_active and self._last_activity == "listening"
        ):
            return "listening", ("listening", "awaiting command")
        if self._last_activity == "activated":
            return "activated", ("activated", "wake word detected")
        return "standby", ("standby", "awaiting wake word")

    # ------------------------------------------------------------------
    # Panel timers
    # ------------------------------------------------------------------
    def _panel_tick(self) -> None:
        """Refresh ACTIVITY panel from transcript + recognition log."""
        try:
            self._refresh_activity_panel()
        except Exception:  # noqa: BLE001
            logger.debug("HUD activity refresh failed", exc_info=True)

    def _refresh_activity_panel(self) -> None:
        transcript = self._transcript or "—"
        intent_name = "—"
        app = "—"
        dest = "—"
        status = "—"
        result = "—"

        # Latest recognition outcome (deque tail).
        entry = None
        if self._session is not None:
            try:
                log = self._session.recognition_log()
                if log:
                    entry = log[-1]
            except Exception:  # noqa: BLE001
                pass
        if entry is not None:
            self._last_recognition = entry
            intent_name = entry.get("intent") or "—"
            outcome = entry.get("outcome") or ""
            detail = entry.get("detail") or ""
            status = outcome.replace("_", " ").upper() or "—"
            if detail:
                result = detail[:80]

        # Intent entities for APPLICATION / DESTINATION.
        last_intent = getattr(self._session, "last_intent", None)
        if last_intent is not None:
            entities = getattr(last_intent, "entities", {}) or {}
            name = getattr(last_intent, "name", "") or ""
            if name and name != "UNKNOWN":
                intent_name = name
            app_val = (
                entities.get("app_name")
                or entities.get("app")
                or entities.get("application")
            )
            if app_val is None:
                info = entities.get("_app_info")
                app_val = getattr(info, "name", None)
            if app_val:
                app = str(app_val)[:60]
            mon = entities.get("monitor")
            if mon is not None:
                dest = getattr(mon, "device", None) or str(mon)
            elif entities.get("folder") or entities.get("path"):
                dest = str(
                    entities.get("folder") or entities.get("path"))[:60]
            elif entities.get("query"):
                dest = f"query: {entities['query']}"
            params = [
                f"{k}={v}" for k, v in entities.items()
                if not k.startswith("_") and v not in (None, "")
            ]
            if params and dest == "—":
                dest = ", ".join(params[:3])[:60]

        # History: last few recognition entries.
        lines = []
        if self._session is not None:
            try:
                for e in list(self._session.recognition_log())[-5:]:
                    txt = (e.get("text") or "")[:38]
                    it = e.get("intent") or "?"
                    oc = (e.get("outcome") or "")[:18]
                    lines.append(f"{oc} · {it} · {txt}")
            except Exception:  # noqa: BLE001
                pass
        self._activity_lines = lines or self._activity_lines[-5:]

        self._hud.set_activity(
            transcript, intent_name, app, dest, status, result)
        self._hud.set_history(self._activity_lines)

    def _desktop_tick(self) -> None:
        try:
            snap = self._desktop_probe.snapshot()
            mons = (snap.get("monitors") or "").split(" · ")
            mon1 = mons[0] if len(mons) > 0 else "—"
            mon2 = mons[1] if len(mons) > 1 else "—"
            self._hud.set_desktop_rows(
                snap.get("active_app", "—"),
                snap.get("window_count", "—"),
                mon1, mon2,
                snap.get("system_state", "—"),
            )
        except Exception:  # noqa: BLE001
            logger.debug("HUD desktop tick failed", exc_info=True)

    def _system_tick(self) -> None:
        try:
            snap = self._system_probe.snapshot()
            # Wake-word detector row: processing state + match count +
            # overflow counter from the shared KWS stats. Never claims a
            # match — only an actual keyword emission sets "matched:".
            try:
                st = (
                    self._mic.audio_stats()
                    if self._mic is not None else {}
                ) or {}
            except Exception:  # noqa: BLE001
                st = {}
            if self._session_active:
                self._wake_status = "suspended · session active"
            else:
                proc = st.get("frames_processed") or 0
                det = st.get("detections") or 0
                drops = st.get("frames_dropped") or 0
                if proc:
                    self._wake_status = (
                        f"processing · {det} match(es)"
                        + (f" · drops {drops}" if drops else "")
                    )
                else:
                    self._wake_status = "idle"
            stt_name = getattr(self._stt, "engine_name", "—") or "—"
            tts_name = getattr(self._tts, "engine_name", "—") or "—"
            self._hud.set_system_rows(
                self._mic_status or "—",
                self._wake_status or "—",
                stt_name,
                tts_name,
                snap.get("cpu", "—"),
                snap.get("ram", "—"),
            )
        except Exception:  # noqa: BLE001
            logger.debug("HUD system tick failed", exc_info=True)

    def _refresh_static_rows(self) -> None:
        self._desktop_tick()
        self._system_tick()
        self._panel_tick()

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Stop timers; HUD window lifetime is owned by the controller."""
        for timer in (self._level_timer, self._panel_timer,
                      self._desktop_timer, self._system_timer):
            try:
                timer.stop()
            except Exception:  # noqa: BLE001
                pass
