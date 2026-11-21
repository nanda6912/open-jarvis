"""Microphone controller: orchestrates capture + wake-word detection.

Bridges the audio stack to the application: enable/disable listening,
persist preferences, select devices, and emit status/wake events via
callbacks (marshalled to the GUI thread by the caller).

Device selection modes:

- ``auto`` — follow the current Windows default input device. A monitor
  thread polls the default input ~1x/second; when Windows switches the
  default (earphones plugged/unplugged) or the active device disappears,
  the shared capture stream is restarted on the new input. Wake-word
  detection keeps running because it consumes the same frame queue —
  only the PortAudio stream is swapped, never duplicated.
- ``device`` — pinned to a specific device id. If that device vanishes,
  the controller falls back to the Windows default and reports the
  fallback in status/diagnostics instead of silently claiming the
  requested device is active.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Callable, Optional

from src.voice.audio_capture_service import AudioCaptureService
from src.voice.audio_device_manager import (
    AUTO_DEVICE_ID,
    AudioDevice,
    AudioDeviceManager,
    AudioError,
)
from src.voice.wake_word_detection_service import WakeWordDetectionService
from src.voice.wake_word_engine import WakeWordEngine, create_engine

logger = logging.getLogger(__name__)

STATUS_DISABLED = "disabled"
STATUS_IDLE = "idle"
STATUS_LISTENING = "listening"
STATUS_ERROR = "error"

MODE_AUTO = "auto"
MODE_DEVICE = "device"

# Standby watchdog tuning (seconds / counts).
_STALL_MAX_RETRIES = 3
_STALL_WINDOW_SECONDS = 120.0
_STALL_MIN_INTERVAL = 8.0
_STALL_BACKOFF_BASE = 30.0
_STALL_BACKOFF_MAX = 300.0
_DETECT_MAX_RETRIES = 3
_DETECT_WINDOW_SECONDS = 120.0
_DETECT_BACKOFF_SECONDS = 60.0
# How long ``_detection`` may stay None while status is LISTENING
# before the monitor treats it as a failed restart rather than the
# normal suspend/disable detach window (bounded by the detection
# stop() joins, ~4.5 s worst case).
_DETECTION_MISSING_GRACE = 8.0

SETTING_SELECTION = "voice.microphone_selection"
SETTING_DEVICE = "voice.microphone_device"
SETTING_ENABLED = "voice.microphone_enabled"


class MicrophoneController:
    """High-level microphone lifecycle controller."""

    def __init__(
        self,
        device_manager: Optional[AudioDeviceManager] = None,
        capture: Optional[AudioCaptureService] = None,
        engine: Optional[WakeWordEngine] = None,
        detection: Optional[WakeWordDetectionService] = None,
        settings_get: Optional[Callable[[str], Optional[str]]] = None,
        settings_set: Optional[Callable[[str, str], None]] = None,
        threshold: float = 0.25,
        cooldown_seconds: float = 3.0,
        monitor_interval: float = 1.0,
        callback_stall_threshold: float = 5.0,
        detection_stall_threshold: float = 15.0,
    ) -> None:
        self._device_manager = device_manager or AudioDeviceManager()
        self._capture = capture or AudioCaptureService(
            device_manager=self._device_manager
        )
        self._engine = engine if engine is not None else create_engine()
        self._settings_get = settings_get or (lambda k: None)
        self._settings_set = settings_set or (lambda k, v: None)
        self._threshold = threshold
        self._cooldown = cooldown_seconds
        self._monitor_interval = monitor_interval
        self._callback_stall_threshold = callback_stall_threshold
        self._detection_stall_threshold = detection_stall_threshold
        self._on_wake_word: Optional[Callable[[str], None]] = None
        self._on_status: Optional[Callable[[str, str], None]] = None
        self._on_error: Optional[Callable[[str], None]] = None
        self._on_stats: Optional[Callable[[dict], None]] = None
        self._detection: Optional[WakeWordDetectionService] = None
        # Detectors whose stop()/join timed out (or is still in
        # progress) and whose worker thread may still be draining the
        # shared frame queue. Marked under ``_restart_lock`` at detach
        # time — BEFORE stop() — so a concurrent installer can never
        # miss a live consumer; entries are purged lazily once
        # ``is_running`` goes false. Ownership invariant: at most one
        # live detector worker may consume frames, so a new
        # WakeWordDetectionService is installed only while this list
        # holds no live worker.
        self._retiring_detectors: list = []
        # Increments each time standby detection is (re)created so logs
        # and stats can distinguish per-session processing counters.
        self._detection_generation = 0
        self._status = STATUS_DISABLED
        self._status_detail = "not started"

        # Selection mode. Backward compatibility: an explicit persisted
        # device id with no selection key means the user pinned a device.
        mode = (self._settings_get(SETTING_SELECTION) or "").strip().lower()
        if mode not in (MODE_AUTO, MODE_DEVICE):
            stored_id = self._settings_get(SETTING_DEVICE)
            mode = MODE_DEVICE if stored_id else MODE_AUTO
        self._selection_mode = mode
        self._selected_device_id: Optional[str] = (
            self._settings_get(SETTING_DEVICE)
            if self._selection_mode == MODE_DEVICE
            else None
        )

        # Dynamic-device monitor state.
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self._restart_lock = threading.RLock()
        self._fallback_notice: Optional[str] = None
        self._switch_count = 0
        self._last_switch_monotonic = 0.0
        self._last_restart_attempt = 0.0

        # Standby watchdog state (callback + detection liveness).
        # Bounded retries inside a sliding window, then exponential
        # backoff so a persistently broken endpoint cannot restart-loop.
        self._stall_restarts = 0
        self._stall_window_start = 0.0
        self._stall_backoff_until = 0.0
        self._stall_backoff_seconds = 0.0
        self._detect_restarts = 0
        self._detect_window_start = 0.0
        self._detect_backoff_until = 0.0
        self._detect_proc_seen = -1
        self._detect_proc_seen_at = 0.0
        self._detection_none_since = 0.0
        self._last_wake_detected_monotonic = 0.0
        self._last_wake_keyword: Optional[str] = None
        self._last_wake_detection: Optional[dict] = None

        # OS suspend/resume detection. Primary signal: Windows
        # WM_POWERBROADCAST notifications forwarded via
        # notify_system_resume() (see main.py). Fallback: wall-clock
        # vs monotonic drift — sleep / Modern Standby / hibernate
        # freezes every thread, so time.time() jumps far ahead of
        # time.monotonic(). Drift alone is only a *candidate*: NTP
        # steps and manual clock changes produce the same signature,
        # so the candidate is confirmed against pipeline evidence
        # (callbacks/frames must actually be stale) before any
        # recovery runs.
        self._last_wall_time = time.time()
        self._last_mono_time = time.monotonic()
        self._resume_verify_until = 0.0
        self._resume_events = 0
        self._resume_drift_threshold = 3.0
        self._resume_stall_threshold = 2.0
        self._resume_verify_seconds = 15.0
        self._resume_candidate: Optional[dict] = None
        self._resume_candidate_seconds = 2.0
        self._resume_candidate_fresh_seconds = 1.5
        self._last_resume_handled = 0.0
        self._resume_notify_debounce = 10.0
        self._suspend_notified = False
        # Post-resume readiness tracking: milestones with monotonic +
        # wall timestamps, evaluated every monitor tick until the
        # pipeline proves READY (not merely PortAudio active=True).
        self._resume_pending = False
        self._resume_milestones: dict = {}
        self._last_resume_report: Optional[dict] = None
        self._resume_failure_logged = False
        self._readiness_mark: Optional[dict] = None
        self._frames_advancing = False
        self._detector_consuming = False
        self._on_system_resume: Optional[Callable[[], None]] = None
        self._last_readiness: dict = {
            "ready": False,
            "checks": {},
            "failure_reason": "not evaluated",
        }
        self._last_desired = None

    # ------------------------------------------------------------------
    # Configuration / callbacks
    # ------------------------------------------------------------------
    def set_callbacks(
        self,
        on_wake_word: Optional[Callable[[str], None]] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_stats: Optional[Callable[[dict], None]] = None,
    ) -> None:
        """Register status/event callbacks (thread-safe emitters)."""
        self._on_wake_word = on_wake_word
        self._on_status = on_status
        self._on_error = on_error
        self._on_stats = on_stats

    def set_resume_listener(
        self, listener: Optional[Callable[[], None]]
    ) -> None:
        """Register a callback invoked on confirmed OS resume.

        Used to reset session-scoped consumers (e.g. the STT service's
        VAD noise floor and partial utterance) so the first post-resume
        command is judged on fresh audio only.
        """
        self._on_system_resume = listener

    @property
    def status(self) -> str:
        return self._status

    @property
    def status_detail(self) -> str:
        return self._status_detail

    @property
    def is_listening(self) -> bool:
        return self._status == STATUS_LISTENING

    @property
    def selection_mode(self) -> str:
        """Return ``"auto"`` or ``"device"``."""
        return self._selection_mode

    @property
    def selection_key(self) -> str:
        """Return the UI/persistence key for the current selection."""
        if self._selection_mode == MODE_AUTO:
            return AUTO_DEVICE_ID
        return self._selected_device_id or AUTO_DEVICE_ID

    @property
    def selected_device_id(self) -> Optional[str]:
        return self._selected_device_id

    @property
    def devices(self) -> list[AudioDevice]:
        return self._device_manager.list_input_devices()

    @property
    def active_device(self) -> Optional[AudioDevice]:
        """Return the device currently being captured, if any."""
        return self._capture.device

    @property
    def capture(self) -> AudioCaptureService:
        """Return the underlying capture service (pause/resume gate)."""
        return self._capture

    @property
    def engine_name(self) -> str:
        """Return the wake-word engine identifier."""
        return getattr(self._engine, "name", "unknown")

    @property
    def engine_model_dir(self) -> Optional[str]:
        """Return the engine model directory, if the engine exposes one."""
        model_dir = getattr(self._engine, "model_dir", None)
        return str(model_dir) if model_dir is not None else None

    def audio_stats(self) -> dict:
        """Return capture + detection pipeline diagnostics."""
        if self._detection is not None and hasattr(self._detection, "stats"):
            return self._detection.stats()
        if hasattr(self._capture, "stats"):
            return self._capture.stats()
        return {}

    def diagnostics(self) -> dict:
        """Return active-microphone diagnostics for UI/status display."""
        active = self._capture.device
        return {
            "mode": self._selection_mode,
            "requested_device_id": self._selected_device_id,
            "active_device": active.name if active else None,
            "active_device_id": active.device_id if active else None,
            "connected": bool(
                active is not None and self._capture.is_running
            ),
            "fallback": self._fallback_notice,
            "status": self._status,
            "status_detail": self._status_detail,
            "switches": self._switch_count,
            "watchdog": self.watchdog_stats(),
        }

    def last_wake_detection_monotonic(self) -> Optional[float]:
        """Monotonic timestamp of the most recent emitted wake event."""
        return self._last_wake_detected_monotonic or None

    def last_wake_detection(self) -> Optional[dict]:
        """Structured detection record for the most recent wake event."""
        if isinstance(self._last_wake_detection, dict):
            return dict(self._last_wake_detection)
        return None

    def watchdog_stats(self) -> dict:
        """Standby-watchdog counters for diagnostics/tests."""
        return {
            "callback_stall_threshold": self._callback_stall_threshold,
            "detection_stall_threshold": self._detection_stall_threshold,
            "stall_restarts": self._stall_restarts,
            "stall_backoff_until": self._stall_backoff_until or None,
            "stall_backoff_seconds": self._stall_backoff_seconds or None,
            "detect_restarts": self._detect_restarts,
            "detect_backoff_until": self._detect_backoff_until or None,
            "last_wake_detected_monotonic":
                self._last_wake_detected_monotonic or None,
            "last_wake_keyword": self._last_wake_keyword,
            "detection_generation": self._detection_generation,
            "retiring_detectors": len(self._retiring_detectors),
            "resume_events": self._resume_events,
            "suspend_notified": self._suspend_notified,
            "resume_pending": self._resume_pending,
            "resume_milestones": dict(self._resume_milestones),
            "last_resume_report": self._last_resume_report,
            "audio_readiness": self._last_readiness.get("ready"),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def enable(self) -> bool:
        """Start microphone capture and wake-word detection.

        Returns True on success; errors are reported via callbacks and
        reflected in status rather than raised.
        """
        if self._status in (STATUS_LISTENING, STATUS_IDLE):
            # IDLE means a voice session is active: capture runs for
            # the STT consumer and wake detection is intentionally
            # suspended. Re-running enable() here would install a
            # second queue consumer mid-session.
            return True
        try:
            device = self._start_capture()
        except AudioError as exc:
            self._set_status(STATUS_ERROR, str(exc))
            self._emit_error(str(exc))
            return False
        if device is None:
            return False

        self._settings_set(SETTING_ENABLED, "true")
        self._settings_set(SETTING_SELECTION, self._selection_mode)

        # Serialize the detector swap with every other lifecycle path.
        # A concurrent enable() may have installed a detector while
        # capture.start() ran — never create a second consumer.
        with self._restart_lock:
            if self._status in (STATUS_LISTENING, STATUS_IDLE):
                return True
            # Detach-then-stop any surviving detection worker before
            # installing the replacement: paths like a capture error
            # can leave ``_detection`` alive while status is
            # ERROR/DISABLED, and overwriting the reference without
            # stopping it orphans a consumer that keeps draining the
            # shared frame queue. The worker is marked retiring BEFORE
            # stop() so the slot check below can never miss it.
            stale_detection = self._detection
            self._detection = None
            self._mark_retiring(stale_detection)
            self._stop_detector(stale_detection)
            self._detection_generation += 1
            generation = self._detection_generation
            if self._detector_slot_available():
                self._detection = self._new_detection_service(generation)
                self._detection.start()
            else:
                # Fail closed: a retired worker still owns the frame
                # queue. Leave ``_detection`` empty and LISTENING; the
                # standby watchdog's missing-detector recovery installs
                # a replacement once the survivor actually exits.
                logger.error(
                    "Wake-word detector install deferred "
                    "(generation=%d): retiring worker(s) still alive; "
                    "watchdog will retry", generation,
                )
            self._set_status(STATUS_LISTENING, self._detail_for(device))
        self._start_monitor()
        return True

    def disable(self) -> None:
        """Immediately stop capture, inference and release the device."""
        self._stop_monitor()
        # Detach under the restart lock (same protocol as
        # suspend_detection): the standby watchdog treats
        # ``_detection is None`` as "nothing to restart", so it cannot
        # spawn a replacement worker that the ``= None`` below would
        # orphan. The worker is marked retiring BEFORE the stop/join
        # (which runs outside the lock so a worker blocked in a
        # callback that needs this lock cannot stall the join) so no
        # concurrent installer can miss a still-live consumer.
        with self._restart_lock:
            detection = self._detection
            self._detection = None
            self._mark_retiring(detection)
        if detection is not None:
            self._stop_detector(detection)
        # Serialize the capture shutdown + terminal status under the
        # restart lock: an in-flight _restart_for_selection() holds the
        # same lock, so it either completes before this (and is then
        # stopped here) or starts after and sees DISABLED and aborts.
        # Without this, a restart finishing after capture.stop() would
        # leave a live stream behind a "microphone off" status.
        with self._restart_lock:
            self._capture.stop()
            self._engine.reset()
            self._settings_set(SETTING_ENABLED, "false")
            self._set_status(STATUS_DISABLED, "microphone off")

    def suspend_detection(self, pause_capture: bool = True) -> None:
        """Pause wake-word inference after activation.

        Args:
            pause_capture: When True (default) incoming frames are
                discarded. Pass False when a command-session listener
                (e.g. the STT service) will consume the same frame queue.
        """
        # Atomically detach the detection service under the restart
        # lock before stopping it. The standby watchdog evaluates
        # liveness under the same lock and treats ``_detection is
        # None`` as "nothing to restart", so it can no longer observe
        # the just-stopped worker as dead while status is still
        # LISTENING and spawn a replacement that the old ``= None``
        # would orphan (orphaned workers kept consuming frames and
        # emitted duplicate wake events). The worker is marked retiring
        # at detach time — before the stop/join, which runs outside the
        # lock so a worker blocked in a callback that needs this lock
        # cannot stall the join — so resume_detection() can never
        # install a second consumer while this one may still be alive.
        with self._restart_lock:
            detection = self._detection
            self._detection = None
            self._mark_retiring(detection)
        if detection is not None:
            detection.suspend()
        if pause_capture:
            self._capture.pause()
        # Serialize the terminal transition with disable(): if disable
        # won the race, do not resurrect the controller to IDLE.
        with self._restart_lock:
            if self._status == STATUS_DISABLED:
                return
            self._set_status(STATUS_IDLE, "activated")

    def resume_detection(self) -> None:
        """Resume wake-word detection after the session ends.

        V10.7.5 handover order (serialized, no stale audio):
        1. Close the capture gate so no session audio can queue.
        2. Flush whatever the session left in the frame queue.
        3. Reset the KWS stream (engine lock serializes this against
           any in-flight inference).
        4. Start the new standby detection generation.
        5. Re-open the capture gate — only fresh frames reach the
           new worker. READY is reported after the first frame is
           actually processed, not at thread start.
        """
        # Serialize the whole handover with disable()/restarts: the
        # status re-check inside the lock ensures a disable() that
        # completed while we were paused cannot be overwritten by a
        # stale IDLE->LISTENING transition or a orphaned detector.
        with self._restart_lock:
            if not (
                self._status == STATUS_IDLE
                and self._capture.is_running
            ):
                return
            logger.info(
                "Standby handover: closing capture gate + flushing "
                "stale session audio"
            )
            pause = getattr(self._capture, "pause", None)
            if pause is not None:
                pause()
            # Drop stale frames queued while the session was active so
            # the wake-word engine does not process old audio.
            flush = getattr(self._capture, "flush", None)
            if flush is not None:
                flush()
            # V10.6: reset the KWS stream on standby handover. Frames fed
            # between the detection and suspend_detection() (the worker
            # keeps consuming until the thread stops) remain in the
            # engine stream; without this reset they are decoded as
            # stale audio at the start of the next standby period.
            # V10.7.5: engine.reset() is serialized by the engine lock,
            # so it cannot interleave with an in-flight process().
            try:
                self._engine.reset()
            except Exception:  # noqa: BLE001
                logger.debug("KWS engine reset on standby failed",
                             exc_info=True)
            existing = self._detection
            if existing is not None and getattr(
                existing, "is_running", False
            ):
                # A live detector survived suspension — reuse it.
                # Replacing it would orphan a second consumer of the
                # shared frame queue (normal paths always leave
                # _detection=None here, so this is a defensive guard).
                generation = getattr(
                    existing, "_generation", self._detection_generation
                )
                logger.info(
                    "Standby handover: reusing surviving detector "
                    "(generation=%s)", generation,
                )
            else:
                if existing is not None:
                    # Dead reference only (the reuse branch above takes
                    # live workers): stop is a no-op safety net. Mark
                    # before stop so the slot check below sees any
                    # survivor of OTHER retiring workers too.
                    self._mark_retiring(existing)
                    self._stop_detector(existing)
                self._detection_generation += 1
                generation = self._detection_generation
                if self._detector_slot_available():
                    self._detection = self._new_detection_service(
                        generation,
                        on_first_frame=lambda: self._log_detector_ready(
                            generation
                        ),
                    )
                    self._detection.start()
                else:
                    # Fail closed: a retired worker still owns the
                    # frame queue. Leave ``_detection`` empty; the
                    # standby watchdog's missing-detector recovery
                    # installs a replacement once it exits.
                    self._detection = None
                    logger.error(
                        "Standby handover: detector install deferred "
                        "(generation=%d): retiring worker(s) still "
                        "alive; watchdog will retry", generation,
                    )
            # Re-open the gate only after the detector exists: frames
            # captured from here on are fresh standby audio.
            resume = getattr(self._capture, "resume", None)
            if resume is not None:
                resume()
            device = self._capture.device
            self._set_status(
                STATUS_LISTENING, self._detail_for(device)
            )
            logger.info(
                "Standby handover armed (generation=%s, threshold=%.2f, "
                "variants=%s); READY follows first processed frame",
                generation,
                self._threshold,
                getattr(self._engine, "keyword_count", None),
            )

    def _log_detector_ready(self, generation: int) -> None:
        """Report KWS READY only after real audio was processed."""
        try:
            det = self._detection
            logger.info(
                "KWS detector READY (generation=%d, threshold=%.2f, "
                "variants=%s, first_frame_rms=%.4f)",
                generation,
                self._threshold,
                getattr(self._engine, "keyword_count", None),
                getattr(det, "first_frame_rms", 0.0) or 0.0,
            )
        except Exception:  # noqa: BLE001
            logger.debug("Detector READY log failed", exc_info=True)

    def select_device(self, device_id: str) -> bool:
        """Select a microphone (or ``"auto"``) and restart capture on it.

        Persists the choice and, when the controller is active, restarts
        the shared capture stream on the new input. Returns False if an
        explicit device id is not currently available.
        """
        if device_id == AUTO_DEVICE_ID:
            self._selection_mode = MODE_AUTO
            self._selected_device_id = None
            self._settings_set(SETTING_SELECTION, MODE_AUTO)
            if self._status in (STATUS_LISTENING, STATUS_IDLE):
                return self._restart_for_selection("user selected Auto")
            return True

        found = self._device_manager.find_device(device_id)
        if found is None or not self._id_name_matches(device_id, found):
            logger.warning("Cannot select unknown microphone %r", device_id)
            return False

        self._selection_mode = MODE_DEVICE
        self._selected_device_id = device_id
        self._settings_set(SETTING_SELECTION, MODE_DEVICE)
        self._settings_set(SETTING_DEVICE, device_id)

        if self._status in (STATUS_LISTENING, STATUS_IDLE):
            return self._restart_for_selection(
                f"user selected {device_id}"
            )
        return True

    def shutdown(self) -> None:
        """Full cleanup: stop workers and release the microphone."""
        self.disable()
        try:
            self._engine.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # Capture (re)start helpers
    # ------------------------------------------------------------------
    def _target_device_id(self) -> Optional[str]:
        """Return the device id to request, or None for Windows default."""
        if self._selection_mode == MODE_DEVICE and self._selected_device_id:
            return self._selected_device_id
        return None

    def _detail_for(self, device: Optional[AudioDevice]) -> str:
        """Build the human-readable status detail for a device."""
        name = device.name if device is not None else "unknown"
        if self._fallback_notice:
            return f"{name} ({self._fallback_notice})"
        if self._selection_mode == MODE_AUTO:
            return f"{name} [auto]"
        return name

    @staticmethod
    def _id_name_matches(device_id: str, device: AudioDevice) -> bool:
        """Strict check: does ``device`` have the name stored in the id?

        ``hostapi:index`` matching is only a last resort and can bind a
        *different* physical microphone after hot-unplug index reuse, so
        honesty decisions (fallback reporting, persistence) require the
        stored name to match when the id carries one. Uses the device
        manager's tolerant comparison (trademark markers stripped,
        Windows name truncation tolerated).
        """
        parts = device_id.split(":", 2)
        if len(parts) < 3 or not parts[2]:
            return True  # legacy id without a name: cannot verify
        return AudioDeviceManager._names_match(parts[2], device.name)

    def _record_active_device(self, device: AudioDevice) -> None:
        """Persist the active device for diagnostics/backward compat."""
        if self._selection_mode == MODE_DEVICE:
            requested = self._selected_device_id
            # Strict = the captured device IS the configured endpoint by
            # stable identity (host API + normalized name), not by the
            # ephemeral hostapi:index pair. A name-resolved device at a
            # new PortAudio index is the configured microphone, not a
            # fallback — previously the index check made every reshuffle
            # report a false "fallback: ... unavailable".
            strict = bool(
                requested
                and self._device_manager.matches_stable_id(device, requested)
            )
            if strict:
                self._settings_set(SETTING_DEVICE, device.device_id)
                self._fallback_notice = None
            elif requested:
                # capture.start() fell back to another device: record the
                # truth and never overwrite the user's pinned choice.
                self._fallback_notice = (
                    f"fallback: {requested} unavailable"
                )
        else:
            # Auto mode: the stored id is informational (last used mic).
            self._settings_set(SETTING_DEVICE, device.device_id)

    def _start_capture(self) -> Optional[AudioDevice]:
        """Open the capture stream for the current selection.

        Returns the active AudioDevice, or None when the stream could
        not be opened (status/error already reported).
        """
        requested_id = self._target_device_id()
        try:
            device = self._capture.start(
                device_id=requested_id,
                on_error=self._handle_capture_error,
            )
        except AudioError as exc:
            # Safe fallback: if the requested device failed, try the
            # Windows default input once before giving up.
            if requested_id is None:
                raise
            logger.warning(
                "Selected microphone %r failed to open (%s); "
                "falling back to Windows default input",
                requested_id, exc,
            )
            self._fallback_notice = f"fallback: {exc}"
            try:
                device = self._capture.start(
                    device_id=None,
                    on_error=self._handle_capture_error,
                )
            except AudioError:
                self._fallback_notice = None
                raise
        self._record_active_device(device)
        return device

    def _restart_for_selection(self, reason: str) -> bool:
        """Restart the shared capture stream after a selection change."""
        with self._restart_lock:
            # A restart that was already in flight when disable() ran
            # must not reopen the stream afterwards: DISABLED is
            # terminal until the next explicit enable().
            if self._status == STATUS_DISABLED:
                logger.info(
                    "Ignoring capture restart (%s): microphone disabled",
                    reason,
                )
                return False
            self._last_restart_attempt = time.monotonic()
            logger.info("Microphone restart: %s", reason)
            was_idle = self._status == STATUS_IDLE
            # Always close the old stream first: capture.start() is a
            # no-op while a stream is still active, and stop() drains
            # stale frames queued from the previous device.
            self._capture.stop()
            try:
                device = self._start_capture()
            except AudioError as exc:
                self._set_status(STATUS_ERROR, str(exc))
                self._emit_error(str(exc))
                return False
            if device is None:
                return False
            # Drop frames queued from the old device so detection/STT
            # never process audio from a microphone that is no longer
            # selected.
            flush = getattr(self._capture, "flush", None)
            if flush is not None:
                flush()
            # The audio source just changed: reset the KWS decoder
            # stream (same rule as the standby handover — engine.reset
            # is serialized against in-flight inference by the engine
            # lock) and notify session-scoped consumers so STT VAD state
            # and any partial utterance spanning the old stream are
            # discarded. Without this, pre-restart audio leaks into the
            # new stream's first recognition cycle.
            try:
                self._engine.reset()
            except Exception:  # noqa: BLE001
                logger.debug("KWS engine reset on capture restart failed",
                             exc_info=True)
            if self._on_system_resume is not None:
                try:
                    self._on_system_resume()
                except Exception:  # noqa: BLE001
                    logger.debug("Stream-restart listener failed",
                                 exc_info=True)
            self._switch_count += 1
            self._last_switch_monotonic = time.monotonic()
            new_status = STATUS_IDLE if was_idle else STATUS_LISTENING
            self._set_status(new_status, self._detail_for(device))
            return True

    def _desired_device(self) -> Optional[AudioDevice]:
        """Return the input device the current selection should use."""
        if self._selection_mode == MODE_DEVICE and self._selected_device_id:
            device = self._device_manager.find_device(
                self._selected_device_id
            )
            if device is not None and self._id_name_matches(
                self._selected_device_id, device
            ):
                return device
            # Selected device disappeared: fall back to the Windows
            # default and say so — never pretend it is still active.
            logger.warning(
                "Selected microphone %r is no longer available; "
                "falling back to Windows default input",
                self._selected_device_id,
            )
            self._fallback_notice = (
                f"fallback: {self._selected_device_id} unavailable"
            )
        else:
            self._fallback_notice = None
        try:
            return self._device_manager.default_input_device()
        except Exception:  # noqa: BLE001
            logger.debug("Cannot query default input device",
                         exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Dynamic device monitor
    # ------------------------------------------------------------------
    def _start_monitor(self) -> None:
        """Start the background thread watching for device changes."""
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._monitor_stop.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="mic-device-monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _stop_monitor(self) -> None:
        """Stop the device monitor thread."""
        self._monitor_stop.set()
        thread, self._monitor_thread = self._monitor_thread, None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)

    def _monitor_loop(self) -> None:
        while not self._monitor_stop.wait(self._monitor_interval):
            try:
                self._check_system_resume()
                self._check_device()
                self._update_readiness()
            except Exception:  # noqa: BLE001 - monitor must never die
                logger.exception("Microphone monitor check failed")

    # ------------------------------------------------------------------
    # OS suspend/resume: notifications + validated drift fallback
    # ------------------------------------------------------------------
    def notify_system_suspend(self, source: str = "power") -> None:
        """Windows reported the system is suspending.

        Called from the Qt native-event filter on PBT_APMSUSPEND.
        Nothing is recovered here — the process is about to freeze —
        but the flag lets the drift fallback treat the next large
        wall-clock jump as an expected resume rather than suspect.
        """
        self._suspend_notified = True
        logger.info("System suspend notified (%s)", source)

    def notify_system_resume(self, source: str = "power") -> None:
        """Windows reported the system resumed from sleep.

        Trusted signal (WM_POWERBROADCAST PBT_APMRESUME*): skips the
        drift-candidate phase and goes straight to recovery. Debounced
        so repeated notifications cannot trigger repeated flushes or
        restarts.
        """
        now = time.monotonic()
        if now - self._last_resume_handled < self._resume_notify_debounce:
            logger.debug("Resume notification (%s) debounced", source)
            return
        if self._status not in (STATUS_LISTENING, STATUS_IDLE):
            self._last_resume_handled = now
            logger.info(
                "Resume notification (%s) while status=%s — ignored",
                source, self._status,
            )
            return
        self._suspend_notified = False
        self._resume_candidate = None
        self._resume_events += 1
        logger.warning(
            "OS resume notification (%s) — recovering audio pipeline",
            source,
        )
        self._handle_system_resume()

    def _check_system_resume(self) -> None:
        """Drift-based suspend/resume fallback (validated).

        A large wall-vs-monotonic discrepancy is only a *candidate*:
        NTP synchronization and manual clock changes produce the same
        signature while audio keeps flowing. The candidate is
        confirmed after ``_resume_candidate_seconds`` only if the
        pipeline actually went stale (callbacks/frames stopped
        advancing). A healthy pipeline means it was a clock step.
        """
        now_wall = time.time()
        now_mono = time.monotonic()
        wall_dt = now_wall - self._last_wall_time
        mono_dt = now_mono - self._last_mono_time
        self._last_wall_time = now_wall
        self._last_mono_time = now_mono
        if wall_dt - mono_dt > self._resume_drift_threshold:
            self._note_resume_candidate(now_mono, wall_dt, mono_dt)
        self._evaluate_resume_candidate(now_mono)

    def _note_resume_candidate(
        self, now_mono: float, wall_dt: float, mono_dt: float
    ) -> None:
        if self._status not in (STATUS_LISTENING, STATUS_IDLE):
            return
        if self._resume_candidate is not None:
            return
        cap = self._capture
        self._resume_candidate = {
            "since": now_mono,
            "wall_dt": wall_dt,
            "mono_dt": mono_dt,
            "frames": getattr(cap, "frames_captured", 0),
            "seq": getattr(cap, "callback_seq", 0),
        }
        logger.info(
            "Possible OS suspend/resume (unconfirmed): wall +%.1fs "
            "vs monotonic +%.1fs — validating pipeline",
            wall_dt, mono_dt,
        )

    def _evaluate_resume_candidate(self, now_mono: float) -> None:
        cand = self._resume_candidate
        if cand is None:
            return
        if now_mono - cand["since"] < self._resume_candidate_seconds:
            return
        cap = self._capture
        frames_now = getattr(cap, "frames_captured", 0)
        seq_now = getattr(cap, "callback_seq", 0)
        stall_fn = getattr(cap, "callback_stall_seconds", None)
        stall = stall_fn() if stall_fn is not None else None
        advanced = (
            frames_now > cand["frames"] or seq_now > cand["seq"]
        )
        fresh = (
            stall is not None
            and stall < self._resume_candidate_fresh_seconds
        )
        self._resume_candidate = None
        if advanced or fresh:
            # Clock stepped but audio is healthy — NTP/manual change,
            # not a suspend. No flush, no restart.
            logger.debug(
                "Clock drift was not a suspend: frames %+d, seq %+d, "
                "callback_age=%s",
                frames_now - cand["frames"],
                seq_now - cand["seq"],
                stall,
            )
            return
        self._resume_events += 1
        logger.warning(
            "OS suspend/resume confirmed (wall +%.1fs vs monotonic "
            "+%.1fs, frames %+d): recovering audio pipeline",
            cand["wall_dt"], cand["mono_dt"],
            frames_now - cand["frames"],
        )
        self._handle_system_resume()

    def _handle_system_resume(self) -> None:
        """Post-resume recovery: drop stale audio, verify callbacks.

        Frames queued before the freeze are seconds-to-minutes old and
        must never reach the wake-word engine or STT. The tighter
        stall threshold used for the next ``_resume_verify_seconds``
        makes the watchdog restart a dead stream quickly instead of
        waiting the full standby threshold. Readiness milestones are
        tracked until the pipeline proves READY.
        """
        now_mono = time.monotonic()
        self._last_resume_handled = now_mono
        self._resume_pending = True
        self._resume_failure_logged = False
        self._resume_milestones = {
            "resume_detected": {
                "monotonic": now_mono,
                "wall": datetime.now().isoformat(timespec="milliseconds"),
            }
        }
        flush = getattr(self._capture, "flush", None)
        if flush is not None:
            try:
                flush()
            except Exception:  # noqa: BLE001
                logger.debug("Post-resume queue flush failed",
                             exc_info=True)
        if self._on_system_resume is not None:
            try:
                self._on_system_resume()
            except Exception:  # noqa: BLE001
                logger.debug("Resume listener failed", exc_info=True)
        self._resume_verify_until = (
            now_mono + self._resume_verify_seconds
        )

    # ------------------------------------------------------------------
    # Readiness: evidence-based READY, not just PortAudio active=True
    # ------------------------------------------------------------------
    def audio_readiness(self) -> dict:
        """Structured readiness snapshot for diagnostics/UI."""
        return {
            "ready": bool(self._last_readiness.get("ready")),
            "checks": dict(self._last_readiness.get("checks") or {}),
            "failure_reason":
                self._last_readiness.get("failure_reason"),
            "status": self._status,
            "status_detail": self._status_detail,
            "resume_pending": self._resume_pending,
            "resume_milestones": dict(self._resume_milestones),
            "last_resume_report": self._last_resume_report,
        }

    def _update_readiness(self) -> None:
        """Evaluate pipeline readiness once per monitor tick.

        READY requires evidence, not just ``stream.active``:
        expected device resolved, callbacks arriving, captured and
        processed frame counters advancing, and the wake-word engine
        operational. While a resume is pending, each newly-true check
        records a milestone timestamp; full readiness closes the
        report. A permanently unavailable microphone can never reach
        READY — the failure reason is reported instead.
        """
        now = time.monotonic()
        cap = self._capture
        det = self._detection
        cap_stats = cap.stats() if hasattr(cap, "stats") else {}
        det_stats = det.stats() if det is not None else {}

        desired = self._last_desired
        device_available = (
            desired is not None or cap_stats.get("device") is not None
        )
        expected_device = (
            getattr(desired, "name", None)
            or cap_stats.get("device")
        )
        stream_running = bool(cap_stats.get("running"))
        cb_age = cap_stats.get("callback_age_seconds")
        stall_fn = getattr(cap, "callback_stall_seconds", None)
        stall = stall_fn() if stall_fn is not None else cb_age
        callbacks_fresh = (
            stall is not None and stall < self._callback_stall_threshold
        )
        det_running = det is not None and det.is_running
        engine_name = getattr(self._engine, "name", "null")
        engine_ok = engine_name != "null"

        # Counter advancement is measured over a ~1 s window: the
        # mark is only refreshed at window boundaries so fast monitor
        # ticks don't shrink the comparison interval to ~0.
        mark = self._readiness_mark
        captured = cap_stats.get("frames_captured") or 0
        processed = det_stats.get("frames_processed") or 0
        if mark is None:
            self._readiness_mark = {
                "mono": now, "captured": captured,
                "processed": processed,
            }
        elif now - mark["mono"] >= 1.0:
            self._frames_advancing = captured > mark["captured"]
            self._detector_consuming = (
                processed > mark["processed"]
                or (
                    det_stats.get("last_processed_age_seconds")
                    is not None
                    and det_stats["last_processed_age_seconds"] < 2.0
                )
            )
            self._readiness_mark = {
                "mono": now, "captured": captured,
                "processed": processed,
            }
        frames_advancing = getattr(self, "_frames_advancing", False)
        detector_consuming = getattr(
            self, "_detector_consuming", False
        )

        should_run = self._status in (STATUS_LISTENING, STATUS_IDLE)
        checks = {
            "device_available": bool(device_available),
            "expected_device": expected_device,
            "stream_running": stream_running,
            "callbacks_fresh": bool(callbacks_fresh),
            "frames_advancing": bool(frames_advancing),
            "detector_running": bool(det_running),
            "detector_consuming": bool(detector_consuming),
            "engine_operational": bool(engine_ok),
            "status_allows_capture": bool(should_run),
        }
        ready = all(
            v for k, v in checks.items()
            if k != "expected_device"
        )
        failure = None if ready else next(
            (k for k, v in checks.items()
             if k != "expected_device" and not v),
            None,
        )
        self._last_readiness = {
            "ready": ready,
            "checks": checks,
            "failure_reason": failure,
        }

        if not self._resume_pending:
            return
        # Record milestones for checks that just became true.
        for key in (
            "device_available", "stream_running", "callbacks_fresh",
            "frames_advancing", "detector_running",
            "detector_consuming", "engine_operational",
        ):
            if checks.get(key) and key not in self._resume_milestones:
                self._resume_milestones[key] = {
                    "monotonic": now,
                    "wall": datetime.now().isoformat(
                        timespec="milliseconds"),
                }
        if ready:
            self._resume_pending = False
            self._last_resume_report = {
                "result": "READY",
                "completed_monotonic": now,
                "completed_wall": datetime.now().isoformat(
                    timespec="milliseconds"),
                "milestones": dict(self._resume_milestones),
            }
            logger.info(
                "Post-resume audio pipeline READY: %s",
                {
                    k: v["wall"]
                    for k, v in self._resume_milestones.items()
                },
            )
        elif (
            now - self._resume_milestones["resume_detected"]["monotonic"]
            > self._resume_verify_seconds * 2
            and not self._resume_failure_logged
        ):
            self._resume_failure_logged = True
            self._last_resume_report = {
                "result": "FAILED",
                "failure_reason": failure,
                "checks": dict(checks),
                "completed_wall": datetime.now().isoformat(
                    timespec="milliseconds"),
            }
            logger.warning(
                "Post-resume audio pipeline NOT ready: %s "
                "(status=%s detail=%s)",
                failure, self._status, self._status_detail,
            )

    def _check_device(self) -> None:
        """One monitor pass: restart capture if the input changed/died."""
        if self._status == STATUS_DISABLED:
            self._last_desired = None
            return
        now = time.monotonic()
        # A stall-backoff period that has expired resumes LISTENING so
        # the liveness check below gets another bounded recovery cycle
        # (with doubled backoff if it stalls again).
        if (
            self._status == STATUS_ERROR
            and self._stall_backoff_until
            and now >= self._stall_backoff_until
        ):
            self._stall_backoff_until = 0.0
            self._stall_restarts = 0
            self._stall_window_start = 0.0
            self._set_status(
                STATUS_LISTENING, self._detail_for(self._capture.device)
            )
        # Back off after a failed restart so a persistently broken device
        # does not cause a tight reopen loop.
        if (
            self._status == STATUS_ERROR
            and now - self._last_restart_attempt < 5.0
        ):
            return

        should_run = self._status in (STATUS_LISTENING, STATUS_IDLE)
        # Hold the restart lock across the whole check: without it the
        # monitor can observe the stop->start gap of a concurrent
        # select_device()/restart and fire a redundant restart that
        # drains the STT/wake-word queue.
        with self._restart_lock:
            # A dead stream must be recovered even from STATUS_ERROR:
            # a callback exception or a failed restart can leave the
            # controller in ERROR while ``is_running`` is False, and
            # without this check nothing ever retries — standby stays
            # deaf until the user toggles the microphone.
            stream_dead = (
                should_run or self._status == STATUS_ERROR
            ) and not self._capture.is_running
            desired = self._desired_device()
            self._last_desired = desired
            active = self._capture.device

            if desired is None:
                if should_run or stream_dead:
                    if (
                        not self._capture.is_running
                        and self._status != STATUS_ERROR
                    ):
                        self._set_status(
                            STATUS_ERROR, "no input device available"
                        )
                        self._emit_error("no input device available")
                return

            if stream_dead:
                logger.warning(
                    "Capture stream is no longer running; restarting on %r",
                    desired.name,
                )
                self._restart_for_selection("stream stopped unexpectedly")
                return

            if active is None:
                if self._status != STATUS_DISABLED:
                    self._restart_for_selection("no active capture device")
                return

            # Standby watchdog: the stream can stay PortAudio-active
            # while the driver stops invoking callbacks (endpoint stall,
            # power management, glitch). Without this check the wake-word
            # pipeline starves silently — the exact long-standby failure.
            if should_run:
                self._check_pipeline_liveness(now)

            if not self._device_manager.same_physical_device(
                desired, active
            ):
                reason = (
                    f"default input changed: {active.name!r} -> "
                    f"{desired.name!r}"
                ) if self._selection_mode == MODE_AUTO else (
                    f"selected device changed: {active.name!r} -> "
                    f"{desired.name!r}"
                )
                self._restart_for_selection(reason)

    # ------------------------------------------------------------------
    # Standby watchdog: capture-callback and detection-worker liveness
    # ------------------------------------------------------------------
    def _check_pipeline_liveness(self, now: float) -> None:
        """Detect silent pipeline stalls while standby listening."""
        stall_fn = getattr(self._capture, "callback_stall_seconds", None)
        stall = stall_fn() if stall_fn is not None else None
        threshold = self._callback_stall_threshold
        if now < self._resume_verify_until:
            # Post OS-resume window: the endpoint is the most likely
            # to be dead, so verify with a tighter threshold.
            threshold = min(threshold, self._resume_stall_threshold)
        if stall is not None and stall >= threshold:
            self._handle_callback_stall(now, stall)
        # NOTE: healthy readings do NOT reset the retry budget — a
        # flapping endpoint that stalls again inside the window must
        # still exhaust its bounded retries. The window itself expires
        # after _STALL_WINDOW_SECONDS of stall-free operation.
        self._check_detection_liveness(now)

    def _handle_callback_stall(self, now: float, stall: float) -> None:
        """Restart the capture stream after callback inactivity.

        Bounded: at most ``_STALL_MAX_RETRIES`` restarts per
        ``_STALL_WINDOW_SECONDS``; on exhaustion the controller enters
        ERROR with exponential backoff instead of looping forever.
        """
        if now < self._stall_backoff_until:
            return
        if now - self._last_restart_attempt < _STALL_MIN_INTERVAL:
            return
        if (
            self._stall_window_start == 0.0
            or now - self._stall_window_start > _STALL_WINDOW_SECONDS
        ):
            self._stall_window_start = now
            self._stall_restarts = 0
        if self._stall_restarts >= _STALL_MAX_RETRIES:
            self._stall_backoff_seconds = min(
                (self._stall_backoff_seconds or _STALL_BACKOFF_BASE) * 2,
                _STALL_BACKOFF_MAX,
            )
            self._stall_backoff_until = now + self._stall_backoff_seconds
            self._stall_restarts = 0
            self._stall_window_start = now
            logger.error(
                "Capture callback stall persists (%.1fs); auto-recovery "
                "paused for %.0fs",
                stall, self._stall_backoff_seconds,
            )
            self._set_status(
                STATUS_ERROR,
                f"microphone stalled; retrying in "
                f"{self._stall_backoff_seconds:.0f}s",
            )
            self._emit_error("microphone capture stalled")
            return
        self._stall_restarts += 1
        logger.warning(
            "Capture callback stall %.1fs — restarting stream "
            "(attempt %d/%d)",
            stall, self._stall_restarts, _STALL_MAX_RETRIES,
        )
        self._restart_for_selection(
            f"audio callback stall ({stall:.1f}s)"
        )

    def _check_detection_liveness(self, now: float) -> None:
        """Restart the wake-word worker if it silently stopped.

        Only evaluated while standby LISTENING is expected (detection
        service present, capture callbacks fresh). A starved-but-alive
        worker is distinguished from a dead thread.
        """
        if self._status != STATUS_LISTENING:
            self._detect_proc_seen = -1
            self._detect_proc_seen_at = 0.0
            self._detection_none_since = 0.0
            return
        # Hold the restart lock across the whole check (same pattern as
        # _check_capture_liveness): _restart_detection publishes
        # _detection=None while it rebuilds the worker, and an
        # unsynchronized observer can read that gap and either skip a
        # needed restart or observe a half-restarted service.
        with self._restart_lock:
            detection = self._detection
            if detection is None:
                self._detect_proc_seen = -1
                self._detect_proc_seen_at = 0.0
                # LISTENING with no detector is legitimate only inside
                # the brief suspend/disable detach window (those paths
                # flip status right after stopping the worker). If it
                # persists, a detection restart failed and left standby
                # permanently deaf — rebuild through the bounded
                # detect-restart budget/backoff.
                if self._detection_none_since == 0.0:
                    self._detection_none_since = now
                elif (
                    now - self._detection_none_since
                    >= _DETECTION_MISSING_GRACE
                ):
                    self._detection_none_since = 0.0
                    self._restart_detection("detection worker missing")
                return
            self._detection_none_since = 0.0
            if now < self._detect_backoff_until:
                return

            thread_dead = not detection.is_running
            proc = int(getattr(detection, "frames_processed", 0) or 0)
            proc_stalled = False
            if proc != self._detect_proc_seen:
                self._detect_proc_seen = proc
                self._detect_proc_seen_at = now
            elif self._detect_proc_seen_at:
                proc_stalled = (
                    now - self._detect_proc_seen_at
                    >= self._detection_stall_threshold
                )
            # Only call it a consumer stall when audio is actually
            # flowing: a capture stall is handled by the callback
            # watchdog instead.
            cap_stall_fn = getattr(
                self._capture, "callback_stall_seconds", None
            )
            cap_stall = cap_stall_fn() if cap_stall_fn else None
            capture_flowing = (
                cap_stall is not None and cap_stall < 2.0
            )
            queue_waiting = False
            try:
                queue_waiting = (
                    self._capture.stats().get("queue_depth") or 0
                ) > 0
            except Exception:  # noqa: BLE001
                pass

            if thread_dead or (
                proc_stalled
                and capture_flowing
                and queue_waiting
            ):
                self._restart_detection(
                    "worker thread dead" if thread_dead
                    else f"no frames processed for "
                         f"{self._detection_stall_threshold:.0f}s"
                )

    def _restart_detection(self, reason: str) -> None:
        """Recreate the wake-word detection worker (bounded retries).

        Ownership invariant: a replacement consumer is installed only
        when no retired worker may still be alive — a stop()/join
        timeout never justifies a second consumer. While a survivor
        lives, the previous reference is kept (or ``None`` on the
        missing-detector path) and later watchdog ticks retry; once
        every retired worker has actually exited the slot frees and
        the bounded retry budget installs the replacement.
        """
        now = time.monotonic()
        with self._restart_lock:
            if now < self._detect_backoff_until:
                return
            if not self._detector_slot_available():
                # A retired worker still owns the frame queue. Defer
                # without consuming the retry budget — this is not a
                # failed restart, it is an ownership wait.
                logger.warning(
                    "Wake-word detection restart deferred (%s): %d "
                    "retiring worker(s) still alive",
                    reason, len(self._retiring_detectors),
                )
                return
            if (
                self._detect_window_start == 0.0
                or now - self._detect_window_start > _DETECT_WINDOW_SECONDS
            ):
                self._detect_window_start = now
                self._detect_restarts = 0
            if self._detect_restarts >= _DETECT_MAX_RETRIES:
                self._detect_backoff_until = now + _DETECT_BACKOFF_SECONDS
                self._detect_restarts = 0
                self._detect_window_start = now
                logger.error(
                    "Wake-word detection restart failed repeatedly; "
                    "backing off %.0fs",
                    _DETECT_BACKOFF_SECONDS,
                )
                return
            self._detect_restarts += 1
            self._detection_generation += 1
            generation = self._detection_generation
            old = self._detection
            self._detection = None
            # Mark BEFORE stop: if the join times out with the worker
            # still alive, the slot check below must see it.
            self._mark_retiring(old)
            self._stop_detector(old)
            if self._detector_slot_available():
                try:
                    self._detection = self._new_detection_service(
                        generation
                    )
                    self._detection.start()
                    logger.warning(
                        "Wake-word detection restarted (%s, attempt "
                        "%d/%d)",
                        reason, self._detect_restarts, _DETECT_MAX_RETRIES,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("Wake-word detection restart failed")
            else:
                # Fail closed: the old worker survived its join and may
                # still be the only live consumer. Keep ownership of
                # its reference — never install a second consumer —
                # and let later watchdog ticks retry once it exits.
                self._detection = old
                logger.error(
                    "Wake-word detection restart deferred (%s): old "
                    "worker generation=%s survived stop(); keeping "
                    "ownership until it exits",
                    reason, getattr(old, "_generation", None),
                )

    # ------------------------------------------------------------------
    # Detector ownership: at most one live frame-queue consumer
    # ------------------------------------------------------------------
    def _mark_retiring(self, service) -> None:
        """Track a detector whose worker may still consume frames.

        Called under ``_restart_lock`` at detach time — BEFORE
        ``stop()`` — so a concurrent installer can never miss a
        still-live consumer during the stop/join window. Only
        currently-running workers are tracked; dead threads cannot
        consume frames.
        """
        if service is None:
            return
        if not getattr(service, "is_running", False):
            return
        if service not in self._retiring_detectors:
            self._retiring_detectors.append(service)
            logger.info(
                "Wake-word detector generation=%s marked retiring",
                getattr(service, "_generation", None),
            )

    def _detector_slot_available(self) -> bool:
        """True when no retired worker may still drain the frame queue.

        Purges dead entries, so a survivor that has actually exited
        frees the slot for the next install attempt. Must be called
        with ``_restart_lock`` held.
        """
        self._retiring_detectors[:] = [
            s for s in self._retiring_detectors
            if getattr(s, "is_running", False)
        ]
        return not self._retiring_detectors

    @staticmethod
    def _stop_detector(service) -> None:
        """Stop a detector worker with the bounded service join."""
        if service is None:
            return
        try:
            service.stop()
        except Exception:  # noqa: BLE001
            logger.debug("Detection stop failed", exc_info=True)

    def _new_detection_service(
        self,
        generation: int,
        on_first_frame: Optional[Callable[[], None]] = None,
    ) -> WakeWordDetectionService:
        """Build a detection worker whose wake callback is bound to this
        exact service instance.

        ``_handle_wake_word`` ignores emissions from any service that is
        no longer ``self._detection``, so a retired or orphaned worker
        can never deliver a duplicate wake event nor suspend the newer
        detector that replaced it.
        """
        holder: dict = {}

        def _on_detected(keyword: str) -> None:
            self._handle_wake_word(holder.get("service"), keyword)

        service = WakeWordDetectionService(
            capture=self._capture,
            engine=self._engine,
            threshold=self._threshold,
            cooldown_seconds=self._cooldown,
            on_detected=_on_detected,
            on_stats=self._on_stats,
            on_first_frame=on_first_frame,
            generation=generation,
        )
        holder["service"] = service
        return service

    def _handle_wake_word(
        self,
        service: Optional[WakeWordDetectionService],
        keyword: str,
    ) -> None:
        # Stale-service guard: only the currently installed detector may
        # activate a session. Emissions from a retired/orphaned worker
        # (e.g. one whose stop() outlived a watchdog restart) are dropped
        # before they can suspend the replacement detector or reach the
        # session callback a second time.
        if service is not self._detection:
            logger.info(
                "Ignoring wake-word emission from retired detector "
                "(generation=%s, keyword=%r)",
                getattr(service, "_generation", None), keyword,
            )
            return
        # Prefer the detection worker's emission timestamp over the
        # callback/Qt-queue receipt time so session telemetry measures
        # the real wake->activation gap.
        detection = self._detection
        last = getattr(detection, "last_detection", None)
        detected_at = (
            last.get("monotonic")
            if isinstance(last, dict) else None
        )
        self._last_wake_detected_monotonic = (
            float(detected_at) if detected_at else time.monotonic()
        )
        self._last_wake_keyword = keyword
        if isinstance(last, dict):
            info = dict(last)
            info.setdefault("keyword", keyword)
        else:
            info = {
                "monotonic": self._last_wake_detected_monotonic,
                "keyword": keyword,
            }
        self._last_wake_detection = info
        self.suspend_detection()
        if self._on_wake_word is not None:
            try:
                self._on_wake_word(keyword)
            except Exception:  # noqa: BLE001
                logger.exception("Wake-word callback failed")

    def _handle_capture_error(self, exc: Exception) -> None:
        self._set_status(STATUS_ERROR, str(exc))
        self._emit_error(str(exc))

    def _emit_error(self, message: str) -> None:
        if self._on_error is not None:
            try:
                self._on_error(message)
            except Exception:  # noqa: BLE001
                pass

    def _set_status(self, status: str, detail: str) -> None:
        self._status = status
        self._status_detail = detail
        logger.info("Microphone status: %s (%s)", status, detail)
        if self._on_status is not None:
            try:
                self._on_status(status, detail)
            except Exception:  # noqa: BLE001
                logger.exception("Status callback failed")
