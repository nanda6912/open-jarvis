"""Utterance-level speech recognition worker.

Consumes 16 kHz frames from an :class:`AudioCaptureService` queue while a
voice session is active, performs lightweight energy-based voice
activity detection, and transcribes each completed utterance with an
injected :class:`SpeechRecognizer`.

The worker runs on a dedicated thread; all callbacks are invoked from
that thread — receivers must marshal to the GUI thread themselves.
No raw audio is stored: utterance samples live only in memory for the
duration of one decode.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np

from src.voice import recognition_taxonomy
from src.voice.audio_capture_service import AudioCaptureService
from src.voice.stt_engine import SpeechRecognizer

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


class SpeechRecognitionService:
    """Background worker that turns captured audio into text.

    Args:
        capture: Frame source (``AudioCaptureService`` or compatible
            object exposing ``get_frame(timeout)``).
        recognizer: STT backend implementing ``transcribe()``.
        speech_threshold: Minimum frame RMS to count as speech.
        silence_hangover: Seconds of silence that ends an utterance.
        min_utterance: Minimum seconds of detected speech to transcribe.
        max_utterance: Hard cap on utterance length (seconds).
        pre_roll: Seconds of audio kept before speech onset.
    """

    def __init__(
        self,
        capture: AudioCaptureService,
        recognizer: SpeechRecognizer,
        speech_threshold: float = 0.01,
        silence_hangover: float = 1.0,
        min_utterance: float = 0.3,
        max_utterance: float = 15.0,
        pre_roll: float = 0.3,
        frame_timeout: float = 0.25,
        vad_min_threshold: float = 0.003,
        vad_max_threshold: float = 0.05,
        noise_window: int = 120,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_no_speech: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_speech_activity: Optional[Callable[[bool], None]] = None,
    ) -> None:
        self._capture = capture
        self._recognizer = recognizer
        self._speech_threshold = speech_threshold
        self._silence_hangover = silence_hangover
        self._min_utterance = min_utterance
        self._max_utterance = max_utterance
        self._frame_timeout = frame_timeout
        self._on_transcript = on_transcript
        self._on_no_speech = on_no_speech
        self._on_error = on_error
        self._on_speech_activity = on_speech_activity

        # Adaptive VAD: the configured ``speech_threshold`` is the
        # ceiling; the effective threshold tracks 6x the median ambient
        # frame RMS, bounded by [vad_min, min(vad_max, configured)].
        # Quiet microphones (e.g. Intel SST arrays) get a lower floor;
        # noisy rooms push the threshold up instead of false-triggering.
        self._vad_floor = vad_min_threshold
        self._vad_cap = min(vad_max_threshold, speech_threshold)
        self._noise_window = noise_window
        self._noise_floor: deque[float] = deque(maxlen=noise_window)
        self._effective_threshold = speech_threshold

        # Pre-roll keeps the last ``pre_roll`` seconds of audio before
        # speech onset. Frame duration varies with the capture rate, so
        # the deque is oversized and trimmed by sample count on onset.
        self._pre_roll_seconds = pre_roll
        self._pre_roll: deque[np.ndarray] = deque(maxlen=400)
        self._thread: Optional[threading.Thread] = None
        # Per-run stop event + generation. A single shared Event would
        # be cleared by the next start() while a previous worker is
        # still inside a slow decode (stop() joins only 2 s), silently
        # resurrecting it into a second consumer of the same frame
        # queue — duplicate transcripts and stale results delivered
        # into a NEW voice session. Each run gets its own Event that is
        # never cleared again, plus a generation counter so an
        # in-flight decode from a retired run drops its result instead
        # of emitting it into the new session.
        self._run_stop: Optional[threading.Event] = None
        self._run_generation = 0
        self._lock = threading.RLock()
        self._collecting = False
        self._utterance: list[np.ndarray] = []
        self._utterance_samples = 0
        self._speech_started = 0.0
        self._last_voiced = 0.0

        # Diagnostics (no audio retained).
        self.utterances_completed = 0
        self.utterances_dropped = 0
        self.inference_errors = 0
        self.last_rms = 0.0
        self.last_transcript = ""
        self.last_inference_monotonic = 0.0
        self.last_utterance_rms = 0.0
        self.last_utterance_peak = 0.0
        # Per-utterance frame diagnostics.
        self._frames_seen = 0
        self._session_frames = 0
        self._voiced_frames = 0
        self._rejected_frames = 0
        self._utterance_voiced = 0
        self._utt_rms: list = []          # per-frame RMS inside utterance
        self._last_frame_monotonic = 0.0
        self._session_started_monotonic = 0.0
        self._starved = False
        self._starved_since = 0.0
        self._below_vad_emitted = False
        # Emit "no audio" only after this much sustained starvation so
        # brief queue gaps never masquerade as a dead capture stream.
        self._starvation_emit_after = 2.0
        self._below_vad_emitted = False
        self._below_vad_streak = 0
        # Per-utterance stats snapshot taken before _reset_utterance()
        # clears the live counters (V10.5 diagnostics fix).
        self._pending_stats: Optional[dict] = None
        # V10.8 Task 4: per-attempt diagnostics. Every recognition
        # attempt (utterance, VAD rejection, starvation, STT failure)
        # gets one structured record classifiable by processing stage.
        # ``last_attempt`` is the most recent record; ``_outcome_counts``
        # aggregates canonical categories for stats().
        self._attempt_seq = 0
        self.last_attempt: Optional[dict] = None
        self._outcome_counts: dict = {}

    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def engine_name(self) -> str:
        return getattr(self._recognizer, "name", "unknown")

    @property
    def model_dir(self) -> Optional[str]:
        md = getattr(self._recognizer, "model_dir", None)
        return str(md) if md is not None else None

    def set_callbacks(
        self,
        on_transcript: Optional[Callable[[str], None]] = None,
        on_no_speech: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
        on_speech_activity: Optional[Callable[[bool], None]] = None,
    ) -> None:
        """Register or replace event callbacks."""
        self._on_transcript = on_transcript
        self._on_no_speech = on_no_speech
        self._on_error = on_error
        self._on_speech_activity = on_speech_activity

    def stats(self) -> dict:
        return {
            "recognizer": self.engine_name,
            "utterances_completed": self.utterances_completed,
            "utterances_dropped": self.utterances_dropped,
            "inference_errors": self.inference_errors,
            "last_rms": self.last_rms,
            "last_transcript": self.last_transcript,
            "last_utterance_rms": self.last_utterance_rms,
            "last_utterance_peak": self.last_utterance_peak,
            "collecting": self._collecting,
            "frames_seen": self._frames_seen,
            "voiced_frames": self._voiced_frames,
            "rejected_frames": self._rejected_frames,
            "vad_threshold": self._effective_threshold,
            "noise_floor": (
                sorted(self._noise_floor)[len(self._noise_floor) // 2]
                if self._noise_floor else None
            ),
            "device": getattr(
                getattr(self._capture, "device", None), "name", None
            ),
            "device_stable_id": getattr(
                getattr(self._capture, "device", None), "stable_id", None
            ),
            "sample_rate": getattr(self._capture, "sample_rate", None),
            "stream_active": bool(
                getattr(getattr(self._capture, "_stream", None),
                        "active", False)
            ),
            "input_channels": getattr(self._capture, "input_channels",
                                      None),
            "session_frames": self._session_frames,
            "last_frame_age_seconds": (
                time.monotonic() - self._last_frame_monotonic
                if self._last_frame_monotonic else None
            ),
            "session_started_monotonic": (
                self._session_started_monotonic or None
            ),
            # V10.8 Task 4: per-attempt diagnostics.
            "last_attempt": self.last_attempt,
            "outcome_counts": dict(self._outcome_counts),
        }

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the recognition worker (idempotent)."""
        if self.is_running:
            return
        self._run_generation += 1
        self._run_stop = threading.Event()
        self._reset_utterance()
        # Fresh VAD calibration per session: a noise floor learned
        # hours ago (or on a different microphone) must not gate the
        # first command after standby/resume/device change.
        self._noise_floor.clear()
        self._effective_threshold = self._speech_threshold
        self._session_frames = 0
        self._session_started_monotonic = time.monotonic()
        generation = self._run_generation
        stop_event = self._run_stop
        self._thread = threading.Thread(
            target=self._run,
            args=(generation, stop_event),
            name="speech-recognition",
            daemon=True,
        )
        self._thread.start()
        self._log_session_banner()
        logger.info(
            "Speech recognition started (engine=%s, threshold=%.4f)",
            self.engine_name,
            self._speech_threshold,
        )

    def stop(self) -> None:
        """Stop the worker and discard any partial utterance."""
        stop_event = self._run_stop
        self._run_stop = None
        if stop_event is not None:
            stop_event.set()
        thread, self._thread = self._thread, None
        self._reset_utterance()
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=2.0)
        logger.info("Speech recognition stopped")

    def shutdown(self) -> None:
        """Stop the worker and release the recognizer."""
        self.stop()
        try:
            self._recognizer.close()
        except Exception:  # noqa: BLE001
            pass

    def notify_system_resume(self) -> None:
        """Reset utterance/VAD state after an OS suspend/resume.

        Called by the microphone controller's resume path. Any
        partially-collected utterance spans the freeze and must be
        discarded; the noise floor is recalibrated from fresh audio
        so a stale estimate cannot reject the first post-resume
        command. Safe to call while running — state resets are
        locked where required.
        """
        self._reset_utterance()
        self._noise_floor.clear()
        self._effective_threshold = self._speech_threshold
        self._starved = False
        self._starved_since = 0.0
        self._starvation_emitted = False
        self._below_vad_streak = 0
        self._below_vad_emitted = False
        logger.info("STT state reset after system resume")

    # ------------------------------------------------------------------
    def _log_session_banner(self) -> None:
        """Priority-1 proof block: exactly which code/device is live."""
        import os
        import sys
        from pathlib import Path

        module_path = Path(__file__).resolve()
        try:
            mtime = module_path.stat().st_mtime
        except OSError:
            mtime = 0.0
        build_id = os.environ.get("VOICEOS_BUILD_ID", "dev")
        cap = self._capture
        dev = getattr(cap, "device", None)
        stream = getattr(cap, "_stream", None)
        logger.info(
            "STT session banner: module=%s module_mtime=%.0f "
            "build=%s pid=%d python=%s project_root=%s "
            "adaptive_vad=ENABLED(vad_min=%.4f vad_max=%.4f) "
            "device=%r device_id=%r stable_id=%r sample_rate=%s "
            "max_channels=%s stream_channels=%s stream_active=%s "
            "configured_threshold=%.4f "
            "effective_threshold=%.4f noise_floor=%s "
            "calibration_frames=%d",
            module_path,
            mtime,
            build_id,
            os.getpid(),
            sys.executable,
            Path(__file__).resolve().parents[2],
            self._vad_floor,
            self._vad_cap,
            getattr(dev, "name", None),
            getattr(dev, "device_id", None),
            getattr(dev, "stable_id", None),
            getattr(cap, "sample_rate", None),
            getattr(dev, "max_input_channels", None),
            getattr(cap, "input_channels", None),
            bool(getattr(stream, "active", False)),
            self._speech_threshold,
            self._effective_threshold,
            (
                sorted(self._noise_floor)[len(self._noise_floor) // 2]
                if self._noise_floor else None
            ),
            len(self._noise_floor),
        )

    # ------------------------------------------------------------------
    # V10.8 Task 4: structured attempt diagnostics
    # ------------------------------------------------------------------
    def _stt_log(self, event: str, **fields) -> None:
        """Emit one structured STT trace event (no PCM, no transcript).

        Format mirrors the KWS ``EVENT key=value`` convention so a
        developer can follow one attempt end to end.
        """
        parts = []
        for key, value in fields.items():
            if isinstance(value, str):
                parts.append(f"{key}={value!r}")
            elif isinstance(value, float):
                parts.append(f"{key}={value:.4g}")
            else:
                parts.append(f"{key}={value}")
        logger.info("%s %s", event, " ".join(parts))

    def _capture_meta(self) -> dict:
        """Best-effort capture identity/health snapshot (no audio)."""
        try:
            cap_stats = (
                self._capture.stats()
                if hasattr(self._capture, "stats")
                else {}
            )
        except Exception:  # noqa: BLE001
            cap_stats = {}
        dev = getattr(self._capture, "device", None)
        stream = getattr(self._capture, "_stream", None)
        stream_active = cap_stats.get("stream_active")
        if stream_active is None and stream is not None:
            stream_active = bool(getattr(stream, "active", False))
        return {
            "device": cap_stats.get("device") or getattr(
                dev, "name", None),
            "device_stable_id": getattr(dev, "stable_id", None),
            "sample_rate": cap_stats.get("sample_rate") or getattr(
                self._capture, "sample_rate", None),
            "input_channels": getattr(
                self._capture, "input_channels", None),
            "stream_active": stream_active,
            "capture_running": cap_stats.get("running"),
            "frames_captured": cap_stats.get("frames_captured"),
            "frames_queued": cap_stats.get("frames_queued"),
            "frames_dropped": cap_stats.get("frames_dropped"),
        }

    def _record_attempt(self, outcome: str, **fields) -> dict:
        """Store one recognition-attempt record and bump counters.

        ``outcome`` is the canonical taxonomy category when one applies
        (e.g. ``stt_empty_transcript``); free-form detail goes in
        ``fields``. Returns the record for optional further use.
        """
        self._attempt_seq += 1
        category = recognition_taxonomy.is_category(outcome)
        record = {
            "attempt_id": f"stt-utt-{self._attempt_seq}",
            "monotonic": round(time.monotonic(), 3),
            "wall": time.time(),
            "outcome": outcome,
            "category": outcome if category else None,
            "engine": self.engine_name,
            "model_dir": self.model_dir,
            "session_frames": self._session_frames,
            "frames_seen": self._frames_seen,
            "voiced_frames": self._voiced_frames,
            "rejected_frames": self._rejected_frames,
            "vad_enabled": True,
            "vad_threshold_configured": self._speech_threshold,
            "vad_threshold_effective": self._effective_threshold,
            "noise_floor": self.stats().get("noise_floor"),
            **self._capture_meta(),
        }
        record.update(fields)
        self.last_attempt = record
        key = record["category"] or outcome
        self._outcome_counts[key] = self._outcome_counts.get(key, 0) + 1
        return record

    def _rms_snapshot(self, rms_list: list) -> dict:
        """min/max/mean/median over VALID (utterance) frame RMS only."""
        vals = sorted(float(v) for v in rms_list if v is not None)
        if not vals:
            return {
                "rms_min": None, "rms_max": None,
                "rms_mean": None, "rms_median": None,
            }
        return {
            "rms_min": round(vals[0], 5),
            "rms_max": round(vals[-1], 5),
            "rms_mean": round(sum(vals) / len(vals), 5),
            "rms_median": round(vals[len(vals) // 2], 5),
        }

    def _reset_utterance(self) -> None:
        with self._lock:
            self._collecting = False
            self._utterance = []
            self._utterance_samples = 0
            self._utterance_voiced = 0
            self._utt_rms = []
            self._pre_roll.clear()
            self._speech_started = 0.0
            self._last_voiced = 0.0

    def _emit_allowed(
        self,
        generation: Optional[int],
        stop_event: Optional[threading.Event],
    ) -> bool:
        """True while this worker run is still the current one.

        A worker whose run was stopped (its private ``stop_event`` stays
        set forever — it is never cleared by a later ``start()``) must
        not emit transcripts, no-speech or error callbacks: doing so
        would deliver a stale result into a NEW voice session.
        ``generation=None`` marks an untracked/direct invocation (unit
        tests, manual calls) and is always allowed.
        """
        if generation is None:
            return True
        return (
            generation == self._run_generation
            and stop_event is not None
            and not stop_event.is_set()
        )

    def _run(
        self, generation: int, stop_event: Optional[threading.Event]
    ) -> None:
        while (
            stop_event is not None and not stop_event.is_set()
        ):
            frame = self._capture.get_frame(timeout=self._frame_timeout)
            if frame is None:
                # No audio arriving (paused gate, stalled device): still
                # honour the silence hangover so utterances cannot get
                # stuck open forever, and report starvation distinctly
                # from "VAD rejected everything".
                self._check_hangover(generation, stop_event)
                self._check_starvation(generation, stop_event)
                continue
            self._starved = False
            self._starved_since = 0.0
            self._starvation_emitted = False
            try:
                self._process_frame(frame, generation, stop_event)
            except Exception:  # noqa: BLE001
                logger.exception("Speech recognition frame error")

    def _check_hangover(
        self,
        generation: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Finalize a collecting utterance if its silence timed out.

        V10.5: while the capture gate is paused (TTS playback) no frames
        arrive at all, so wall-clock "silence" is meaningless — finalizing
        here would truncate the utterance at the playback boundary and
        hand STT a clipped fragment. The utterance stays open; the first
        post-resume frames either continue it (speech) or trip the normal
        hangover (silence).
        """
        if getattr(self._capture, "paused", False):
            return
        now = time.monotonic()
        utterance = None
        duration = 0.0
        with self._lock:
            if not self._collecting:
                return
            silence = now - self._last_voiced
            duration = self._utterance_samples / SAMPLE_RATE
            if silence >= self._silence_hangover or (
                duration >= self._max_utterance
            ):
                utterance = self._utterance
                self._pending_stats = {
                    "voiced": self._utterance_voiced,
                    "rms": list(self._utt_rms),
                    "started": self._speech_started,
                }
                self._reset_utterance()
                self._emit_activity(False)
        if utterance is not None:
            reason = "max" if duration >= self._max_utterance else "silence"
            self._finalize(
                utterance, reason=reason,
                generation=generation, stop_event=stop_event,
            )

    def _check_starvation(
        self,
        generation: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        """Log once when the capture queue stops delivering frames."""
        # The TTS gate intentionally pauses capture during playback —
        # an empty queue then is expected, not a stalled stream.
        if getattr(self._capture, "paused", False):
            self._starved = False
            self._starved_since = 0.0
            self._starvation_emitted = False
            return
        now = time.monotonic()
        if not self._starved:
            self._starved = True
            self._starved_since = now
            return
        if now - self._starved_since < self._starvation_emit_after:
            return
        if getattr(self, "_starvation_emitted", False):
            return
        self._starvation_emitted = True
        try:
            cap_stats = (
                self._capture.stats()
                if hasattr(self._capture, "stats")
                else {}
            )
        except Exception:  # noqa: BLE001
            cap_stats = {}
        device = cap_stats.get("device") or getattr(
            getattr(self._capture, "device", None), "name", "unknown"
        )
        # V10.8 Task 4: distinguish a dead capture stream
        # (capture_failed) from a healthy stream that simply delivers
        # no frames (no_audio_received). A paused gate never reaches
        # here (early return above). Dead requires POSITIVE evidence —
        # an explicit running=False, a stats-reported stream_active=
        # False, or a real stream object reporting inactive. Missing
        # stats/stream objects mean "unknown", which stays
        # no_audio_received so fakes/minimal captures cannot fake a
        # dead device (and, downstream, cannot reset the session
        # inactivity timeout via the capture_failed touch path).
        stream = getattr(self._capture, "_stream", None)
        stream_active = cap_stats.get("stream_active")
        if stream_active is None and stream is not None:
            stream_active = bool(getattr(stream, "active", False))
        running = cap_stats.get("running")
        capture_dead = (
            running is False
            or cap_stats.get("stream_active") is False
            or (stream is not None and stream_active is False)
        )
        reason = "capture_failed" if capture_dead else "no_audio_frames"
        logger.warning(
            "No audio frames reaching STT (%s): "
            "device=%r running=%s stream_active=%s frames_captured=%s "
            "queued=%s frame_age=%s",
            reason,
            device,
            running,
            stream_active,
            cap_stats.get("frames_captured"),
            cap_stats.get("frames_queued"),
            cap_stats.get("frame_age_seconds"),
        )
        attempt = self._record_attempt(
            "capture_failed" if capture_dead else "no_audio_received",
            reason=reason,
            audio_seconds=0.0,
            samples=0,
            stt_invoked=False,
        )
        self._stt_log(
            "STT_NO_AUDIO",
            attempt_id=attempt["attempt_id"],
            reason=reason,
            device=device,
            running=running,
            stream_active=stream_active,
        )
        if self._emit_allowed(generation, stop_event):
            self._emit_no_speech(reason)

    def _adapt_threshold(self, rms: float, collecting: bool) -> None:
        """Update the noise-floor estimate and effective VAD threshold.

        Only non-collecting frames feed the estimator so speech never
        inflates the floor; the median is robust to transient loud
        sounds (keyboard clicks, TTS bleed). V10.5: frames at or above
        the current effective threshold are also excluded — otherwise
        every utterance onset (a speech-level frame seen while not yet
        collecting) contaminates the floor and can pin it near speech
        RMS (observed: noise_floor=0.0955 vs true ambient ~0.0001).
        """
        if collecting:
            return
        if rms >= self._effective_threshold:
            return
        self._noise_floor.append(rms)
        if len(self._noise_floor) < 10:
            return
        s = sorted(self._noise_floor)
        noise = s[len(s) // 2]
        self._effective_threshold = min(
            self._speech_threshold,
            max(self._vad_floor, noise * 6.0),
        )

    def _process_frame(
        self,
        frame: np.ndarray,
        generation: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        samples = np.asarray(frame, dtype=np.float32).reshape(-1)
        if samples.size == 0:
            return
        rms = float(np.sqrt(np.mean(samples * samples)))
        self.last_rms = rms
        self._frames_seen += 1
        self._session_frames += 1
        self._last_frame_monotonic = time.monotonic()
        voiced = rms >= self._effective_threshold
        if voiced:
            self._voiced_frames += 1
            self._below_vad_streak = 0
            self._below_vad_emitted = False
        else:
            self._rejected_frames += 1
            self._below_vad_streak += 1
            # Distinct outcome: audio IS arriving but sits below the
            # VAD threshold — report once per quiet stretch.
            if (self._below_vad_streak >= 30
                    and not self._below_vad_emitted):
                self._below_vad_emitted = True
                logger.info(
                    "Audio below VAD threshold: rms=%.4f threshold=%.4f "
                    "device=%r rate=%s",
                    rms, self._effective_threshold,
                    getattr(getattr(self._capture, "device", None),
                            "name", None),
                    getattr(self._capture, "sample_rate", None),
                )
                # V10.8 Task 4: VAD rejection is its own category —
                # distinct from starvation and from STT-empty.
                attempt = self._record_attempt(
                    recognition_taxonomy.VAD_REJECTED,
                    reason="audio_below_vad_threshold",
                    audio_seconds=0.0,
                    samples=0,
                    frame_rms=rms,
                    stt_invoked=False,
                )
                self._stt_log(
                    "STT_VAD_REJECTED",
                    attempt_id=attempt["attempt_id"],
                    rms=rms,
                    threshold=self._effective_threshold,
                    streak=self._below_vad_streak,
                )
                if self._emit_allowed(generation, stop_event):
                    self._emit_no_speech("audio_below_vad_threshold")
        now = time.monotonic()

        with self._lock:
            collecting = self._collecting
            self._adapt_threshold(rms, collecting)
            voiced = rms >= self._effective_threshold
            if not self._collecting:
                self._pre_roll.append(samples)
                if voiced:
                    self._collecting = True
                    self._utterance = list(self._pre_roll)
                    # Trim pre-roll to the configured window by sample
                    # count — frame sizes vary with capture rate.
                    keep = int(self._pre_roll_seconds * SAMPLE_RATE)
                    total = sum(len(f) for f in self._utterance)
                    while total > keep and len(self._utterance) > 1:
                        total -= len(self._utterance[0])
                        self._utterance.pop(0)
                    self._utterance_samples = sum(
                        len(f) for f in self._utterance
                    )
                    self._utt_rms = [rms]
                    # The onset frame is part of the utterance: count it
                    # as voiced when it crossed the effective threshold.
                    self._utterance_voiced = 1 if voiced else 0
                    self._pre_roll.clear()
                    self._speech_started = now
                    self._last_voiced = now
                    self._emit_activity(True)
                return

            self._utterance.append(samples)
            self._utterance_samples += samples.size
            self._utt_rms.append(rms)
            if voiced:
                self._last_voiced = now
                self._utterance_voiced += 1

            duration = self._utterance_samples / SAMPLE_RATE
            silence = now - self._last_voiced
            finalize = silence >= self._silence_hangover or (
                duration >= self._max_utterance
            )
            if finalize:
                utterance = self._utterance
                # V10.5: snapshot per-utterance stats BEFORE reset —
                # _reset_utterance() clears them, which previously made
                # every empty-transcript log show voiced=0/frames=1.
                self._pending_stats = {
                    "voiced": self._utterance_voiced,
                    "rms": list(self._utt_rms),
                    "started": self._speech_started,
                }
                self._reset_utterance()
                self._emit_activity(False)
            else:
                utterance = None

        if utterance is not None:
            reason = "max" if duration >= self._max_utterance else "silence"
            self._finalize(
                utterance, reason=reason,
                generation=generation, stop_event=stop_event,
            )

    def _finalize(
        self,
        frames: list[np.ndarray],
        reason: str,
        generation: Optional[int] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> None:
        if not frames:
            return
        audio = np.concatenate(frames).astype(np.float32)
        duration = len(audio) / SAMPLE_RATE
        if audio.size:
            self.last_utterance_rms = float(
                np.sqrt(np.mean(audio * audio))
            )
            self.last_utterance_peak = float(np.max(np.abs(audio)))
        self.utterances_completed += 1
        # V10.8 Task 4: per-attempt metadata snapshot. ``rms_list`` holds
        # one RMS per collected frame (valid VAD input only — rejected
        # frames never enter an utterance).
        pending = getattr(self, "_pending_stats", None) or {}
        rms_list = pending.get("rms") or []
        if not rms_list:
            rms_list = [self.last_utterance_rms or 0.0]
        voiced_count = pending.get("voiced", self._utterance_voiced)
        attempt_base = {
            "finalize_reason": reason,
            "audio_seconds": round(duration, 3),
            "samples": int(audio.size),
            "frames": len(frames),
            "rms_frames": len(rms_list),
            "voiced_frames_utterance": voiced_count,
            "rejected_frames_utterance": max(
                0, len(rms_list) - voiced_count),
            "speech_started_monotonic": pending.get("started") or None,
            "utterance_rms": self.last_utterance_rms,
            "utterance_peak": self.last_utterance_peak,
            **self._rms_snapshot(rms_list),
        }
        if duration < self._min_utterance:
            self.utterances_dropped += 1
            attempt = self._record_attempt(
                "utterance_too_short", stt_invoked=False, **attempt_base)
            self._stt_log(
                "STT_UTTERANCE_DROPPED",
                attempt_id=attempt["attempt_id"],
                reason="too_short",
                audio_seconds=round(duration, 3),
            )
            logger.debug(
                "Utterance too short (%.2fs); ignored", duration
            )
            return
        self._stt_log(
            "STT_DECODE_START",
            attempt_seq=self._attempt_seq + 1,
            audio_seconds=round(duration, 3),
            samples=int(audio.size),
        )
        t0 = time.monotonic()
        try:
            text = self._recognizer.transcribe(audio)
        except Exception as exc:  # noqa: BLE001
            stt_seconds = time.monotonic() - t0
            self.inference_errors += 1
            if not self._emit_allowed(generation, stop_event):
                return
            attempt = self._record_attempt(
                recognition_taxonomy.STT_EXCEPTION,
                stt_invoked=True,
                stt_duration_seconds=round(stt_seconds, 4),
                error_type=type(exc).__name__,
                error=str(exc)[:200],
                **attempt_base,
            )
            self._stt_log(
                "STT_EXCEPTION",
                attempt_id=attempt["attempt_id"],
                error_type=type(exc).__name__,
                stt_duration_seconds=round(stt_seconds, 4),
                audio_seconds=round(duration, 3),
            )
            logger.exception("STT inference error")
            self._emit_error(f"Speech recognition failed: {exc}")
            return
        stt_seconds = time.monotonic() - t0
        self.last_inference_monotonic = time.monotonic()
        # A decode that outlived its worker run (stop() join timeout)
        # must never emit into a session that started afterwards.
        if not self._emit_allowed(generation, stop_event):
            return
        text = (text or "").strip()
        self.last_transcript = text
        if text:
            attempt = self._record_attempt(
                recognition_taxonomy.STT_TRANSCRIPT_PRODUCED,
                stt_invoked=True,
                stt_duration_seconds=round(stt_seconds, 4),
                transcript_chars=len(text),
                **attempt_base,
            )
            self._stt_log(
                "STT_TRANSCRIPT",
                attempt_id=attempt["attempt_id"],
                transcript_chars=len(text),
                stt_duration_seconds=round(stt_seconds, 4),
                audio_seconds=round(duration, 3),
            )
            logger.info("Recognized speech: %r", text)
            if self._on_transcript is not None:
                try:
                    self._on_transcript(text)
                except Exception:  # noqa: BLE001
                    logger.exception("Transcript callback failed")
        else:
            attempt = self._record_attempt(
                recognition_taxonomy.STT_EMPTY_TRANSCRIPT,
                stt_invoked=True,
                stt_duration_seconds=round(stt_seconds, 4),
                **attempt_base,
            )
            self._stt_log(
                "STT_EMPTY_TRANSCRIPT",
                attempt_id=attempt["attempt_id"],
                stt_duration_seconds=round(stt_seconds, 4),
                audio_seconds=round(duration, 3),
                voiced_frames=voiced_count,
            )
            try:
                cap_stats = (
                    self._capture.stats()
                    if hasattr(self._capture, "stats")
                    else {}
                )
            except Exception:  # noqa: BLE001
                cap_stats = {}
            voiced_count = pending.get(
                "voiced", self._utterance_voiced
            )
            # V10.8: one sorted snapshot drives min/max/mean/median —
            # the previous code reported chronological endpoints as
            # min/max, which could invert them.
            srt = sorted(float(v) for v in rms_list)
            rms_min = srt[0]
            rms_max = srt[-1]
            rms_mean = sum(srt) / len(srt)
            med = srt[len(srt) // 2]
            logger.info(
                "No speech recognized in utterance (%.1fs, rms=%.4f, "
                "peak=%.3f, voiced=%d, rejected=%d, frames=%d, "
                "rms_min=%.4f rms_max=%.4f rms_mean=%.4f rms_med=%.4f, "
                "vad_threshold=%.4f, noise_floor=%s, device=%r, "
                "vad_result=accepted stt_called=True "
                "rate=%s, stream=%s, transcript=empty)",
                duration,
                self.last_utterance_rms,
                self.last_utterance_peak,
                voiced_count,
                max(0, len(rms_list) - voiced_count),
                len(rms_list),
                rms_min,
                rms_max,
                rms_mean,
                med,
                self._effective_threshold,
                self.stats().get("noise_floor"),
                cap_stats.get("device"),
                cap_stats.get("sample_rate"),
                cap_stats.get("running"),
            )
            self._emit_no_speech("vad_accepted_stt_empty")

    # ------------------------------------------------------------------
    def _emit_activity(self, active: bool) -> None:
        if self._on_speech_activity is not None:
            try:
                self._on_speech_activity(active)
            except Exception:  # noqa: BLE001
                pass

    def _emit_no_speech(self, reason: str) -> None:
        if self._on_no_speech is not None:
            try:
                self._on_no_speech(reason)
            except Exception:  # noqa: BLE001
                logger.exception("No-speech callback failed")

    def _emit_error(self, message: str) -> None:
        if self._on_error is not None:
            try:
                self._on_error(message)
            except Exception:  # noqa: BLE001
                pass
