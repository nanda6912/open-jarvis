"""Wake-word detection worker: consumes audio frames, applies threshold
and cooldown, and emits a single activation event per utterance."""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Callable, Optional

import numpy as np

from src.voice import kws_parity_diagnostic, kws_rescore
from src.voice.audio_capture_service import AudioCaptureService
from src.voice.wake_word_engine import WakeWordEngine

logger = logging.getLogger(__name__)


class WakeWordDetectionService:
    """Background thread running wake-word inference on captured frames.

    Callbacks are invoked from the worker thread — receivers must marshal
    to the GUI thread themselves (e.g. via Qt signals).
    """

    def __init__(
        self,
        capture: AudioCaptureService,
        engine: WakeWordEngine,
        threshold: float = 0.25,
        cooldown_seconds: float = 3.0,
        on_detected: Optional[Callable[[str], None]] = None,
        on_confidence: Optional[Callable[[float], None]] = None,
        on_stats: Optional[Callable[[dict], None]] = None,
        on_first_frame: Optional[Callable[[], None]] = None,
        stats_interval: float = 0.5,
        stats_log_interval: float = 30.0,
        generation: int = 0,
        rescore_policy: Optional[kws_rescore.RescorePolicy] = None,
    ) -> None:
        self._capture = capture
        self._generation = int(generation)
        self._engine = engine
        self._threshold = threshold
        self._cooldown = cooldown_seconds
        self._on_detected = on_detected
        self._on_confidence = on_confidence
        self._on_stats = on_stats
        self._on_first_frame = on_first_frame
        self._stats_interval = stats_interval
        self._stats_log_interval = stats_log_interval
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._last_detection = 0.0
        self._detection_count = 0
        # Diagnostic counters.
        self.frames_processed = 0
        self.inference_errors = 0
        self.last_inference_monotonic = 0.0
        self.last_keyword: Optional[str] = None
        self._last_stats_emit = 0.0
        self._last_stats_log = 0.0
        # Inference timing (EMA) and dequeue timestamp for
        # capture->inference latency diagnostics.
        self._inference_seconds_ema = 0.0
        self.last_frame_dequeued_monotonic = 0.0
        # KWS input quality: stats of the exact samples handed to the
        # engine, so "no detection" can be classified as no-signal vs
        # signal-present-but-not-detected vs bad formatting.
        self.last_kws_rms = 0.0
        self.last_kws_peak = 0.0
        self.last_kws_samples = 0
        self.max_kws_rms = 0.0
        # V10.4 rolling-window diagnostics: short windows (default 1 s)
        # of KWS input level + inference cost, so a missed detection
        # can be classified from recent evidence instead of the 30 s
        # aggregate log line alone.
        self._window_seconds = 1.0
        self._win_start = 0.0
        self._win_frames = 0
        self._win_inferences = 0
        self._win_inf_ms = 0.0
        self._win_kws_rms_max = 0.0
        self._win_kws_peak_max = 0.0
        self._win_kws_rms_sum = 0.0
        self._win_selected_channel: Optional[int] = None
        # Speech-end tracker: last time the KWS input carried speech-level
        # energy. Used to estimate end-of-phrase -> detection latency.
        self._speech_rms_threshold = 0.008
        self._last_voiced_monotonic = 0.0
        # Record of the most recent detection for latency reporting.
        self.last_detection: Optional[dict] = None
        self.detection_latencies: list[float] = []
        # V10.3.3 lifecycle diagnostics: per-service (per standby
        # generation) timestamps proving fresh audio reaches the KWS.
        self.started_monotonic = 0.0
        self.first_frame_monotonic = 0.0
        self.first_frame_rms = 0.0
        self._first_frame_logged = False
        # V10.6 diagnostics -------------------------------------------------
        # Bounded in-memory ring of the exact mono float32 frames handed
        # to the KWS engine (~5 s). Diagnostic analysis only: never
        # written to disk, never logged as PCM, overwritten by age and
        # cleared on stop/session transitions.
        self._diag_buffer_seconds = 5.0
        self._diag_buffer: deque = deque()
        self._diag_buffer_samples = 0
        # V10.7.1: per-activity-window tracker. An "activity window" is a
        # stretch of KWS input whose RMS exceeds the activity gate,
        # bounded by silence hangover. It measures AUDIO ENERGY only —
        # not confirmed speech and not a wake-word attempt. Each closed
        # window produces one summary record, including windows that
        # produced NO keyword result.
        self._activity_threshold = 0.008
        self._activity_hangover = 1.5
        self._activity_open = False
        self._activity_last_active = 0.0
        self._activity_frames = 0
        self._activity_active_frames = 0
        self._activity_active_samples = 0
        self._activity_rms_max = 0.0
        self._activity_rms_sum = 0.0
        self._activity_peak_max = 0.0
        self._activity_inf_ms = 0.0
        self._activity_queue_max = 0
        self._activity_dropped_start = 0
        self._activity_channel = None
        self._activity_channel_rms = None
        self._activity_channel_peaks = None
        self._activity_detected = False
        self._activity_keyword = None
        self._activity_seq = 0
        # KWS input is 16 kHz mono float32 (capture TARGET_SAMPLE_RATE);
        # used to convert active samples into real seconds.
        self._kws_sample_rate = 16000
        self.activity_windows: deque = deque(maxlen=100)
        self.last_activity_window: Optional[dict] = None
        # V10.7.5: bounded in-memory copy of the KWS input belonging to
        # the OPEN activity window. When the window closes without a
        # keyword, this audio is replayed through fresh KWS stream
        # anchors (engine.rescore) to recover detections lost to
        # stream-position sensitivity (V10.7.3). Never persisted.
        self._window_audio: deque = deque()
        self._window_audio_samples = 0
        self._window_audio_max_samples = 12 * self._kws_sample_rate
        # Rescore policy: fresh anchors spread across the chaotic
        # position grid measured in V10.7.3/V10.7.5 sweeps — a single
        # anchor can land in a dead zone, so ~10 anchors give the
        # borderline pronunciation several independent chances.
        # Trailing pad mirrors the validated replay path (0.5 s zeros).
        self._rescore_lead_seconds = (
            0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 1.8,
        )
        self._rescore_trail_seconds = 0.5
        self._rescore_min_seconds = 0.3
        self._rescore_max_seconds = 8.0
        # V10.7.5.1 rescore candidates: besides the full window, try
        # (a) the window audio trimmed to its first clear-speech frame
        # and (b) the speech-trimmed tail of the 5 s diagnostic ring,
        # which retains pre-window context the activity gate clipped.
        # A wake phrase whose onset sits below the activity gate is
        # otherwise unrecoverable (verified: onset-cut fixture misses
        # on every anchor).
        self._rescore_speech_gate = 0.02
        self._rescore_trim_margin_seconds = 0.4
        self._rescore_trim_min_drop_seconds = 0.3
        self._window_open_monotonic = 0.0
        # V10.8: aggregate report for the most recent rescore attempt.
        # Kept separate from the activity-window record so a miss still
        # exposes candidates/anchors actually attempted.
        self._last_rescore_report: Optional[dict] = None
        # V10.8 Task 3: bounded asynchronous rescore scheduling. Rescore
        # runs on a dedicated worker so multi-second anchor sweeps never
        # delay live KWS frames on the detection worker; the engine lock
        # is released between 100 ms decode chunks so live process()
        # interleaves instead of starving.
        self._rescore_policy = rescore_policy or kws_rescore.RescorePolicy(
            lead_seconds=self._rescore_lead_seconds,
            trail_seconds=self._rescore_trail_seconds,
            min_seconds=self._rescore_min_seconds,
            max_seconds=self._rescore_max_seconds,
            speech_gate=self._rescore_speech_gate,
            trim_margin_seconds=self._rescore_trim_margin_seconds,
            trim_min_drop_seconds=self._rescore_trim_min_drop_seconds,
        )
        self._rescore_queue: "queue.Queue" = queue.Queue(
            maxsize=self._rescore_policy.queue_maxsize
        )
        self._rescore_thread: Optional[threading.Thread] = None
        self._rescore_stop = threading.Event()
        # Completed rescore hits awaiting emission on the DETECTION
        # worker — keeps the on_detected callback on the same thread as
        # the live path.
        self._rescore_results: "queue.Queue" = queue.Queue()
        # window_id -> pending job bookkeeping for duplicate/coalescing
        # checks (one job per activity window).
        self._rescore_pending_lock = threading.Lock()
        self._rescore_pending_ids: set = set()
        # Scheduler accounting (bounded counters).
        self._rescore_submitted = 0
        self._rescore_completed = 0
        self._rescore_rejected = 0
        self._rescore_stale = 0
        self._rescore_aborted = 0
        self._rescore_hits = 0

    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def detection_count(self) -> int:
        return self._detection_count

    @property
    def threshold(self) -> float:
        return self._threshold

    def stats(self) -> dict:
        """Return pipeline diagnostics (capture + inference counters)."""
        now = time.monotonic()
        capture_stats = (
            self._capture.stats()
            if hasattr(self._capture, "stats")
            else {}
        )
        engine_inferences = getattr(self._engine, "inferences", None)
        engine_last = getattr(self._engine, "last_inference_monotonic", 0.0)
        return {
            **capture_stats,
            # V10.8: explicit counter scopes — capture stats are lifetime
            # for the shared stream; inference/detection counters are
            # per detector generation; activity windows are per window.
            "detector_generation": self._generation,
            "lifetime_frames_captured": capture_stats.get("frames_captured"),
            "lifetime_frames_dropped": capture_stats.get("frames_dropped"),
            "generation_frames_processed": self.frames_processed,
            "generation_inference_errors": self.inference_errors,
            "generation_detections": self._detection_count,
            "activity_window_seq": self._activity_seq,
            "frames_processed": self.frames_processed,
            "inference_errors": self.inference_errors,
            "engine_inferences": engine_inferences,
            "detections": self._detection_count,
            "threshold": self._threshold,
            "last_keyword": self.last_keyword,
            "last_inference_age_seconds": (
                now - self.last_inference_monotonic
                if self.last_inference_monotonic
                else None
            ),
            "engine_last_inference_age_seconds": (
                now - engine_last if engine_last else None
            ),
            "last_processed_age_seconds": (
                now - self.last_inference_monotonic
                if self.last_inference_monotonic
                else None
            ),
            "inference_seconds": (
                self._inference_seconds_ema or None
            ),
            # Exact audio handed to the KWS engine on the last frame.
            "kws_input_rms": self.last_kws_rms or None,
            "kws_input_peak": self.last_kws_peak or None,
            "kws_input_samples": self.last_kws_samples or None,
            "kws_max_rms": self.max_kws_rms or None,
            # Approximate capture->inference latency: queued frames
            # each represent one frame period, so queue depth x period
            # estimates how stale the audio being inferred is.
            "capture_to_inference_estimate_seconds": (
                round(
                    (capture_stats.get("queue_depth") or 0)
                    * self._frame_period_seconds(capture_stats),
                    3,
                )
            ),
            # V10.4 additions.
            "keywords_file": (
                str(getattr(self._engine, "keywords_file", "") or "")
                or None
            ),
            "keyword_count": getattr(self._engine, "keyword_count", None),
            "keywords_sha256": getattr(
                self._engine, "keywords_sha256", None
            ),
            "engine_inference_ms_ema": getattr(
                self._engine, "inference_ms_ema", None
            ),
            "window": self._window_stats(),
            "last_detection": self.last_detection,
            "last_speech_end_to_detection_seconds": (
                self.last_detection.get("speech_end_to_detection_seconds")
                if self.last_detection else None
            ),
            # V10.3.3: per-generation lifecycle timestamps (monotonic).
            "service_started_monotonic": (
                self.started_monotonic or None
            ),
            "first_frame_monotonic": (
                self.first_frame_monotonic or None
            ),
            "first_frame_rms": self.first_frame_rms or None,
            # V10.6/V10.7.1: activity-window diagnostics + buffer state.
            "diag_buffer_seconds": round(
                self._diag_buffer_samples / 16000.0, 2
            ),
            "activity_windows_recorded": len(self.activity_windows),
            "activity_windows_without_keyword": sum(
                1 for a in self.activity_windows
                if not a.get("keyword_detected")
            ),
            "last_activity_window": self.last_activity_window,
            # V10.8 async rescore scheduler state.
            "rescore_queue_depth": self._rescore_queue.qsize(),
            "rescore_pending": len(self._rescore_pending_ids),
            "rescore_submitted": self._rescore_submitted,
            "rescore_completed": self._rescore_completed,
            "rescore_rejected": self._rescore_rejected,
            "rescore_stale": self._rescore_stale,
            "rescore_aborted": self._rescore_aborted,
            "rescore_hits": self._rescore_hits,
            "rescore_worker_alive": bool(
                self._rescore_thread is not None
                and self._rescore_thread.is_alive()
            ),
            # Legacy aliases (V10.6 names) for compatibility.
            "attempts_recorded": len(self.activity_windows),
            "attempts_missed": sum(
                1 for a in self.activity_windows
                if not a.get("keyword_detected")
            ),
            "last_attempt": self.last_activity_window,
        }

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # V10.7.5.1 structured wake-trace events (no PCM, no speech content)
    # ------------------------------------------------------------------
    def _kws_log(self, event: str, **fields) -> None:
        """Emit one structured KWS trace event.

        Format: ``EVENT key=value key='str' ...`` — consistent event
        names make the production wake path greppable end to end.
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

    def _window_uid(self, window_id: Optional[int]) -> Optional[str]:
        """Generation-scoped window correlation id (window ids reset)."""
        if window_id is None:
            return None
        return f"gen{self._generation}-win{window_id}"

    def _capture_info(self) -> None:
        """Log the capture configuration seen by this generation."""
        try:
            stats = self._capture.stats()
        except Exception:  # noqa: BLE001
            stats = {}
        self._kws_log(
            "KWS_CAPTURE_INFO",
            generation=self._generation,
            device=stats.get("device"),
            sample_rate=stats.get("sample_rate"),
            input_channels=stats.get("input_channels"),
            selected_channel=stats.get("selected_channel"),
            dtype="float32",
            target_rate=self._kws_sample_rate,
        )

    # V10.6 diagnostic helpers (no audio leaves this object)
    # ------------------------------------------------------------------
    def diagnostic_audio(self) -> Optional[np.ndarray]:
        """Return a copy of the buffered KWS-input audio (<=5 s).

        For offline diagnostic analysis/tests only — the buffer is
        bounded, never persisted, and cleared on stop/session end.
        """
        if not self._diag_buffer:
            return None
        return np.concatenate(list(self._diag_buffer)).astype(np.float32)

    def clear_diagnostic_audio(self) -> None:
        """Drop all buffered diagnostic audio immediately."""
        self._diag_buffer.clear()
        self._diag_buffer_samples = 0

    def _diag_append(self, samples: np.ndarray) -> None:
        """Append one KWS-input frame; evict oldest past ~5 s."""
        if samples.size == 0:
            return
        self._diag_buffer.append(samples)
        self._diag_buffer_samples += int(samples.size)
        limit = int(self._diag_buffer_seconds * 16000)
        while self._diag_buffer_samples > limit and self._diag_buffer:
            old = self._diag_buffer.popleft()
            self._diag_buffer_samples -= int(old.size)

    # V10.6 compatibility aliases: the tracker measures audio activity
    # windows, not wake-word attempts. Old attribute names remain
    # readable so external diagnostics keep working.
    @property
    def attempts(self) -> deque:
        return self.activity_windows

    @property
    def last_attempt(self) -> Optional[dict]:
        return self.last_activity_window

    def _activity_frame(
        self, rms: float, peak: float, samples: int, now: float
    ) -> None:
        """Fold one KWS-input frame into the open activity window.

        A frame is "active" when its RMS crosses the activity gate —
        this is an energy measurement only, not a speech/VAD verdict.
        """
        active = rms >= self._activity_threshold
        if active:
            if not self._activity_open:
                self._activity_open = True
                self._activity_seq += 1
                self._window_open_monotonic = now
                self._activity_frames = 0
                self._activity_active_frames = 0
                self._activity_active_samples = 0
                self._activity_rms_max = 0.0
                self._activity_rms_sum = 0.0
                self._activity_peak_max = 0.0
                self._activity_inf_ms = 0.0
                self._activity_queue_max = 0
                try:
                    cap = self._capture.stats()
                except Exception:  # noqa: BLE001
                    cap = {}
                self._activity_dropped_start = int(
                    cap.get("frames_dropped") or 0
                )
                self._activity_channel = cap.get("selected_channel")
                self._activity_channel_rms = cap.get("channel_rms")
                self._activity_channel_peaks = cap.get("channel_peaks")
                self._activity_detected = False
                self._activity_keyword = None
                # V10.7.5: fresh rescore buffer for the new window.
                self._window_audio.clear()
                self._window_audio_samples = 0
                self._kws_log(
                    "KWS_ACTIVITY_OPEN",
                    window_id=self._activity_seq,
                    window_uid=self._window_uid(self._activity_seq),
                    generation=self._generation,
                    monotonic=round(now, 3),
                    rms=round(rms, 5),
                    peak=round(peak, 4),
                    samples=int(samples or 0),
                    selected_channel=self._activity_channel,
                    threshold=self._activity_threshold,
                )
            self._activity_last_active = now
            self._activity_active_frames += 1
            self._activity_active_samples += int(samples or 0)
            if rms > self._activity_rms_max:
                self._activity_rms_max = rms
            if peak > self._activity_peak_max:
                self._activity_peak_max = peak
        if self._activity_open:
            self._activity_frames += 1
            self._activity_rms_sum += rms
            try:
                qd = int(self._capture.stats().get("queue_depth") or 0)
            except Exception:  # noqa: BLE001
                qd = 0
            if qd > self._activity_queue_max:
                self._activity_queue_max = qd

    def _activity_close_check(self, now: float) -> None:
        """Close the window after the silence hangover elapses."""
        if not self._activity_open:
            return
        if now - self._activity_last_active < self._activity_hangover:
            return
        self._close_activity_window(now)

    def _close_activity_window(
        self, now: float, allow_rescore: bool = True
    ) -> None:
        """Finalize the current activity window into the record list."""
        if not self._activity_open:
            return
        self._activity_open = False
        window_id = self._activity_seq
        window_samples = self._window_audio_samples
        # V10.7.5/V10.8: the continuous KWS stream is position-sensitive —
        # a borderline utterance can miss at its live anchor yet detect
        # on a fresh one (V10.7.3). Before recording a miss, the window's
        # audio is replayed through fresh stream anchors on the DEDICATED
        # rescore worker — never on this detection thread, where a
        # multi-second anchor sweep starved live inference (observed
        # queue_depth ~= 37 and seconds-long processing stalls). Never
        # runs during shutdown/stop or when the window already detected.
        detected = bool(self._activity_detected)
        keyword = self._activity_keyword
        skip_reason = None
        if detected:
            skip_reason = "already_detected"
        elif not allow_rescore:
            skip_reason = "not_allowed"
        elif self._stop.is_set():
            skip_reason = "stopping"
        elif not self._window_audio:
            skip_reason = "empty_buffer"
        elif window_samples < int(
            self._rescore_min_seconds * self._kws_sample_rate
        ):
            skip_reason = "audio_too_short"
        elif getattr(self._engine, "rescore", None) is None:
            skip_reason = "engine_no_rescore"
        job = None
        if skip_reason is None:
            job = self._prepare_rescore_job(window_id, now)
            if job is None:
                skip_reason = getattr(
                    self, "_last_rescore_skip_reason", "submit_failed"
                )
        elif not detected:
            self._kws_log(
                "KWS_RESCORE_SKIPPED",
                window_id=window_id,
                window_uid=self._window_uid(window_id),
                generation=self._generation,
                reason=skip_reason,
                audio_seconds=round(
                    window_samples / float(self._kws_sample_rate), 3),
                samples=window_samples,
            )
        # Duration from the actual KWS-input sample count at 16 kHz —
        # not an assumed frame period. This measures how long the audio
        # stayed above the energy gate; it is NOT verified speech time.
        active_seconds = round(
            self._activity_active_samples / float(self._kws_sample_rate),
            2,
        )
        rec = {
            "window_id": self._activity_seq,
            "event": (
                "WAKE_MATCHED" if detected else "ACTIVITY_WITHOUT_KEYWORD"
            ),
            "end_monotonic": now,
            "frames": self._activity_frames,
            "active_frames": self._activity_active_frames,
            "active_audio_seconds": active_seconds,
            "rms_max": round(self._activity_rms_max, 5),
            "rms_mean": round(
                self._activity_rms_sum / max(1, self._activity_frames), 5
            ),
            "peak_max": round(self._activity_peak_max, 4),
            "inferences": self._activity_frames,
            "inference_ms_total": round(self._activity_inf_ms, 1),
            "queue_depth_max": self._activity_queue_max,
            "frames_dropped_during": None,
            "selected_channel": self._activity_channel,
            "channel_rms": self._activity_channel_rms,
            "channel_peaks": self._activity_channel_peaks,
            "keyword_detected": detected,
            "rescored": False,
            "rescore_lead_seconds": None,
            # V10.8 async rescore: fields below are populated when the
            # scheduled job completes on the rescore worker; until then
            # they describe the PENDING state, not a finished attempt.
            "rescore_candidates": (
                len(job["candidates"]) if job is not None else 0
            ),
            "rescore_scheduled": job is not None,
            "window_uid": self._window_uid(window_id),
            # V10.8 rescore lifecycle aggregation (diagnostics only).
            "rescore_attempted": job is not None,
            "rescore_succeeded": (
                None if job is not None else False
            ),
            "rescore_candidates_attempted": 0,
            "rescore_candidates_completed": 0,
            "rescore_anchors_requested": (
                len(job["candidates"])
                * len(self._rescore_policy.lead_seconds)
                if job is not None else 0
            ),
            "rescore_anchors_evaluated": None,
            "rescore_anchors_evaluated_complete": job is None,
            "rescore_elapsed_ms": None,
            "rescore_error": None,
            "rescore_hit_candidate": None,
            "rescore_aborted": None,
            "detection_method": (
                "live" if detected
                else "rescore_pending" if job is not None
                else "none"
            ),
            # V10.6-compatible aliases.
            "detected": detected,
            "keyword": keyword,
            "threshold": self._threshold,
        }
        try:
            dropped_now = int(self._capture.stats().get(
                "frames_dropped") or 0)
            rec["frames_dropped_during"] = (
                dropped_now - self._activity_dropped_start
            )
        except Exception:  # noqa: BLE001
            pass
        self.activity_windows.append(rec)
        self.last_activity_window = rec
        # sherpa-onnx 2.0.1 KeywordSpotter.get_result() returns a
        # keyword string or "" — detection is binary, there is no
        # confidence score. keyword_detected reflects an actual
        # nonempty keyword emission only.
        logger.info(
            "Audio activity window #%d closed: event=%s "
            "active_frames=%d active_audio_seconds=%.2f "
            "rms_max=%.4f peak_max=%.3f selected_channel=%s "
            "inferences=%d queue_depth_max=%d frames_dropped=%s "
            "keyword_detected=%s%s%s",
            rec["window_id"], rec["event"],
            rec["active_frames"], rec["active_audio_seconds"],
            rec["rms_max"], rec["peak_max"], rec["selected_channel"],
            rec["inferences"], rec["queue_depth_max"],
            rec["frames_dropped_during"], detected,
            f" keyword={keyword!r}" if detected else "",
            " rescore=scheduled" if job is not None else "",
        )
        # V10.7.5.1: one structured close event carrying everything
        # needed to classify the window without raw audio.
        window_audio_n = int(self._window_audio_samples)
        self._kws_log(
            "KWS_ACTIVITY_CLOSE",
            window_id=rec["window_id"],
            window_uid=rec["window_uid"],
            generation=self._generation,
            open_monotonic=round(self._window_open_monotonic, 3),
            close_monotonic=round(now, 3),
            duration_seconds=round(
                now - self._window_open_monotonic, 3),
            frames=rec["frames"],
            active_frames=rec["active_frames"],
            samples=window_audio_n,
            sample_rate=self._kws_sample_rate,
            dtype="float32",
            audio_seconds=round(
                window_audio_n / float(self._kws_sample_rate), 3),
            rms_max=rec["rms_max"],
            rms_mean=rec["rms_mean"],
            peak_max=rec["peak_max"],
            selected_channel=rec["selected_channel"],
            channel_rms=rec["channel_rms"],
            channel_peaks=rec["channel_peaks"],
            enough_audio=window_audio_n >= int(
                self._rescore_min_seconds * self._kws_sample_rate),
            keyword_detected=detected,
            keyword=keyword,
            rescored=False,
            rescore_scheduled=job is not None,
            rescore_lead_seconds=None,
            rescore_candidates=(
                len(job["candidates"]) if job is not None else 0
            ),
            rescore_attempted=rec["rescore_attempted"],
            rescore_succeeded=rec["rescore_succeeded"],
            rescore_candidates_attempted=rec[
                "rescore_candidates_attempted"
            ],
            rescore_candidates_completed=rec[
                "rescore_candidates_completed"
            ],
            rescore_anchors_requested=rec["rescore_anchors_requested"],
            rescore_anchors_evaluated=rec["rescore_anchors_evaluated"],
            rescore_anchors_evaluated_complete=rec[
                "rescore_anchors_evaluated_complete"
            ],
            rescore_elapsed_ms=rec["rescore_elapsed_ms"],
            rescore_error=rec["rescore_error"],
            rescore_hit_candidate=rec["rescore_hit_candidate"],
            detection_method=rec["detection_method"],
        )
        self._activity_detected = False
        self._activity_keyword = None
        if job is not None:
            # Attach the window record and enqueue; the rescore worker
            # performs the anchor sweep off the detection thread and the
            # detection worker emits any hit via _drain_rescore_results.
            job["record"] = rec
            self._enqueue_rescore_job(job)

    # ------------------------------------------------------------------
    # V10.8 asynchronous rescore scheduling
    # ------------------------------------------------------------------
    def _prepare_rescore_job(
        self, window_id: int, now: float
    ) -> Optional[dict]:
        """Snapshot window audio into rescore candidates (no inference).

        Returns a job dict (not yet queued) or None; on failure
        ``self._last_rescore_skip_reason`` carries the reason and a
        ``KWS_RESCORE_SKIPPED`` event is logged.
        """
        self._last_rescore_skip_reason = "submit_failed"
        uid = self._window_uid(window_id)
        try:
            window_audio = kws_rescore.concat_float32(self._window_audio)
        except Exception as exc:  # noqa: BLE001
            self._last_rescore_skip_reason = "concat_error"
            self._kws_log(
                "KWS_RESCORE_SKIPPED", window_id=window_id, window_uid=uid,
                generation=self._generation, reason="concat_error",
                stage="concat", error=repr(exc))
            logger.debug("KWS rescore buffer concat failed",
                         exc_info=True)
            return None
        diag_audio = None
        if self._diag_buffer:
            try:
                diag_audio = kws_rescore.concat_float32(self._diag_buffer)
            except Exception:  # noqa: BLE001
                diag_audio = None
        candidates, plan_error = kws_rescore.plan_candidates(
            window_audio,
            diag_audio,
            self._rescore_policy,
            self._kws_sample_rate,
            window_reference_samples=self._window_audio_samples,
        )
        if not candidates:
            reason = plan_error or "empty_buffer"
            self._last_rescore_skip_reason = reason
            self._kws_log(
                "KWS_RESCORE_SKIPPED", window_id=window_id, window_uid=uid,
                generation=self._generation, reason=reason,
                stage="plan")
            return None
        return {
            "window_id": window_id,
            "window_uid": uid,
            "generation": self._generation,
            "submitted_monotonic": now,
            "deadline_monotonic": (
                now + self._rescore_policy.job_ttl_seconds
            ),
            "budget_seconds": self._rescore_policy.job_budget_seconds,
            "started_monotonic": None,
            "candidates": candidates,
            "record": None,
        }

    def _enqueue_rescore_job(self, job: dict) -> bool:
        """Queue a prepared rescore job; bounded, never blocks."""
        window_id = job["window_id"]
        try:
            self._rescore_queue.put_nowait(job)
        except queue.Full:
            self._rescore_rejected += 1
            rec = job.get("record")
            if isinstance(rec, dict):
                rec["rescore_attempted"] = False
                rec["rescore_scheduled"] = False
                rec["rescore_succeeded"] = False
                rec["rescore_candidates"] = 0
                rec["rescore_anchors_requested"] = 0
                rec["rescore_anchors_evaluated_complete"] = True
                rec["detection_method"] = (
                    "live" if rec.get("keyword_detected") else "none"
                )
            self._kws_log(
                "KWS_RESCORE_SKIPPED",
                window_id=window_id,
                window_uid=job.get("window_uid"),
                generation=self._generation,
                reason="queue_full",
                stage="enqueue",
                queue_depth=self._rescore_queue.qsize(),
            )
            return False
        with self._rescore_pending_lock:
            self._rescore_pending_ids.add(window_id)
        self._rescore_submitted += 1
        self._kws_log(
            "KWS_RESCORE_QUEUED",
            window_id=window_id,
            window_uid=job.get("window_uid"),
            generation=self._generation,
            candidates=len(job["candidates"]),
            anchors_requested=len(job["candidates"])
            * len(self._rescore_policy.lead_seconds),
            queue_depth=self._rescore_queue.qsize(),
        )
        return True

    def _rescore_worker(self) -> None:
        """Dedicated rescore thread: runs anchor sweeps off the
        detection worker so live KWS inference is never starved."""
        while not self._rescore_stop.is_set():
            try:
                job = self._rescore_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            try:
                self._execute_rescore_job(job)
            except Exception:  # noqa: BLE001
                logger.exception("KWS rescore job failed")
                self._rescore_aborted += 1
                self._rescore_results.put((job, self._rescore_error_report(job)))
            finally:
                try:
                    self._rescore_queue.task_done()
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _rescore_error_report(job: dict) -> dict:
        return {
            "attempted": True,
            "succeeded": False,
            "result": None,
            "candidates_attempted": len(job.get("candidates") or []),
            "candidates_completed": 0,
            "anchors_requested": 0,
            "anchors_evaluated": 0,
            "anchors_evaluated_complete": False,
            "elapsed_ms": None,
            "error": "worker_exception",
            "hit_candidate": None,
            "aborted": True,
            "budget_exceeded": False,
        }

    def _rescore_should_abort(self, job: dict) -> bool:
        """Abort predicate checked between anchors/chunks by the engine."""
        if self._rescore_stop.is_set() or self._stop.is_set():
            return True
        now = time.monotonic()
        if now > job["deadline_monotonic"]:
            return True
        started = job.get("started_monotonic")
        if (
            started is not None
            and now - started > job["budget_seconds"]
        ):
            return True
        return False

    def _execute_rescore_job(self, job: dict) -> None:
        """Run one rescore job on the rescore worker thread."""
        job["started_monotonic"] = time.monotonic()
        window_id = job["window_id"]
        uid = job.get("window_uid")

        def _event(kind: str, fields: dict) -> None:
            # Map shared-orchestration events onto the production
            # KWS_RESCORE_* trace names (identical fields).
            if kind == "candidate_start":
                self._kws_log(
                    "KWS_RESCORE_START",
                    window_id=window_id, window_uid=uid,
                    generation=self._generation,
                    candidate=fields.get("candidate"),
                    audio_seconds=fields.get("audio_seconds"),
                    samples=fields.get("samples"),
                    sample_rate=fields.get("sample_rate"),
                    dtype="float32",
                    anchors_requested=fields.get("anchors_requested"),
                    trail_seconds=fields.get("trail_seconds"),
                )
            elif kind == "anchor":
                self._kws_log(
                    "KWS_RESCORE_ANCHOR",
                    window_id=window_id, window_uid=uid,
                    generation=self._generation,
                    candidate=fields.get("candidate"),
                    lead_seconds=fields.get("lead_seconds"),
                    padded_seconds=fields.get("padded_seconds"),
                    audio_seconds=fields.get("audio_seconds"),
                    hit=fields.get("hit"),
                    keyword=fields.get("keyword"),
                    elapsed_ms=fields.get("elapsed_ms"),
                    error=fields.get("error"),
                )
            elif kind == "skipped":
                self._kws_log(
                    "KWS_RESCORE_SKIPPED",
                    window_id=window_id, window_uid=uid,
                    generation=self._generation,
                    candidate=fields.get("candidate"),
                    reason=fields.get("reason"),
                    stage=fields.get("stage"),
                )
            elif kind == "candidate_complete":
                self._kws_log(
                    "KWS_RESCORE_COMPLETE",
                    window_id=window_id, window_uid=uid,
                    generation=self._generation,
                    candidate=fields.get("candidate"),
                    anchors_requested=fields.get("anchors_requested"),
                    anchors_evaluated=fields.get("anchors_evaluated"),
                    hit=fields.get("hit"),
                    keyword=fields.get("keyword"),
                    lead_seconds=fields.get("lead_seconds"),
                    elapsed_ms=fields.get("elapsed_ms"),
                )
            elif kind == "candidate_failed":
                self._kws_log(
                    "KWS_RESCORE_FAILED",
                    window_id=window_id, window_uid=uid,
                    generation=self._generation,
                    candidate=fields.get("candidate"),
                    error=fields.get("error"),
                    elapsed_ms=fields.get("elapsed_ms"),
                )

        report = kws_rescore.run_candidates(
            self._engine,
            job["candidates"],
            self._rescore_policy,
            event=_event,
            should_abort=lambda: self._rescore_should_abort(job),
            sample_rate=self._kws_sample_rate,
        )
        # Normalize to the _last_rescore_report shape used by the
        # synchronous path and window-record diagnostics.
        self._last_rescore_report = {
            "attempted": True,
            "succeeded": bool(report.get("succeeded")),
            "result": report.get("result"),
            "candidates_attempted": report.get("candidates_attempted", 0),
            "candidates_completed": report.get("candidates_completed", 0),
            "anchors_requested": report.get("anchors_requested", 0),
            "anchors_evaluated": report.get("anchors_evaluated"),
            "anchors_evaluated_complete": report.get(
                "anchors_evaluated_complete", True
            ),
            "elapsed_ms": report.get("elapsed_ms"),
            "error": report.get("error"),
            "hit_candidate": report.get("hit_candidate"),
            "aborted": report.get("aborted", False),
            "budget_exceeded": report.get("budget_exceeded", False),
        }
        self._rescore_completed += 1
        if report.get("result"):
            self._rescore_hits += 1
        self._rescore_results.put((job, self._last_rescore_report))

    def _apply_rescore_report(self, rec: dict, report: dict) -> None:
        """Fold a completed rescore report into its window record."""
        rec["rescore_attempted"] = True
        rec["rescore_succeeded"] = bool(report.get("succeeded"))
        rec["rescore_candidates_attempted"] = int(
            report.get("candidates_attempted") or 0
        )
        rec["rescore_candidates_completed"] = int(
            report.get("candidates_completed") or 0
        )
        rec["rescore_candidates"] = rec["rescore_candidates_completed"]
        rec["rescore_anchors_requested"] = int(
            report.get("anchors_requested") or 0
        )
        ev = report.get("anchors_evaluated")
        rec["rescore_anchors_evaluated"] = (
            int(ev) if ev is not None else None
        )
        rec["rescore_anchors_evaluated_complete"] = bool(
            report.get("anchors_evaluated_complete", True)
        )
        rec["rescore_elapsed_ms"] = report.get("elapsed_ms")
        rec["rescore_error"] = report.get("error")
        rec["rescore_hit_candidate"] = report.get("hit_candidate")
        rec["rescore_aborted"] = bool(report.get("aborted", False))
        result = report.get("result")
        if result:
            keyword = result[0]
            lead = result[1] if len(result) > 1 else None
            rec["rescored"] = True
            rec["rescore_lead_seconds"] = lead
            rec["keyword"] = keyword
            rec["detected"] = True
            rec["keyword_detected"] = True
            rec["event"] = "WAKE_MATCHED"
            rec["detection_method"] = "rescore"
        else:
            rec["detection_method"] = (
                "live" if rec.get("keyword_detected") else "none"
            )

    def _drain_rescore_results(self) -> None:
        """Emit completed rescore hits on the DETECTION worker thread.

        Keeps ``on_detected`` on the same thread as the live path and
        preserves chronological ordering relative to live frames.
        """
        while True:
            try:
                job, report = self._rescore_results.get_nowait()
            except queue.Empty:
                return
            window_id = job.get("window_id")
            with self._rescore_pending_lock:
                self._rescore_pending_ids.discard(window_id)
            rec = job.get("record")
            if isinstance(rec, dict):
                try:
                    self._apply_rescore_report(rec, report)
                except Exception:  # noqa: BLE001
                    logger.debug("rescore record update failed",
                                 exc_info=True)
                # The window record was closed (and KWS_ACTIVITY_CLOSE
                # logged) while the rescore was still pending; emit the
                # final aggregated result as its own event so the trace
                # carries truthful post-rescore state.
                self._kws_log(
                    "KWS_RESCORE_RESULT",
                    window_id=window_id,
                    window_uid=job.get("window_uid"),
                    generation=self._generation,
                    succeeded=rec.get("rescore_succeeded"),
                    candidates_attempted=rec.get(
                        "rescore_candidates_attempted"),
                    candidates_completed=rec.get(
                        "rescore_candidates_completed"),
                    anchors_requested=rec.get(
                        "rescore_anchors_requested"),
                    anchors_evaluated=rec.get(
                        "rescore_anchors_evaluated"),
                    anchors_evaluated_complete=rec.get(
                        "rescore_anchors_evaluated_complete"),
                    elapsed_ms=rec.get("rescore_elapsed_ms"),
                    error=rec.get("rescore_error"),
                    hit_candidate=rec.get("rescore_hit_candidate"),
                    keyword=rec.get("keyword"),
                    detection_method=rec.get("detection_method"),
                    aborted=rec.get("rescore_aborted"),
                )
            if self._stop.is_set():
                self._rescore_stale += 1
                continue
            result = report.get("result")
            if not result:
                continue
            keyword = result[0]
            # A rescored hit is a real detection: same cooldown,
            # bookkeeping and callback path as a live hit.
            self._emit_detection(
                keyword, time.monotonic(), rescored=True,
                diag_frame=None, raw_keyword=keyword,
                window_id=window_id,
            )

    def _rescore_activity_audio(
        self, window_id: int
    ) -> Optional[tuple]:
        """Replay captured KWS input on fresh stream anchors.

        V10.7.5.1: up to three candidates are tried in order —

        - ``full``: the activity-window audio (tail-capped).
        - ``speech_trimmed``: the same audio starting at its first
          clear-speech frame, so leading non-speech cannot push the
          phrase outside anchor coverage.
        - ``recent_context``: the speech-trimmed tail of the ~5 s
          diagnostic ring, which retains PRE-WINDOW audio the
          activity gate clipped. A wake phrase whose onset sits below
          the gate is unrecoverable from the window buffer alone
          (verified on the real-voice fixture: onset-cut audio misses
          on every anchor).

        Returns ``((keyword, lead_seconds), candidates_completed)``
        or ``None``. Synchronous path retained for diagnostics/tests;
        production window-close rescoring runs asynchronously via
        ``_prepare_rescore_job``/``_rescore_worker``. Audio stays in
        memory only. V10.8: every attempt also populates
        ``self._last_rescore_report`` so close-out diagnostics remain
        accurate on misses, early exits and failures.
        """
        report = {
            "attempted": False,
            "succeeded": False,
            "candidates_attempted": 0,
            "candidates_completed": 0,
            "anchors_requested": 0,
            "anchors_evaluated": 0,
            "anchors_evaluated_complete": True,
            "elapsed_ms": None,
            "error": None,
            "hit_candidate": None,
        }
        self._last_rescore_report = report
        rescore = getattr(self._engine, "rescore", None)
        if rescore is None:
            return None
        report["attempted"] = True
        t0 = time.monotonic()

        def _finish() -> None:
            report["elapsed_ms"] = round(
                (time.monotonic() - t0) * 1000.0, 2
            )

        try:
            window_audio = kws_rescore.concat_float32(self._window_audio)
        except Exception as exc:  # noqa: BLE001
            report["error"] = repr(exc)
            _finish()
            self._kws_log(
                "KWS_RESCORE_FAILED", window_id=window_id,
                window_uid=self._window_uid(window_id),
                generation=self._generation, stage="concat",
                error=repr(exc), elapsed_ms=report["elapsed_ms"])
            logger.debug("KWS rescore buffer concat failed",
                         exc_info=True)
            return None
        # V10.8: candidate preparation is shared with the async
        # scheduler (kws_rescore.plan_candidates) — identical order:
        # full -> speech_trimmed -> recent_context.
        diag_audio = None
        if self._diag_buffer:
            try:
                diag_audio = kws_rescore.concat_float32(
                    self._diag_buffer
                )
            except Exception:  # noqa: BLE001
                diag_audio = None
        candidates, plan_error = kws_rescore.plan_candidates(
            window_audio,
            diag_audio,
            self._rescore_policy,
            self._kws_sample_rate,
            window_reference_samples=self._window_audio_samples,
        )
        if not candidates:
            report["error"] = plan_error or "empty_buffer"
            _finish()
            return None
        for cand_name, cand_audio in candidates:
            report["candidates_attempted"] += 1
            result, cand_stats = self._rescore_candidate(
                rescore, cand_name, cand_audio, window_id)
            report["candidates_completed"] += 1
            req = cand_stats.get("anchors_requested")
            if req is not None:
                report["anchors_requested"] += int(req)
            ev = cand_stats.get("anchors_evaluated")
            if ev is None:
                report["anchors_evaluated_complete"] = False
            else:
                report["anchors_evaluated"] += int(ev)
            if cand_stats.get("error") and report["error"] is None:
                report["error"] = cand_stats["error"]
            if result is not None:
                report["succeeded"] = True
                report["hit_candidate"] = cand_name
                _finish()
                return result, report["candidates_completed"]
        _finish()
        return None

    def _speech_trimmed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Return window audio starting near the first clear-speech
        frame, or None when no meaningful trim exists.

        Only 100 ms hops at/above ``_rescore_speech_gate`` count as
        speech onset; a margin of pre-speech audio is kept so a quiet
        phrase onset is never cut. Returns None when the trim would
        drop less than ``_rescore_trim_min_drop_seconds`` — i.e. the
        candidate would be effectively identical to the full window.
        """
        hop = int(self._kws_sample_rate * 0.1)
        first = None
        for off in range(0, audio.size - hop + 1, hop):
            seg = audio[off:off + hop]
            rms = float(np.sqrt(np.mean(seg * seg)))
            if rms >= self._rescore_speech_gate:
                first = off
                break
        if first is None:
            return None
        margin = int(self._rescore_trim_margin_seconds
                     * self._kws_sample_rate)
        start = max(0, first - margin)
        min_drop = int(self._rescore_trim_min_drop_seconds
                       * self._kws_sample_rate)
        if start < min_drop:
            return None
        return audio[start:]

    def _recent_context_trimmed(self) -> Optional[np.ndarray]:
        """Speech-trimmed tail of the diagnostic ring (~5 s KWS input).

        Returns None unless the ring holds meaningfully MORE audio
        than the window buffer — i.e. it contains pre-window context
        the window clip lost — and a clear-speech onset exists to trim
        to.
        """
        if not self._diag_buffer:
            return None
        try:
            recent = np.concatenate(
                [np.asarray(f, dtype=np.float32)
                 for f in self._diag_buffer]
            )
        except Exception:  # noqa: BLE001
            return None
        min_n = int(self._rescore_min_seconds * self._kws_sample_rate)
        if recent.size <= (
            self._window_audio_samples
            + int(0.3 * self._kws_sample_rate)
        ):
            return None
        trimmed = self._speech_trimmed(recent)
        if trimmed is None or trimmed.size < min_n:
            return None
        return trimmed

    def _rescore_candidate(
        self, rescore, cand_name: str, audio: np.ndarray,
        window_id: int,
    ) -> tuple:
        """Run one rescore candidate; return ``(result, stats)``."""
        audio_seconds = audio.size / float(self._kws_sample_rate)
        cand_stats = {
            "candidate": cand_name,
            "anchors_requested": len(self._rescore_lead_seconds),
            "anchors_evaluated": None,
            "anchor_events": 0,
            "hit": False,
            "elapsed_ms": None,
            "error": None,
        }
        uid = self._window_uid(window_id)

        def _trace(event: str, **fields) -> None:
            if event == "anchor":
                cand_stats["anchor_events"] += 1
                self._kws_log(
                    "KWS_RESCORE_ANCHOR",
                    window_id=window_id,
                    window_uid=uid,
                    generation=self._generation,
                    candidate=cand_name,
                    lead_seconds=fields.get("lead_seconds"),
                    padded_seconds=fields.get("padded_seconds"),
                    audio_seconds=fields.get("audio_seconds"),
                    hit=fields.get("hit"),
                    keyword=fields.get("keyword"),
                    elapsed_ms=fields.get("elapsed_ms"),
                    error=fields.get("error"),
                )
            elif event == "skipped":
                self._kws_log(
                    "KWS_RESCORE_SKIPPED",
                    window_id=window_id,
                    window_uid=uid,
                    generation=self._generation,
                    candidate=cand_name,
                    reason=fields.get("reason"),
                    stage=fields.get("stage"),
                )
            elif event == "complete":
                req = fields.get("anchors_requested")
                ev = fields.get("anchors_evaluated")
                if req is not None:
                    cand_stats["anchors_requested"] = int(req)
                if ev is not None:
                    cand_stats["anchors_evaluated"] = int(ev)
                cand_stats["hit"] = bool(fields.get("hit"))
                self._kws_log(
                    "KWS_RESCORE_COMPLETE",
                    window_id=window_id,
                    window_uid=uid,
                    generation=self._generation,
                    candidate=cand_name,
                    anchors_requested=fields.get("anchors_requested"),
                    anchors_evaluated=fields.get("anchors_evaluated"),
                    hit=fields.get("hit"),
                    keyword=fields.get("keyword"),
                    lead_seconds=fields.get("lead_seconds"),
                    elapsed_ms=fields.get("elapsed_ms"),
                )

        self._kws_log(
            "KWS_RESCORE_START",
            window_id=window_id,
            window_uid=uid,
            generation=self._generation,
            candidate=cand_name,
            audio_seconds=round(audio_seconds, 3),
            samples=int(audio.size),
            sample_rate=self._kws_sample_rate,
            dtype="float32",
            anchors_requested=len(self._rescore_lead_seconds),
            trail_seconds=self._rescore_trail_seconds,
        )
        t0 = time.monotonic()

        def _fail(exc: Exception) -> tuple:
            elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)
            cand_stats["elapsed_ms"] = elapsed_ms
            cand_stats["error"] = repr(exc)
            cand_stats["hit"] = False
            if cand_stats["anchors_evaluated"] is None:
                cand_stats["anchors_evaluated"] = (
                    cand_stats["anchor_events"]
                    or 0
                )
            self._kws_log(
                "KWS_RESCORE_FAILED", window_id=window_id,
                window_uid=uid, generation=self._generation,
                candidate=cand_name, error=repr(exc),
                elapsed_ms=elapsed_ms)
            logger.exception("KWS rescore failed")
            return None, cand_stats

        try:
            result = rescore(
                audio,
                lead_seconds=self._rescore_lead_seconds,
                trail_seconds=self._rescore_trail_seconds,
                trace=_trace,
            )
        except TypeError:
            # Engines predating the trace parameter.
            try:
                result = rescore(
                    audio,
                    lead_seconds=self._rescore_lead_seconds,
                    trail_seconds=self._rescore_trail_seconds,
                )
            except Exception as exc:  # noqa: BLE001
                return _fail(exc)
        except Exception as exc:  # noqa: BLE001
            return _fail(exc)
        elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)
        cand_stats["elapsed_ms"] = elapsed_ms
        cand_stats["hit"] = bool(result)
        if cand_stats["anchors_evaluated"] is None:
            if cand_stats["anchor_events"]:
                cand_stats["anchors_evaluated"] = cand_stats[
                    "anchor_events"
                ]
            elif result is None:
                # A full miss on a conforming engine evaluated every
                # requested anchor; a hit's exact count is unknown
                # without the trace callback, so leave it None.
                cand_stats["anchors_evaluated"] = cand_stats[
                    "anchors_requested"
                ]
        if result:
            logger.info(
                "Activity window rescored: keyword=%r lead=%.2fs "
                "audio=%.2fs elapsed=%.0fms candidate=%s",
                result[0], result[1], audio_seconds, elapsed_ms, cand_name,
            )
        return result, cand_stats

    @staticmethod
    def _frame_period_seconds(capture_stats: dict) -> float:
        frames = capture_stats.get("callback_frames") or 0
        rate = capture_stats.get("sample_rate") or 0
        if frames and rate:
            return float(frames) / float(rate)
        return 0.1  # ~100 ms frames at 16 kHz

    def _window_stats(self) -> dict:
        """Snapshot of the current rolling measurement window."""
        if not self._win_frames:
            return {"frames": 0}
        return {
            "window_seconds": self._window_seconds,
            "frames": self._win_frames,
            "inferences": self._win_inferences,
            "inference_ms_avg": (
                round(self._win_inf_ms / self._win_inferences, 3)
                if self._win_inferences else None
            ),
            "kws_rms_max": round(self._win_kws_rms_max, 5),
            "kws_rms_mean": round(
                self._win_kws_rms_sum / self._win_frames, 5
            ),
            "kws_peak_max": round(self._win_kws_peak_max, 4),
            "selected_channel": self._win_selected_channel,
        }

    def _roll_window(self, now: float) -> None:
        """Reset the rolling window when it has expired."""
        if self._win_start and now - self._win_start >= self._window_seconds:
            self._win_start = now
            self._win_frames = 0
            self._win_inferences = 0
            self._win_inf_ms = 0.0
            self._win_kws_rms_max = 0.0
            self._win_kws_peak_max = 0.0
            self._win_kws_rms_sum = 0.0
            self._win_selected_channel = None
        elif not self._win_start:
            self._win_start = now

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the detection worker thread (idempotent)."""
        if self.is_running:
            return
        self._stop.clear()
        self._last_detection = 0.0
        self.started_monotonic = time.monotonic()
        self.first_frame_monotonic = 0.0
        self.first_frame_rms = 0.0
        self._first_frame_logged = False
        self._thread = threading.Thread(
            target=self._run, name="wakeword-detect", daemon=True
        )
        # V10.7.4.1: emit the lifecycle event before the worker starts
        # so diagnostic windows never race the start boundary.
        kws_parity_diagnostic.detector_event(self, "start")
        self._thread.start()
        # V10.8: dedicated rescore worker — anchor sweeps never run on
        # the detection thread, so live KWS inference stays responsive.
        self._rescore_stop.clear()
        self._rescore_thread = threading.Thread(
            target=self._rescore_worker, name="kws-rescore",
            daemon=True,
        )
        self._rescore_thread.start()
        self._capture_info()
        kw_file = getattr(self._engine, "keywords_file", None)
        kw_count = getattr(self._engine, "keyword_count", None)
        kw_sha = getattr(self._engine, "keywords_sha256", None)
        logger.info(
            "Wake-word detection started (threshold=%.2f, keywords=%s, "
            "variants=%s, sha256=%s)",
            self._threshold,
            kw_file,
            kw_count,
            kw_sha,
        )

    def stop(self) -> None:
        """Stop the worker and wait briefly for it to finish."""
        kws_parity_diagnostic.detector_event(self, "stop")
        self._stop.set()
        self._rescore_stop.set()
        # V10.7.1: close any open activity window and drop diagnostic
        # audio on every stop (covers suspend/session transitions and
        # shutdown).
        try:
            self._close_activity_window(
                time.monotonic(), allow_rescore=False
            )
        except Exception:  # noqa: BLE001
            pass
        self.clear_diagnostic_audio()
        thread, self._thread = self._thread, None
        # May be called from the worker thread itself (e.g. after a
        # detection); joining self would raise RuntimeError.
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=2.0)
            if thread.is_alive():
                logger.warning(
                    "Wake-word worker did not exit within 2s; it will "
                    "finish its current frame then stop"
                )
        # V10.8: stop the rescore worker. should_abort makes an
        # in-flight engine.rescore() bail between 100 ms decode chunks,
        # so this join stays short even mid-sweep.
        rthread, self._rescore_thread = self._rescore_thread, None
        if (
            rthread is not None
            and rthread.is_alive()
            and rthread is not threading.current_thread()
        ):
            rthread.join(timeout=2.0)
            if rthread.is_alive():
                logger.warning(
                    "KWS rescore worker did not exit within 2s"
                )
        with self._rescore_pending_lock:
            self._rescore_pending_ids.clear()
        logger.info("Wake-word detection stopped")

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            # Emit completed async rescore hits before consuming the
            # next frame (keeps on_detected on this thread).
            self._drain_rescore_results()
            frame = self._capture.get_frame(timeout=0.25)
            if frame is None:
                self._maybe_emit_stats()
                self._activity_close_check(time.monotonic())
                continue
            dequeued_at = time.monotonic()
            self.last_frame_dequeued_monotonic = dequeued_at
            self._roll_window(dequeued_at)
            # KWS input diagnostics (no audio persisted).
            try:
                fa = np.asarray(frame, dtype=np.float32).reshape(-1)
                self.last_kws_samples = int(fa.size)
                if fa.size:
                    self.last_kws_rms = float(
                        np.sqrt(np.mean(fa * fa)))
                    self.last_kws_peak = float(np.max(np.abs(fa)))
                    if self.last_kws_rms > self.max_kws_rms:
                        self.max_kws_rms = self.last_kws_rms
                    # Rolling window + speech-end tracking.
                    self._win_frames += 1
                    self._win_kws_rms_sum += self.last_kws_rms
                    if self.last_kws_rms > self._win_kws_rms_max:
                        self._win_kws_rms_max = self.last_kws_rms
                    if self.last_kws_peak > self._win_kws_peak_max:
                        self._win_kws_peak_max = self.last_kws_peak
                    if self.last_kws_rms >= self._speech_rms_threshold:
                        self._last_voiced_monotonic = dequeued_at
                    cap_stats = (
                        self._capture.stats()
                        if hasattr(self._capture, "stats") else {}
                    )
                    self._win_selected_channel = cap_stats.get(
                        "selected_channel"
                    )
                    # V10.6/V10.7.1: ring buffer + activity-window
                    # tracking on the exact
                    # KWS input (bounded, in-memory only).
                    self._diag_append(fa)
                    self._activity_frame(
                        self.last_kws_rms, self.last_kws_peak,
                        self.last_kws_samples, dequeued_at,
                    )
                    # V10.7.5: keep the window's exact KWS input for a
                    # possible post-window rescore. Bounded to ~12 s;
                    # oldest frames evicted first.
                    if self._activity_open:
                        self._window_audio.append(fa)
                        self._window_audio_samples += int(fa.size)
                        while (
                            self._window_audio_samples
                            > self._window_audio_max_samples
                            and self._window_audio
                        ):
                            old = self._window_audio.popleft()
                            self._window_audio_samples -= int(old.size)
            except Exception:  # noqa: BLE001
                pass
            self._activity_close_check(time.monotonic())
            # V10.7.4: opt-in parity diagnostic — captures the exact
            # frame about to enter engine.process() plus raw sherpa
            # outcomes. No-op unless VOICEOS_KWS_PARITY_DIAGNOSTIC=1.
            _diag_frame = kws_parity_diagnostic.before_process(
                self, frame, dequeued_at)
            try:
                t0 = time.monotonic()
                confidence = self._engine.process(frame)
                elapsed = time.monotonic() - t0
                kws_parity_diagnostic.after_process(
                    _diag_frame, confidence=confidence,
                    elapsed=elapsed, error=None)
                self._win_inferences += 1
                self._win_inf_ms += elapsed * 1000.0
                if self._activity_open:
                    self._activity_inf_ms += elapsed * 1000.0
                self._inference_seconds_ema = (
                    elapsed
                    if not self._inference_seconds_ema
                    else 0.8 * self._inference_seconds_ema
                    + 0.2 * elapsed
                )
            except Exception as exc:  # noqa: BLE001
                kws_parity_diagnostic.after_process(
                    _diag_frame, confidence=None, elapsed=None,
                    error=repr(exc))
                self.inference_errors += 1
                logger.exception("Wake-word inference error")
                kws_parity_diagnostic.frame_complete(_diag_frame)
                continue
            self.frames_processed += 1
            self.last_inference_monotonic = time.monotonic()
            if not self._first_frame_logged:
                # First fresh KWS input after (re)start: proves the
                # handover delivered live audio, not a stale queue.
                self._first_frame_logged = True
                self.first_frame_monotonic = dequeued_at
                self.first_frame_rms = self.last_kws_rms
                logger.info(
                    "First fresh KWS input after detector start: "
                    "rms=%.4f peak=%.3f samples=%d",
                    self.last_kws_rms, self.last_kws_peak,
                    self.last_kws_samples,
                )
                if self._on_first_frame is not None:
                    try:
                        self._on_first_frame()
                    except Exception:  # noqa: BLE001
                        logger.debug(
                            "on_first_frame callback failed",
                            exc_info=True,
                        )
            if self._on_confidence is not None:
                try:
                    self._on_confidence(confidence)
                except Exception:  # noqa: BLE001
                    pass
            if confidence >= self._threshold:
                _kw_raw = self._engine.keyword_detected()
                keyword = _kw_raw or "hello jarvis"
                emitted = self._emit_detection(
                    keyword, time.monotonic(),
                    diag_frame=_diag_frame, raw_keyword=_kw_raw)
                if not emitted:
                    # Cooldown-suppressed: preserve the original
                    # frame_complete + continue behaviour.
                    kws_parity_diagnostic.frame_complete(_diag_frame)
                    continue
            kws_parity_diagnostic.frame_complete(_diag_frame)
            self._maybe_emit_stats()

    def _emit_detection(
        self,
        keyword: str,
        now: float,
        *,
        diag_frame=None,
        raw_keyword: Optional[str] = None,
        rescored: bool = False,
        window_id: Optional[int] = None,
    ) -> bool:
        """Shared post-keyword path: cooldown, bookkeeping, callback.

        Used by the live inference path (``diag_frame`` set) and by
        the V10.7.5 activity-window rescore path (``rescored=True``).
        Returns True when the detection was emitted, False when the
        cooldown suppressed it.
        """
        via = "rescore" if rescored else "live"
        wid = (
            window_id if window_id is not None
            else (self._activity_seq if self._activity_open else None)
        )
        if now - self._last_detection < self._cooldown:
            if diag_frame is not None:
                kws_parity_diagnostic.note_threshold(
                    diag_frame, raw_keyword=raw_keyword,
                    effective_keyword=keyword, cooldown_blocked=True)
            logger.info(
                "Wake-word detection suppressed by cooldown "
                "(%.2fs since last)", self._cooldown
            )
            self._kws_log(
                "KWS_DETECTION_BLOCKED",
                window_id=wid,
                window_uid=self._window_uid(wid),
                generation=self._generation,
                keyword=keyword,
                via=via,
                reason="cooldown",
                cooldown_seconds=self._cooldown,
                seconds_since_last_detection=round(
                    now - self._last_detection, 3),
            )
            self._engine.reset()
            if diag_frame is not None:
                kws_parity_diagnostic.note_reset(diag_frame)
            return False
        if diag_frame is not None:
            kws_parity_diagnostic.note_threshold(
                diag_frame, raw_keyword=raw_keyword,
                effective_keyword=keyword, cooldown_blocked=False)
        self._last_detection = now
        self._detection_count += 1
        self.last_keyword = keyword
        # V10.7.1: tag the open window so its record shows the
        # keyword that closed it. Rescored hits arrive asynchronously
        # after their window closed — tagging whatever window happens
        # to be open now would misattribute the detection, and the
        # originating record was already updated by
        # _apply_rescore_report.
        if self._activity_open and not rescored:
            self._activity_detected = True
            self._activity_keyword = keyword
        # V10.4 detection record: timestamps + input level so
        # end-of-phrase -> detection latency is measurable.
        speech_end_gap = (
            now - self._last_voiced_monotonic
            if self._last_voiced_monotonic else None
        )
        self.last_detection = {
            "monotonic": now,
            "wall": time.time(),
            "keyword": keyword,
            "kws_rms": self.last_kws_rms,
            "kws_peak": self.last_kws_peak,
            "audio_age_seconds": (
                now - self.last_frame_dequeued_monotonic
                if self.last_frame_dequeued_monotonic else None
            ),
            "speech_end_to_detection_seconds": speech_end_gap,
            "threshold": self._threshold,
            "rescored": rescored,
            "detection_method": via,
            "generation": self._generation,
            "window_uid": self._window_uid(wid),
        }
        if speech_end_gap is not None:
            self.detection_latencies.append(speech_end_gap)
            # Bound the sample list; reports use all retained.
            if len(self.detection_latencies) > 1000:
                del self.detection_latencies[:500]
        logger.info(
            "Wake word detected: %r kws_rms=%.4f kws_peak=%.3f "
            "audio_age=%.3fs speech_end_to_detection=%s%s",
            keyword,
            self.last_kws_rms,
            self.last_kws_peak,
            self.last_detection["audio_age_seconds"] or -1.0,
            (
                f"{speech_end_gap:.3f}s"
                if speech_end_gap is not None else "n/a"
            ),
            " via=rescore" if rescored else "",
        )
        self._kws_log(
            "KWS_DETECTION_EMITTED",
            window_id=wid,
            window_uid=self._window_uid(wid),
            generation=self._generation,
            keyword=keyword,
            via=via,
            rescored=rescored,
            detection_count=self._detection_count,
        )
        callback_error = None
        if self._on_detected is not None:
            if diag_frame is not None:
                kws_parity_diagnostic.note_on_detected(
                    diag_frame, keyword)
            try:
                self._on_detected(keyword)
            except Exception as exc:  # noqa: BLE001
                callback_error = repr(exc)
                if diag_frame is not None:
                    kws_parity_diagnostic.note_on_detected_error(
                        diag_frame)
                logger.exception("Wake-word callback failed")
        self._kws_log(
            "KWS_SESSION_ACTIVATED",
            window_id=wid,
            window_uid=self._window_uid(wid),
            generation=self._generation,
            keyword=keyword,
            via=via,
            callback_ok=callback_error is None,
            error=callback_error,
        )
        # Suspend detection until reset()/restart (one shot).
        self._engine.reset()
        if diag_frame is not None:
            kws_parity_diagnostic.note_reset(diag_frame)
        return True

    def _maybe_emit_stats(self) -> None:
        """Emit throttled pipeline stats to the UI and the log."""
        now = time.monotonic()
        if self._on_stats is not None and (
            now - self._last_stats_emit >= self._stats_interval
        ):
            self._last_stats_emit = now
            try:
                self._on_stats(self.stats())
            except Exception:  # noqa: BLE001
                pass
        if now - self._last_stats_log >= self._stats_log_interval:
            self._last_stats_log = now
            s = self.stats()
            logger.info(
                "Pipeline stats: captured=%s queued=%s(depth=%s) "
                "dropped=%s processed=%s rms=%.4f peak=%.3f "
                "max_rms=%.4f signal=%s clipped=%s detections=%s "
                "kws_rms=%.4f kws_peak=%.3f ch_rms=%s sel_ch=%s",
                s.get("frames_captured"),
                s.get("frames_queued"),
                s.get("queue_depth", "?"),
                s.get("frames_dropped"),
                s.get("frames_processed"),
                s.get("rms") or 0.0,
                s.get("peak") or 0.0,
                s.get("max_rms") or 0.0,
                s.get("signal_frames"),
                s.get("clipped_frames"),
                s.get("detections"),
                s.get("kws_input_rms") or 0.0,
                s.get("kws_input_peak") or 0.0,
                s.get("channel_rms"),
                s.get("selected_channel"),
            )

    def suspend(self) -> None:
        """Stop detection after a trigger (mic capture may continue)."""
        kws_parity_diagnostic.detector_event(self, "suspend")
        self.stop()

    def resume(self) -> None:
        """Restart detection after suspension."""
        self.start()
