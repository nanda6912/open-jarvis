"""V10.7.4/V10.7.4.1 — opt-in KWS parity diagnostic.

Enabled only when ``VOICEOS_KWS_PARITY_DIAGNOSTIC=1``. Records the
*exact* audio frames handed to ``WakeWordEngine.process()`` plus the
raw Sherpa-ONNX outcomes at every processing boundary, then — after a
bounded 450-frame (45 s) window fills — replays the identical samples
through independent fresh engines (plain replay and lifecycle replay)
and verifies speech segments with the production STT engine.

V10.7.4.1 reliability fixes:

* ``diagnostic_status`` section exposes collector health so an empty
  ``windows`` array is never ambiguous (disabled / no frames / partial
  window / pending analysis / failed analysis / shutdown).
* Atomic report checkpoints (temp file + ``os.replace``) written by a
  single dedicated writer thread; the KWS worker never does file I/O.
* Window lifecycle is service-generation aware: a window stays open
  across ``stop``/``suspend`` (recording the lifecycle trail) and is
  sealed as an explicit ``interrupted_partial`` entry when the next
  ``start`` arrives or the process shuts down — partial windows are
  preserved with their interruption reason instead of silently
  mixing stream histories.
* Window finalization is deferred to ``frame_complete()`` so the
  450th frame's detection bookkeeping (cooldown, emitted, callback)
  is captured before the snapshot.
* Analysis runs on a single daemon worker fed by a bounded queue;
  every finalized window is submitted exactly once, failures are
  recorded on the window entry, and shutdown marks pending work as
  interrupted instead of pretending it completed.
* HUD/performance metadata is collected outside the audio path
  (process CPU/RSS via psutil when available, KWS inference EMA,
  queue depth, dropped frames, plus an optional HUD snapshot hook).

Hard guarantees:

* Disabled by default; every public function is a no-op (and fully
  exception-shielded) unless the env var equals ``"1"``.
* No second microphone stream, no production behaviour change.
* PCM stays in bounded memory only (~2.9 MB per 45 s window); the
  JSON report contains hashes/metadata/transcripts, never samples.
* Analysis never blocks the KWS worker or the PortAudio callback.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import queue
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from src.core.paths import runtime_dir

logger = logging.getLogger(__name__)

ENV_FLAG = "VOICEOS_KWS_PARITY_DIAGNOSTIC"
MAX_WINDOW_SECONDS = 45.0
FRAME_RATE = 16000
MAX_FRAMES = int(MAX_WINDOW_SECONDS * FRAME_RATE / 1600)  # 450
CHECKPOINT_EVERY_FRAMES = 50
MAX_PENDING_ANALYSIS = 3
MAX_REPORTED_WINDOWS = 8
MAX_KEPT_WINDOWS = 16

REPORT_PATH = (
    runtime_dir() / "reports" / "v1074_production_kws_parity.json"
)

_ENABLED_CACHE: bool | None = None


def enabled() -> bool:
    """True only when VOICEOS_KWS_PARITY_DIAGNOSTIC=1 (cached)."""
    global _ENABLED_CACHE
    if _ENABLED_CACHE is None:
        _ENABLED_CACHE = os.environ.get(ENV_FLAG, "") == "1"
    return _ENABLED_CACHE


def _reset_enabled_cache() -> None:
    """Test hook: re-read the environment flag."""
    global _ENABLED_CACHE
    _ENABLED_CACHE = None


def _safe(fn, *args, **kwargs):
    """Run fn swallowing all errors — diagnostics must never break KWS."""
    try:
        return fn(*args, **kwargs)
    except Exception:  # noqa: BLE001
        logger.debug("KWS parity diagnostic hook failed", exc_info=True)
        return None


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


class _Recorder:
    """Single shared recorder; all mutable state is lock-guarded."""

    def __init__(self) -> None:
        self._init_state()

    # --------------------------------------------------------------
    # state lifecycle
    # --------------------------------------------------------------
    def _init_state(self) -> None:
        self._lock = threading.Lock()
        self._io_lock = threading.Lock()
        # current (open) window buffers
        self._frames: list[dict] = []
        self._audio: list[np.ndarray] = []
        self._engine_events: list[dict] = []
        self._detector_events: list[dict] = []
        self._open: dict[int, dict] = {}
        self._next_frame = 0
        # window identity / lifecycle
        self._next_window_index = 1
        self._cur_meta: dict = self._fresh_window_meta()
        self._stopped_reason: str | None = None
        # finalized windows (immutable entry dicts; swaps replace them)
        self._windows: deque[dict] = deque(maxlen=MAX_KEPT_WINDOWS)
        # counters
        self._frames_seen = 0
        self._frames_recorded = 0
        self._finalize_after_idx: int | None = None
        self._finalizing = False
        # analysis pipeline
        self._analysis_queue: queue.Queue = queue.Queue()
        self._analysis_thread: threading.Thread | None = None
        self._analysis_pending = 0
        self._analysis_running = 0
        self._analysis_completed = 0
        self._analysis_failed = 0
        self._analysis_skipped = 0
        # report writer
        self._writer_thread: threading.Thread | None = None
        self._writer_event = threading.Event()
        self._last_report_write: float | None = None
        # misc status
        self._last_cap_stats: dict = {}
        self._engine_meta: dict = {}
        self._service_kinds: dict[int, dict] = {}
        self._kws_elapsed_ema: float | None = None
        self._last_frame_mono: float | None = None
        self._last_frame_wall: float | None = None
        self._last_error: str | None = None
        self._shutdown = False
        self._superseded = False
        self._migrated = False
        # HUD / performance metadata (collected off the audio path)
        self._hud_latest: dict | None = None
        self._hud_snapshots: deque = deque(maxlen=50)
        self._hud_probe_cache: dict | None = None
        self._hud_probe_at = 0.0
        self._proc_snapshot: dict | None = None
        self._proc_at = 0.0
        self._model_files_cache: dict | None = None
        self._proc_obj = None

    @staticmethod
    def _fresh_window_meta() -> dict:
        return {
            "index": None,
            "started_monotonic": None,
            "started_wall": None,
            "first_frame_monotonic": None,
            "last_frame_monotonic": None,
            "last_frame_wall": None,
            "service_ids": [],
            "stream_generations": [],
        }

    # --------------------------------------------------------------
    # live hooks (called from production threads; must be cheap)
    # --------------------------------------------------------------
    def before_process(self, service, frame, dequeued_at):
        if not enabled():
            return None
        return self._guard(self._before_process, service, frame,
                           dequeued_at)

    def _guard(self, fn, *args, **kwargs):
        if self._shutdown:
            return None
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            with self._lock:
                self._last_error = repr(exc)
            logger.debug("KWS parity diagnostic hook failed",
                         exc_info=True)
            return None

    def _before_process(self, service, frame, dequeued_at):
        arr = np.asarray(frame, dtype=np.float32).reshape(-1)
        contiguous = np.ascontiguousarray(arr, dtype=np.float32)
        digest = hashlib.sha256(contiguous.tobytes()).hexdigest()
        cap_stats = {}
        try:
            cap_stats = service._capture.stats() or {}
        except Exception:  # noqa: BLE001
            pass
        self._last_cap_stats = {
            k: cap_stats.get(k)
            for k in (
                "sample_rate", "input_channels", "selected_channel",
                "queue_depth", "frames_captured", "frames_dropped",
                "overflow_callbacks", "status_warnings",
                "channel_switch_count",
            )
        }
        engine = getattr(service, "_engine", None)
        engine_meta = {}
        if engine is not None:
            engine_meta = {
                "name": getattr(engine, "name", None),
                "model_dir": str(getattr(engine, "_model_dir", "") or ""),
                "keyword_count": getattr(engine, "keyword_count", None),
                "keywords_sha256": getattr(
                    engine, "keywords_sha256", None),
                "keywords_file": str(
                    getattr(engine, "keywords_file", "") or ""),
            }
        rec = {
            "idx": self._next_frame,
            "sha256": digest,
            "n_samples": int(arr.size),
            "rms": float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0,
            "peak": float(np.max(np.abs(arr))) if arr.size else 0.0,
            "queue_depth": self._last_cap_stats.get("queue_depth"),
            "selected_channel": self._last_cap_stats.get(
                "selected_channel"),
            "stream_rate": self._last_cap_stats.get("sample_rate"),
            "service_id": id(service),
            "thread_id": threading.get_ident(),
            "thread_name": threading.current_thread().name,
            "stream_generation": getattr(
                engine, "_stream_generation", None),
            "engine_name": getattr(engine, "name", None),
            "dequeue_monotonic": dequeued_at,
            "process_entry_monotonic": time.monotonic(),
            "process_exit_monotonic": None,
            "process_seconds": None,
            "decode_calls": None,
            "raw_keyword": None,
            "raw_keyword_type": None,
            "confidence": None,
            "threshold_passed": None,
            "emitted": False,
            "cooldown_blocked": False,
            "on_detected_called": False,
            "on_detected_error": False,
            "engine_reset_after": False,
            "inference_error": None,
        }
        now_mono = time.monotonic()
        now_wall = time.time()
        need_checkpoint = False
        need_finalize = False
        with self._lock:
            self._frames_seen += 1
            self._engine_meta = engine_meta or self._engine_meta
            meta = self._cur_meta
            new_window = meta["index"] is None
            if new_window:
                meta["index"] = self._next_window_index
                self._next_window_index += 1
                meta["started_monotonic"] = now_mono
                meta["started_wall"] = now_wall
            sid = id(service)
            if sid not in meta["service_ids"]:
                meta["service_ids"].append(sid)
            gen = rec["stream_generation"]
            if gen is not None and gen not in meta["stream_generations"]:
                if len(meta["stream_generations"]) < 64:
                    meta["stream_generations"].append(gen)
            meta["last_frame_monotonic"] = now_mono
            meta["last_frame_wall"] = now_wall
            if meta["first_frame_monotonic"] is None:
                meta["first_frame_monotonic"] = now_mono
            self._last_frame_mono = now_mono
            self._last_frame_wall = now_wall
            self._frames.append(rec)
            self._audio.append(contiguous.copy())
            self._open[threading.get_ident()] = rec
            self._next_frame += 1
            self._frames_recorded += 1
            if len(self._frames) >= MAX_FRAMES:
                if self._finalize_after_idx is None:
                    self._finalize_after_idx = rec["idx"]
            elif (
                self._finalize_after_idx is not None
                and self._next_frame > self._finalize_after_idx + 5
            ):
                # Defensive: frame_complete was missed; do not let the
                # window grow unboundedly.
                need_finalize = True
            if self._next_frame % CHECKPOINT_EVERY_FRAMES == 0 \
                    or new_window:
                need_checkpoint = True
        if need_finalize:
            self._spawn_finalize()
        if need_checkpoint:
            self._request_checkpoint()
        return rec

    def after_process(self, rec, *, confidence, elapsed, error):
        if rec is None:
            return
        self._guard(self._after_process, rec, confidence, elapsed,
                    error)

    def _after_process(self, rec, confidence, elapsed, error):
        rec["process_exit_monotonic"] = time.monotonic()
        rec["process_seconds"] = elapsed
        rec["confidence"] = confidence
        rec["threshold_passed"] = (
            confidence is not None and confidence >= 0.25)
        if error is not None:
            rec["inference_error"] = error
        with self._lock:
            self._open.pop(threading.get_ident(), None)
            if elapsed:
                ms = float(elapsed)
                self._kws_elapsed_ema = (
                    ms if self._kws_elapsed_ema is None
                    else 0.8 * self._kws_elapsed_ema + 0.2 * ms
                )

    def frame_complete(self, rec):
        """Called by the service at the end of each processed frame.

        Triggers window finalization only after ALL per-frame diagnostic
        hooks for that frame (threshold/cooldown/emission/callback) have
        run, so the snapshot is complete.
        """
        if not enabled() or rec is None:
            return
        self._guard(self._frame_complete, rec)

    def _frame_complete(self, rec):
        spawn = False
        with self._lock:
            if (
                self._finalize_after_idx is not None
                and rec.get("idx") == self._finalize_after_idx
            ):
                spawn = True
        if spawn:
            self._spawn_finalize()

    # engine-level raw sherpa boundary (same worker thread)
    def engine_result(self, engine, *, raw, decode_calls, elapsed):
        if not enabled():
            return
        if getattr(engine, "_parity_replay", False):
            return
        self._guard(self._engine_result, engine, raw, decode_calls,
                    elapsed)

    def _engine_result(self, engine, raw, decode_calls, elapsed):
        rec = self._open.get(threading.get_ident())
        if rec is None:
            return
        rec["decode_calls"] = decode_calls
        rec["raw_keyword"] = raw if isinstance(raw, str) else repr(raw)
        rec["raw_keyword_type"] = type(raw).__name__

    def on_stream_event(self, engine, reason: str):
        if not enabled():
            return
        if getattr(engine, "_parity_replay", False):
            return
        self._guard(self._on_stream_event, engine, reason)

    def _on_stream_event(self, engine, reason: str):
        ident = threading.get_ident()
        with self._lock:
            open_rec = self._open.get(ident)
            ev = {
                "seq": len(self._engine_events),
                "reason": reason,
                "stream_generation": getattr(
                    engine, "_stream_generation", None),
                "monotonic": time.monotonic(),
                "wall": time.time(),
                "thread_id": ident,
                "during_frame": (
                    open_rec["idx"] if open_rec is not None else None
                ),
                "applies_before_frame": (
                    open_rec["idx"] + 1 if open_rec is not None
                    else self._next_frame
                ),
            }
            self._engine_events.append(ev)
            if open_rec is not None:
                open_rec["engine_reset_after"] = True

    def detector_event(self, service, kind: str):
        if not enabled():
            return
        self._guard(self._detector_event, service, kind)

    def _detector_event(self, service, kind: str):
        sid = id(service)
        with self._lock:
            self._service_kinds[sid] = {
                "kind": kind,
                "monotonic": time.monotonic(),
                "wall": time.time(),
            }
        if kind == "start":
            # Seal the previous window only when a stop/suspend really
            # opened a lifecycle boundary. Frames that arrive before the
            # start event (thread.start() races ahead) belong to the
            # same collection and must not spawn a spurious partial.
            with self._lock:
                was_stopped = self._stopped_reason is not None
            if was_stopped:
                self._seal_window()
            with self._lock:
                self._stopped_reason = None
        elif kind in ("stop", "suspend"):
            with self._lock:
                if self._stopped_reason is None:
                    self._stopped_reason = kind
        with self._lock:
            self._detector_events.append({
                "seq": len(self._detector_events),
                "kind": kind,
                "service_id": sid,
                "monotonic": time.monotonic(),
                "wall": time.time(),
                "thread_id": threading.get_ident(),
                "frames_recorded": len(self._frames),
            })
        self._request_checkpoint()

    def note_threshold(self, rec, *, raw_keyword, effective_keyword,
                       cooldown_blocked):
        if rec is None:
            return
        self._guard(self._note_threshold, rec, raw_keyword,
                    effective_keyword, cooldown_blocked)

    def _note_threshold(self, rec, raw_keyword, effective_keyword,
                        cooldown_blocked):
        rec["raw_keyword_detected"] = raw_keyword
        rec["effective_keyword"] = effective_keyword
        rec["cooldown_blocked"] = bool(cooldown_blocked)
        rec["emitted"] = not cooldown_blocked

    def note_on_detected(self, rec, keyword):
        if rec is None:
            return
        self._guard(self._note_on_detected, rec, keyword)

    def _note_on_detected(self, rec, keyword):
        rec["on_detected_called"] = True
        rec["on_detected_keyword"] = keyword

    def note_on_detected_error(self, rec):
        if rec is None:
            return
        self._guard(self._note_on_detected_error, rec)

    def _note_on_detected_error(self, rec):
        rec["on_detected_error"] = True

    def note_reset(self, rec):
        if rec is None:
            return
        self._guard(self._note_reset, rec)

    def _note_reset(self, rec):
        rec["engine_reset_after"] = True

    # --------------------------------------------------------------
    # HUD / performance metadata (off the audio path)
    # --------------------------------------------------------------
    def record_hud_snapshot(self, **fields):
        """Optional HUD telemetry hook (HUD code is not modified).

        Accepts arbitrary fields; a ``monotonic``/``wall`` timestamp is
        added automatically. Bounded to the last 50 snapshots.
        """
        if not enabled():
            return
        self._guard(self._record_hud_snapshot, fields)

    def _record_hud_snapshot(self, fields):
        snap = dict(fields)
        snap["monotonic"] = time.monotonic()
        snap["wall"] = time.time()
        with self._lock:
            self._hud_latest = snap
            self._hud_snapshots.append(snap)

    # --------------------------------------------------------------
    # window sealing / finalization
    # --------------------------------------------------------------
    def _snapshot_open_window(self):
        """Detach current buffers. Caller must hold no lock."""
        with self._lock:
            frames = list(self._frames)
            audio = [a.copy() for a in self._audio]
            events = list(self._engine_events)
            det_events = list(self._detector_events)
            cap_stats = dict(self._last_cap_stats)
            engine_meta = dict(self._engine_meta)
            meta = dict(self._cur_meta)
            meta["service_ids"] = list(meta["service_ids"])
            meta["stream_generations"] = list(
                meta["stream_generations"])
            self._frames.clear()
            self._audio.clear()
            self._engine_events.clear()
            self._detector_events.clear()
            self._open.clear()
            self._next_frame = 0
            self._cur_meta = self._fresh_window_meta()
            self._finalize_after_idx = None
        return frames, audio, events, det_events, cap_stats, \
            engine_meta, meta

    def _seal_window(self, reason: str | None = None):
        """Close the open window as an interrupted partial (if it has
        frames) and preserve it with its interruption reason."""
        with self._lock:
            has_frames = bool(self._frames)
            stopped = self._stopped_reason
            self._stopped_reason = None
        if not has_frames:
            # No audio captured: drop event-only residue silently but
            # still reset buffers so the next window starts clean.
            self._snapshot_open_window()
            return
        frames, audio, events, det_events, cap_stats, engine_meta, \
            meta = self._snapshot_open_window()
        why = stopped or reason or "detector_restart"
        entry = _entry_base(
            meta, frames, events, det_events, cap_stats, engine_meta,
            status="interrupted_partial",
            analysis={"status": "skipped",
                      "reason": f"partial_window:{why}"},
            interruption={
                "reason": why,
                "monotonic": time.monotonic(),
                "wall": time.time(),
                "frames": len(frames),
            },
        )
        with self._lock:
            self._windows.append(entry)
        self._request_checkpoint()

    def _spawn_finalize(self):
        with self._lock:
            if self._finalizing:
                return
            if (
                self._finalize_after_idx is None
                and len(self._frames) < MAX_FRAMES
            ):
                return
            self._finalizing = True
        try:
            frames, audio, events, det_events, cap_stats, engine_meta, \
                meta = self._snapshot_open_window()
            entry = _entry_base(
                meta, frames, events, det_events, cap_stats,
                engine_meta, status="pending_analysis",
                analysis={
                    "status": "pending",
                    "submitted_monotonic": time.monotonic(),
                },
            )
            with self._lock:
                if self._analysis_queue.qsize() >= MAX_PENDING_ANALYSIS:
                    entry = dict(entry)
                    entry["status"] = "analysis_skipped"
                    entry["analysis"] = {
                        "status": "skipped",
                        "reason": "analysis_backlog",
                    }
                    self._windows.append(entry)
                    self._analysis_skipped += 1
                else:
                    self._windows.append(entry)
                    self._analysis_pending += 1
                    self._analysis_queue.put((entry, frames, audio,
                                              events))
                    self._ensure_analysis_worker()
        finally:
            with self._lock:
                self._finalizing = False
        self._request_checkpoint()

    # --------------------------------------------------------------
    # analysis worker (single daemon; bounded queue)
    # --------------------------------------------------------------
    def _ensure_analysis_worker(self):
        t = self._analysis_thread
        if t is not None and t.is_alive():
            return
        self._analysis_thread = threading.Thread(
            target=self._analysis_main,
            name="kws-parity-analysis", daemon=True,
        )
        self._analysis_thread.start()

    def _analysis_main(self):
        while True:
            try:
                job = self._analysis_queue.get(timeout=0.5)
            except queue.Empty:
                with self._lock:
                    if self._shutdown:
                        return
                continue
            entry, frames, audio, events = job
            with self._lock:
                self._analysis_running += 1
            running = dict(entry)
            running["status"] = "analyzing"
            running["analysis"] = {
                "status": "running",
                "started_monotonic": time.monotonic(),
            }
            self._replace_entry(entry, running)
            entry = running
            try:
                result = self._analyze_window(
                    entry["window"]["index"], frames, audio, events)
            except Exception as exc:  # noqa: BLE001
                done = dict(entry)
                done["status"] = "failed"
                done["analysis"] = {
                    "status": "failed",
                    "error": repr(exc),
                    "finished_monotonic": time.monotonic(),
                }
                self._replace_entry(entry, done)
                with self._lock:
                    self._analysis_failed += 1
                    self._analysis_pending -= 1
                    self._analysis_running -= 1
                    self._last_error = repr(exc)
                logger.exception("KWS parity analysis failed")
            else:
                done = dict(entry)
                done.update(result)
                done["status"] = "analyzed"
                done["analysis"] = {
                    "status": "completed",
                    "finished_monotonic": time.monotonic(),
                }
                self._replace_entry(entry, done)
                with self._lock:
                    self._analysis_completed += 1
                    self._analysis_pending -= 1
                    self._analysis_running -= 1
            self._request_checkpoint()

    def _replace_entry(self, old: dict, new: dict) -> bool:
        with self._lock:
            for i, e in enumerate(self._windows):
                if e is old:
                    self._windows[i] = new
                    return True
        return False

    # --------------------------------------------------------------
    # report writing (single writer thread + atomic replace)
    # --------------------------------------------------------------
    def _request_checkpoint(self):
        self._ensure_writer()
        self._writer_event.set()

    def _ensure_writer(self):
        with self._lock:
            t = self._writer_thread
            if t is not None and t.is_alive():
                return
            self._writer_thread = threading.Thread(
                target=self._writer_main,
                name="kws-parity-report", daemon=True,
            )
            self._writer_thread.start()

    def _writer_main(self):
        while True:
            self._writer_event.wait(timeout=30.0)
            self._writer_event.clear()
            if self._superseded:
                return
            try:
                self._write_report()
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._last_error = repr(exc)
                logger.debug("KWS parity report write failed",
                             exc_info=True)
            with self._lock:
                if self._shutdown:
                    return

    def _write_report(self):
        with self._io_lock:
            report = self._build_report()
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = REPORT_PATH.with_name(REPORT_PATH.name + ".tmp")
            tmp.write_text(
                json.dumps(report, indent=2, default=str),
                encoding="utf-8",
            )
            os.replace(tmp, REPORT_PATH)
            with self._lock:
                self._last_report_write = time.time()

    # --------------------------------------------------------------
    # report assembly
    # --------------------------------------------------------------
    def _build_report(self) -> dict:
        with self._lock:
            migrated = self._migrated
        if not migrated:
            # One-time import of windows from a previous report file.
            # File I/O happens outside the recorder lock.
            self._migrate_existing_windows()
        with self._lock:
            status = self._status_dict()
            active = self._active_window_dict()
            windows = list(self._windows)
            hud_latest = self._hud_latest
            hud_snaps = list(self._hud_snapshots)
            cap = dict(self._last_cap_stats)
            kws_ema = self._kws_elapsed_ema
            shutdown = self._shutdown
        pending = [
            {
                "window_index": w.get("window", {}).get("index"),
                "frames": w.get("window", {}).get("frames"),
                "analysis_status": (w.get("analysis") or {}).get(
                    "status"),
            }
            for w in windows
            if w.get("status") in (
                "pending_analysis", "analyzing")
        ]
        return {
            "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "env_flag": f"{ENV_FLAG}=1",
            "environment": _environment(),
            "model_files": self._model_files(),
            "diagnostic_status": status,
            "active_window": active,
            "pending_analysis": pending,
            "windows": windows[-MAX_REPORTED_WINDOWS:],
            "hud": {
                "latest_snapshot": hud_latest,
                "probe": self._hud_probe(),
                "snapshots": hud_snaps[-20:],
                "note": (
                    "HUD code is not instrumented directly; populate "
                    "via record_hud_snapshot() or read the passive "
                    "probe fields."
                ),
            },
            "performance": {
                "process": self._process_snapshot(),
                "kws": {
                    "process_seconds_ema": kws_ema,
                    "queue_depth": cap.get("queue_depth"),
                    "frames_dropped": cap.get("frames_dropped"),
                    "overflow_callbacks": cap.get(
                        "overflow_callbacks"),
                    "status_warnings": cap.get("status_warnings"),
                },
            },
            "shutdown": shutdown,
            "note": (
                "diagnostic_status.collection_state explains why "
                "windows may be empty: disabled / no frames / partial "
                "window / pending or failed analysis / shutdown. "
                "PCM is never persisted."
            ),
        }

    def _migrate_existing_windows(self):
        """Preserve windows from a previous report file (one-time)."""
        with self._lock:
            if self._migrated:
                return
        old = self._read_prev_windows()
        with self._lock:
            if self._migrated:
                return
            for w in old:
                self._windows.appendleft(w)
            self._migrated = True

    @staticmethod
    def _read_prev_windows() -> list:
        try:
            if not REPORT_PATH.exists():
                return []
            prev = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return []
        out = []
        for w in (prev.get("windows") or [])[-MAX_KEPT_WINDOWS:]:
            if not isinstance(w, dict):
                continue
            w = dict(w)
            w.setdefault("status", "analyzed")
            w.setdefault("analysis", {"status": "completed",
                                      "migrated": True})
            out.append(w)
        return out

    def _status_dict(self) -> dict:
        analyzing = self._analysis_running > 0
        pending_jobs = self._analysis_pending
        frames_open = len(self._frames)
        if not enabled():
            state = "disabled"
        elif self._shutdown:
            state = "stopped"
        elif analyzing:
            state = "analysis_running"
        elif pending_jobs:
            state = "window_finalized_pending_analysis"
        elif frames_open:
            state = (
                "detector_stopped_window_open"
                if self._stopped_reason else "collecting"
            )
        elif self._service_kinds:
            state = "service_attached_no_frames"
        else:
            state = "no_production_frames_received"
        analyzed = sum(
            1 for w in self._windows if w.get("status") == "analyzed")
        interrupted = sum(
            1 for w in self._windows
            if w.get("status") == "interrupted_partial")
        failed = sum(
            1 for w in self._windows if w.get("status") == "failed")
        return {
            "enabled": enabled(),
            "initialized": True,
            "recorder_attached": enabled(),
            "production_service_attached": bool(self._service_kinds),
            "hooks_invoked": (
                self._frames_seen > 0
                or bool(self._engine_events)
                or bool(self._detector_events)
                or bool(self._service_kinds)
            ),
            "frames_seen": self._frames_seen,
            "frames_recorded": self._frames_recorded,
            "current_window_frames": frames_open,
            "required_window_frames": MAX_FRAMES,
            "current_window_index": self._cur_meta.get("index"),
            "windows_total": len(self._windows),
            "completed_windows": analyzed,
            "interrupted_windows": interrupted,
            "failed_windows": failed,
            "analysis_jobs_pending": pending_jobs,
            "analysis_jobs_running": self._analysis_running,
            "analysis_jobs_completed": self._analysis_completed,
            "analysis_jobs_failed": self._analysis_failed,
            "analysis_jobs_skipped": self._analysis_skipped,
            "last_frame_timestamp": (
                {
                    "wall": _iso(self._last_frame_wall),
                    "monotonic": self._last_frame_mono,
                }
                if self._last_frame_wall is not None else None
            ),
            "last_report_write_timestamp": _iso(
                self._last_report_write),
            "last_error": self._last_error,
            "collection_state": state,
            "stopped_reason": self._stopped_reason,
            "shutdown": self._shutdown,
        }

    def _active_window_dict(self) -> dict | None:
        meta = self._cur_meta
        n = len(self._frames)
        if meta.get("index") is None and n == 0:
            return None
        return {
            "index": meta.get("index"),
            "status": (
                "stopped_open" if self._stopped_reason else "collecting"
            ),
            "frames": n,
            "required_frames": MAX_FRAMES,
            "seconds": round(
                sum(f["n_samples"] for f in self._frames)
                / FRAME_RATE, 2) if self._frames else 0.0,
            "started_monotonic": meta.get("started_monotonic"),
            "started_wall": _iso(meta.get("started_wall")),
            "last_frame_monotonic": meta.get("last_frame_monotonic"),
            "last_frame_wall": _iso(meta.get("last_frame_wall")),
            "service_ids": list(meta.get("service_ids") or []),
            "stream_generations": list(
                meta.get("stream_generations") or []),
            "stopped_reason": self._stopped_reason,
        }

    # --------------------------------------------------------------
    # passive HUD / process probes (never on the audio path)
    # --------------------------------------------------------------
    def _hud_probe(self) -> dict | None:
        now = time.monotonic()
        with self._lock:
            if self._hud_probe_cache is not None \
                    and now - self._hud_probe_at < 10.0:
                return self._hud_probe_cache
        probe = self._probe_hud()
        with self._lock:
            self._hud_probe_cache = probe
            self._hud_probe_at = now
        return probe

    @staticmethod
    def _probe_hud() -> dict | None:
        """Read-only HUD state discovery; never modifies the HUD."""
        try:
            import gc
            from src.ui.hologram.hud_window import HologramHUD
            huds = [
                o for o in gc.get_objects()
                if isinstance(o, HologramHUD)
            ]
            if not huds:
                return {"hud_instances": 0}
            h = huds[0]
            orb = getattr(h, "orb", None)
            timer = getattr(orb, "_timer", None) if orb else None
            interval = timer.interval() if timer is not None else None
            return {
                "hud_instances": len(huds),
                "visible": bool(h.isVisible()),
                "target_fps": (
                    round(1000.0 / interval, 1) if interval else None),
                "render_timer_active": (
                    bool(timer.isActive())
                    if timer is not None else None),
                "orb_mode": getattr(orb, "mode", None)
                if orb is not None else None,
            }
        except Exception as exc:  # noqa: BLE001
            return {"error": repr(exc)}

    def _process_snapshot(self) -> dict | None:
        now = time.monotonic()
        with self._lock:
            if self._proc_snapshot is not None \
                    and now - self._proc_at < 10.0:
                return self._proc_snapshot
        snap = self._probe_process()
        with self._lock:
            self._proc_snapshot = snap
            self._proc_at = now
        return snap

    @staticmethod
    def _probe_process() -> dict | None:
        try:
            import psutil
            proc = psutil.Process()
            mem = proc.memory_info()
            return {
                "cpu_percent": proc.cpu_percent(interval=None),
                "memory_rss_bytes": mem.rss,
                "memory_vms_bytes": mem.vms,
                "num_threads": proc.num_threads(),
                "create_time": proc.create_time(),
            }
        except Exception:  # noqa: BLE001
            return None

    def _model_files(self) -> dict:
        with self._lock:
            if self._model_files_cache is not None:
                return self._model_files_cache
        data = _model_hashes()
        with self._lock:
            self._model_files_cache = data
        return data

    # --------------------------------------------------------------
    # shutdown / test reset
    # --------------------------------------------------------------
    def shutdown(self, reason: str = "process_shutdown") -> None:
        """Seal any open window, mark pending analysis interrupted and
        write a final report. Safe to call multiple times."""
        if not enabled():
            return
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        try:
            self._seal_window(reason)
        except Exception:  # noqa: BLE001
            pass
        # Jobs still queued can never be analyzed now: mark them
        # explicitly interrupted rather than leaving them "pending".
        while True:
            try:
                job = self._analysis_queue.get_nowait()
            except queue.Empty:
                break
            entry = job[0]
            done = dict(entry)
            done["status"] = "interrupted"
            done["analysis"] = {
                "status": "interrupted",
                "reason": reason,
                "finished_monotonic": time.monotonic(),
            }
            self._replace_entry(entry, done)
            with self._lock:
                self._analysis_pending -= 1
        try:
            self._write_report()
        except Exception:  # noqa: BLE001
            logger.debug("KWS parity final report write failed",
                         exc_info=True)
        self._writer_event.set()

    def _supersede(self):
        """Test hook: stop writer/worker without a final report write."""
        with self._lock:
            self._superseded = True
            self._shutdown = True
        self._writer_event.set()


# Module-level singleton used by production hooks.
recorder = _Recorder()


def _atexit_shutdown() -> None:
    try:
        if enabled():
            recorder.shutdown()
    except Exception:  # noqa: BLE001
        pass


atexit.register(_atexit_shutdown)


def reset_for_tests() -> None:
    """Replace the singleton with a fresh recorder (test isolation).

    The previous recorder is superseded so its writer exits without
    overwriting the report file.
    """
    global recorder, _ENABLED_CACHE
    _ENABLED_CACHE = None
    try:
        recorder._supersede()
    except Exception:  # noqa: BLE001
        pass
    recorder = _Recorder()


# ----------------------------------------------------------------------
# Analysis helpers (unchanged from V10.7.4)
# ----------------------------------------------------------------------
def _entry_base(meta, frames, events, det_events, cap_stats,
                engine_meta, *, status, analysis, interruption=None):
    """Static portion of a window report entry (never mutated later —
    updates replace the whole dict so concurrent report serialization
    always sees a consistent snapshot)."""
    live = {
        "frames": len(frames),
        "detections_emitted": [
            f["idx"] for f in frames if f.get("emitted")],
        "raw_keyword_frames": [
            {"frame": f["idx"], "keyword": f.get("raw_keyword")}
            for f in frames
            if f.get("raw_keyword") not in (None, "")],
        "threshold_passed_frames": [
            f["idx"] for f in frames if f.get("threshold_passed")],
        "inference_errors": [
            {"frame": f["idx"], "error": f.get("inference_error")}
            for f in frames if f.get("inference_error")],
        "cooldown_blocked": [
            f["idx"] for f in frames if f.get("cooldown_blocked")],
        "on_detected_errors": [
            f["idx"] for f in frames if f.get("on_detected_error")],
    }
    entry = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "env_flag": f"{ENV_FLAG}=1",
        "status": status,
        "window": {
            "index": meta.get("index"),
            "frames": len(frames),
            "seconds": round(
                sum(f["n_samples"] for f in frames) / FRAME_RATE, 2),
            "first_frame_sha256": (
                frames[0]["sha256"] if frames else None),
            "last_frame_sha256": (
                frames[-1]["sha256"] if frames else None),
            "started_monotonic": meta.get("started_monotonic"),
            "started_wall": _iso(meta.get("started_wall")),
            "last_frame_monotonic": meta.get("last_frame_monotonic"),
            "last_frame_wall": _iso(meta.get("last_frame_wall")),
            "service_ids": list(meta.get("service_ids") or []),
            "stream_generations": list(
                meta.get("stream_generations") or []),
        },
        "environment": _environment(),
        "engine": engine_meta,
        "capture": cap_stats,
        "frame_records": frames,
        "engine_events": events,
        "detector_events": det_events,
        "live_summary": live,
        "analysis": analysis,
    }
    if interruption is not None:
        entry["interruption"] = interruption
    return entry


# Attach analysis implementation to the class (kept separate for
# readability; identical logic to V10.7.4).
def _rec_analyze_window(self, widx, frames, audio, events) -> dict:
    live = {
        "frames": len(frames),
        "detections_emitted": [
            f["idx"] for f in frames if f.get("emitted")],
        "raw_keyword_frames": [
            {"frame": f["idx"], "keyword": f.get("raw_keyword")}
            for f in frames
            if f.get("raw_keyword") not in (None, "")],
        "threshold_passed_frames": [
            f["idx"] for f in frames if f.get("threshold_passed")],
        "inference_errors": [
            {"frame": f["idx"], "error": f.get("inference_error")}
            for f in frames if f.get("inference_error")],
        "cooldown_blocked": [
            f["idx"] for f in frames if f.get("cooldown_blocked")],
        "on_detected_errors": [
            f["idx"] for f in frames if f.get("on_detected_error")],
    }

    # ---- replay B: fresh engine, identical frames ---------------
    rep_b = self._replay_plain(frames, audio)
    # ---- replay C: lifecycle replay (generation-matched) --------
    rep_c = self._replay_lifecycle(frames, audio, events)

    first_div = self._first_divergence(frames, rep_b, rep_c)

    stt = self._stt_verify(audio, frames, rep_b)

    return {
        "replay_B_fresh_engine": rep_b["summary"],
        "replay_B_per_frame": rep_b["per_frame"],
        "replay_C_lifecycle": rep_c["summary"],
        "replay_C_per_frame": rep_c["per_frame"],
        "first_divergence": first_div,
        "stt_verification": stt,
        "conclusion": _conclusion(live, rep_b["summary"], stt,
                                 first_div),
    }


_Recorder._analyze_window = _rec_analyze_window


def _rec_replay_plain(self, frames, audio):
    from src.voice.wake_word_engine import create_engine  # local

    eng = create_engine()
    eng._parity_replay = True
    per_frame = []
    try:
        for i, arr in enumerate(audio):
            conf = eng.process(np.asarray(arr, dtype=np.float32))
            kw = eng.keyword_detected() if conf and conf >= 0.25 \
                else None
            per_frame.append({
                "frame": i, "confidence": conf, "keyword": kw,
                "sha256": frames[i]["sha256"]
                if i < len(frames) else None,
            })
    finally:
        eng.close()
    dets = [{"frame": p["frame"], "keyword": p["keyword"]}
            for p in per_frame if p["keyword"]]
    return {
        "summary": {
            "engine": getattr(eng, "name", None),
            "detections": dets,
            "frames": len(per_frame),
        },
        "per_frame": per_frame,
    }


def _rec_replay_lifecycle(self, frames, audio, events):
    from src.voice.wake_word_engine import create_engine  # local

    eng = create_engine()
    eng._parity_replay = True
    # map: applies_before_frame -> count of fresh streams required
    need: dict[int, int] = {}
    for ev in events:
        fidx = ev.get("applies_before_frame")
        if isinstance(fidx, int):
            need[fidx] = need.get(fidx, 0) + 1
    live_gens = [f.get("stream_generation") for f in frames]
    per_frame = []
    resets_applied = 0
    try:
        for i, arr in enumerate(audio):
            # catch up to the live stream generation, then apply
            # any lifecycle events scheduled before this frame
            target = live_gens[i] if i < len(live_gens) else None
            while target is not None:
                cur = getattr(eng, "_stream_generation", None)
                if cur is None or cur >= target:
                    break
                eng.reset()
                if getattr(eng, "_stream_generation", cur) == cur:
                    break  # engine cannot fresh-stream; avoid spin
                resets_applied += 1
            for _ in range(need.get(i, 0)):
                eng.reset()
                resets_applied += 1
            conf = eng.process(np.asarray(arr, dtype=np.float32))
            kw = eng.keyword_detected() if conf and conf >= 0.25 \
                else None
            per_frame.append({
                "frame": i, "confidence": conf, "keyword": kw,
                "stream_generation": getattr(
                    eng, "_stream_generation", None),
                "live_generation": live_gens[i] if i < len(
                    live_gens) else None,
            })
    finally:
        eng.close()
    dets = [{"frame": p["frame"], "keyword": p["keyword"]}
            for p in per_frame if p["keyword"]]
    return {
        "summary": {
            "engine": getattr(eng, "name", None),
            "detections": dets,
            "frames": len(per_frame),
            "resets_applied": resets_applied,
            "engine_events_replayed": len(events),
        },
        "per_frame": per_frame,
    }


_Recorder._replay_plain = _rec_replay_plain
_Recorder._replay_lifecycle = _rec_replay_lifecycle


def _rec_first_divergence(self, frames, rep_b, rep_c):
    """First frame where live and fresh-replay outcomes diverge."""
    bmap = {p["frame"]: p for p in rep_b["per_frame"]}
    cmap = {p["frame"]: p for p in rep_c["per_frame"]}
    for f in frames:
        i = f["idx"]
        live_raw = bool(f.get("raw_keyword"))
        live_emit = bool(f.get("emitted"))
        b = bmap.get(i)
        c = cmap.get(i)
        b_det = bool(b and b.get("keyword"))
        c_det = bool(c and c.get("keyword"))
        if live_raw != b_det:
            return {
                "frame": i,
                "boundary": "engine.process raw get_result",
                "live_raw_keyword": f.get("raw_keyword"),
                "live_raw_keyword_type": f.get("raw_keyword_type"),
                "live_emitted": live_emit,
                "live_cooldown_blocked": f.get("cooldown_blocked"),
                "replay_B_keyword": b.get("keyword") if b else None,
                "replay_C_keyword": c.get("keyword") if c else None,
            }
        if live_emit != b_det and not f.get("cooldown_blocked"):
            return {
                "frame": i,
                "boundary": "application emission",
                "live_raw_keyword": f.get("raw_keyword"),
                "live_emitted": live_emit,
                "replay_B_keyword": b.get("keyword") if b else None,
                "replay_C_keyword": c.get("keyword") if c else None,
            }
    # lifecycle replay divergence (secondary)
    for f in frames:
        i = f["idx"]
        live_emit = bool(f.get("emitted"))
        c = cmap.get(i)
        c_det = bool(c and c.get("keyword"))
        if live_emit != c_det:
            return {
                "frame": i,
                "boundary": "lifecycle replay",
                "live_raw_keyword": f.get("raw_keyword"),
                "live_emitted": live_emit,
                "replay_C_keyword": c.get("keyword") if c else None,
            }
    return None


_Recorder._first_divergence = _rec_first_divergence


def _rec_stt_verify(self, audio, frames, rep_b):
    """Moonshine STT on RMS-gated speech segments (memory only)."""
    if not audio:
        return {"skipped": "no audio"}
    mono = np.concatenate(audio).astype(np.float32)
    hop = FRAME_RATE // 10
    n = len(mono) // hop
    if n == 0:
        return {"skipped": "audio too short"}
    w = mono[: n * hop].reshape(n, hop)
    wrms = np.sqrt(np.mean(w * w, axis=1))
    active = wrms > 0.02
    segs, start, gap = [], None, 0
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            gap = 0
        elif start is not None:
            gap += 1
            if gap > 6:
                segs.append((start, i - gap + 1))
                start, gap = None, 0
    if start is not None:
        segs.append((start, n))
    segs = [(s, e) for s, e in segs if (e - s) >= 3]
    if not segs:
        return {"segments": [], "note": "no speech-like segments"}

    bmap = {p["frame"]: p for p in rep_b["per_frame"]}
    out = []
    try:
        from src.voice.stt_engine import create_recognizer  # local
        rec = create_recognizer()
    except Exception as exc:  # noqa: BLE001
        return {"segments_found": len(segs),
                "error": f"STT unavailable: {exc}"}
    pad = np.zeros(hop, dtype=np.float32)
    for s, e in segs[:12]:
        a, b = s * hop, e * hop
        seg = mono[max(0, a - hop): min(len(mono), b + hop)]
        txt = ""
        try:
            txt = rec.transcribe(np.concatenate([pad, seg, pad]))
        except Exception as exc:  # noqa: BLE001
            txt = f"<stt error: {exc}>"
        # KWS frames overlapping this segment: hop == 1600 samples
        # == one KWS frame, so segment indices map 1:1 to frame idx.
        f0, f1 = s * hop // 1600, e * hop // 1600
        live_kws = [
            {"frame": f["idx"], "keyword": f.get("raw_keyword")}
            for f in frames
            if f0 <= f["idx"] <= f1 and f.get("raw_keyword")]
        rep_kws = [
            {"frame": p["frame"], "keyword": p["keyword"]}
            for p in bmap.values()
            if f0 <= p["frame"] <= f1 and p.get("keyword")]
        out.append({
            "start_s": round(s * 0.1, 1),
            "end_s": round(e * 0.1, 1),
            "dur_s": round((e - s) * 0.1, 1),
            "peak_rms": round(float(wrms[s:e].max()), 4),
            "transcript": txt,
            "live_kws_raw": live_kws,
            "replay_B_kws": rep_kws,
        })
    try:
        rec.close()
    except Exception:  # noqa: BLE001
        pass
    return {"engine": getattr(rec, "name", "stt"),
            "segments": out}


_Recorder._stt_verify = _rec_stt_verify


def _environment() -> dict:
    def ver(pkg):
        try:
            return importlib.metadata.version(pkg)
        except Exception:  # noqa: BLE001
            return None
    return {
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "sherpa_onnx": ver("sherpa-onnx"),
        "numpy": ver("numpy"),
        "sounddevice": ver("sounddevice"),
        "onnxruntime_separate": ver("onnxruntime"),
    }


def _model_hashes() -> dict:
    """SHA-256 of every KWS model file the engine could select."""
    try:
        from src.voice.wake_word_engine import DEFAULT_MODEL_DIR
        d = Path(DEFAULT_MODEL_DIR)
    except Exception:  # noqa: BLE001
        return {"error": "model dir unresolved"}
    out = {"model_dir": str(d), "files": {}}
    for pat in ("encoder*.onnx", "decoder*.onnx", "joiner*.onnx",
                "tokens.txt", "keywords_jarvis.txt", "keywords.txt"):
        matches = sorted(d.rglob(pat))
        for m in matches:
            try:
                h = hashlib.sha256(m.read_bytes()).hexdigest()
                out["files"][m.name] = {
                    "path": str(m), "sha256": h,
                    "size": m.stat().st_size,
                }
            except OSError:
                continue
    return out


def _conclusion(live, rep_b, stt, first_div) -> str:
    emitted = live["detections_emitted"]
    b_dets = rep_b.get("detections", [])
    raw_kws = live["raw_keyword_frames"]
    if emitted and b_dets:
        return ("Detection path consistent in this window: live and "
                "fresh replay agree. Earlier failure not reproduced "
                "here — treat as intermittent/model-level, keep "
                "collecting windows.")
    if raw_kws and not emitted:
        return ("Sherpa emitted a keyword but the application did NOT "
                "propagate it — inspect cooldown_blocked / "
                "on_detected_error fields; this is an app-level defect.")
    if not emitted and b_dets:
        return ("Live missed while fresh replay detected on identical "
                "samples — points to production engine state/stream "
                "history or lifecycle divergence (see first_divergence "
                "and engine_events).")
    if not emitted and not b_dets:
        segs = (stt or {}).get("segments") or []
        confirmed = any(
            "jarvis" in (s.get("transcript") or "").lower()
            or "hello" in (s.get("transcript") or "").lower()
            for s in segs)
        if confirmed:
            return ("Live AND replay missed while STT confirms a wake "
                    "phrase — KWS model-recognition robustness issue, "
                    "not an application pipeline defect.")
        return ("Live and replay both missed and STT does not confirm "
                "a wake phrase in this window — insufficient evidence "
                "of a valid attempt; keep collecting windows.")
    if emitted and not b_dets:
        return ("Live detected but fresh replay did not — unexpected; "
                "inspect frame hashes and lifecycle events.")
    return "Inconclusive — inspect frame_records manually."


# ----------------------------------------------------------------------
# Public hook surface (all no-op when the env flag is unset).
# ----------------------------------------------------------------------
def before_process(service, frame, dequeued_at):
    return recorder.before_process(service, frame, dequeued_at)


def after_process(rec, *, confidence, elapsed, error):
    return recorder.after_process(
        rec, confidence=confidence, elapsed=elapsed, error=error)


def frame_complete(rec):
    return recorder.frame_complete(rec)


def engine_result(engine, *, raw, decode_calls, elapsed):
    return recorder.engine_result(
        engine, raw=raw, decode_calls=decode_calls, elapsed=elapsed)


def on_stream_event(engine, reason: str):
    return recorder.on_stream_event(engine, reason)


def detector_event(service, kind: str):
    return recorder.detector_event(service, kind)


def note_threshold(rec, *, raw_keyword, effective_keyword,
                   cooldown_blocked):
    return recorder.note_threshold(
        rec, raw_keyword=raw_keyword,
        effective_keyword=effective_keyword,
        cooldown_blocked=cooldown_blocked)


def note_on_detected(rec, keyword):
    return recorder.note_on_detected(rec, keyword)


def note_on_detected_error(rec):
    return recorder.note_on_detected_error(rec)


def note_reset(rec):
    return recorder.note_reset(rec)


def record_hud_snapshot(**fields):
    return recorder.record_hud_snapshot(**fields)


def shutdown_diagnostic():
    return recorder.shutdown()


def report_status() -> dict:
    """Return the current diagnostic_status dict (tests/diagnostics)."""
    if not enabled():
        return {"enabled": False, "collection_state": "disabled"}
    return recorder._status_dict()
