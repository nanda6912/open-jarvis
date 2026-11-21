"""V10.8 async rescore regressions.

Root cause under test: activity-window rescoring previously ran
*synchronously on the detection worker* — a multi-second anchor sweep
stopped live KWS inference, backlogged the capture queue (observed
depth ~= 37) and made wake-word recognition position-sensitive.

These tests pin the async contract:

* rescore executes on a dedicated worker; live ``engine.process()``
  keeps consuming frames while a rescore job is in flight.
* a completed rescore hit still travels the normal ``on_detected``
  callback path (same thread as live detections).
* results completed after ``stop()`` are discarded, never emitted.
* the rescore queue is bounded: a full queue rejects the job instead
  of blocking the detection worker.
* ``stop()`` leaves no rescore thread behind.
"""

from __future__ import annotations

import threading
import time

import numpy as np

from src.voice.wake_word_detection_service import WakeWordDetectionService

from tests.test_dynamic_mic import FakeEngine, wait_for
from tests.test_v1075_reliability import (
    CHUNK,
    SR,
    _QueueCapture,
    _RescoringEngine,
    _feed,
    _silence_frame,
    _speech_frame,
)


class _GatedRescoreEngine(FakeEngine):
    """Live stream always misses; rescore blocks until released.

    ``inflight`` is set when the rescore worker enters ``rescore()``;
    ``release`` must be set for it to return. Models the multi-second
    anchor sweeps observed in production.
    """

    def __init__(self, rescore_keyword="HELLO JARVIS", gate_timeout=8.0):
        super().__init__(score=0.0)
        self.rescore_keyword = rescore_keyword
        self.rescore_calls = 0
        self.inflight = threading.Event()
        self.release = threading.Event()
        self._gate_timeout = gate_timeout

    def rescore(self, samples, lead_seconds=(0.5,), trail_seconds=0.5,
                **kwargs):
        self.rescore_calls += 1
        self.inflight.set()
        self.release.wait(timeout=self._gate_timeout)
        if self.rescore_keyword is None:
            return None
        return self.rescore_keyword, float(lead_seconds[0])


def _service(cap, engine, detected=None, **kw):
    return WakeWordDetectionService(
        capture=cap,
        engine=engine,
        threshold=0.25,
        cooldown_seconds=0.0,
        on_detected=detected.append if detected is not None else None,
        **kw,
    )


def _close_one_window(cap):
    """Feed speech + hangover silence so one activity window closes."""
    _feed(cap, [_speech_frame() for _ in range(6)])
    _feed(cap, [_silence_frame() for _ in range(20)])


def test_rescore_does_not_block_live_frame_processing():
    """While a rescore job is in flight, the detection worker must keep
    consuming frames and calling engine.process() — the V10.8 fix."""
    cap = _QueueCapture()
    engine = _GatedRescoreEngine()
    detected: list = []
    svc = _service(cap, engine, detected)
    try:
        svc.start()
        _close_one_window(cap)
        assert wait_for(lambda: engine.inflight.is_set(), timeout=10.0), (
            "rescore job never started on the rescore worker"
        )
        # Rescore is blocked inside engine.rescore(); live inference
        # must continue unaffected.
        proc_before = engine.processed
        frames_before = svc.frames_processed
        _feed(cap, [_silence_frame() for _ in range(15)])
        assert wait_for(
            lambda: engine.processed >= proc_before + 10, timeout=10.0
        ), "live engine.process() starved during rescore"
        assert svc.frames_processed > frames_before
        assert not detected, (
            "rescore hit emitted while the job was still gated"
        )
        engine.release.set()
        assert wait_for(lambda: len(detected) >= 1, timeout=10.0)
        assert detected[0] == "HELLO JARVIS"
        assert wait_for(
            lambda: svc.last_activity_window is not None
            and svc.last_activity_window.get("rescore_succeeded")
            is not None,
            timeout=10.0,
        )
        win = svc.last_activity_window
        assert win["rescored"] is True
        assert win["detection_method"] == "rescore"
    finally:
        engine.release.set()
        svc.stop()


def test_rescore_hit_emitted_only_after_window_close():
    """Ordering: the activity window closes first (record persisted),
    then the async rescore hit reaches on_detected."""
    cap = _QueueCapture()
    engine = _GatedRescoreEngine()
    detected: list = []
    svc = _service(cap, engine, detected)
    try:
        svc.start()
        _close_one_window(cap)
        assert wait_for(
            lambda: svc.last_activity_window is not None, timeout=10.0
        )
        closed = svc.last_activity_window
        assert closed["keyword_detected"] is False
        assert closed["rescore_scheduled"] is True
        # Give the rescore worker a moment: even if it finished, the
        # hit cannot be emitted before the window record exists.
        assert wait_for(lambda: engine.inflight.is_set(), timeout=10.0)
        engine.release.set()
        assert wait_for(lambda: len(detected) >= 1, timeout=10.0)
        assert svc.last_activity_window is closed
        assert closed["rescored"] is True
        assert closed["keyword_detected"] is True
    finally:
        engine.release.set()
        svc.stop()


def test_rescore_result_after_stop_is_discarded():
    """A rescore completing during/after stop() must not emit a
    detection or resurrect the worker."""
    cap = _QueueCapture()
    engine = _GatedRescoreEngine()
    detected: list = []
    svc = _service(cap, engine, detected)
    svc.start()
    _close_one_window(cap)
    assert wait_for(lambda: engine.inflight.is_set(), timeout=10.0)
    # Let the gated rescore finish shortly after stop begins; the
    # result lands in _rescore_results but the detection worker is
    # already stopping, so _drain_rescore_results drops it as stale.
    timer = threading.Timer(0.2, engine.release.set)
    timer.start()
    svc.stop()
    timer.join(timeout=5.0)
    time.sleep(0.3)
    assert not detected, "stale rescore result emitted a detection"
    assert svc._rescore_thread is None
    assert not svc.is_running


def test_rescore_queue_full_rejects_without_blocking():
    """A saturated rescore queue rejects the job immediately — the
    detection worker never waits on rescore capacity."""
    cap = _QueueCapture()
    svc = _service(cap, _RescoringEngine())
    try:
        # Prime window audio directly (service never started).
        n = int(0.8 * SR)
        svc._window_audio.append(
            np.full(n, 0.05, dtype=np.float32)
        )
        svc._window_audio_samples = n
        svc._diag_buffer.clear()
        svc._diag_buffer_samples = 0
        job = svc._prepare_rescore_job(1, time.monotonic())
        assert job is not None
        # Saturate the bounded queue.
        while True:
            try:
                svc._rescore_queue.put_nowait(object())
            except Exception:
                break
        rec = {"keyword_detected": False, "rescored": False}
        job["record"] = rec
        assert svc._enqueue_rescore_job(job) is False
        assert svc._rescore_rejected == 1
        assert rec["rescore_attempted"] is False
        assert rec["rescore_scheduled"] is False
        assert rec["detection_method"] == "none"
    finally:
        svc.stop()


def test_stop_leaves_no_rescore_worker():
    cap = _QueueCapture()
    svc = _service(cap, _RescoringEngine())
    svc.start()
    assert svc.stats()["rescore_worker_alive"] is True
    svc.stop()
    assert svc._rescore_thread is None
    assert svc.stats()["rescore_worker_alive"] is False
    assert svc._rescore_queue.empty()


def _prime_rescore_job(svc):
    """Build a real rescore job without starting the service."""
    n = int(0.8 * SR)
    svc._window_audio.append(np.full(n, 0.05, dtype=np.float32))
    svc._window_audio_samples = n
    svc._diag_buffer.clear()
    svc._diag_buffer_samples = 0
    job = svc._prepare_rescore_job(1, time.monotonic())
    assert job is not None
    return job


def test_expired_queued_job_aborts_before_inference():
    """2026-11-20 anomaly: a job whose deadline passed while queued
    (e.g. across a ~7 min suspend) must abort BEFORE any
    ``engine.rescore()`` call — deterministic, no threads."""
    cap = _QueueCapture()
    engine = _RescoringEngine()
    svc = _service(cap, engine)
    try:
        job = _prime_rescore_job(svc)
        # Simulate the job sitting in the queue past its TTL.
        job["deadline_monotonic"] = time.monotonic() - 1.0
        svc._execute_rescore_job(job)
        assert engine.rescore_calls == 0, (
            "expired job still ran engine inference"
        )
        report = svc._last_rescore_report
        assert report["aborted"] is True
        assert report["candidates_attempted"] == 0
        assert report["succeeded"] is False
    finally:
        svc.stop()


def test_zero_valued_callbacks_do_not_block_recovery():
    """2026-11-20 anomaly: ~60 min of callbacks delivering exact-zero
    buffers (rms=0, peak=0, counters still increasing) followed by
    normal speech. The pipeline must keep consuming frames and a
    later activity window must still close, rescore and detect —
    digital silence is just quiet audio, not a wedge."""
    cap = _QueueCapture()
    engine = _RescoringEngine()
    detected: list = []
    svc = _service(cap, engine, detected)
    try:
        svc.start()
        # ~30 s of exact-zero callbacks (all channels digital silence).
        _feed(cap, [_silence_frame() for _ in range(300)])
        assert svc.frames_processed > 0
        # Real speech returns: window opens, closes on hangover, the
        # rescore worker recovers the wake hit.
        _close_one_window(cap)
        assert wait_for(lambda: len(detected) >= 1, timeout=10.0)
        assert detected[0] == "HELLO JARVIS"
        assert wait_for(
            lambda: svc.last_activity_window is not None
            and svc.last_activity_window.get("rescored") is True,
            timeout=10.0,
        )
    finally:
        svc.stop()


def test_stop_returns_while_rescore_inflight():
    """stop() must not hang on a stalled rescore. NOTE: the abort is
    cooperative — ``should_abort`` is only checked between anchors and
    100 ms decode chunks, and a native ``decode_stream`` call that
    never returns cannot be interrupted or killed. The bounded
    ``join(timeout=2.0)`` plus daemon thread is the actual guarantee;
    this pins that contract only."""
    cap = _QueueCapture()
    engine = _GatedRescoreEngine(gate_timeout=30.0)
    svc = _service(cap, engine)
    try:
        svc.start()
        _close_one_window(cap)
        assert wait_for(lambda: engine.inflight.is_set(), timeout=10.0)
        t0 = time.monotonic()
        svc.stop()
        elapsed = time.monotonic() - t0
        assert elapsed < 10.0, f"stop() blocked {elapsed:.1f}s"
        assert svc._rescore_thread is None
        # The OS thread may still be inside the fake's gate wait; it
        # is a daemon and its late result is discarded.
        engine.release.set()
    finally:
        engine.release.set()
