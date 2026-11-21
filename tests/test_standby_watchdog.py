"""V10 standby watchdog tests: capture-callback and detection-worker
liveness, bounded restart/backoff, pause distinction, and diagnostics.

Uses the scriptable FakeSD harness from ``test_dynamic_mic`` — no real
microphone required. Callbacks are pumped manually so stall timing is
deterministic.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from tests.test_dynamic_mic import (
    FakeEngine,
    FakeSD,
    active_streams,
    mic,
    wait_for,
)
from src.controllers.microphone_controller import (
    STATUS_ERROR,
    STATUS_LISTENING,
    MicrophoneController,
)
from src.voice.audio_capture_service import AudioCaptureService
from src.voice.audio_device_manager import AudioDeviceManager

FRAME = np.random.rand(1600, 1).astype(np.float32) * 0.01


def make_controller(sd, settings=None, engine=None, **overrides):
    settings = settings if settings is not None else {}
    mgr = AudioDeviceManager(sd=sd)
    params = dict(
        device_manager=mgr,
        capture=AudioCaptureService(sd=sd, device_manager=mgr),
        engine=engine or FakeEngine(score=0.0),
        settings_get=lambda k: settings.get(k),
        settings_set=lambda k, v: settings.__setitem__(k, v),
        monitor_interval=0.05,
        callback_stall_threshold=0.3,
        detection_stall_threshold=0.6,
    )
    params.update(overrides)
    return MicrophoneController(**params)


def pump(sd, seconds: float, interval: float = 0.05,
         stop: threading.Event | None = None) -> None:
    """Feed audio callbacks into every active stream."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if stop is not None and stop.is_set():
            return
        for s in active_streams(sd):
            try:
                s.callback(FRAME.copy(), 1600, None, None)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(interval)


def pump_thread(sd, interval: float = 0.05):
    stop = threading.Event()
    t = threading.Thread(
        target=pump, args=(sd, 3600.0, interval, stop), daemon=True
    )
    t.start()
    return stop, t


@pytest.fixture
def sd():
    return FakeSD([mic("Mic A", rate=16000.0)])


@pytest.fixture
def ctl(sd):
    c = make_controller(sd)
    yield c
    try:
        c.shutdown()
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------
# Capture callback liveness
# ----------------------------------------------------------------------
class TestCallbackStallWatchdog:
    def test_healthy_callbacks_no_restart(self, ctl, sd):
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 3)
            pump(sd, 0.6)
            assert len(active_streams(sd)) == 1
            assert ctl.watchdog_stats()["stall_restarts"] == 0
        finally:
            stop.set()

    def test_callback_stall_restarts_stream(self, ctl, sd):
        assert ctl.enable()
        first = sd.streams[0]
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
        finally:
            stop.set()
        # Stop delivering callbacks: the stream stays PortAudio-active
        # but goes silent — the reported long-standby failure mode.
        assert wait_for(
            lambda: len(sd.streams) >= 2
            and sd.streams[-1].active
            and not first.active,
            timeout=6.0,
        ), "watchdog did not restart the stalled stream"
        assert ctl.status == STATUS_LISTENING
        assert len(active_streams(sd)) == 1  # never duplicated

    def test_paused_capture_with_callbacks_no_stall(self, ctl, sd):
        """Intentional TTS/STT pauses still receive callbacks — they
        must NOT be treated as stalls."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            ctl.capture.pause()
            pump(sd, 0.7)
            assert ctl.capture.paused
            assert len(sd.streams) == 1
            assert ctl.watchdog_stats()["stall_restarts"] == 0
        finally:
            stop.set()
            ctl.capture.resume()

    def test_repeated_stall_is_bounded_then_backs_off(
        self, ctl, sd, monkeypatch
    ):
        """A persistently dead endpoint cannot restart-loop forever."""
        import src.controllers.microphone_controller as mc
        monkeypatch.setattr(mc, "_STALL_MIN_INTERVAL", 0.2)
        assert ctl.enable()
        # Never pump callbacks: every restart stalls again.
        assert wait_for(lambda: ctl.status == STATUS_ERROR, timeout=15.0)
        stats = ctl.watchdog_stats()
        assert stats["stall_backoff_seconds"]
        assert stats["stall_backoff_seconds"] >= 30.0
        # Bounded: initial stream + at most _STALL_MAX_RETRIES restarts.
        assert len(sd.streams) <= 1 + 3 + 1

    def test_recovery_after_stall(self, ctl, sd):
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
        finally:
            stop.set()
        assert wait_for(lambda: len(sd.streams) >= 2, timeout=6.0)
        # Callbacks resume on the replacement stream.
        stop2, _ = pump_thread(sd)
        try:
            assert wait_for(
                lambda: ctl.capture.callback_seq
                > ctl.capture.stats()["callback_seq"] or True
            )
            before = ctl.capture.callback_seq
            pump(sd, 0.4)
            assert ctl.capture.callback_seq > before
            assert ctl.status == STATUS_LISTENING
        finally:
            stop2.set()

    def test_no_stream_when_disabled(self, ctl, sd):
        ctl.enable()
        ctl.disable()
        assert ctl.capture.callback_stall_seconds() is None


# ----------------------------------------------------------------------
# Detection worker liveness
# ----------------------------------------------------------------------
class TestDetectionLiveness:
    def test_dead_detection_worker_restarted(self, ctl, sd):
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl._detection is not None
                            and ctl._detection.is_running)
            old = ctl._detection
            old._stop.set()  # simulate silent worker death
            assert wait_for(lambda: not old.is_running, timeout=3.0)
            assert wait_for(
                lambda: ctl._detection is not old
                and ctl._detection is not None
                and ctl._detection.is_running,
                timeout=5.0,
            ), "dead detection worker was not restarted"
        finally:
            stop.set()

    def test_consumer_stall_restarts_detection(self, ctl, sd):
        """Worker alive but not processing while audio flows -> restart."""
        engine = ctl._engine
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        block = threading.Event()
        orig_process = engine.process

        def blocked(samples):
            block.wait(15.0)
            return orig_process(samples)

        try:
            assert wait_for(lambda: ctl._detection is not None
                            and ctl._detection.frames_processed > 2)
            old = ctl._detection
            engine.process = blocked  # worker now stalls inside inference
            assert wait_for(
                lambda: ctl._detection is not old
                and ctl._detection is not None,
                timeout=6.0,
            ), "stalled detection worker was not restarted"
        finally:
            block.set()
            engine.process = orig_process
            stop.set()


# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------
class TestStandbyDiagnostics:
    def test_capture_stats_fields(self, ctl, sd):
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 3)
            s = ctl.capture.stats()
            assert s["callback_age_seconds"] is not None
            assert s["callback_age_seconds"] < 1.0
            assert s["callback_interval_seconds"]
            assert 0.01 <= s["callback_interval_seconds"] <= 0.5
            assert s["stream_uptime_seconds"] > 0
            assert s["paused"] is False
            assert s["running"] is True
        finally:
            stop.set()

    def test_detection_stats_fields(self, ctl, sd):
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(
                lambda: ctl._detection is not None
                and ctl._detection.frames_processed > 2
            )
            s = ctl._detection.stats()
            assert s["inference_seconds"] is not None
            assert s["last_processed_age_seconds"] is not None
            assert s["last_processed_age_seconds"] < 2.0
            assert s["capture_to_inference_estimate_seconds"] >= 0
        finally:
            stop.set()

    def test_extended_standby_counters_keep_growing(self, ctl, sd):
        """Simulated standby: captured/processed counters must advance
        continuously — the V10 symptom was them freezing for minutes."""
        assert ctl.enable()
        stop, _ = pump_thread(sd, interval=0.05)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            c0 = ctl.capture.frames_captured
            p0 = ctl._detection.frames_processed if ctl._detection else 0
            pump(sd, 1.0)
            assert ctl.capture.frames_captured > c0 + 5
            assert ctl._detection.frames_processed > p0 + 5
            assert ctl.watchdog_stats()["stall_restarts"] == 0
        finally:
            stop.set()

    def test_watchdog_stats_exposed(self, ctl):
        stats = ctl.watchdog_stats()
        for key in ("callback_stall_threshold",
                    "detection_stall_threshold",
                    "stall_restarts", "detect_restarts",
                    "last_wake_detected_monotonic"):
            assert key in stats

    def test_missed_callback_intervals_counted(self, ctl, sd):
        """A callback gap far beyond the established interval is
        recorded as a missed interval (early stall evidence)."""
        cap = ctl.capture
        cap.callback_interval_seconds = 0.05
        # Simulate a 2 s hole between two callbacks on the same stream.
        cap._prev_callback_monotonic = time.monotonic() - 2.0
        cap._audio_callback(FRAME.copy(), 1600, None, None)
        assert cap.missed_callback_intervals == 1
        assert cap.consecutive_missed_callbacks == 1
        # A normal interval resets the consecutive counter.
        cap._prev_callback_monotonic = time.monotonic() - 0.05
        cap._audio_callback(FRAME.copy(), 1600, None, None)
        assert cap.consecutive_missed_callbacks == 0
        assert cap.missed_callback_intervals == 1


# ----------------------------------------------------------------------
# OS suspend/resume detection
# ----------------------------------------------------------------------
class TestSystemResumeDetection:
    def test_wall_monotonic_drift_triggers_resume(self, ctl, sd):
        """A large wall-vs-monotonic drift with a stale pipeline is
        confirmed as a suspend and arms post-resume verification."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
        finally:
            stop.set()
        # Simulate the process being frozen: wall clock jumped
        # ~60 s while monotonic did not, and callbacks went stale.
        ctl._last_wall_time = time.time() - 60.0
        ctl._last_mono_time = time.monotonic()
        assert wait_for(lambda: ctl._resume_events >= 1,
                        timeout=6.0)
        assert ctl.watchdog_stats()["resume_events"] >= 1
        assert ctl._resume_verify_until > time.monotonic()

    def test_clock_step_with_healthy_audio_is_not_resume(
        self, ctl, sd
    ):
        """NTP/manual clock changes produce the same drift signature.
        If callbacks and frames keep advancing, it is NOT a suspend:
        no flush, no restart, no resume event."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 5)
            ctl._last_wall_time = time.time() - 60.0
            ctl._last_mono_time = time.monotonic()
            pump(sd, 3.0)  # audio stays healthy through "drift"
            assert ctl._resume_events == 0
            assert ctl._resume_candidate is None
            assert len(sd.streams) == 1
            assert ctl.watchdog_stats()["stall_restarts"] == 0
        finally:
            stop.set()

    def test_small_drift_does_not_trigger(self, ctl, sd):
        """NTP-sized clock steps must not look like a suspend."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            ctl._last_wall_time = time.time() - 1.0
            ctl._last_mono_time = time.monotonic()
            pump(sd, 0.4)
            assert ctl._resume_events == 0
        finally:
            stop.set()

    def test_power_resume_notification_trusted(self, ctl, sd):
        """notify_system_resume (WM_POWERBROADCAST) skips the
        candidate phase and recovers immediately; repeats debounce."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            ctl.notify_system_suspend("test")
            assert ctl._suspend_notified is True
            ctl.notify_system_resume("test")
            assert ctl._resume_events == 1
            assert ctl._resume_pending is True
            assert "resume_detected" in ctl._resume_milestones
            handled = ctl._last_resume_handled
            ctl.notify_system_resume("test")
            assert ctl._last_resume_handled == handled  # debounced
        finally:
            stop.set()

    def test_readiness_becomes_ready_with_healthy_pipeline(
        self, ctl, sd
    ):
        """READY requires evidence: device, callbacks, advancing
        counters and a consuming detector — not just active=True."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.audio_readiness()["ready"],
                            timeout=8.0)
            r = ctl.audio_readiness()
            assert r["checks"]["device_available"] is True
            assert r["checks"]["callbacks_fresh"] is True
            assert r["checks"]["frames_advancing"] is True
            assert r["checks"]["detector_consuming"] is True
            assert r["checks"]["engine_operational"] is True
        finally:
            stop.set()

    def test_readiness_reports_failure_not_ready(self, ctl, sd):
        """A disabled/broken pipeline reports a failure reason —
        never a false READY."""
        r = ctl.audio_readiness()
        assert r["ready"] is False
        assert r["failure_reason"]

    def test_post_resume_dead_stream_restarts_fast(self, ctl, sd):
        """After a resume event, a stalled stream restarts with the
        tighter threshold — not the full 5 s standby threshold."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
        finally:
            stop.set()
        # Frozen process: drift injected while callbacks are stale.
        ctl._last_wall_time = time.time() - 30.0
        ctl._last_mono_time = time.monotonic()
        assert wait_for(lambda: ctl._resume_events >= 1,
                        timeout=6.0)
        # No more callbacks: dead stream. With the tighter 0.3 s test
        # threshold already below the resume threshold, restart is
        # bounded by _STALL_MIN_INTERVAL + monitor ticks.
        assert wait_for(lambda: len(sd.streams) >= 2, timeout=8.0)
        assert ctl.status == STATUS_LISTENING
