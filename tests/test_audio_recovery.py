"""Phase 1 Task 5 — audio recovery & watchdog hardening tests.

Deterministic coverage for the stream-health invariants:

- at most one active capture stream / one recovery owner at a time
- bounded retries with the existing backoff policy
- no restart of a disabled/shut-down microphone by stale paths
- no pre-restart audio or consumer state leaking into a new cycle
- intentional TTS pauses are not mistaken for stream failure
- suspend/resume and disable races cannot resurrect audio

Uses the scriptable FakeSD harness — callbacks are pumped manually.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import src.controllers.microphone_controller as mc
from src.controllers.microphone_controller import (
    STATUS_DISABLED,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_LISTENING,
    MicrophoneController,
)
from src.voice.audio_capture_service import AudioCaptureService
from src.voice.audio_device_manager import AudioDeviceManager
from tests.test_dynamic_mic import (
    FakeEngine,
    FakeSD,
    active_streams,
    mic,
    wait_for,
)
from tests.test_standby_watchdog import (
    FRAME,
    make_controller,
    pump,
    pump_thread,
)


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
# enable() lifecycle races
# ----------------------------------------------------------------------
class TestEnableLifecycle:
    def test_enable_during_active_session_is_noop(self, ctl, sd):
        """enable() while a voice session owns the queue (IDLE) must not
        install a second detection consumer."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            ctl.suspend_detection()
            assert ctl.status == STATUS_IDLE
            assert ctl._detection is None
            assert ctl.enable() is True
            assert ctl.status == STATUS_IDLE
            assert ctl._detection is None
        finally:
            stop.set()

    def test_enable_from_error_stops_orphaned_detection(self, ctl, sd):
        """A capture error leaves detection alive; re-enabling must stop
        the old worker instead of orphaning a second queue consumer."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(
                lambda: ctl._detection is not None
                and ctl._detection.is_running
            )
            old = ctl._detection
            ctl._handle_capture_error(RuntimeError("boom"))
            assert ctl.status == STATUS_ERROR
            assert ctl.enable() is True
            assert ctl._detection is not old
            assert not old.is_running
            assert ctl._detection.is_running
            assert ctl.status == STATUS_LISTENING
            assert len(active_streams(sd)) == 1
        finally:
            stop.set()


# ----------------------------------------------------------------------
# Detection-worker recovery
# ----------------------------------------------------------------------
class TestDetectionRecovery:
    def test_failed_detection_restart_is_retried(
        self, ctl, sd, monkeypatch
    ):
        """A failed _restart_detection leaves _detection=None; the
        monitor must rebuild it instead of staying deaf forever."""
        monkeypatch.setattr(mc, "_DETECTION_MISSING_GRACE", 0.3)
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(
                lambda: ctl._detection is not None
                and ctl._detection.is_running
            )
            orig = ctl._new_detection_service

            def boom(generation, on_first_frame=None):
                raise RuntimeError("factory exploded")

            ctl._new_detection_service = boom
            try:
                ctl._restart_detection("forced failure")
            finally:
                ctl._new_detection_service = orig
            assert ctl._detection is None
            assert ctl.status == STATUS_LISTENING
            assert wait_for(
                lambda: ctl._detection is not None
                and ctl._detection.is_running,
                timeout=5.0,
            ), "missing detection worker was never rebuilt"
        finally:
            stop.set()

    def test_missing_detection_does_not_fire_during_suspend(
        self, ctl, sd, monkeypatch
    ):
        """The legitimate suspend detach window (status still
        LISTENING, _detection None) must not trigger a spurious
        replacement worker mid-suspend."""
        monkeypatch.setattr(mc, "_DETECTION_MISSING_GRACE", 0.3)
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl._detection is not None)
            ctl.suspend_detection()
            assert ctl.status == STATUS_IDLE
            # Give the monitor several ticks: the missing-detector
            # path must stay disarmed outside LISTENING.
            pump(sd, 0.5)
            assert ctl._detection is None
            assert ctl._detection_none_since == 0.0
        finally:
            stop.set()


# ----------------------------------------------------------------------
# Capture-stream recovery from ERROR
# ----------------------------------------------------------------------
class TestErrorStateRecovery:
    def test_error_status_dead_stream_recovers(self, ctl, sd):
        """Callback error -> ERROR, then the PortAudio stream dies:
        the watchdog must restart it instead of staying deaf."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            ctl._handle_capture_error(RuntimeError("callback blew up"))
            assert ctl.status == STATUS_ERROR
            sd.streams[-1].active = False  # silent stream death
            assert wait_for(
                lambda: ctl.status == STATUS_LISTENING
                and len(active_streams(sd)) == 1,
                timeout=6.0,
            ), "dead stream in ERROR status was not recovered"
            assert len(sd.streams) == 2  # exactly one replacement
        finally:
            stop.set()

    def test_error_status_live_stream_stays_put(self, ctl, sd):
        """ERROR with a healthy stream must not flap into restarts
        every monitor tick."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            ctl._handle_capture_error(RuntimeError("transient"))
            n_streams = len(sd.streams)
            pump(sd, 0.5)
            assert len(sd.streams) == n_streams
        finally:
            stop.set()


# ----------------------------------------------------------------------
# Restart safety: stale callbacks, queue/state reset, disable races
# ----------------------------------------------------------------------
class TestRestartIsolation:
    def test_stale_stream_callback_dropped_after_restart(self, ctl, sd):
        """A callback from the replaced stream must not feed the
        shared queue (stream-generation guard)."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            old_stream = sd.streams[-1]
            ctl._restart_for_selection("device swap")
            assert len(active_streams(sd)) == 1
            cap = ctl.capture
            cap.flush()
            seq0 = cap.callback_seq
            q0 = cap._queue.qsize()
            old_stream.callback(FRAME.copy(), 1600, None, None)
            assert cap.callback_seq == seq0
            assert cap._queue.qsize() == q0
        finally:
            stop.set()

    def test_stale_stream_callback_dropped_after_stop(self, ctl, sd):
        """Callbacks in flight while stop() runs cannot enqueue frames
        that would leak into the next start()."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            s = sd.streams[-1]
            cap = ctl.capture
            cap.stop()
            cap.flush()
            seq0 = cap.callback_seq
            s.callback(FRAME.copy(), 1600, None, None)
            assert cap.callback_seq == seq0
            assert cap._queue.qsize() == 0
        finally:
            stop.set()
            ctl.enable()

    def test_restart_resets_engine_and_stream_consumers(
        self, ctl, sd
    ):
        """A capture restart must reset the KWS decoder stream and
        notify session-scoped consumers (STT VAD/utterance) so
        pre-restart audio cannot leak into the new cycle."""
        engine = ctl._engine
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            engine.reset_count = 0
            calls = []
            ctl.set_resume_listener(lambda: calls.append(1))
            ctl._restart_for_selection("test restart")
            assert engine.reset_count == 1
            assert calls == [1]
            # Queue was flushed: no pre-restart frames remain.
            assert ctl.capture._queue.qsize() == 0
        finally:
            stop.set()

    def test_concurrent_restart_requests_single_stream(
        self, ctl, sd
    ):
        """Two recovery paths racing must serialize on the restart
        lock: exactly one active stream, no exceptions."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        errors = []
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)

            def go():
                try:
                    ctl._restart_for_selection("racer")
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            t1 = threading.Thread(target=go)
            t2 = threading.Thread(target=go)
            t1.start(); t2.start()
            t1.join(10.0); t2.join(10.0)
            assert not errors
            assert len(active_streams(sd)) == 1
            assert ctl.status == STATUS_LISTENING
        finally:
            stop.set()

    def test_disable_during_restart_leaves_no_stream(self, ctl, sd):
        """disable() racing an in-flight restart must end with no
        active stream and DISABLED status — the restart must not
        reopen capture afterwards."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            entered = threading.Event()
            release = threading.Event()
            orig_start = ctl._capture.start

            def slow_start(*a, **kw):
                entered.set()
                release.wait(10.0)
                return orig_start(*a, **kw)

            ctl._capture.start = slow_start
            result = {}

            def restart():
                result["ok"] = ctl._restart_for_selection("racer")

            t = threading.Thread(target=restart)
            t.start()
            assert entered.wait(5.0)
            ctl.disable()
            release.set()
            t.join(10.0)
            assert ctl.status == STATUS_DISABLED
            assert not ctl.capture.is_running
            assert len(active_streams(sd)) == 0
            assert ctl._detection is None
        finally:
            release.set()
            stop.set()

    def test_restart_after_disable_is_ignored(self, ctl, sd):
        """A stale watchdog/restart call after disable() must abort
        instead of reopening the microphone."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            ctl.disable()
            assert ctl._restart_for_selection("stale watchdog") is False
            assert ctl.status == STATUS_DISABLED
            assert not ctl.capture.is_running
            assert len(active_streams(sd)) == 0
        finally:
            stop.set()

    def test_suspend_during_disable_does_not_resurrect_idle(
        self, ctl, sd
    ):
        """suspend_detection() completing after disable() must not
        flip the controller back to IDLE."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            ctl.disable()
            assert ctl.status == STATUS_DISABLED
            ctl.suspend_detection()
            assert ctl.status == STATUS_DISABLED
            assert ctl._detection is None
        finally:
            stop.set()

    def test_resume_detection_after_disable_is_noop(self, ctl, sd):
        """A stale standby handover after disable() must not create a
        detector or flip status to LISTENING."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            ctl.suspend_detection()
            assert ctl.status == STATUS_IDLE
            ctl.disable()
            assert ctl.status == STATUS_DISABLED
            ctl.resume_detection()
            assert ctl.status == STATUS_DISABLED
            assert ctl._detection is None
            assert not ctl.capture.is_running
        finally:
            stop.set()


# ----------------------------------------------------------------------
# Intentional pauses and suspend/resume cycles
# ----------------------------------------------------------------------
class TestPauseAndResume:
    def test_tts_pause_with_callbacks_no_stall_no_restart(
        self, ctl, sd
    ):
        """Paused capture still receives callbacks: the watchdog must
        not treat the intentional TTS pause as a stream failure."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            ctl.capture.pause()
            pump(sd, 0.6)
            assert ctl.watchdog_stats()["stall_restarts"] == 0
            assert len(sd.streams) == 1
            assert ctl.status == STATUS_LISTENING
        finally:
            stop.set()
            ctl.capture.resume()

    def test_repeated_suspend_resume_single_stream_single_worker(
        self, ctl, sd
    ):
        """Session handover cycles must never duplicate the capture
        stream or orphan detection workers."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            for _ in range(3):
                assert wait_for(
                    lambda: ctl._detection is not None
                    and ctl._detection.is_running
                )
                ctl.suspend_detection()
                assert ctl.status == STATUS_IDLE
                ctl.resume_detection()
                assert ctl.status == STATUS_LISTENING
                assert len(active_streams(sd)) == 1
            # Exactly one live detection worker, no orphans.
            det = ctl._detection
            assert det is not None and det.is_running
            pump(sd, 0.3)
            assert len(active_streams(sd)) == 1
        finally:
            stop.set()

    def test_os_resume_flush_and_single_stream(self, ctl, sd):
        """notify_system_resume drops stale queued audio and keeps a
        single stream/worker — no duplicate recovery."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl.capture.callback_seq > 2)
            ctl.notify_system_suspend("test")
            ctl.notify_system_resume("test")
            assert ctl._resume_events == 1
            assert ctl.capture._queue.qsize() == 0
            pump(sd, 0.3)
            assert len(active_streams(sd)) == 1
            assert ctl.status == STATUS_LISTENING
        finally:
            stop.set()


# ----------------------------------------------------------------------
# Standby handover: surviving-detector safety
# ----------------------------------------------------------------------
class TestResumeDetectionSafety:
    def test_resume_reuses_surviving_live_detector(self, ctl, sd):
        """resume_detection must not replace a detector that survived
        suspension — a second worker would orphan a duplicate consumer
        of the shared frame queue."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl._detection is not None
                            and ctl._detection.is_running)
            ctl.suspend_detection()
            assert ctl.status == STATUS_IDLE
            # Simulate a detector that survived the suspend detach.
            survivor = ctl._new_detection_service(
                ctl._detection_generation + 100
            )
            survivor.start()
            ctl._detection = survivor
            ctl.resume_detection()
            assert ctl.status == STATUS_LISTENING
            assert ctl._detection is survivor
            assert survivor.is_running
            # Exactly one live detector; no replacement was spawned.
            pump(sd, 0.3)
            assert len(active_streams(sd)) == 1
        finally:
            stop.set()

    def test_resume_replaces_dead_detector_safely(self, ctl, sd):
        """A corpse left in _detection is stopped before the new
        standby worker is installed — never an orphan."""
        assert ctl.enable()
        stop, _ = pump_thread(sd)
        try:
            assert wait_for(lambda: ctl._detection is not None
                            and ctl._detection.is_running)
            ctl.suspend_detection()
            assert ctl.status == STATUS_IDLE
            corpse = ctl._new_detection_service(
                ctl._detection_generation + 100
            )
            corpse.start()
            corpse._stop.set()  # simulate dead worker
            assert wait_for(lambda: not corpse.is_running, timeout=3.0)
            ctl._detection = corpse
            ctl.resume_detection()
            assert ctl.status == STATUS_LISTENING
            assert ctl._detection is not corpse
            assert ctl._detection.is_running
        finally:
            stop.set()
