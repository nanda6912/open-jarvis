"""Wake-word detector ownership tests (Task 8).

Invariant under test: **at most one live WakeWordDetectionService
worker may consume the shared audio frame queue at any time.**

A detector must never be replaced merely because its stop()/join
timeout expired. While a retired worker may still be alive, no second
consumer may be installed; the controller fails closed and the
bounded watchdog retries once the survivor actually exits.

All races are reproduced deterministically with a fake detector whose
``is_running`` is an explicit event — no timing sleeps.
"""

from __future__ import annotations

import threading
import time

import src.controllers.microphone_controller as mc_mod
from src.controllers.microphone_controller import (
    STATUS_DISABLED,
    STATUS_IDLE,
    STATUS_LISTENING,
    MicrophoneController,
)
from src.voice.audio_capture_service import AudioCaptureService
from src.voice.audio_device_manager import AudioDeviceManager

from tests.test_dynamic_mic import FakeEngine, FakeSD, mic


class FakeDetector:
    """Scriptable stand-in for WakeWordDetectionService.

    ``is_running`` is backed by an explicit Event: ``stop()`` clears
    it immediately (clean exit) unless ``survive`` is set, modelling a
    worker that outlived its join timeout. ``terminate()`` models the
    worker finally exiting on its own.
    """

    instances: list["FakeDetector"] = []

    def __init__(self, capture=None, engine=None, threshold=0.25,
                 cooldown_seconds=3.0, on_detected=None, on_stats=None,
                 on_first_frame=None, generation=0, **_kw):
        self._generation = generation
        self.on_detected = on_detected
        self.on_first_frame = on_first_frame
        self._alive = threading.Event()
        self._survive = False
        self.stop_calls = 0
        self.frames_processed = 0
        FakeDetector.instances.append(self)

    # -- lifecycle -----------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._alive.is_set()

    def start(self):
        self._alive.set()
        if self.on_first_frame is not None:
            self.on_first_frame()

    def stop(self):
        self.stop_calls += 1
        if not self._survive:
            self._alive.clear()

    def suspend(self):
        self.stop()

    def terminate(self):
        """Model the worker finally exiting after a survived stop()."""
        self._alive.clear()

    def survive_stop(self):
        self._survive = True

    # -- introspection used by the controller --------------------------
    def stats(self):
        return {
            "frames_processed": self.frames_processed,
            "last_processed_age_seconds": 0.0,
        }

    @property
    def last_detection(self):
        return None


def _live_detectors() -> list[FakeDetector]:
    return [d for d in FakeDetector.instances if d.is_running]


def _make_ctl(monkeypatch) -> MicrophoneController:
    FakeDetector.instances.clear()
    sd = FakeSD([mic("Mic A", rate=16000.0, ch=1)])
    mgr = AudioDeviceManager(sd=sd)
    ctl = MicrophoneController(
        device_manager=mgr,
        capture=AudioCaptureService(sd=sd, device_manager=mgr),
        engine=FakeEngine(score=0.0),
        settings_get=lambda k: None,
        settings_set=lambda k, v: None,
        monitor_interval=0.05,
        callback_stall_threshold=60.0,
        detection_stall_threshold=60.0,
    )
    monkeypatch.setattr(mc_mod, "WakeWordDetectionService", FakeDetector)
    return ctl


def _enable_listening(ctl) -> FakeDetector:
    assert ctl.enable()
    assert ctl.status == STATUS_LISTENING
    det = ctl._detection
    assert isinstance(det, FakeDetector) and det.is_running
    return det


class TestRepeatedAndConcurrentResume:
    def test_repeated_resume_creates_no_duplicate(self, monkeypatch):
        ctl = _make_ctl(monkeypatch)
        try:
            _enable_listening(ctl)
            ctl.suspend_detection()
            assert ctl.status == STATUS_IDLE
            n = len(FakeDetector.instances)
            ctl.resume_detection()
            first = ctl._detection
            assert isinstance(first, FakeDetector)
            # Second resume: status is already LISTENING -> no-op.
            ctl.resume_detection()
            ctl.resume_detection()
            assert ctl._detection is first
            assert len(FakeDetector.instances) == n + 1
            assert len(_live_detectors()) == 1
        finally:
            ctl.shutdown()

    def test_concurrent_resumes_install_one_worker(self, monkeypatch):
        ctl = _make_ctl(monkeypatch)
        try:
            _enable_listening(ctl)
            ctl.suspend_detection()
            barrier = threading.Barrier(3)
            errors = []

            def _resume():
                try:
                    barrier.wait(timeout=5)
                    ctl.resume_detection()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            threads = [threading.Thread(target=_resume) for _ in range(2)]
            for t in threads:
                t.start()
            barrier.wait(timeout=5)
            for t in threads:
                t.join(timeout=10)
            assert not errors
            assert ctl.status == STATUS_LISTENING
            assert len(_live_detectors()) == 1
            assert ctl._detection in _live_detectors()
        finally:
            ctl.shutdown()


class TestSurvivingWorkerOwnership:
    def test_restart_does_not_replace_surviving_worker(self, monkeypatch):
        """A worker that outlives stop() keeps ownership; no second
        consumer is installed until it actually exits."""
        ctl = _make_ctl(monkeypatch)
        try:
            old = _enable_listening(ctl)
            old.survive_stop()
            ctl._restart_detection("test stall")
            # Old worker survived the join: it is still the owner.
            assert ctl._detection is old
            assert old.is_running
            assert len(FakeDetector.instances) == 1, (
                "a second consumer was installed while the old "
                "worker was still alive"
            )
            assert len(_live_detectors()) == 1
            # Recovery: once the survivor exits, the watchdog path
            # installs exactly one replacement.
            old.terminate()
            ctl._restart_detection("retry after exit")
            new = ctl._detection
            assert isinstance(new, FakeDetector)
            assert new is not old and new.is_running
            assert len(_live_detectors()) == 1
            assert ctl._detection is new
        finally:
            ctl.shutdown()

    def test_survivor_does_not_consume_retry_budget(self, monkeypatch):
        """Deferrals are ownership waits, not failed restarts."""
        ctl = _make_ctl(monkeypatch)
        try:
            old = _enable_listening(ctl)
            old.survive_stop()
            ctl._restart_detection("first")
            restarts_after_first = ctl._detect_restarts
            ctl._restart_detection("second")
            ctl._restart_detection("third")
            # Pre-check deferral returns before the budget counter.
            assert ctl._detect_restarts == restarts_after_first
        finally:
            ctl.shutdown()

    def test_resume_blocked_while_suspended_worker_alive(self, monkeypatch):
        """suspend -> worker survives -> resume must NOT install a
        second consumer; the missing-detector watchdog retries."""
        ctl = _make_ctl(monkeypatch)
        try:
            old = _enable_listening(ctl)
            old.survive_stop()
            ctl.suspend_detection()
            assert ctl.status == STATUS_IDLE
            assert old.is_running  # still draining frames
            n = len(FakeDetector.instances)
            ctl.resume_detection()
            # Slot blocked: LISTENING but no detector installed yet.
            assert ctl.status == STATUS_LISTENING
            assert ctl._detection is None
            assert len(FakeDetector.instances) == n, (
                "resume installed a second consumer"
            )
            # Watchdog missing-detector path also defers while the
            # survivor lives.
            ctl._detection_none_since = time.monotonic() - 60.0
            ctl._check_detection_liveness(time.monotonic())
            assert ctl._detection is None
            assert len(FakeDetector.instances) == n
            # Survivor exits -> watchdog installs the replacement.
            old.terminate()
            ctl._detection_none_since = time.monotonic() - 60.0
            ctl._check_detection_liveness(time.monotonic())
            new = ctl._detection
            assert isinstance(new, FakeDetector) and new.is_running
            assert len(_live_detectors()) == 1
        finally:
            ctl.shutdown()

    def test_watchdog_cannot_add_consumer_while_owner_alive(
        self, monkeypatch
    ):
        """Repeated watchdog ticks during the survivor window never
        create an additional consumer."""
        ctl = _make_ctl(monkeypatch)
        try:
            old = _enable_listening(ctl)
            old.survive_stop()
            ctl._restart_detection("stall")
            n = len(FakeDetector.instances)
            for _ in range(3):
                ctl._check_detection_liveness(time.monotonic())
            assert ctl._detection is old
            assert len(FakeDetector.instances) == n
            assert len(_live_detectors()) == 1
        finally:
            ctl.shutdown()


class TestDisableAndStaleEmissions:
    def test_disable_terminal_against_concurrent_resume(self, monkeypatch):
        """DISABLED stays terminal: a resume racing disable() can
        neither install a detector nor resurrect the status."""
        ctl = _make_ctl(monkeypatch)
        try:
            _enable_listening(ctl)
            ctl.suspend_detection()
            assert ctl.status == STATUS_IDLE
            barrier = threading.Barrier(3)
            errors = []

            def _resume():
                try:
                    barrier.wait(timeout=5)
                    ctl.resume_detection()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            def _disable():
                try:
                    barrier.wait(timeout=5)
                    ctl.disable()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)

            t1 = threading.Thread(target=_resume)
            t2 = threading.Thread(target=_disable)
            t1.start()
            t2.start()
            barrier.wait(timeout=5)
            t1.join(timeout=10)
            t2.join(timeout=10)
            assert not errors
            assert ctl.status == STATUS_DISABLED
            assert ctl._detection is None
            # No consumer may remain: every created detector is dead.
            assert not _live_detectors()
            # A later resume is a no-op while DISABLED.
            ctl.resume_detection()
            assert ctl.status == STATUS_DISABLED
            assert ctl._detection is None
        finally:
            ctl.shutdown()

    def test_stale_detector_emission_is_dropped(self, monkeypatch):
        """A retired worker's late wake callback must never reach the
        session or suspend the current detector."""
        ctl = _make_ctl(monkeypatch)
        wakes = []
        try:
            ctl.set_callbacks(on_wake_word=wakes.append)
            old = _enable_listening(ctl)
            old.terminate()
            ctl._restart_detection("replace dead worker")
            new = ctl._detection
            assert new is not old and new.is_running
            # Old worker's callback fires after ownership changed.
            old.on_detected("HELLO JARVIS")
            assert wakes == []
            assert ctl._last_wake_keyword is None
            assert ctl.status == STATUS_LISTENING
            assert ctl._detection is new and new.is_running
            # Sanity: the CURRENT detector's emission still works.
            new.on_detected("HELLO JARVIS")
            assert wakes == ["HELLO JARVIS"]
            # ...and it suspended standby as designed.
            assert ctl.status == STATUS_IDLE
        finally:
            ctl.shutdown()

    def test_shutdown_leaves_no_orphan_consumers(self, monkeypatch):
        ctl = _make_ctl(monkeypatch)
        old = _enable_listening(ctl)
        ctl.shutdown()
        assert ctl.status == STATUS_DISABLED
        assert ctl._detection is None
        assert not old.is_running
        assert not _live_detectors()
        # Every created detector is accounted for: stopped, none alive.
        assert all(not d.is_running for d in FakeDetector.instances)


class TestNormalHandoverStillWorks:
    def test_suspend_resume_replaces_dead_worker_once(self, monkeypatch):
        """Regression: the ordinary session handover still installs
        exactly one new standby detector."""
        ctl = _make_ctl(monkeypatch)
        try:
            _enable_listening(ctl)
            ctl.suspend_detection()
            assert ctl.status == STATUS_IDLE
            ctl.resume_detection()
            assert ctl.status == STATUS_LISTENING
            det = ctl._detection
            assert isinstance(det, FakeDetector) and det.is_running
            assert len(_live_detectors()) == 1
        finally:
            ctl.shutdown()

    def test_enable_after_error_replaces_stale_worker(self, monkeypatch):
        """enable() still performs its detach-stop-install sequence
        when a stale detector survived a previous error state."""
        ctl = _make_ctl(monkeypatch)
        try:
            old = _enable_listening(ctl)
            old.survive_stop()
            # Simulate the error-state survivor: status knocked to
            # ERROR while the detector reference is still attached.
            ctl._set_status(STATUS_DISABLED, "test")
            # NOTE: _detection still references `old`; enable() must
            # mark+stop it and — because it survives — defer install.
            ctl.enable()
            assert ctl.status == STATUS_LISTENING
            # Old survived -> deferred, no second consumer yet.
            if old.is_running:
                assert ctl._detection is None
                assert len(_live_detectors()) == 1
                old.terminate()
                ctl._restart_detection("post-enable retry")
                assert isinstance(ctl._detection, FakeDetector)
                assert len(_live_detectors()) == 1
        finally:
            ctl.shutdown()
