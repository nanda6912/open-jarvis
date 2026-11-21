"""V10.7.5 — wake-word reliability + HUD display stability regressions.

Covers:

- Standby handover ordering (pause -> flush -> engine reset -> new
  generation -> capture resume) in ``MicrophoneController``.
- KWS engine serialization: ``process()``/``reset()``/``rescore()``
  cannot interleave (engine lock).
- Activity-window rescoring: a window the live stream missed is
  replayed on fresh anchors and can still emit a detection.
- Silence never triggers rescoring or watchdog recovery.
- Dead-worker recovery via ``_restart_detection``.
- HUD placement: no reposition while the display signature is
  unchanged, reposition on real display change, bounded retries when
  geometry cannot converge, debounced display notifications.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QObject, QRect
from PySide6.QtWidgets import QApplication

from src.controllers.microphone_controller import (
    STATUS_IDLE,
    STATUS_LISTENING,
    MicrophoneController,
)
from src.voice.audio_capture_service import AudioCaptureService
from src.voice.audio_device_manager import AudioDeviceManager
from src.voice.wake_word_detection_service import WakeWordDetectionService
from src.voice.wake_word_engine import create_engine

from tests.test_dynamic_mic import (
    FakeEngine,
    FakeSD,
    mic,
    wait_for,
)

ROOT = Path(__file__).resolve().parent.parent
WAKE_WAV = ROOT / "tests" / "fixtures" / "commands" / "hello_jarvis.wav"
SR = 16000
CHUNK = 1600


def _load_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sw == 4:
        a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        a = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        a = (a - 128.0) / 128.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if rate != SR:
        from src.voice.audio_capture_service import resample_to_16k

        a = resample_to_16k(a, float(rate))
    return a.astype(np.float32)


class _QueueCapture:
    """Capture fake with an ordered event log for handover assertions."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
        self.events: list[str] = []
        self.paused = False
        self.is_running = True
        self.device = None
        self.flush_count = 0

    def get_frame(self, timeout=0.5):
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def pause(self):
        self.events.append("pause")
        self.paused = True

    def resume(self):
        self.events.append("resume")
        self.paused = False

    def flush(self):
        self.events.append("flush")
        self.flush_count += 1
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def stats(self):
        return {"queue_depth": self._q.qsize(), "running": True}

    def push(self, frame: np.ndarray):
        self._q.put(frame.astype(np.float32))


class _RescoringEngine(FakeEngine):
    """Fake engine whose live stream always misses but whose rescore
    hits — isolates the V10.7.5 recovery path."""

    def __init__(self, rescore_keyword="HELLO JARVIS"):
        super().__init__(score=0.0)
        self.rescore_keyword = rescore_keyword
        self.rescore_calls = 0
        self.rescore_leads = []

    def rescore(self, samples, lead_seconds=(0.5,), trail_seconds=0.5):
        self.rescore_calls += 1
        self.rescore_leads.append(tuple(lead_seconds))
        if self.rescore_keyword is None:
            return None
        return self.rescore_keyword, float(lead_seconds[0])


def _speech_frame(amplitude=0.05):
    t = np.arange(CHUNK) / SR
    return (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _silence_frame():
    return np.zeros(CHUNK, dtype=np.float32)


def _feed(cap: _QueueCapture, frames):
    for f in frames:
        cap.push(f)


# ----------------------------------------------------------------------
# 1. Standby handover ordering
# ----------------------------------------------------------------------
class TestStandbyHandoverOrdering:
    def _controller(self, sd, engine):
        mgr = AudioDeviceManager(sd=sd)
        return MicrophoneController(
            device_manager=mgr,
            capture=AudioCaptureService(sd=sd, device_manager=mgr),
            engine=engine,
            settings_get=lambda k: None,
            settings_set=lambda k, v: None,
            monitor_interval=0.05,
            callback_stall_threshold=60.0,
            detection_stall_threshold=60.0,
        )

    def test_handover_pauses_flushes_resets_then_resumes(self):
        """resume_detection must: pause gate -> flush -> engine reset
        -> start new generation -> resume gate."""
        sd = FakeSD([mic("Mic A", rate=16000.0, ch=1)])
        engine = FakeEngine(score=0.0)
        ctl = self._controller(sd, engine)
        assert ctl.enable()
        # Simulate session end state via the real path: suspend
        # detaches _detection so resume_detection builds a fresh
        # generation (a surviving detector is intentionally reused).
        ctl.suspend_detection()
        resets_before = engine.reset_count

        events: list[str] = []
        orig_pause = ctl.capture.pause
        orig_resume = ctl.capture.resume
        orig_flush = ctl.capture.flush
        ctl.capture.pause = lambda: (events.append("pause"),
                                     orig_pause())[1]
        ctl.capture.resume = lambda: (events.append("resume"),
                                      orig_resume())[1]
        ctl.capture.flush = lambda: (events.append("flush"),
                                     orig_flush())[1]

        import src.controllers.microphone_controller as mc_mod

        real_cls = mc_mod.WakeWordDetectionService

        class _SpyService(real_cls):
            def start(self):
                events.append("service_start")
                super().start()

        orig_cls = mc_mod.WakeWordDetectionService
        mc_mod.WakeWordDetectionService = _SpyService
        try:
            ctl.resume_detection()
        finally:
            mc_mod.WakeWordDetectionService = orig_cls

        assert ctl.status == STATUS_LISTENING
        assert ctl._detection is not None and ctl._detection.is_running
        # Engine reset happened during handover.
        assert engine.reset_count > resets_before
        # Ordering: pause first, flush before resume, service started
        # before the gate reopened.
        assert events[0] == "pause"
        assert events.index("flush") < events.index("resume")
        assert events.index("service_start") < events.index("resume")
        ctl.shutdown()

    def test_ready_logged_only_after_first_frame(self):
        sd = FakeSD([mic("Mic A", rate=16000.0, ch=1)])
        engine = FakeEngine(score=0.0)
        ctl = self._controller(sd, engine)
        assert ctl.enable()
        # Real session-end path: detach the standby detector so
        # resume_detection creates the spy service under test.
        ctl.suspend_detection()
        ready = []
        import src.controllers.microphone_controller as mc_mod

        real_cls = mc_mod.WakeWordDetectionService

        class _ReadySpy(real_cls):
            def __init__(self, *a, **kw):
                kw["on_first_frame"] = lambda: ready.append(1)
                super().__init__(*a, **kw)

        orig_cls = mc_mod.WakeWordDetectionService
        mc_mod.WakeWordDetectionService = _ReadySpy
        try:
            ctl.resume_detection()
            assert not ready, "READY fired before any frame processed"
            # Feed one frame through the real capture callback.
            end = time.monotonic() + 2.0
            frame = np.zeros((CHUNK, 1), dtype=np.float32)
            while time.monotonic() < end and not ready:
                for s in sd.streams:
                    try:
                        s.callback(frame.copy(), CHUNK, None, None)
                    except Exception:  # noqa: BLE001
                        pass
                time.sleep(0.02)
            assert ready, "first-frame callback never fired"
        finally:
            mc_mod.WakeWordDetectionService = orig_cls
            ctl.shutdown()


# ----------------------------------------------------------------------
# 2. Engine serialization
# ----------------------------------------------------------------------
class TestEngineSerialization:
    def _engine(self):
        engine = create_engine()
        if getattr(engine, "name", "") != "sherpa-onnx-kws":
            pytest.skip("sherpa-onnx KWS model unavailable")
        return engine

    def test_process_and_reset_are_mutually_exclusive(self):
        engine = self._engine()
        lock = getattr(engine, "_lock", None)
        assert lock is not None, "engine has no serialization lock"
        frame = np.zeros(CHUNK, dtype=np.float32)

        done = threading.Event()

        def worker():
            try:
                engine.process(frame)
            finally:
                done.set()

        # Hold the engine lock: process() must block until release.
        with lock:
            t = threading.Thread(target=worker, daemon=True)
            t.start()
            assert not done.wait(timeout=0.5), (
                "process() ran while engine lock was held"
            )
        assert done.wait(timeout=10.0), "process() did not finish"

    def test_rescore_does_not_touch_live_stream(self):
        engine = self._engine()
        gen_before = getattr(engine, "_stream_generation", None)
        audio = np.zeros(SR, dtype=np.float32)
        result = engine.rescore(audio, lead_seconds=(0.5,))
        assert result is None  # silence never produces a keyword
        assert getattr(engine, "_stream_generation", None) == gen_before

    def test_rescore_recovers_fixture_keyword(self):
        if not WAKE_WAV.exists():
            pytest.skip("fixture missing")
        engine = self._engine()
        audio = _load_wav_16k(WAKE_WAV)
        result = engine.rescore(
            audio,
            lead_seconds=(0.1, 0.25, 0.5, 0.8, 1.2),
            trail_seconds=0.5,
        )
        assert result is not None, "rescore missed the fixture wake phrase"
        keyword, lead = result
        assert "JARVIS" in keyword.upper()


# ----------------------------------------------------------------------
# 3. Activity-window rescoring in the detection service
# ----------------------------------------------------------------------
class TestActivityRescore:
    def test_missed_window_rescored_and_emitted(self):
        """Live stream misses; rescore recovers -> detection emitted."""
        cap = _QueueCapture()
        engine = _RescoringEngine()
        detected = []
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0, on_detected=detected.append,
        )
        try:
            svc.start()
            # ~0.6 s speech-level audio then >hangover silence so the
            # activity window opens and closes without a live hit.
            _feed(cap, [_speech_frame() for _ in range(6)])
            _feed(cap, [_silence_frame() for _ in range(20)])
            assert wait_for(lambda: len(detected) >= 1, timeout=10.0), (
                "rescored detection was not emitted"
            )
            assert detected[0] == "HELLO JARVIS"
            assert engine.rescore_calls >= 1
            win = svc.last_activity_window
            assert win is not None and win["keyword_detected"]
            assert win["rescored"] is True
        finally:
            svc.stop()

    def test_silence_never_rescores(self):
        cap = _QueueCapture()
        engine = _RescoringEngine()
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0,
        )
        try:
            svc.start()
            _feed(cap, [_silence_frame() for _ in range(40)])
            time.sleep(0.5)
            assert engine.rescore_calls == 0
            assert not svc.activity_windows
        finally:
            svc.stop()

    def test_rescore_failure_is_swallowed(self):
        cap = _QueueCapture()
        engine = _RescoringEngine(rescore_keyword=None)

        def boom(*a, **kw):
            raise RuntimeError("rescore exploded")

        engine.rescore = boom
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0,
        )
        try:
            svc.start()
            _feed(cap, [_speech_frame() for _ in range(6)])
            _feed(cap, [_silence_frame() for _ in range(20)])
            time.sleep(2.5)  # > 1.5 s hangover so the window closes
            assert svc.inference_errors == 0
        finally:
            svc.stop()

    def test_window_audio_buffer_bounded(self):
        cap = _QueueCapture()
        engine = _RescoringEngine()
        svc = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0,
        )
        try:
            svc.start()
            # Long activity: 20 s of speech-level frames.
            _feed(cap, [_speech_frame() for _ in range(200)])
            time.sleep(0.5)
            assert svc._window_audio_samples <= (
                svc._window_audio_max_samples
            )
        finally:
            svc.stop()


# ----------------------------------------------------------------------
# 4. Worker failure recovery + stale-frame rejection
# ----------------------------------------------------------------------
class TestWorkerRecovery:
    def _controller(self, sd, engine):
        mgr = AudioDeviceManager(sd=sd)
        return MicrophoneController(
            device_manager=mgr,
            capture=AudioCaptureService(sd=sd, device_manager=mgr),
            engine=engine,
            settings_get=lambda k: None,
            settings_set=lambda k, v: None,
            monitor_interval=0.05,
            callback_stall_threshold=60.0,
            detection_stall_threshold=60.0,
        )

    def test_dead_worker_is_recreated(self):
        sd = FakeSD([mic("Mic A", rate=16000.0, ch=1)])
        ctl = self._controller(sd, FakeEngine(score=0.0))
        assert ctl.enable()
        old = ctl._detection
        old.stop()  # simulate dead worker
        assert not old.is_running
        ctl._check_detection_liveness(time.monotonic())
        new = ctl._detection
        assert new is not None and new is not old
        assert new.is_running
        ctl.shutdown()

    def test_silence_does_not_trigger_detection_restart(self):
        """Silent standby must not look like a worker failure."""
        sd = FakeSD([mic("Mic A", rate=16000.0, ch=1)])
        ctl = self._controller(sd, FakeEngine(score=0.0))
        assert ctl.enable()
        det = ctl._detection
        # Callbacks keep arriving (silence) — worker alive and idle.
        end = time.monotonic() + 0.4
        frame = np.zeros((CHUNK, 1), dtype=np.float32)
        while time.monotonic() < end:
            for s in sd.streams:
                try:
                    s.callback(frame.copy(), CHUNK, None, None)
                except Exception:  # noqa: BLE001
                    pass
            time.sleep(0.02)
        for _ in range(3):
            ctl._check_detection_liveness(time.monotonic())
        assert ctl._detection is det
        ctl.shutdown()


# ----------------------------------------------------------------------
# 5. HUD single-monitor stability
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class FakeScreen:
    def __init__(self, geom: QRect):
        self._geom = geom
        self._avail = QRect(geom.x(), geom.y(),
                            geom.width(), geom.height() - 48)

    def geometry(self):
        return self._geom

    def availableGeometry(self):
        return self._avail

    def name(self):
        return "FakeScreen"

    def devicePixelRatio(self):
        return 1.0


class GeomFakeHud(QObject):
    """QWidget-like fake with real geometry semantics.

    Inherits QObject so ``QTimer(self.hud)`` (debounce/placement
    timers) has a valid parent.
    """

    def __init__(self, converge=True):
        super().__init__()
        self._geom = QRect(0, 0, 10, 10)
        self._converge = converge
        self.set_calls: list[QRect] = []
        self._visible = False

    def geometry(self):
        return self._geom

    def setGeometry(self, rect):
        r = QRect(rect)
        self.set_calls.append(r)
        if self._converge:
            self._geom = r
        # non-converging fake keeps its old geometry

    def show(self):
        self._visible = True

    def hide(self):
        self._visible = False

    def isVisible(self):
        return self._visible

    def raise_(self):
        pass

    def activateWindow(self):
        pass

    def set_topmost(self, on, block=True):
        pass

    def set_fullscreen(self, on, block=True):
        pass


class _OneMonitor:
    def __init__(self, bounds):
        self._bounds = bounds

    def monitors(self, refresh=True):
        from src.controllers.monitor_manager import MonitorInfo, MonitorRect

        b = self._bounds
        return [MonitorInfo(
            index=1, device="\\\\.\\DISPLAY1",
            bounds=MonitorRect(b[0], b[1], b[2], b[3]),
            work_area=MonitorRect(b[0], b[1], b[2], b[3] - 60),
            primary=True,
        )]


def _hud_controller(hud, screen, monitor_bounds=(0, 0, 800, 600)):
    from src.ui.hologram.hud_controller import HudController

    store = {}
    ctl = HudController(
        settings_get=lambda k: store.get(k),
        settings_set=lambda k, v: store.__setitem__(k, v),
        monitor_manager=_OneMonitor(monitor_bounds),
        hud_factory=lambda **kw: hud,
        bridge_factory=lambda h, **kw: None,
    )
    ctl.hud = hud
    ctl._shown = True
    ctl._fullscreen_pref = True
    ctl._screens = lambda: [screen]
    # Make _target_screen resolve our fake screen regardless of the
    # Win32-vs-Qt coordinate mismatch.
    ctl._target_screen = lambda: screen
    return ctl


class TestHudPlacement:
    def test_unchanged_display_never_repositions(self, qapp):
        screen = FakeScreen(QRect(0, 0, 800, 600))
        hud = GeomFakeHud()
        ctl = _hud_controller(hud, screen)
        ctl._recheck_placement()   # initial placement
        assert hud.set_calls, "initial placement did not run"
        n = len(hud.set_calls)
        for _ in range(5):
            ctl._recheck_placement()
        assert len(hud.set_calls) == n, (
            "HUD re-placed on an unchanged single-monitor desktop"
        )

    def test_real_display_change_repositions(self, qapp):
        screen = FakeScreen(QRect(0, 0, 800, 600))
        hud = GeomFakeHud()
        ctl = _hud_controller(hud, screen)
        ctl._recheck_placement()
        n = len(hud.set_calls)
        # Resolution change: new geometry -> new signature.
        screen._geom = QRect(0, 0, 1024, 768)
        screen._avail = QRect(0, 0, 1024, 720)
        ctl._recheck_placement()
        assert len(hud.set_calls) > n
        # V10.8: the HUD is a compact square overlay, not a fullscreen
        # window — it must land on the desired rect for the new display.
        assert hud.geometry() == ctl._desired_geometry(screen)

    def test_nonconverging_geometry_stops_after_budget(self, qapp):
        screen = FakeScreen(QRect(0, 0, 800, 600))
        hud = GeomFakeHud(converge=False)
        ctl = _hud_controller(hud, screen)
        for _ in range(10):
            ctl._recheck_placement()
        n = len(hud.set_calls)
        # Bounded: initial 3 attempts + one slow-retry round, then the
        # 120 s backoff keeps it silent.
        assert n <= 6, f"unbounded re-placement: {n} calls"
        for _ in range(5):
            ctl._recheck_placement()
        assert len(hud.set_calls) == n, (
            "re-placement continued during backoff"
        )

    def test_display_change_notification_is_debounced(self, qapp):
        from PySide6.QtTest import QTest

        screen = FakeScreen(QRect(0, 0, 800, 600))
        hud = GeomFakeHud()
        ctl = _hud_controller(hud, screen)
        ctl._shown = True
        ctl.hud = hud
        ctl._recheck_placement()  # establish baseline placement
        n = len(hud.set_calls)
        # Simulate a burst of display notifications.
        for _ in range(5):
            ctl._on_display_change()
        QTest.qWait(600)
        qapp.processEvents()
        # Debounce collapses the burst into (at most) one evaluation;
        # geometry unchanged -> no additional setGeometry calls.
        assert len(hud.set_calls) == n
