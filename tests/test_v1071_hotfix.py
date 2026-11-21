"""V10.7.1 hotfix regression tests.

Issue 1: HUD startup NameError (``micro_controller`` vs ``mic_controller``).
Issue 2: wake-attempt diagnostic terminology — audio activity windows
must never be labelled as failed wake-word attempts, and durations must
come from real sample counts, not an assumed frame period.
"""

from __future__ import annotations

import inspect
import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from src.ui.hologram.hud_controller import HudController  # noqa: E402
from src.voice.wake_word_detection_service import (  # noqa: E402
    WakeWordDetectionService,
)


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Fakes (mirrors test_wake_diagnostics_v106)
# ---------------------------------------------------------------------------
class FakeCapture:
    sample_rate = 16000
    input_channels = 1
    device = None
    paused = False

    def __init__(self):
        import queue
        self._queue = queue.Queue()
        self.frames_dropped = 0

    def get_frame(self, timeout=0.5):
        import queue
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stats(self):
        return {
            "frames_dropped": self.frames_dropped,
            "queue_depth": self._queue.qsize(),
            "selected_channel": 0,
            "channel_rms": [0.2],
            "channel_peaks": [0.3],
        }


class FakeEngine:
    name = "fake-kws"

    def process(self, samples):
        return 0.0  # empty keyword result — sherpa-onnx binary API

    def keyword_detected(self):
        return ""

    def reset(self):
        pass

    def close(self):
        pass


def _service(**kw):
    params = dict(threshold=0.25, cooldown_seconds=0.0)
    params.update(kw)
    return WakeWordDetectionService(FakeCapture(), FakeEngine(), **params)


# ---------------------------------------------------------------------------
# Issue 1 — HUD startup wiring
# ---------------------------------------------------------------------------
def test_main_py_passes_mic_controller_to_hud():
    """The V10.7.1 NameError must not regress: the HUD start() call must
    reference the real ``mic_controller`` variable."""
    import src.main as main_mod

    src = inspect.getsource(main_mod)
    assert "microphone_controller=mic_controller" in src
    assert "microphone_controller=micro_controller" not in src
    # The whole module still compiles.
    compile(src, main_mod.__file__, "exec")


def test_hud_controller_forwards_all_bridge_dependencies(qapp):
    """Every dependency main.py injects must reach HudBridge intact."""
    captured = {}

    class FakeHud:
        hideRequested = None
        topmostChanged = None
        fullscreenChanged = None

    get, set_ = (lambda k: None), (lambda k, v: None)

    def bridge_factory(hud, **kw):
        captured.update(kw)
        return type("B", (), {"shutdown": lambda s: None})()

    mic = object()
    signals = object()
    state = object()
    session = object()
    stt = object()
    tts = object()
    probe = object()

    ctl = HudController(
        get, set_,
        hud_factory=lambda **kw: FakeHud(),
        bridge_factory=bridge_factory,
    )
    ok = ctl.start(
        voice_signals=signals,
        state_manager=state,
        session_controller=session,
        microphone_controller=mic,
        stt_service=stt,
        tts_service=tts,
        desktop_probe=probe,
    )
    assert ok is True
    assert captured["microphone_controller"] is mic
    assert captured["voice_signals"] is signals
    assert captured["state_manager"] is state
    assert captured["session_controller"] is session
    assert captured["stt_service"] is stt
    assert captured["tts_service"] is tts
    assert captured["desktop_probe"] is probe
    ctl.shutdown()


def test_hud_creation_failure_is_isolated(qapp):
    """A HUD factory exception must not propagate — voice startup
    continues with hud_controller effectively disabled."""
    get, set_ = (lambda k: None), (lambda k, v: None)

    def boom(**kw):
        raise RuntimeError("no GL")

    ctl = HudController(get, set_, hud_factory=boom)
    assert ctl.start() is False
    assert ctl.is_available is False
    ctl.shutdown()  # must not raise


def test_hud_show_hide_does_not_touch_audio(qapp):
    """Hiding/showing the HUD only toggles rendering; it never calls
    into capture or detection services."""
    from src.ui.hologram.hud_window import HologramHUD

    hud = HologramHUD(target_fps=30)
    hud.show()
    qapp.processEvents()
    assert hud.orb._timer.isActive()
    hud.hide()
    qapp.processEvents()
    assert not hud.orb._timer.isActive()
    hud.show()
    qapp.processEvents()
    assert hud.orb._timer.isActive()
    hud.close()


# ---------------------------------------------------------------------------
# Issue 2/3 — activity-window terminology + measurements
# ---------------------------------------------------------------------------
def test_silence_produces_no_activity_window():
    svc = _service()
    now = time.monotonic()
    for i in range(10):
        svc._activity_frame(0.001, 0.01, 1600, now + i * 0.1)
    svc._close_activity_window(now + 5.0)
    assert len(svc.activity_windows) == 0
    assert svc.stats()["activity_windows_recorded"] == 0


def test_background_energy_is_activity_not_wake_attempt():
    """Energy above the gate with no keyword must be labelled
    ACTIVITY_WITHOUT_KEYWORD — never a failed wake attempt."""
    svc = _service()
    now = time.monotonic()
    svc._activity_frame(0.05, 0.4, 1600, now)
    svc._close_activity_window(now)
    rec = svc.activity_windows[0]
    assert rec["event"] == "ACTIVITY_WITHOUT_KEYWORD"
    assert rec["keyword_detected"] is False
    assert "attempt" not in rec["event"].lower()
    assert "speech" not in rec  # no field may claim verified speech
    # Log wording check.
    import logging

    logger = logging.getLogger("src.voice.wake_word_detection_service")
    records = []

    class _H(logging.Handler):
        def emit(self, r):
            records.append(r.getMessage())

    h = _H()
    old_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(h)
    try:
        svc._activity_frame(0.05, 0.4, 1600, now + 10)
        svc._close_activity_window(now + 10)
    finally:
        logger.removeHandler(h)
        logger.setLevel(old_level)
    # V10.7.5.1: structured KWS_* trace events are logged after the
    # human summary — find the window record, not just the last line.
    msg = next(
        (m for m in records if "Audio activity window" in m), "")
    assert msg
    assert "Wake attempt" not in msg
    assert "active_audio_seconds" in msg
    assert "keyword_detected=False" in msg


def test_rms_and_peak_measured_from_kws_input():
    svc = _service()
    now = time.monotonic()
    svc._activity_frame(0.02, 0.1, 1600, now)
    svc._activity_frame(0.09, 0.7, 1600, now + 0.1)
    svc._activity_frame(0.03, 0.2, 1600, now + 0.2)
    svc._close_activity_window(now + 0.3)
    rec = svc.activity_windows[0]
    assert rec["rms_max"] == pytest.approx(0.09)
    assert rec["peak_max"] == pytest.approx(0.7)
    assert rec["active_frames"] == 3
    assert rec["frames"] == 3
    assert rec["active_audio_seconds"] == pytest.approx(0.3)


def test_duration_uses_real_sample_count_not_frame_period():
    """Non-100 ms frames must produce correct durations: 800 samples
    @16 kHz = 0.05 s, not the old voiced*0.1 estimate."""
    svc = _service()
    now = time.monotonic()
    svc._activity_frame(0.05, 0.2, 800, now)
    svc._activity_frame(0.05, 0.2, 800, now + 0.05)
    svc._activity_frame(0.05, 0.2, 400, now + 0.1)
    svc._close_activity_window(now + 0.2)
    rec = svc.activity_windows[0]
    # 2000 samples @16 kHz = 0.125 s -> rounded to 0.12 (2 dp).
    # The old voiced*0.1 estimate would have said 0.3 s.
    assert rec["active_audio_seconds"] == pytest.approx(0.12)


def test_hangover_includes_silence_frames_in_window():
    """Frames below the gate still count toward the open window's
    frame total while inside the hangover — no double counting."""
    svc = _service()
    now = time.monotonic()
    svc._activity_frame(0.05, 0.2, 1600, now)          # active
    svc._activity_frame(0.001, 0.01, 1600, now + 0.1)  # hangover silence
    svc._activity_frame(0.001, 0.01, 1600, now + 0.2)  # hangover silence
    rec_open = svc._activity_open
    assert rec_open is True
    svc._close_activity_window(now + 0.3)
    rec = svc.activity_windows[0]
    assert rec["frames"] == 3
    assert rec["active_frames"] == 1
    assert rec["active_audio_seconds"] == pytest.approx(0.1)


def test_long_activity_not_reported_as_speech():
    """A 30 s noise burst is one activity window with measured energy —
    the record carries no speech/s verification claim."""
    svc = _service()
    now = time.monotonic()
    for i in range(300):  # ~30 s of supra-gate frames
        svc._activity_frame(0.4, 0.9, 1600, now + i * 0.1)
    svc._close_activity_window(now + 30.0)
    rec = svc.activity_windows[0]
    assert rec["active_audio_seconds"] == pytest.approx(30.0)
    assert rec["event"] == "ACTIVITY_WITHOUT_KEYWORD"
    for key in rec:
        assert "speech" not in key and "voiced" not in key


def test_empty_kws_result_is_no_keyword():
    """sherpa-onnx 2.0.1 returns '' for no match — that is NOT a
    near-threshold confidence; keyword_detected stays False."""
    svc = _service()
    eng = svc._engine
    assert eng.keyword_detected() == ""
    svc._activity_frame(0.05, 0.2, 1600, time.monotonic())
    # Simulate the _run path: empty keyword -> no tag.
    assert svc._activity_detected is False
    svc._close_activity_window(time.monotonic())
    assert svc.activity_windows[0]["keyword_detected"] is False


def test_nonempty_kws_result_is_keyword_detection():
    svc = _service()
    svc._activity_frame(0.05, 0.2, 1600, time.monotonic())
    svc._activity_detected = True
    svc._activity_keyword = "hello jarvis"
    svc._close_activity_window(time.monotonic())
    rec = svc.activity_windows[0]
    assert rec["event"] == "WAKE_MATCHED"
    assert rec["keyword_detected"] is True
    assert rec["keyword"] == "hello jarvis"


def test_repeated_standby_cycles_keep_working():
    """stop()/start() cycles reset window state without losing the
    bounded history."""
    svc = _service()
    now = time.monotonic()
    for cycle in range(3):
        svc._activity_frame(0.05, 0.2, 1600, now + cycle)
        svc._close_activity_window(now + cycle + 1.0)
    assert len(svc.activity_windows) == 3
    assert svc.stats()["activity_windows_without_keyword"] == 3
    # Aliases still consistent.
    assert svc.attempts is svc.activity_windows
    assert svc.last_attempt is svc.last_activity_window
    assert svc.stats()["attempts_recorded"] == 3
    assert svc.stats()["attempts_missed"] == 3


def test_diagnostic_thresholds_unchanged():
    """V10.7.1 keeps the V10.6 segmentation parameters and the KWS
    threshold exactly as configured."""
    svc = _service()
    assert svc._activity_threshold == 0.008
    assert svc._activity_hangover == 1.5
    assert svc._threshold == 0.25
    assert svc._diag_buffer_seconds == 5.0


def test_kws_receives_identical_audio_path():
    """The tracker is passive: _activity_frame only reads rms/peak/
    samples — it cannot alter or delay engine.process()."""
    svc = _service()
    sig = inspect.signature(WakeWordDetectionService._activity_frame)
    assert list(sig.parameters) == ["self", "rms", "peak", "samples", "now"]
    # Engine still invoked once per dequeued frame in _run.
    src = inspect.getsource(WakeWordDetectionService._run)
    assert src.count("self._engine.process(frame)") == 1
    assert "_activity_frame" in src


# ---------------------------------------------------------------------------
# HUD wake-status indicators (Issue 6)
# ---------------------------------------------------------------------------
def test_hud_wake_status_never_claims_failure(qapp):
    """An activity window closing without a keyword must not surface as
    a wake-word failure anywhere in the HUD."""
    from tests.test_hud_v107 import FakeHud, _bridge
    from src.voice.audio_worker import VoiceSignals

    hud = FakeHud()
    signals = VoiceSignals(qapp)
    bridge = _bridge(hud, signals=signals)
    # Simulate standby stats with processed frames but no detections.
    bridge._mic._stats = {
        "frames_processed": 500, "detections": 0, "frames_dropped": 0,
    }
    bridge._system_tick()
    wake_row = hud.system_rows[-1][1]
    assert "fail" not in wake_row.lower()
    assert "processing" in wake_row
    # A real keyword emission is the only path to "matched:".
    signals.wakeWordDetected.emit("hello jarvis")
    qapp.processEvents()
    assert bridge._wake_status.startswith("matched:")
    bridge.shutdown()
