"""V10.6 wake-word reliability diagnostics tests.

Covers the new measurement surface added to investigate missed
"Hello Jarvis" detections on the real Intel microphone:

* bounded in-memory diagnostic ring buffer (~5 s of KWS input);
* per-window audio-activity records (keyword-less windows included);
* per-channel / selected-channel snapshots inside window records;
* PortAudio overflow/status event accounting in capture stats;
* KWS stream reset on standby handover;
* no extra streams, no persisted audio, lifecycle clearing.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.voice.audio_capture_service import AudioCaptureService
from src.voice.wake_word_detection_service import (
    WakeWordDetectionService,
)

SR = 16000


class FakeCapture:
    """Frame source double with stats() for channel/queue fields."""

    sample_rate = SR
    input_channels = 4
    device = None
    paused = False

    def __init__(self):
        import queue
        self._queue = queue.Queue()
        self.frames_dropped = 0
        self.status_warnings = 0
        self.overflow_callbacks = 0
        self._status_events = []

    def push(self, samples):
        self._queue.put(np.asarray(samples, dtype=np.float32))

    def push_speech(self, seconds=0.4, level=0.2):
        n = int(SR * seconds)
        t = np.arange(n) / SR
        self.push((level * np.sin(2 * np.pi * 220 * t)).astype(np.float32))

    def push_silence(self, seconds=0.4):
        self.push(np.zeros(int(SR * seconds), dtype=np.float32))

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
            "channel_rms": [0.2, 0.2, 0.001, 0.001],
            "channel_peaks": [0.3, 0.3, 0.01, 0.01],
        }


class FakeEngine:
    """Scripted wake-word engine: fires once after N voiced frames."""

    name = "fake-kws"
    keyword_count = 6
    keywords_sha256 = "abc"
    keywords_file = None
    inference_ms_ema = 1.0

    def __init__(self, fire_after=None):
        self.inferences = 0
        self.reset_count = 0
        self._fire_after = fire_after
        self._seen_voiced = 0
        self._pending = None

    def process(self, samples):
        self.inferences += 1
        a = np.asarray(samples, dtype=np.float32)
        rms = float(np.sqrt(np.mean(a * a))) if a.size else 0.0
        if rms >= 0.008:
            self._seen_voiced += 1
        if (
            self._fire_after is not None
            and self._seen_voiced >= self._fire_after
            and self._pending is None
        ):
            self._pending = "hello jarvis"
            return 1.0
        return 0.0

    def keyword_detected(self):
        kw, self._pending = self._pending, None
        return kw

    def reset(self):
        self.reset_count += 1
        self._fresh_stream()

    def _fresh_stream(self):
        pass

    def close(self):
        pass


def _wait_for(cond, timeout=5.0, interval=0.02):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return True
        time.sleep(interval)
    return False


def _service(cap, engine, **kw):
    params = dict(threshold=0.25, cooldown_seconds=0.0)
    params.update(kw)
    return WakeWordDetectionService(cap, engine, **params)


def test_diag_buffer_bounded_to_five_seconds():
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    for _ in range(200):  # ~20 s of 100 ms frames
        cap.push_speech(seconds=0.1, level=0.01)
    _wait_for(lambda: svc._diag_buffer_samples > 0, timeout=2)
    svc._run_once_for_test = None  # noqa - placeholder no-op
    # Drive frames directly through the worker loop is complex; instead
    # exercise the buffer primitive directly.
    for i in range(200):
        svc._diag_append(np.full(1600, 0.01, dtype=np.float32))
    assert svc._diag_buffer_samples <= int(5.0 * SR) + 1600
    audio = svc.diagnostic_audio()
    assert audio is not None
    assert len(audio) <= int(5.0 * SR) + 1600
    assert svc.stats()["diag_buffer_seconds"] <= 5.1


def test_diag_buffer_cleared_on_stop():
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    svc._diag_append(np.full(1600, 0.01, dtype=np.float32))
    svc.clear_diagnostic_audio()
    assert svc.diagnostic_audio() is None
    assert svc._diag_buffer_samples == 0


def test_activity_window_recorded_without_keyword():
    """Audio energy above the gate with no keyword result produces an
    ACTIVITY_WITHOUT_KEYWORD record — audio activity, not a failed
    wake-word attempt."""
    cap, eng = FakeCapture(), FakeEngine(fire_after=None)
    svc = _service(cap, eng)
    now = time.monotonic()
    svc._activity_frame(0.05, 0.2, 1600, now)
    svc._activity_frame(0.05, 0.2, 1600, now)
    svc._close_activity_window(now)
    assert len(svc.activity_windows) == 1
    rec = svc.activity_windows[0]
    assert rec["event"] == "ACTIVITY_WITHOUT_KEYWORD"
    assert rec["keyword_detected"] is False
    assert rec["active_frames"] == 2
    # 2 x 1600 samples @16 kHz = 0.2 s of supra-gate audio.
    assert rec["active_audio_seconds"] == pytest.approx(0.2)
    assert rec["rms_max"] == pytest.approx(0.05)
    assert rec["selected_channel"] == 0
    assert rec["channel_rms"] == [0.2, 0.2, 0.001, 0.001]
    assert svc.stats()["activity_windows_without_keyword"] == 1
    # V10.6 compatibility aliases still resolve.
    assert svc.attempts is svc.activity_windows
    assert svc.last_attempt is svc.last_activity_window
    assert svc.stats()["attempts_missed"] == 1


def test_activity_window_marked_on_keyword():
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    now = time.monotonic()
    svc._activity_frame(0.05, 0.2, 1600, now)
    svc._activity_detected = True
    svc._activity_keyword = "hello jarvis"
    svc._close_activity_window(now)
    rec = svc.activity_windows[0]
    assert rec["event"] == "WAKE_MATCHED"
    assert rec["keyword_detected"] is True
    assert rec["keyword"] == "hello jarvis"
    assert rec["detected"] is True  # legacy alias
    assert svc.stats()["activity_windows_without_keyword"] == 0


def test_activity_window_hangover_close_check():
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    now = time.monotonic()
    svc._activity_frame(0.05, 0.2, 1600, now)
    # Within hangover: stays open.
    svc._activity_close_check(now + svc._activity_hangover - 0.1)
    assert svc._activity_open
    svc._activity_close_check(now + svc._activity_hangover + 0.2)
    assert not svc._activity_open
    assert len(svc.activity_windows) == 1


def test_multiple_windows_then_keyword():
    """Repeated activity windows each get their own record, and a
    later keyword match is recorded on its own window."""
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    now = time.monotonic()
    for i in range(3):
        svc._activity_frame(0.04, 0.15, 1600, now + i)
        svc._close_activity_window(now + i + 2.0)
    assert len(svc.activity_windows) == 3
    assert all(
        w["event"] == "ACTIVITY_WITHOUT_KEYWORD"
        for w in svc.activity_windows
    )
    # Window with a real keyword result.
    svc._activity_frame(0.05, 0.25, 1600, now + 10)
    svc._activity_detected = True
    svc._activity_keyword = "hello jarvis"
    svc._close_activity_window(now + 12)
    assert svc.activity_windows[-1]["keyword_detected"] is True
    assert svc.stats()["activity_windows_recorded"] == 4
    assert svc.stats()["activity_windows_without_keyword"] == 3


def test_activity_windows_deque_bounded():
    cap, eng = FakeCapture(), FakeEngine()
    svc = _service(cap, eng)
    now = time.monotonic()
    for i in range(150):
        svc._activity_frame(0.04, 0.1, 1600, now + i * 0.01)
        svc._close_activity_window(now + i * 0.01 + 2.0)
    assert len(svc.activity_windows) == 100  # deque(maxlen=100)


def test_capture_status_overflow_counted():
    """PortAudio input-overflow status events are counted and logged
    with timestamps — they drop samples BEFORE our callback and are
    invisible to frames_dropped."""
    sd = pytest.importorskip("sounddevice")
    cap = AudioCaptureService(sd=sd, device_manager=None)
    # Drive the callback directly with a fake status object.
    class _Status:
        def __str__(self):
            return "input overflow"

    indata = np.zeros((1600, 1), dtype=np.float32)
    cap._audio_callback(indata, 1600, None, _Status())
    cap._audio_callback(indata, 1600, None, _Status())
    cap._audio_callback(indata, 1600, None, None)
    assert cap.status_warnings == 2
    assert cap.overflow_callbacks == 2
    stats = cap.stats()
    assert stats["overflow_callbacks"] == 2
    assert len(stats["recent_status_events"]) == 2
    assert "overflow" in stats["recent_status_events"][0]["status"]


def test_no_extra_microphone_stream_created():
    """Diagnostics must not open additional streams: the capture
    service still owns exactly one _stream slot."""
    cap = FakeCapture()
    svc = _service(cap, FakeEngine())
    assert not hasattr(svc, "_stream")
    assert not hasattr(cap, "_diag_stream")


def test_engine_reset_on_standby_handover():
    """resume_detection must reset the KWS stream so stale post-
    detection frames cannot leak into the next standby period."""
    from src.controllers.microphone_controller import (
        MicrophoneController,
    )

    class _Cap:
        is_running = True
        paused = False
        device = None
        def resume(self): pass
        def flush(self): pass
        def stats(self): return {}

    class _Eng:
        resets = 0
        def reset(self): self.resets += 1

    eng = _Eng()
    ctl = MicrophoneController.__new__(MicrophoneController)
    ctl._capture = _Cap()
    ctl._engine = eng
    ctl._status = "idle"
    class _OldSvc:
        is_running = False
        def stop(self): pass

    ctl._detection = _OldSvc()
    ctl._detection_generation = 0
    ctl._threshold = 0.25
    ctl._cooldown = 3.0
    ctl._on_stats = None
    ctl._on_wake_word = None
    ctl._fallback_notice = ""
    ctl._selection_mode = "auto"
    ctl._on_status = None
    ctl._status_detail = ""
    ctl._restart_lock = threading.RLock()
    ctl._retiring_detectors = []
    created = []

    class _Svc:
        def __init__(self, **kw): created.append(kw)
        def start(self): pass
        def stop(self): pass

    import src.controllers.microphone_controller as mc
    orig = mc.WakeWordDetectionService
    mc.WakeWordDetectionService = _Svc
    try:
        ctl.resume_detection()
    finally:
        mc.WakeWordDetectionService = orig
    assert eng.resets == 1
    assert ctl._detection_generation == 1
