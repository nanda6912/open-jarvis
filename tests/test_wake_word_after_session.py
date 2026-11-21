"""V10.3.3 — wake-word detection after "Goodbye Jarvis".

Regression coverage for the post-session standby lifecycle:

- Channel selection must NOT adapt while the capture gate is paused
  (TTS playback would otherwise hijack ``_selected_channel`` on
  multi-channel devices and starve the KWS of the user's voice).
- ``MicrophoneController.resume_detection`` must produce a fresh,
  working detection service on every session end (real controller,
  faked audio hardware).
- The real sherpa-onnx KWS engine + detection service must detect
  "Hello Jarvis" again after each simulated suspend/resume cycle.
"""

from __future__ import annotations

import queue
import threading
import time
import wave
from pathlib import Path

import numpy as np
import pytest

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
    active_streams,
    mic,
    wait_for,
)

ROOT = Path(__file__).resolve().parent.parent
WAKE_WAV = ROOT / "tests" / "fixtures" / "commands" / "hello_jarvis.wav"
SR = 16000
CHUNK = 1600  # 100 ms at 16 kHz


def _load_wav_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        ch = w.getnchannels()
        sw = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if sw == 2:
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        a = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    if rate != SR:
        from src.voice.audio_capture_service import resample_to_16k

        a = resample_to_16k(a, float(rate))
    return a.astype(np.float32)


# ----------------------------------------------------------------------
# 1.rotation — capture gate: channel selection frozen while paused
# ----------------------------------------------------------------------
class TestPausedGateChannelFreeze:
    """TTS playback must never teach the channel selector."""

    def _service(self) -> AudioCaptureService:
        svc = AudioCaptureService(sd=object(), device_manager=None)
        svc._sample_rate = float(SR)
        svc._stream_channels = 2
        svc._selected_channel = 0
        return svc

    @staticmethod
    def _frame(ch0: float, ch1: float) -> np.ndarray:
        n = CHUNK
        t = np.arange(n) / SR
        return np.stack(
            [
                (ch0 * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
                (ch1 * np.sin(2 * np.pi * 330 * t)).astype(np.float32),
            ],
            axis=1,
        )

    def test_selection_does_not_adapt_while_paused(self):
        svc = self._service()
        # Unpaused: channel 1 dominates -> selector moves to 1.
        for _ in range(30):
            svc._audio_callback(self._frame(0.001, 0.2), CHUNK, None, None)
        assert svc._selected_channel == 1
        ema_before = list(svc._channel_ema)

        # Paused (TTS playing): channel 0 suddenly dominates — this is
        # speaker bleed, not the user's voice. Selection must freeze.
        svc.pause()
        for _ in range(30):
            svc._audio_callback(self._frame(0.3, 0.001), CHUNK, None, None)
        assert svc._selected_channel == 1
        assert svc._channel_ema == pytest.approx(ema_before, abs=1e-6)
        assert svc.frames_paused >= 30

        # Unpaused again: adaptation resumes normally.
        svc.resume()
        for _ in range(60):
            svc._audio_callback(self._frame(0.3, 0.001), CHUNK, None, None)
        assert svc._selected_channel == 0

    def test_paused_frames_never_reach_queue(self):
        svc = self._service()
        svc.pause()
        svc._audio_callback(self._frame(0.2, 0.2), CHUNK, None, None)
        assert svc._queue.empty()
        svc.resume()
        svc._audio_callback(self._frame(0.2, 0.2), CHUNK, None, None)
        assert not svc._queue.empty()


# ----------------------------------------------------------------------
# 2. MicrophoneController: resume_detection produces a working worker
# ----------------------------------------------------------------------
class _QueueCapture:
    """Minimal capture fake: queue + stats, matching the real surface."""

    def __init__(self):
        self._q: queue.Queue = queue.Queue()
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
        self.paused = True

    def resume(self):
        self.paused = False

    def flush(self):
        self.flush_count += 1
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def stats(self):
        return {"queue_depth": self._q.qsize(), "running": True}


def _make_mic(sd, engine):
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


def _pump(sd, seconds=0.3):
    end = time.monotonic() + seconds
    frame = np.random.rand(CHUNK, 1).astype(np.float32) * 0.01
    while time.monotonic() < end:
        for s in active_streams(sd):
            try:
                s.callback(frame.copy(), CHUNK, None, None)
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.02)


class TestResumeDetectionLifecycle:
    def test_suspend_resume_cycles_recreate_worker(self):
        sd = FakeSD([mic("Mic A", rate=16000.0, ch=1)])
        engine = FakeEngine(score=1.0)  # always "detects"
        wakes = []
        ctl = _make_mic(sd, engine)
        ctl.set_callbacks(on_wake_word=wakes.append)
        assert ctl.enable()
        gen0 = ctl.watchdog_stats()["detection_generation"]

        for cycle in range(3):
            # Simulate activation: KWS fires -> controller suspends.
            _pump(sd, 0.4)
            assert wait_for(lambda: len(wakes) >= cycle + 1), (
                f"cycle {cycle}: no wake detection"
            )
            assert ctl.status == STATUS_IDLE
            # Simulate session end -> standby resume.
            ctl.resume_detection()
            assert ctl.status == STATUS_LISTENING
            assert ctl._detection is not None
            assert ctl._detection.is_running
        assert ctl.watchdog_stats()["detection_generation"] > gen0
        ctl.shutdown()

    def test_resume_detection_noop_when_not_idle(self):
        sd = FakeSD([mic("Mic A", rate=16000.0, ch=1)])
        ctl = _make_mic(sd, FakeEngine(score=0.0))
        assert ctl.enable()
        det = ctl._detection
        # Status LISTENING (not IDLE): resume must be a no-op.
        ctl.resume_detection()
        assert ctl._detection is det
        ctl.shutdown()


# ----------------------------------------------------------------------
# 3. Real KWS engine through the real detection service, per cycle
# ----------------------------------------------------------------------
class _FeederCapture(_QueueCapture):
    pass


def _push_utterance(cap: _QueueCapture, wav: np.ndarray):
    """Queue one wake phrase + 1 s trailing silence (100 ms frames)."""
    audio = np.concatenate([wav, np.zeros(SR, dtype=np.float32)])
    for start in range(0, len(audio) - CHUNK + 1, CHUNK):
        cap._q.put(audio[start:start + CHUNK].astype(np.float32))


@pytest.mark.skipif(not WAKE_WAV.exists(), reason="fixture missing")
class TestRealEnginePostSessionCycles:
    """Real sherpa-onnx KWS must detect again after every suspend."""

    CYCLES = 3

    def test_wake_detected_after_each_suspend_resume(self):
        engine = create_engine()
        if getattr(engine, "name", "") != "sherpa-onnx-kws":
            pytest.skip("sherpa-onnx KWS model unavailable")
        variants = engine.keyword_count
        assert variants and variants >= 5, (
            f"expected >=5 keyword variants, got {variants}"
        )
        wav = _load_wav_16k(WAKE_WAV)
        cap = _FeederCapture()
        detections = []

        service = WakeWordDetectionService(
            capture=cap, engine=engine, threshold=0.25,
            cooldown_seconds=0.0,
            on_detected=lambda kw: detections.append(kw),
        )
        try:
            for cycle in range(self.CYCLES):
                # Standby generation: fresh service, exactly like
                # MicrophoneController.resume_detection() creates.
                if service.is_running:
                    service.stop()
                service = WakeWordDetectionService(
                    capture=cap, engine=engine, threshold=0.25,
                    cooldown_seconds=0.0,
                    on_detected=lambda kw: detections.append(kw),
                )
                service.start()
                cap.flush()  # drop stale frames like resume_detection
                _push_utterance(cap, wav)
                assert wait_for(
                    lambda: len(detections) >= cycle + 1, timeout=15.0
                ), f"cycle {cycle}: wake phrase not detected after resume"
                # Simulate session activation: engine reset + suspend.
                engine.reset()
                service.stop()
                time.sleep(0.05)
        finally:
            service.stop()
        # All variants still loaded; engine reused cleanly.
        assert engine.keyword_count == variants
        assert len(detections) >= self.CYCLES
