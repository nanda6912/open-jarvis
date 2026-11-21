"""V10.3.1 Intel 4-channel array wake-detection tests.

Reproduces the production capture path on a simulated 4-channel
endpoint: verifies the stream opens multichannel, per-channel
selection picks the speech-bearing channel, the KWS worker receives
valid mono float32 16 kHz audio, and detection still fires.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.voice.audio_capture_service import AudioCaptureService
from src.voice.audio_device_manager import AudioDeviceManager
from src.voice.wake_word_detection_service import (
    WakeWordDetectionService,
)
from tests.test_dynamic_mic import FakeEngine, FakeSD, mic

RATE = 16000
FRAMES = 1600
N_CH = 4
SPEECH_CH = 2


def _multichannel_frame(speech_ch=SPEECH_CH, speech_rms=0.08,
                       noise_rms=0.0005, seed=0):
    """(frames, channels) float32 — speech on one channel only."""
    rng = np.random.RandomState(seed)
    arr = rng.randn(FRAMES, N_CH).astype(np.float32)
    arr *= noise_rms
    tone = (np.sin(2 * np.pi * 220 *
                   np.arange(FRAMES) / RATE) * speech_rms
            ).astype(np.float32)
    arr[:, speech_ch] = tone
    return arr


def _make_capture(sd):
    mgr = AudioDeviceManager(sd=sd)
    return AudioCaptureService(sd=sd, device_manager=mgr)


@pytest.fixture
def intel_sd():
    return FakeSD([mic(
        "Intel Smart Sound Technology for Digital Microphones",
        ch=N_CH, rate=float(RATE),
    )])


class TestMultichannelOpen:
    def test_opens_up_to_four_channels_at_16k(self, intel_sd):
        cap = _make_capture(intel_sd)
        dev = cap.start()
        try:
            assert cap.input_channels == N_CH
            assert cap.sample_rate == RATE
            assert dev.max_input_channels == N_CH
        finally:
            cap.stop()


class TestChannelSelection:
    def test_speech_channel_selected_and_mono_delivered(
            self, intel_sd):
        cap = _make_capture(intel_sd)
        cap.start()
        try:
            for i in range(30):
                cap._audio_callback(
                    _multichannel_frame(seed=i), FRAMES, None, None)
            st = cap.stats()
            assert st["input_channels"] == N_CH
            assert len(st["channel_rms"]) == N_CH
            # Speech channel dominates RMS by orders of magnitude.
            assert st["channel_rms"][SPEECH_CH] > 0.05
            assert all(st["channel_rms"][c] < 0.01
                       for c in range(N_CH) if c != SPEECH_CH)
            # Selection converges on the speech channel.
            deadline = time.monotonic() + 3.0
            while (time.monotonic() < deadline
                   and st["selected_channel"] != SPEECH_CH):
                cap._audio_callback(
                    _multichannel_frame(seed=100 + i), FRAMES,
                    None, None)
                i += 1
                st = cap.stats()
            assert st["selected_channel"] == SPEECH_CH
            # Queued frames are 1-D mono float32 at 16 kHz. Drain the
            # backlog (early frames were queued before selection
            # converged), then check the freshest frame content.
            while True:
                try:
                    cap._queue.get_nowait()
                except Exception:
                    break
            for i in range(3):
                cap._audio_callback(
                    _multichannel_frame(seed=400 + i), FRAMES,
                    None, None)
            frame = cap.get_frame(timeout=0.1)
            assert frame is not None
            assert frame.ndim == 1
            assert frame.dtype == np.float32
            assert frame.size == FRAMES
            # Frame content matches the speech channel, not a mix.
            assert np.allclose(frame, _last_tone(), atol=1e-4)
        finally:
            cap.stop()

    def test_channel_switch_follows_speech(self, intel_sd):
        cap = _make_capture(intel_sd)
        cap.start()
        try:
            for i in range(40):
                cap._audio_callback(
                    _multichannel_frame(speech_ch=SPEECH_CH, seed=i),
                    FRAMES, None, None)
            assert cap.stats()["selected_channel"] == SPEECH_CH
            # Move the speech to channel 0.
            for i in range(40):
                cap._audio_callback(
                    _multichannel_frame(speech_ch=0, seed=200 + i),
                    FRAMES, None, None)
            deadline = time.monotonic() + 3.0
            while (time.monotonic() < deadline
                   and cap.stats()["selected_channel"] != 0):
                cap._audio_callback(
                    _multichannel_frame(speech_ch=0, seed=300 + i),
                    FRAMES, None, None)
                time.sleep(0.01)
            assert cap.stats()["selected_channel"] == 0
            assert cap.stats()["channel_switch_count"] >= 1
        finally:
            cap.stop()


def _last_tone():
    return (np.sin(2 * np.pi * 220 *
                   np.arange(FRAMES) / RATE) * 0.08
            ).astype(np.float32)


class TestKwsInputValidity:
    def test_detection_worker_gets_mono_float32_and_detects(
            self, intel_sd):
        """End-to-end: 4ch callbacks -> queue -> detection worker ->
        engine.process receives 1-D float32 frames and a detection
        configured above threshold fires."""
        cap = _make_capture(intel_sd)
        seen_shapes = []

        class ShapeSpy(FakeEngine):
            def process(self, samples):
                seen_shapes.append(
                    (np.asarray(samples).shape,
                     np.asarray(samples).dtype))
                return super().process(samples)

        spy = ShapeSpy(score=1.0)  # always "detect"
        det = WakeWordDetectionService(
            cap, spy, threshold=0.25, cooldown_seconds=0.0,
            on_detected=lambda kw: None,
        )
        cap.start()
        det.start()
        try:
            deadline = time.monotonic() + 5.0
            while (time.monotonic() < deadline
                   and spy.processed < 5):
                cap._audio_callback(
                    _multichannel_frame(seed=int(time.time() * 1000) % (2**32)),
                    FRAMES, None, None)
                time.sleep(0.01)
            assert spy.processed >= 5
            for shape, dtype in seen_shapes:
                assert shape == (FRAMES,)
                assert dtype == np.float32
            st = det.stats()
            assert st["kws_input_samples"] == FRAMES
            assert st["kws_input_rms"] > 0.05  # speech channel level
            assert st["kws_input_peak"] > 0.05
        finally:
            det.stop()
            cap.stop()

    def test_silence_produces_zero_kws_signal(self, intel_sd):
        """No speech -> KWS input RMS stays at noise floor; the
        'no usable signal' vs 'speech not detected' distinction is
        observable in stats."""
        cap = _make_capture(intel_sd)
        engine = FakeEngine(score=0.0)
        det = WakeWordDetectionService(cap, engine)
        cap.start()
        det.start()
        try:
            quiet = np.random.RandomState(7).randn(
                FRAMES, N_CH).astype(np.float32) * 0.0005
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                cap._audio_callback(quiet.copy(), FRAMES, None, None)
                time.sleep(0.02)
            st = det.stats()
            assert st["kws_input_rms"] is not None
            assert st["kws_input_rms"] < 0.01
            assert st["detections"] == 0
        finally:
            det.stop()
            cap.stop()


class TestChannelDiagnostics:
    def test_stats_expose_per_channel_metrics(self, intel_sd):
        cap = _make_capture(intel_sd)
        cap.start()
        try:
            for i in range(20):
                cap._audio_callback(
                    _multichannel_frame(seed=i), FRAMES, None, None)
            st = cap.stats()
            for key in ("channel_rms", "channel_peaks", "channel_ema",
                        "selected_channel", "input_channels",
                        "device_stable_id"):
                assert key in st
            assert len(st["channel_peaks"]) == N_CH
            assert st["channel_peaks"][SPEECH_CH] > 0.07
            # EMA of the speech channel should dominate after frames.
            assert int(np.argmax(st["channel_ema"])) == SPEECH_CH
        finally:
            cap.stop()
