"""V10.4 wake-word latency instrumentation tests.

Covers the rolling measurement window, speech-end -> detection latency
estimate, keyword provenance metadata and the enriched stats payload —
all with mocked audio/engines (no microphone or model required).
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.voice.wake_word_detection_service import WakeWordDetectionService
from src.voice.wake_word_engine import NullWakeWordEngine


class FakeSD:
    def __init__(self):
        self._devices = [
            {"name": "Mic A", "max_input_channels": 1,
             "default_samplerate": 16000.0, "hostapi": 0},
        ]
        self.default = MagicMock()
        self.default.device = (0, -1)
        self.streams = []

    def query_devices(self, device=-1, kind=None):
        if device == -1:
            return self._devices
        return self._devices[device]

    def InputStream(self, **kwargs):  # noqa: N802
        outer = self

        class _Stream:
            def __init__(self):
                self.kwargs = kwargs
                self.active = False
                self.callback = kwargs.get("callback")

            def start(self):
                self.active = True

            def stop(self):
                self.active = False

            def close(self):
                self.active = False

        s = _Stream()
        outer.streams.append(s)
        return s


class FrameEngine:
    """Engine that detects on demand and reports inference timing."""

    name = "fake"

    def __init__(self, detect_after=0):
        self.detect_after = detect_after
        self.calls = 0
        self.inferences = 0
        self.last_inference_monotonic = 0.0
        self.last_inference_seconds = 0.0
        self.inference_ms_ema = 0.0
        self.keyword_count = 5
        self.keywords_sha256 = "ab" * 32
        self.keywords_file = "C:/models/kws/keywords_jarvis.txt"

    def process(self, samples):
        self.calls += 1
        self.inferences += 1
        self.last_inference_monotonic = time.monotonic()
        self.last_inference_seconds = 0.001
        self.inference_ms_ema = 1.0
        if self.calls > self.detect_after:
            return 1.0
        return 0.0

    def keyword_detected(self):
        return "hello jarvis" if self.calls > self.detect_after else None

    def reset(self):
        pass

    def close(self):
        pass


def _make_capture():
    from src.voice.audio_capture_service import AudioCaptureService
    from src.voice.audio_device_manager import AudioDeviceManager

    sd = FakeSD()
    mgr = AudioDeviceManager(sd=sd)
    return AudioCaptureService(sd=sd, device_manager=mgr), sd


def _feed(sd, rms=0.02, n=1):
    stream = sd.streams[-1]
    for _ in range(n):
        stream.callback(
            np.full((1600, 1), rms, dtype=np.float32), 1600, None, None
        )


class TestRollingWindow:
    def test_window_accumulates_then_rolls(self):
        capture, sd = _make_capture()
        svc = WakeWordDetectionService(
            capture, FrameEngine(detect_after=10**9)
        )
        svc._window_seconds = 0.2
        svc.start()
        capture.start()
        _feed(sd, n=3)
        deadline = time.time() + 3
        while svc.stats()["window"].get("frames", 0) < 3 \
                and time.time() < deadline:
            _feed(sd)
            time.sleep(0.05)
        win = svc.stats()["window"]
        assert win["frames"] >= 3
        assert win["kws_rms_max"] > 0
        assert win["kws_rms_mean"] > 0
        svc.stop()
        capture.stop()

    def test_window_resets_after_interval(self):
        capture, sd = _make_capture()
        svc = WakeWordDetectionService(
            capture, FrameEngine(detect_after=10**9)
        )
        svc._window_seconds = 0.15
        svc.start()
        capture.start()
        _feed(sd, n=2)
        deadline = time.time() + 3
        while svc.stats()["window"].get("frames", 0) < 2 \
                and time.time() < deadline:
            _feed(sd)
            time.sleep(0.05)
        first = svc.stats()["window"]["frames"]
        # Let the window expire, then feed again: count must restart.
        time.sleep(0.3)
        _feed(sd, n=1)
        deadline = time.time() + 3
        while time.time() < deadline:
            w = svc.stats()["window"]
            if w.get("frames", 0) >= 1 and w["frames"] < first + 2:
                break
            _feed(sd)
            time.sleep(0.05)
        svc.stop()
        capture.stop()


class TestDetectionRecord:
    def test_detection_records_latency_fields(self):
        capture, sd = _make_capture()
        hits = []
        engine = FrameEngine(detect_after=2)
        svc = WakeWordDetectionService(
            capture, engine, threshold=0.5, on_detected=hits.append
        )
        svc.start()
        capture.start()
        # Speech-level frames so the speech-end tracker engages.
        deadline = time.time() + 5
        while not hits and time.time() < deadline:
            _feed(sd, rms=0.03)
            time.sleep(0.05)
        svc.stop()
        capture.stop()
        assert hits == ["hello jarvis"]
        det = svc.last_detection
        assert det is not None
        assert det["keyword"] == "hello jarvis"
        assert det["monotonic"] > 0
        assert det["kws_rms"] > 0
        assert det["audio_age_seconds"] is not None
        assert det["speech_end_to_detection_seconds"] is not None
        assert svc.detection_latencies
        stats = svc.stats()
        assert stats["last_detection"]["keyword"] == "hello jarvis"
        assert stats["keyword_count"] == 5
        assert stats["keywords_sha256"] == "ab" * 32
        assert stats["engine_inference_ms_ema"] == 1.0

    def test_silence_produces_no_voiced_timestamp(self):
        capture, sd = _make_capture()
        engine = FrameEngine(detect_after=10**9)
        svc = WakeWordDetectionService(
            capture, engine, threshold=0.5
        )
        svc.start()
        capture.start()
        _feed(sd, rms=0.0005, n=2)  # below speech threshold
        deadline = time.time() + 2
        while svc.stats()["window"].get("frames", 0) < 2 \
                and time.time() < deadline:
            _feed(sd, rms=0.0005)
            time.sleep(0.05)
        assert svc._last_voiced_monotonic == 0.0
        svc.stop()
        capture.stop()


class TestKeywordProvenance:
    def test_null_engine_metadata(self):
        e = NullWakeWordEngine()
        assert e.keyword_count == 0
        assert e.keywords_sha256 is None
        assert e.inference_ms_ema == 0.0

    def test_start_log_includes_keywords(self, caplog):
        import logging

        capture, sd = _make_capture()
        svc = WakeWordDetectionService(
            capture, FrameEngine(), threshold=0.5
        )
        with caplog.at_level(logging.INFO, logger="src.voice.wake_word_detection_service"):
            svc.start()
            svc.stop()
        text = "\n".join(r.getMessage() for r in caplog.records)
        assert "variants=5" in text
        assert "sha256=" in text
