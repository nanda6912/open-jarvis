"""Tests for microphone capture, wake-word detection and controller.

All audio input and the wake-word model are mocked — no physical
microphone or model files are required.
"""

from __future__ import annotations

import queue
import threading
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.controllers.microphone_controller import (
    MicrophoneController,
    STATUS_DISABLED,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_LISTENING,
)
from src.voice.audio_capture_service import (
    AudioCaptureService,
    resample_to_16k,
)
from src.voice.audio_device_manager import (
    AudioDevice,
    AudioDeviceManager,
    AudioError,
    DeviceUnavailableError,
    PermissionDeniedError,
)
from src.voice.wake_word_detection_service import WakeWordDetectionService
from src.voice.wake_word_engine import (
    ModelUnavailableError,
    NullWakeWordEngine,
    SherpaOnnxKeywordSpotter,
)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeSD:
    """Fake sounddevice module."""

    class error(Exception):
        pass

    def __init__(self, devices=None, fail_on_stream=None):
        self._devices = devices if devices is not None else [
            {"name": "Mic A", "max_input_channels": 2,
             "default_samplerate": 48000.0, "hostapi": 0},
            {"name": "Speakers", "max_input_channels": 0,
             "default_samplerate": 48000.0, "hostapi": 0},
        ]
        self._default = (0, -1)
        self._fail = fail_on_stream
        self.streams = []
        self.default = MagicMock()
        self.default.device = self._default

    def query_devices(self, device=-1, kind=None):
        if device == -1:
            return self._devices
        return self._devices[device]

    streams_created = []

    def InputStream(self, **kwargs):  # noqa: N802
        outer = self
        # Emulate a driver that only accepts the device-native rate:
        # requesting any other samplerate fails so capture falls back
        # to native-rate + software resampling (as on real hardware
        # that rejects non-default rates).
        dev_idx = kwargs.get("device", 0)
        try:
            dev = outer._devices[int(dev_idx)]
            native = float(dev.get("default_samplerate", 0.0))
        except Exception:  # noqa: BLE001
            native = 0.0
        req = float(kwargs.get("samplerate", native) or native)
        if native and req != native:
            raise ValueError(f"unsupported samplerate {req}")

        class _Stream:
            def __init__(self):
                self.kwargs = kwargs
                self.active = False
                self.callback = kwargs.get("callback")

            def start(self):
                if outer._fail:
                    raise outer._fail
                self.active = True

            def stop(self):
                self.active = False

            def close(self):
                self.active = False

        stream = _Stream()
        FakeSD.streams_created.append(stream)
        return stream


@pytest.fixture
def sd():
    FakeSD.streams_created = []
    return FakeSD()


@pytest.fixture
def device_manager(sd):
    return AudioDeviceManager(sd=sd)


@pytest.fixture
def capture(sd, device_manager):
    return AudioCaptureService(sd=sd, device_manager=device_manager)


class FakeEngine:
    name = "fake"

    def __init__(self, score=0.0, keyword="hello jarvis"):
        self.score = score
        self._kw = keyword
        self.processed = 0
        self.reset_count = 0
        self.closed = False

    def process(self, samples):
        self.processed += 1
        if self.score >= 1.0:
            self._pending = self._kw
            return 1.0
        return self.score

    def keyword_detected(self):
        kw = getattr(self, "_pending", None)
        self._pending = None
        return kw

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


# ----------------------------------------------------------------------
# Device manager
# ----------------------------------------------------------------------
class TestDeviceManager:
    def test_lists_only_input_devices(self, device_manager):
        devices = device_manager.list_input_devices()
        assert len(devices) == 1
        assert devices[0].name == "Mic A"
        assert devices[0].max_input_channels == 2

    def test_default_input_device(self, device_manager):
        assert device_manager.default_input_device().name == "Mic A"

    def test_no_devices_raises_on_resolve(self):
        sd = FakeSD(devices=[{"name": "out", "max_input_channels": 0,
                              "default_samplerate": 48000.0, "hostapi": 0}])
        mgr = AudioDeviceManager(sd=sd)
        with pytest.raises(DeviceUnavailableError):
            mgr.resolve_device(None)

    def test_resolve_by_id_and_fallback(self, device_manager):
        dev = device_manager.list_input_devices()[0]
        assert device_manager.resolve_device(dev.device_id).index == dev.index
        assert device_manager.resolve_device("bogus:id").index == dev.index
        assert device_manager.resolve_device(None).index == dev.index

    def test_named_id_with_unknown_name_uses_default_not_index(self):
        # Hot-unplug index reuse: the pinned mic is gone and a
        # DIFFERENT device now sits at its old index. A persisted id
        # that carries a name must never bind that usurper — the
        # explicit default device is used instead.
        sd = FakeSD(devices=[
            {"name": "Usurper Mic", "max_input_channels": 2,
             "default_samplerate": 48000.0, "hostapi": 0},
            {"name": "Default Mic", "max_input_channels": 2,
             "default_samplerate": 48000.0, "hostapi": 0},
        ])
        sd.default.device = (1, -1)
        mgr = AudioDeviceManager(sd=sd)
        got = mgr.resolve_device("0:0:Gone Mic")
        assert got.name == "Default Mic"
        # Nameless legacy ids still resolve via hostapi:index.
        got2 = mgr.resolve_device("0:0")
        assert got2.name == "Usurper Mic"

    def test_stale_index_same_name_still_self_heals(self, device_manager):
        # Ordinary reshuffle: same name, wrong stored index -> name
        # match wins and the id self-heals on next persist.
        dev = device_manager.list_input_devices()[0]
        stale = f"{dev.hostapi}:999:{dev.name}"
        assert device_manager.resolve_device(stale).index == dev.index

    def test_matches_device_id(self, device_manager):
        dev = device_manager.list_input_devices()[0]
        assert device_manager.matches_device_id(dev, dev.device_id)
        assert device_manager.matches_device_id(
            dev, f"{dev.hostapi}:{dev.index}:different name"
        )
        other = AudioDevice(7, "Other", 1, 16000.0, 9)
        assert not device_manager.matches_device_id(other, dev.device_id)


# ----------------------------------------------------------------------
# Capture service
# ----------------------------------------------------------------------
class TestCaptureService:
    def test_start_and_stop(self, capture):
        device = capture.start()
        assert capture.is_running
        assert device.name == "Mic A"
        capture.stop()
        assert not capture.is_running
        assert capture.device is None

    def test_audio_callback_enqueues_16k_frames(self, capture):
        capture.start()
        stream = FakeSD.streams_created[-1]
        stream.callback(np.zeros((4800, 1), dtype=np.float32), 4800, None, None)
        frame = capture.get_frame(timeout=1.0)
        assert frame is not None
        # 4800 samples @48k -> ~1600 @16k
        assert abs(len(frame) - 1600) <= 2

    def test_queue_is_bounded_and_drops_oldest(self, capture):
        capture.start(blocksize=1)
        stream = FakeSD.streams_created[-1]
        for _ in range(600):
            stream.callback(np.zeros((100, 1), dtype=np.float32), 100, None, None)
        assert capture._queue.qsize() <= 250
        assert capture.frames_dropped > 0

    def test_pause_discards_frames(self, capture):
        capture.start()
        capture.pause()
        stream = FakeSD.streams_created[-1]
        stream.callback(np.zeros((1600, 1), dtype=np.float32), 1600, None, None)
        assert capture.get_frame(timeout=0.2) is None
        capture.resume()

    def test_permission_denied(self, sd, device_manager):
        sd._fail = PermissionError("denied")
        cap = AudioCaptureService(sd=sd, device_manager=device_manager)
        with pytest.raises(PermissionDeniedError):
            cap.start()

    def test_device_unavailable(self):
        sd = FakeSD(devices=[])
        cap = AudioCaptureService(sd=sd, device_manager=AudioDeviceManager(sd=sd))
        with pytest.raises(DeviceUnavailableError):
            cap.start()

    def test_generic_audio_error(self, sd, device_manager):
        sd._fail = RuntimeError("boom")
        cap = AudioCaptureService(sd=sd, device_manager=device_manager)
        with pytest.raises(AudioError):
            cap.start()

    def test_stop_without_start_is_safe(self, capture):
        capture.stop()

    def test_queued_frames_own_their_data(self, capture):
        """Queued frames must not alias the recycled PortAudio buffer.

        sounddevice reuses ``indata`` memory across callbacks; a queued
        view would be overwritten by later callbacks, so a consumer that
        falls behind (STT blocked in transcribe) would read corrupted
        audio. Regression test for the wake->STT silence bug.
        """
        capture.start()
        stream = FakeSD.streams_created[-1]
        buf = np.full((1600, 1), 0.5, dtype=np.float32)
        stream.callback(buf, 1600, None, None)
        # Simulate PortAudio reusing the input buffer.
        buf[:] = 0.0
        frame = capture.get_frame(timeout=1.0)
        assert frame is not None
        assert frame.flags.owndata
        assert float(frame.max()) == pytest.approx(0.5)
        capture.stop()

    def test_stats_counters(self, capture):
        capture.start()
        stream = FakeSD.streams_created[-1]
        stream.callback(
            np.full((1600, 1), 0.5, dtype=np.float32), 1600, None, None
        )
        stats = capture.stats()
        assert stats["frames_captured"] == 1
        assert stats["frames_queued"] == 1
        assert stats["frames_dropped"] == 0
        assert stats["rms"] == pytest.approx(0.5, abs=0.01)
        assert stats["peak"] == pytest.approx(0.5, abs=0.01)
        assert stats["device"] == "Mic A"
        capture.stop()


def test_resample_to_16k():
    x = np.ones(4800, dtype=np.float32)
    y = resample_to_16k(x, 48000.0)
    assert abs(len(y) - 1600) <= 2
    assert np.allclose(y, 1.0, atol=0.01)
    assert resample_to_16k(x, 16000.0) is x


# ----------------------------------------------------------------------
# Wake-word engine
# ----------------------------------------------------------------------
class TestWakeWordEngine:
    def test_model_unavailable(self, tmp_path):
        with pytest.raises(ModelUnavailableError):
            SherpaOnnxKeywordSpotter(tmp_path)

    def test_create_engine_falls_back_to_null(self, tmp_path):
        engine = __import__(
            "src.voice.wake_word_engine", fromlist=["create_engine"]
        ).create_engine(tmp_path)
        assert isinstance(engine, NullWakeWordEngine)
        assert engine.process(np.zeros(1600, dtype=np.float32)) == 0.0

    def test_null_engine_never_detects(self):
        e = NullWakeWordEngine()
        e.process(np.zeros(10, dtype=np.float32))
        assert e.keyword_detected() is None

    def _write_model(self, d, empty=False):
        names = ["encoder-1.onnx", "decoder-1.onnx", "joiner-1.onnx",
                 "tokens.txt", "keywords.txt"]
        for n in names:
            (d / n).write_bytes(b"" if empty else b"x" * 16)

    def test_empty_model_files_rejected(self, tmp_path):
        self._write_model(tmp_path, empty=True)
        with pytest.raises(ModelUnavailableError):
            SherpaOnnxKeywordSpotter(tmp_path)

    def test_partial_model_rejected(self, tmp_path):
        (tmp_path / "tokens.txt").write_text("a 1\n")
        with pytest.raises(ModelUnavailableError):
            SherpaOnnxKeywordSpotter(tmp_path)

    def test_corrupt_model_clear_error(self, tmp_path):
        # Files exist but sherpa-onnx cannot parse them -> ModelUnavailableError
        self._write_model(tmp_path)
        try:
            SherpaOnnxKeywordSpotter(tmp_path)
        except ModelUnavailableError:
            pass  # expected: wrapped, not a raw ONNX error
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"unexpected error type: {type(exc).__name__}: {exc}")


# ----------------------------------------------------------------------
# Detection service
# ----------------------------------------------------------------------
class TestDetectionService:
    def _capture_with_frame(self, capture):
        capture.start()
        stream = FakeSD.streams_created[-1]
        stream.callback(
            np.zeros((1600, 1), dtype=np.float32), 1600, None, None
        )

    def test_detection_fires_callback_once(self, capture):
        engine = FakeEngine(score=1.0)
        hits = []
        svc = WakeWordDetectionService(
            capture, engine, threshold=0.5, on_detected=hits.append
        )
        svc.start()
        self._capture_with_frame(capture)
        deadline = time.time() + 3
        while not hits and time.time() < deadline:
            self._capture_with_frame(capture)
            time.sleep(0.05)
        svc.stop()
        capture.stop()
        assert hits == ["hello jarvis"]

    def test_below_threshold_no_detection(self, capture):
        engine = FakeEngine(score=0.1)
        hits = []
        svc = WakeWordDetectionService(
            capture, engine, threshold=0.9, on_detected=hits.append
        )
        svc.start()
        self._capture_with_frame(capture)
        time.sleep(0.4)
        svc.stop()
        capture.stop()
        assert hits == []
        assert engine.processed > 0

    def test_cooldown_blocks_repeat(self, capture):
        engine = FakeEngine(score=1.0)
        hits = []
        svc = WakeWordDetectionService(
            capture, engine, threshold=0.5,
            cooldown_seconds=60.0, on_detected=hits.append,
        )
        svc.start()
        for _ in range(6):
            self._capture_with_frame(capture)
            time.sleep(0.15)
        svc.stop()
        capture.stop()
        assert len(hits) == 1

    def test_stop_terminates_worker(self, capture):
        svc = WakeWordDetectionService(capture, FakeEngine())
        svc.start()
        assert svc.is_running
        svc.stop()
        assert not svc.is_running


# ----------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------
class TestMicrophoneController:
    def _controller(self, sd, device_manager, engine=None, repo=None):
        settings = repo if repo is not None else {}
        return MicrophoneController(
            device_manager=device_manager,
            capture=AudioCaptureService(sd=sd, device_manager=device_manager),
            engine=engine or FakeEngine(score=0.0),
            settings_get=lambda k: settings.get(k),
            settings_set=lambda k, v: settings.__setitem__(k, v),
        )

    def test_enable_disable(self, sd, device_manager):
        settings = {}
        ctl = self._controller(sd, device_manager, repo=settings)
        statuses = []
        ctl.set_callbacks(on_status=lambda s, d: statuses.append(s))
        assert ctl.enable() is True
        assert ctl.is_listening
        assert settings["voice.microphone_enabled"] == "true"
        assert settings["voice.microphone_device"]
        ctl.disable()
        assert ctl.status == STATUS_DISABLED
        assert settings["voice.microphone_enabled"] == "false"
        ctl.shutdown()

    def test_enable_without_mic_reports_error(self):
        sd = FakeSD(devices=[])
        mgr = AudioDeviceManager(sd=sd)
        ctl = self._controller(sd, mgr)
        errors = []
        ctl.set_callbacks(on_error=errors.append)
        assert ctl.enable() is False
        assert ctl.status == STATUS_ERROR
        assert errors

    def test_permission_denied_is_error_not_crash(self, sd, device_manager):
        sd._fail = PermissionError("denied")
        ctl = self._controller(sd, device_manager)
        assert ctl.enable() is False
        assert ctl.status == STATUS_ERROR

    def test_wake_word_suspends_detection(self, sd, device_manager):
        ctl = self._controller(sd, device_manager, engine=FakeEngine(score=1.0))
        wakes = []
        ctl.set_callbacks(on_wake_word=wakes.append)
        assert ctl.enable() is True
        stream = FakeSD.streams_created[-1]
        deadline = time.time() + 4
        while not wakes and time.time() < deadline:
            stream.callback(
                np.zeros((1600, 1), dtype=np.float32), 1600, None, None
            )
            time.sleep(0.05)
        assert wakes == ["hello jarvis"]
        assert ctl.status == STATUS_IDLE
        ctl.shutdown()

    def test_shutdown_releases_everything(self, sd, device_manager):
        engine = FakeEngine()
        ctl = self._controller(sd, device_manager, engine=engine)
        ctl.enable()
        ctl.shutdown()
        assert not ctl.is_listening
        assert engine.closed
        assert not FakeSD.streams_created[-1].active

    def test_fallback_does_not_overwrite_device_preference(self, sd, device_manager):
        # A configured device that no longer exists must not be replaced
        # in settings by whatever fallback device was used.
        settings = {"voice.microphone_device": "9:9:Missing Mic"}
        ctl = self._controller(sd, device_manager, repo=settings)
        assert ctl.enable() is True
        assert settings["voice.microphone_device"] == "9:9:Missing Mic"
        ctl.shutdown()

    def test_select_device_persists_and_restarts(self, sd, device_manager):
        settings = {}
        ctl = self._controller(sd, device_manager, repo=settings)
        ctl.enable()
        device_id = ctl.devices[0].device_id
        assert ctl.select_device(device_id) is True
        assert settings["voice.microphone_device"] == device_id
        assert ctl.is_listening
        assert ctl.select_device("9:9:Missing Mic") is False
        ctl.shutdown()

    def test_audio_stats_passthrough(self, sd, device_manager):
        ctl = self._controller(sd, device_manager)
        ctl.enable()
        stream = FakeSD.streams_created[-1]
        stream.callback(
            np.zeros((1600, 1), dtype=np.float32), 1600, None, None
        )
        time.sleep(0.3)
        stats = ctl.audio_stats()
        assert stats["frames_captured"] >= 1
        ctl.shutdown()

    def test_resume_after_suspend(self, sd, device_manager):
        ctl = self._controller(sd, device_manager)
        ctl.enable()
        ctl.suspend_detection()
        assert ctl.status == STATUS_IDLE
        ctl.resume_detection()
        assert ctl.is_listening
        ctl.shutdown()


# ----------------------------------------------------------------------
# Application state integration
# ----------------------------------------------------------------------
class TestStateIntegration:
    def test_wake_word_transitions_standby_to_activated(self, tmp_path):
        from src.core.application import VoiceOSApplication
        from src.core.state_manager import AppState

        app = VoiceOSApplication(database_path=tmp_path / "s.db")
        app.initialize()
        assert app.on_wake_word() is True
        assert app.state_manager.state is AppState.ACTIVATED
        assert app.session_manager.is_active
        # Idempotent: second call does not raise.
        assert app.on_wake_word() is True

    def test_explicit_disable_stops_capture(self, sd, device_manager):
        ctl = MicrophoneController(
            device_manager=device_manager,
            capture=AudioCaptureService(sd=sd, device_manager=device_manager),
            engine=FakeEngine(),
        )
        ctl.enable()
        stream = FakeSD.streams_created[-1]
        assert stream.active
        ctl.disable()
        assert not stream.active
