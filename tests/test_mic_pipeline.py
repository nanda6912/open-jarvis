"""Tests for the mic pipeline fixes: KWS->STT handoff, adaptive VAD,
device identity, starvation diagnostics, and the registry watcher."""

from __future__ import annotations

import queue
import threading
import time
import types

import numpy as np
import pytest

from src.controllers.application_discovery import (
    ApplicationDiscovery,
    _RegistryWatcher,
)
from src.voice.audio_device_manager import (
    AudioDevice,
    AudioDeviceManager,
    DeviceUnavailableError,
)
from src.voice.speech_recognition_service import (
    SAMPLE_RATE,
    SpeechRecognitionService,
)


def frame(rms: float, n: int = 1600, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n).astype(np.float32)
            * (rms / max(float(np.sqrt(np.mean(
                rng.standard_normal(100) ** 2))), 1e-9)) * 0 + rms)


def const_frame(level: float, n: int = 1600) -> np.ndarray:
    return np.full(n, level, dtype=np.float32)


class FakeCapture:
    def __init__(self, frames=None):
        self._q = queue.Queue()
        for f in frames or []:
            self._q.put(f)
        self.device = types.SimpleNamespace(name="Test Mic")
        self.sample_rate = 16000
        self._stream = types.SimpleNamespace(active=True)
        self.flushed = 0
        self.paused = False

    def get_frame(self, timeout=0.5):
        try:
            return self._q.get(timeout=min(timeout, 0.05))
        except queue.Empty:
            return None

    def stats(self):
        return {"device": "Test Mic", "sample_rate": 16000,
                "running": True, "frames_captured": 10,
                "frames_queued": 10, "frame_age_seconds": 0.01}

    def flush(self):
        self.flushed += 1
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False


class FakeRecognizer:
    name = "fake-stt"

    def __init__(self, text=""):
        self.text = text
        self.calls = []

    def transcribe(self, audio):
        self.calls.append(len(audio))
        return self.text

    def close(self):
        pass


def make_stt(cap, text="", **kw):
    return SpeechRecognitionService(
        cap, FakeRecognizer(text),
        silence_hangover=kw.get("hangover", 0.2),
        min_utterance=kw.get("min_utt", 0.05),
        frame_timeout=0.05,
        **{k: v for k, v in kw.items()
           if k in ("vad_min_threshold", "vad_max_threshold",
                    "noise_window", "speech_threshold")},
    )


# ----------------------------------------------------------------------
# Adaptive VAD
# ----------------------------------------------------------------------
class TestAdaptiveVAD:
    def test_quiet_mic_speech_detected(self):
        """Intel-array-level speech (~0.005 rms) must trigger VAD even
        though it is below the old fixed 0.010 threshold."""
        cap = FakeCapture()
        stt = make_stt(cap, text="open chrome",
                       speech_threshold=0.01, vad_min_threshold=0.003)
        transcripts = []
        stt.set_callbacks(on_transcript=transcripts.append)
        stt.start()
        try:
            # Ambient silence first (calibrates noise floor).
            for _ in range(30):
                cap._q.put(const_frame(0.0002))
            time.sleep(0.4)
            # Quiet speech ~0.005 rms — above adaptive floor, below 0.01.
            for _ in range(10):
                cap._q.put(const_frame(0.005))
            for _ in range(10):
                cap._q.put(const_frame(0.0002))
            deadline = time.time() + 3
            while time.time() < deadline and not transcripts:
                time.sleep(0.05)
            assert transcripts == ["open chrome"]
            assert stt._effective_threshold <= 0.01
        finally:
            stt.stop()

    def test_background_noise_rejected(self):
        """Constant moderate noise must not become 'speech'."""
        cap = FakeCapture()
        stt = make_stt(cap, text="x", speech_threshold=0.01)
        transcripts = []
        stt.set_callbacks(on_transcript=transcripts.append)
        stt.start()
        try:
            for _ in range(120):
                cap._q.put(const_frame(0.0005))
            time.sleep(0.6)
            assert stt._effective_threshold <= 0.05
            # 0.0005 noise stays below threshold -> no collection.
            assert not stt._collecting
        finally:
            stt.stop()

    def test_loud_room_raises_threshold(self):
        """Noisy environment pushes effective threshold toward cap."""
        cap = FakeCapture()
        stt = make_stt(cap, speech_threshold=0.05, vad_max_threshold=0.08)
        stt.start()
        try:
            for _ in range(60):
                cap._q.put(const_frame(0.02))
            time.sleep(0.5)
            # median 0.02 * 6 = 0.12 -> capped at min(0.05, 0.08)=0.05
            assert stt._effective_threshold <= 0.05
        finally:
            stt.stop()

    def test_short_command_preserved(self):
        """A single voiced frame still opens an utterance."""
        cap = FakeCapture()
        stt = make_stt(cap, text="hi", min_utt=0.01)
        transcripts = []
        stt.set_callbacks(on_transcript=transcripts.append)
        stt.start()
        try:
            cap._q.put(const_frame(0.02))   # onset
            for _ in range(8):
                cap._q.put(const_frame(0.0001))
            deadline = time.time() + 3
            while time.time() < deadline and not transcripts:
                time.sleep(0.05)
            assert transcripts == ["hi"]
        finally:
            stt.stop()


# ----------------------------------------------------------------------
# Diagnostics / starvation / empty transcript
# ----------------------------------------------------------------------
class TestDiagnostics:
    def test_stats_fields(self):
        cap = FakeCapture()
        stt = make_stt(cap)
        s = stt.stats()
        for k in ("vad_threshold", "noise_floor", "device",
                  "sample_rate", "stream_active", "voiced_frames",
                  "rejected_frames", "frames_seen"):
            assert k in s

    def test_no_frames_emits_no_audio_frames(self):
        cap = FakeCapture(frames=[])
        stt = make_stt(cap)
        reasons = []
        stt.set_callbacks(on_no_speech=reasons.append)
        stt.start()
        try:
            deadline = time.time() + 4.0
            while time.time() < deadline and not reasons:
                time.sleep(0.05)
            assert "no_audio_frames" in reasons
        finally:
            stt.stop()

    def test_empty_transcript_no_dispatch(self):
        cap = FakeCapture()
        stt = make_stt(cap, text="   ")
        transcripts, no_speech = [], []
        stt.set_callbacks(on_transcript=transcripts.append,
                          on_no_speech=no_speech.append)
        stt.start()
        try:
            for _ in range(5):
                cap._q.put(const_frame(0.02))
            for _ in range(8):
                cap._q.put(const_frame(0.0001))
            deadline = time.time() + 3
            while (time.time() < deadline
                   and stt.utterances_completed < 1):
                time.sleep(0.05)
            time.sleep(0.3)
            assert transcripts == []
            assert any("vad_accepted_stt_empty" in r for r in no_speech)
        finally:
            stt.stop()

    def test_utterance_metrics_logged(self):
        cap = FakeCapture()
        stt = make_stt(cap, text="")
        stt.start()
        try:
            for _ in range(6):
                cap._q.put(const_frame(0.02))
            for _ in range(8):
                cap._q.put(const_frame(0.0001))
            deadline = time.time() + 3
            while time.time() < deadline and stt.utterances_completed < 1:
                time.sleep(0.05)
            assert stt.stats()["voiced_frames"] > 0
            assert stt._utterance_voiced >= 0
        finally:
            stt.stop()


# ----------------------------------------------------------------------
# Device identity
# ----------------------------------------------------------------------
class FakeSD:
    def __init__(self, devices):
        self._devices = devices
        self.default = types.SimpleNamespace(device=(0, 0))

    def query_devices(self):
        return self._devices


def dev(index, name, hapi=2, ch=2):
    return {"name": name, "max_input_channels": ch,
            "default_samplerate": 48000.0, "hostapi": hapi}


class TestDeviceIdentity:
    def _mgr(self, devices):
        return AudioDeviceManager(sd=FakeSD(devices))

    def test_name_match_beats_shifted_index(self):
        devices = [
            dev(0, "Headset Mic (Realtek)", hapi=2),
            dev(1, "Mic Array (Intel)", hapi=2),
        ]
        mgr = self._mgr(devices)
        # Persisted id says index 1 name 'Mic Array (Intel)' but device
        # order swapped: index 1 is now the Realtek.
        swapped = [devices[1], devices[0]]
        mgr2 = self._mgr(swapped)
        # Names in query_devices keep index positions; emulate the
        # persisted id from the OLD enumeration.
        persisted = "2:1:Mic Array (Intel)"
        # device_id name part must match a device on hostapi 2.
        got = mgr2.resolve_device(persisted)
        # name 'Mic Array (Intel)' now sits at index 0 -> name match wins
        assert got.name == "Mic Array (Intel)"

    def test_unmatched_name_uses_default_not_index(self):
        devices = [dev(0, "Mic A"), dev(1, "Mic B")]
        mgr = self._mgr(devices)
        # Named id whose name matches nothing must NOT bind whatever
        # device occupies index 1 (hot-unplug index reuse) — the
        # explicit default (index 0) is used instead.
        got = mgr.resolve_device("2:1:Totally Different Name")
        assert got.index == 0
        assert got.name == "Mic A"

    def test_nameless_legacy_id_still_uses_index(self):
        devices = [dev(0, "Mic A"), dev(1, "Mic B")]
        mgr = self._mgr(devices)
        # "2:1" carries no name at all — hostapi:index is the only
        # identity available, so the index fallback still applies.
        got = mgr.resolve_device("2:1")
        assert got.index == 1
        assert got.name == "Mic B"

    def test_missing_device_raises_when_none(self):
        mgr = self._mgr([])
        with pytest.raises(DeviceUnavailableError):
            mgr.resolve_device(None)


# ----------------------------------------------------------------------
# Registry watcher
# ----------------------------------------------------------------------
class FakeNotifyApi:
    """Scriptable advapi32 fake for watcher tests."""

    def __init__(self, fail_open=False, notify_rc=0, signal_on_arm=0):
        self.opened = []
        self.notified = []
        self.closed_keys = []
        self.closed_handles = []
        self._fail_open = fail_open
        self._notify_rc = notify_rc
        self._signal_on_arm = signal_on_arm
        self._event = object()
        self._wait_count = 0

    def open_key(self, hive, path):
        if self._fail_open:
            raise OSError("access denied")
        self.opened.append((hive, path))
        return object()

    def create_event(self):
        return self._event

    def notify(self, hkey, event):
        self.notified.append(hkey)
        return self._notify_rc

    def wait(self, event, timeout_ms):
        self._wait_count += 1
        # Simulate a registry change on the Nth wait.
        if self._wait_count >= self._signal_on_arm:
            return 0  # WAIT_OBJECT_0
        return 0x102  # WAIT_TIMEOUT

    def close_key(self, hkey):
        self.closed_keys.append(hkey)

    def close_handle(self, handle):
        self.closed_handles.append(handle)


class TestRegistryWatcher:
    def _disc(self, tmp_path):
        return ApplicationDiscovery(
            env={}, file_exists=lambda p: True,
            registry_reader=lambda k: {}, lnk_resolver=lambda p: "",
            dir_lister=lambda d: [], start_apps_lister=lambda: [],
        )

    def test_watcher_uses_advapi32_notify(self):
        api = FakeNotifyApi()
        d = self._disc(None)
        invalidated = []
        stop = d.start_registry_watcher(
            on_invalidate=lambda: invalidated.append(1), api=api)
        try:
            deadline = time.time() + 3
            while time.time() < deadline and not invalidated:
                time.sleep(0.05)
            assert api.opened  # RegOpenKeyExW path used
            assert api.notified  # RegNotifyChangeKeyValue armed
            assert invalidated  # change -> invalidate callback
        finally:
            stop()
            time.sleep(0.8)  # watcher wait timeout is 500 ms
        assert api.closed_keys and api.closed_handles

    def test_watcher_open_failure_falls_back(self):
        api = FakeNotifyApi(fail_open=True)
        d = self._disc(None)
        stop = d.start_registry_watcher(api=api)
        stop()  # clean no-op

    def test_watcher_notify_failure_stops_cleanly(self):
        api = FakeNotifyApi(notify_rc=5)
        d = self._disc(None)
        stop = d.start_registry_watcher(api=api)
        time.sleep(0.3)
        stop()
        time.sleep(0.8)
        assert api.closed_keys

    def test_watcher_clean_shutdown(self):
        api = FakeNotifyApi(signal_on_arm=10**9)  # never signals
        d = self._disc(None)
        stop = d.start_registry_watcher(api=api)
        time.sleep(0.3)
        stop()
        time.sleep(0.3)
        assert api.closed_keys

    def test_real_watcher_starts_on_windows(self):
        d = self._disc(None)
        stop = d.start_registry_watcher()
        time.sleep(0.3)
        stop()


# ----------------------------------------------------------------------
# KWS->STT handoff (controller level, fake mic)
# ----------------------------------------------------------------------
class FakeMic:
    def __init__(self, capture):
        self.capture = capture
        self.suspended = 0
        self.resumed = 0

    def suspend_detection(self, pause_capture=True):
        self.suspended += 1
        if pause_capture:
            self.capture.pause()

    def resume_detection(self):
        self.resumed += 1
        self.capture.resume()


class TestHandoff:
    def test_activation_resumes_capture_and_flushes(self):
        """handle_wake_word must un-pause capture, flush stale frames,
        then start STT — same queue, same device."""
        from src.core.session_manager import SessionManager
        from src.core.state_manager import StateManager
        from src.voice.voice_session_controller import (
            VoiceSessionController,
        )
        from src.voice.speech_output_service import SpeechOutputService

        cap = FakeCapture(frames=[const_frame(0.001)] * 50)
        stt = make_stt(cap, text="")
        tts = SpeechOutputService(
            synthesizer=types.SimpleNamespace(
                synthesize=lambda t: (np.zeros(1600, np.float32),
                                      16000)),
            player=types.SimpleNamespace(play=lambda *a, **k: None),
        )
        mic = FakeMic(cap)
        ctl = VoiceSessionController(
            StateManager(), SessionManager(), stt, tts,
            microphone_controller=mic,
        )
        cap.pause()  # simulate detection-time pause
        assert ctl.handle_wake_word("hello jarvis")
        assert cap.paused is False       # resumed for STT
        assert cap.flushed >= 1           # stale frames dropped
        # Same capture object feeds STT — no second stream.
        assert stt._capture is cap
        assert mic.capture.device.name == "Test Mic"
        ctl.shutdown()
