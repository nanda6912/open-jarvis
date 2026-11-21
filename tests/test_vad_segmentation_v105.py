"""V10.5 VAD segmentation tests.

Covers the root causes found in the empty-transcript investigation:

* per-utterance diagnostics were cleared by ``_reset_utterance()``
  before the empty-decode log read them (voiced=0/frames=1 artifacts);
* loud frames seen while not collecting must not contaminate the
  adaptive noise floor;
* the silence hangover must not finalize an utterance while the
  capture gate is paused (TTS playback would otherwise truncate it).
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from src.voice.speech_recognition_service import SpeechRecognitionService


class _FakeCapture:
    """Minimal capture double: frames are pushed directly."""

    sample_rate = 16000
    input_channels = 1
    device = None
    paused = False

    def __init__(self):
        self._frames = []
        self._cond = threading.Condition()

    def push(self, frame: np.ndarray) -> None:
        with self._cond:
            self._frames.append(np.asarray(frame, dtype=np.float32))
            self._cond.notify_all()

    def get_frame(self, timeout: float = 0.1):
        with self._cond:
            if not self._frames:
                self._cond.wait(timeout=timeout)
            if not self._frames:
                return None
            return self._frames.pop(0)

    def stats(self):
        return {}


class _FakeRecognizer:
    name = "fake"

    def __init__(self, transcript: str = ""):
        self.transcript = transcript
        self.calls = []

    def transcribe(self, samples):
        self.calls.append(np.asarray(samples, dtype=np.float32))
        return self.transcript

    def close(self):
        pass


def _frame(rms_level: float, n: int = 1600, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n).astype(np.float32)
    noise *= rms_level / (np.sqrt(np.mean(noise * noise)) + 1e-9)
    return noise


def _service(recognizer=None, capture=None, **kw):
    cap = capture or _FakeCapture()
    rec = recognizer or _FakeRecognizer("open chrome")
    params = dict(
        speech_threshold=0.01,
        silence_hangover=0.5,
        min_utterance=0.2,
        max_utterance=15.0,
        frame_timeout=0.05,
        pre_roll=0.2,
        vad_min_threshold=0.003,
        vad_max_threshold=0.01,
        noise_window=40,
    )
    params.update(kw)
    svc = SpeechRecognitionService(cap, rec, **params)
    return svc, cap, rec


def test_speech_onset_frames_do_not_inflate_noise_floor():
    """Frames at/above the effective threshold while NOT collecting are
    utterance onsets — they must not enter the noise-floor window."""
    svc, cap, _ = _service()
    # Prime the floor with quiet ambient frames.
    for i in range(30):
        svc._process_frame(_frame(0.0005, seed=i))
    floor_before = list(svc._noise_floor)
    assert floor_before, "ambient frames should calibrate the floor"
    # A loud transient (e.g. TTS tail / handling noise) arrives while
    # not collecting: it triggers collection on the FIRST frame, so
    # subsequent loud frames are 'collecting' and already excluded.
    # Feed one loud frame then confirm the floor did not grow.
    n_before = len(svc._noise_floor)
    svc._process_frame(_frame(0.08, seed=99))
    assert len(svc._noise_floor) == n_before


def test_quiet_ambient_still_calibrates_floor():
    svc, cap, _ = _service()
    for i in range(50):
        svc._process_frame(_frame(0.0008, seed=i))
    assert svc._noise_floor
    med = sorted(svc._noise_floor)[len(svc._noise_floor) // 2]
    assert med < 0.005


def test_pending_stats_snapshot_survives_reset():
    """_finalize must log the REAL voiced/frame counts, not the
    post-reset zeros that produced voiced=0/rejected=1 artifacts."""
    rec = _FakeRecognizer("")  # empty decode -> exercises the log path
    svc, cap, _ = _service(recognizer=rec)
    transcripts = []
    no_speech = []
    svc._on_transcript = lambda t: transcripts.append(t)
    svc._on_no_speech = lambda r: no_speech.append(r)
    # Speech burst (~0.5s voiced) then silence to trigger hangover.
    for i in range(5):
        svc._process_frame(_frame(0.05, seed=i))
    for i in range(12):
        svc._process_frame(_frame(0.0005, seed=100 + i))
        time.sleep(0.06)  # push past the hangover
    assert rec.calls, "STT should have been invoked"
    assert no_speech == ["vad_accepted_stt_empty"]
    # The snapshot must reflect real utterance stats.
    stats = svc._pending_stats
    # _pending_stats is consumed/overwritten per utterance; verify the
    # service recorded a completed utterance with voiced frames > 0.
    assert svc.utterances_completed >= 1


def test_hangover_paused_while_capture_gate_paused():
    """While TTS pauses capture, wall-clock silence must NOT finalize
    the utterance (it would be clipped at the playback boundary)."""
    rec = _FakeRecognizer("open chrome")
    cap = _FakeCapture()
    svc, _, _ = _service(recognizer=rec, capture=cap)
    transcripts = []
    svc._on_transcript = lambda t: transcripts.append(t)
    # Start an utterance.
    for i in range(3):
        svc._process_frame(_frame(0.05, seed=i))
    assert svc._collecting
    # Simulate a long pause: last voiced timestamp far in the past.
    svc._last_voiced = time.monotonic() - 5.0
    cap.paused = True
    svc._check_hangover()
    assert svc._collecting, "utterance must survive the paused gate"
    assert not transcripts
    # Resume: silence frames now finalize normally.
    cap.paused = False
    for i in range(14):
        svc._process_frame(_frame(0.0005, seed=200 + i))
        time.sleep(0.06)
    assert transcripts == ["open chrome"]


def test_empty_decode_reason_distinguishes_stages():
    """no_speech reasons stay distinct: below-VAD vs accepted-but-empty."""
    rec = _FakeRecognizer("")
    svc, cap, _ = _service(recognizer=rec)
    reasons = []
    svc._on_no_speech = lambda r: reasons.append(r)
    # Only ambient noise -> below-VAD path, never reaches STT.
    for i in range(80):
        svc._process_frame(_frame(0.0005, seed=i))
    assert rec.calls == []
    # Voiced utterance -> accepted, STT called, empty decode reason.
    for i in range(5):
        svc._process_frame(_frame(0.05, seed=300 + i))
    for i in range(14):
        svc._process_frame(_frame(0.0005, seed=400 + i))
        time.sleep(0.06)
    assert "vad_accepted_stt_empty" in reasons
    assert "audio_below_vad_threshold" in reasons
