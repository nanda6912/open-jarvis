"""V10.3 VAD / speech-capture recovery tests.

Covers the post-standby recognition path: fresh VAD calibration per
session, resume-time state reset, silence vs speech gating, and the
no-speech outcome taxonomy — all with mocked frames, no microphone.
"""

from __future__ import annotations

import time
import types

import numpy as np
import pytest

from src.voice.speech_recognition_service import (
    SAMPLE_RATE,
    SpeechRecognitionService,
)

from tests.test_mic_pipeline import FakeCapture, FakeRecognizer, make_stt

FRAME_N = 1600  # 100 ms @ 16 kHz
SILENCE = np.zeros(FRAME_N, dtype=np.float32)
SPEECH = (np.random.RandomState(0).randn(FRAME_N)
          .astype(np.float32) * 0.08)


def feed(cap, frames):
    for f in frames:
        cap._q.put(f)


class TestVadCalibration:
    def test_noise_floor_reset_on_start(self):
        """A stale noise floor from a previous session/device must not
        survive start() — VAD recalibrates on fresh audio."""
        cap = FakeCapture()
        stt = make_stt(cap, text="ok", speech_threshold=0.02)
        stt._noise_floor.extend([0.5] * 50)   # stale loud floor
        stt._effective_threshold = 0.02
        stt.start()
        try:
            assert len(stt._noise_floor) == 0
            assert stt._effective_threshold == stt._speech_threshold
            assert stt.stats()["session_frames"] == 0
        finally:
            stt.stop()

    def test_notify_system_resume_resets_vad(self):
        """Post-resume: partial utterance, noise floor and effective
        threshold are all reset so the first command is judged on
        fresh audio only."""
        cap = FakeCapture()
        stt = make_stt(cap, text="ok", speech_threshold=0.02)
        stt.start()
        try:
            feed(cap, [SPEECH] * 5)
            time.sleep(0.4)  # let worker collect an utterance
            stt._noise_floor.extend([0.4] * 30)
            stt._effective_threshold = 0.019
            stt._collecting = True
            stt._utterance = [SPEECH]
            stt.notify_system_resume()
            assert stt._collecting is False
            assert len(stt._noise_floor) == 0
            assert stt._effective_threshold == stt._speech_threshold
        finally:
            stt.stop()

    def test_session_frames_count_fresh_audio(self):
        cap = FakeCapture()
        stt = make_stt(cap, text="", speech_threshold=0.5)
        stt.start()
        try:
            feed(cap, [SILENCE] * 10)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if stt.stats()["session_frames"] >= 10:
                    break
                time.sleep(0.02)
            assert stt.stats()["session_frames"] >= 10
        finally:
            stt.stop()


class TestVadOutcomes:
    def test_silence_produces_no_speech_outcome(self):
        """Below-threshold audio emits audio_below_vad_threshold once
        per quiet stretch — genuine silence is not a capture fault."""
        outcomes = []
        cap = FakeCapture()
        stt = SpeechRecognitionService(
            cap, FakeRecognizer(""), speech_threshold=0.5,
            silence_hangover=0.2, min_utterance=0.05,
            frame_timeout=0.05,
            on_no_speech=lambda r: outcomes.append(r),
        )
        stt._below_vad_streak = 30  # fast-forward the streak
        stt._process_frame(SILENCE.copy())
        assert "audio_below_vad_threshold" in outcomes

    def test_speech_then_silence_finalizes_and_transcribes(self):
        """A voiced utterance followed by hangover silence reaches the
        recognizer exactly once."""
        transcripts = []
        cap = FakeCapture()
        rec = FakeRecognizer("open chrome")
        stt = SpeechRecognitionService(
            cap, rec, speech_threshold=0.02,
            silence_hangover=0.15, min_utterance=0.05,
            frame_timeout=0.05,
            on_transcript=lambda t: transcripts.append(t),
        )
        stt.start()
        try:
            feed(cap, [SPEECH] * 4 + [SILENCE] * 4)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline and not transcripts:
                time.sleep(0.02)
            assert transcripts == ["open chrome"]
            assert len(rec.calls) == 1
        finally:
            stt.stop()

    def test_empty_transcript_is_distinct_outcome(self):
        """VAD-accepted audio with an empty STT result reports
        vad_accepted_stt_empty — different from no-audio and from a
        wrong transcript."""
        outcomes = []
        cap = FakeCapture()
        stt = SpeechRecognitionService(
            cap, FakeRecognizer(""), speech_threshold=0.02,
            silence_hangover=0.15, min_utterance=0.05,
            frame_timeout=0.05,
            on_no_speech=lambda r: outcomes.append(r),
        )
        stt.start()
        try:
            feed(cap, [SPEECH] * 4 + [SILENCE] * 4)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                if "vad_accepted_stt_empty" in outcomes:
                    break
                time.sleep(0.02)
            assert "vad_accepted_stt_empty" in outcomes
        finally:
            stt.stop()
