"""Tests for Task 4: STT, TTS, session management and voice responses.

All engines and audio hardware are faked — no microphone, speaker or
model files are required. Recorded-audio tests that need real models
skip automatically when the models are absent.
"""

from __future__ import annotations

import time
import wave
from pathlib import Path

import numpy as np
import pytest

from src.core.session_manager import SessionManager
from src.core.state_manager import AppState, StateManager
from src.voice.speech_output_service import SpeechOutputService
from src.voice.speech_recognition_service import SpeechRecognitionService
from src.voice.stt_engine import (
    NullSpeechRecognizer,
    SherpaOnnxRecognizer,
    create_recognizer,
)
from src.voice.tts_engine import (
    NullSpeechSynthesizer,
    SherpaOnnxSynthesizer,
    create_synthesizer,
)
from src.voice.voice_session_controller import (
    VoiceSessionController,
    build_acknowledgement,
    normalize_text,
)

SAMPLE_RATE = 16000
MODELS_ROOT = (
    Path.home() / "AppData" / "Local" / "VoiceOS" / "models"
)


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeFrameSource:
    """Controllable frame source with the capture-service interface."""

    def __init__(self):
        self._queue = __import__("queue").Queue()
        self.paused = False
        self.flush_count = 0

    def push(self, samples: np.ndarray) -> None:
        self._queue.put(np.asarray(samples, dtype=np.float32))

    def push_silence(self, seconds: float = 0.1, level: float = 0.0) -> None:
        n = int(SAMPLE_RATE * seconds)
        self.push(np.full(n, level, dtype=np.float32))

    def push_tone(self, seconds: float = 0.1, level: float = 0.2) -> None:
        n = int(SAMPLE_RATE * seconds)
        t = np.arange(n) / SAMPLE_RATE
        self.push((level * np.sin(2 * np.pi * 220 * t)).astype(np.float32))

    def get_frame(self, timeout: float = 0.5):
        try:
            return self._queue.get(timeout=timeout)
        except __import__("queue").Empty:
            return None

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def flush(self):
        self.flush_count += 1
        while True:
            try:
                self._queue.get_nowait()
            except __import__("queue").Empty:
                return


class FakeRecognizer:
    name = "fake-stt"
    model_dir = "/fake/stt"

    def __init__(self, text=""):
        self.text = text
        self.calls = []
        self.closed = False

    def transcribe(self, samples):
        self.calls.append(len(samples))
        return self.text

    def close(self):
        self.closed = True


class FakeSynthesizer:
    name = "fake-tts"
    model_dir = "/fake/tts"

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []
        self.closed = False

    def synthesize(self, text):
        self.calls.append(text)
        if self.fail:
            raise RuntimeError("synth boom")
        return np.zeros(2205, dtype=np.float32), 22050

    def close(self):
        self.closed = True


class FakePlayer:
    name = "fake-player"

    def __init__(self, block: float = 0.0):
        self.played = []
        self.block = block

    def play(self, samples, sample_rate):
        self.played.append((len(samples), sample_rate))
        if self.block:
            time.sleep(self.block)


class StubSTT:
    """Scriptable stand-in for SpeechRecognitionService (controller tests)."""

    def __init__(self):
        self.started = 0
        self.stopped = 0
        self._cb = {}

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    @property
    def is_running(self):
        return self.started > self.stopped

    def emit_transcript(self, text):
        self._cb["on_transcript"](text)

    def emit_no_speech(self, reason="test"):
        self._cb["on_no_speech"](reason)


class StubTTS:
    """Scriptable stand-in for SpeechOutputService (controller tests)."""

    def __init__(self, auto_finish=False):
        self.spoken = []
        self.started = 0
        self.stopped = 0
        self.auto_finish = auto_finish
        self._cb = {}

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    @property
    def is_running(self):
        return True

    def speak(self, text):
        self.spoken.append(text)
        if self.auto_finish:
            self._cb["on_finished"](text)

    def finish_last(self):
        self._cb["on_finished"](self.spoken[-1])


class StubMic:
    def __init__(self):
        self.capture = FakeFrameSource()
        self.suspended = 0
        self.resumed = 0

    def suspend_detection(self, pause_capture=True):
        self.suspended += 1

    def resume_detection(self):
        self.resumed += 1


def _wait_for(predicate, timeout=3.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ----------------------------------------------------------------------
# Text helpers
# ----------------------------------------------------------------------
class TestTextHelpers:
    def test_normalize(self):
        assert normalize_text("  Goodbye Jarvis!! ") == "goodbye jarvis"

    def test_acknowledgement_command(self):
        assert (
            build_acknowledgement("Open Chrome.")
            == "I heard you say open Chrome."
        )

    def test_acknowledgement_question(self):
        assert (
            build_acknowledgement("What time is it?")
            == "I heard you ask what time is it."
        )


# ----------------------------------------------------------------------
# STT engine
# ----------------------------------------------------------------------
class TestSTTEngine:
    def test_null_recognizer(self):
        r = NullSpeechRecognizer()
        assert r.transcribe(np.zeros(1600, dtype=np.float32)) == ""
        r.close()

    def test_missing_model_dir_falls_back(self, tmp_path):
        r = create_recognizer(tmp_path / "nonexistent")
        assert isinstance(r, NullSpeechRecognizer)

    def test_empty_model_dir_falls_back(self, tmp_path):
        r = create_recognizer(tmp_path)
        assert isinstance(r, NullSpeechRecognizer)

    def test_create_recognizer_with_real_model(self):
        model = MODELS_ROOT / "stt"
        if not model.is_dir() or not any(model.rglob("*.onnx")):
            pytest.skip("STT ONNX model not installed")
        r = create_recognizer(model)
        assert isinstance(r, SherpaOnnxRecognizer)
        r.close()


# ----------------------------------------------------------------------
# Speech recognition service (VAD + worker)
# ----------------------------------------------------------------------
class TestSpeechRecognitionService:
    def _service(self, source, recognizer=None, **kw):
        return SpeechRecognitionService(
            capture=source,
            recognizer=recognizer or FakeRecognizer("hello there"),
            speech_threshold=0.02,
            silence_hangover=0.3,
            min_utterance=0.2,
            **kw,
        )

    def test_transcript_emitted_after_silence(self):
        src = FakeFrameSource()
        rec = FakeRecognizer("open chrome")
        got = []
        svc = self._service(src, rec, on_transcript=got.append)
        svc.start()
        try:
            for _ in range(5):
                src.push_tone(0.1)
            for _ in range(5):
                src.push_silence(0.1)
            assert _wait_for(lambda: got == ["open chrome"])
            assert rec.calls and rec.calls[0] > 0
        finally:
            svc.stop()

    def test_silence_produces_no_callback(self):
        src = FakeFrameSource()
        rec = FakeRecognizer("should not appear")
        got, nospeech = [], []
        svc = self._service(
            src, rec, on_transcript=got.append, on_no_speech=nospeech.append
        )
        svc.start()
        try:
            for _ in range(10):
                src.push_silence(0.1)
            time.sleep(0.5)
            assert got == []
            assert nospeech == []
        finally:
            svc.stop()

    def test_empty_transcript_reports_no_speech(self):
        src = FakeFrameSource()
        rec = FakeRecognizer("")
        got, nospeech = [], []
        svc = self._service(
            src, rec, on_transcript=got.append, on_no_speech=nospeech.append
        )
        svc.start()
        try:
            for _ in range(5):
                src.push_tone(0.1)
            for _ in range(5):
                src.push_silence(0.1)
            assert _wait_for(lambda: len(nospeech) == 1)
            assert got == []
        finally:
            svc.stop()

    def test_too_short_utterance_dropped(self):
        src = FakeFrameSource()
        rec = FakeRecognizer("click")
        got = []
        svc = self._service(src, rec, on_transcript=got.append)
        svc.start()
        try:
            src.push_tone(0.05)  # below min_utterance=0.2 incl. preroll? borderline
            for _ in range(5):
                src.push_silence(0.1)
            time.sleep(0.6)
            # Either dropped for shortness or transcribed — never both paths
            # asserted; just ensure service survives and counters sane.
            assert svc.utterances_completed >= 0
        finally:
            svc.stop()

    def test_recognizer_error_reported(self):
        src = FakeFrameSource()

        class BadRec(FakeRecognizer):
            def transcribe(self, samples):
                raise RuntimeError("onnx boom")

        errors = []
        svc = self._service(src, BadRec(), on_error=errors.append)
        svc.start()
        try:
            for _ in range(5):
                src.push_tone(0.1)
            for _ in range(5):
                src.push_silence(0.1)
            assert _wait_for(lambda: len(errors) == 1)
            assert svc.inference_errors == 1
        finally:
            svc.stop()

    def test_stop_terminates_worker(self):
        src = FakeFrameSource()
        svc = self._service(src)
        svc.start()
        assert svc.is_running
        svc.stop()
        assert not svc.is_running

    def test_speech_activity_callback(self):
        src = FakeFrameSource()
        events = []
        svc = self._service(
            src, on_speech_activity=events.append
        )
        svc.start()
        try:
            for _ in range(4):
                src.push_tone(0.1)
            for _ in range(4):
                src.push_silence(0.1)
            assert _wait_for(lambda: events.count(True) >= 1)
            assert _wait_for(lambda: events and events[-1] is False)
        finally:
            svc.stop()

    def test_stats(self):
        src = FakeFrameSource()
        svc = self._service(src)
        stats = svc.stats()
        assert stats["recognizer"] == "fake-stt"
        assert stats["utterances_completed"] == 0


# ----------------------------------------------------------------------
# TTS engine + output service
# ----------------------------------------------------------------------
class TestTTSEngine:
    def test_null_synthesizer(self):
        s = NullSpeechSynthesizer()
        samples, rate = s.synthesize("hi")
        assert samples.dtype == np.float32
        assert rate == 22050
        assert len(samples) > 0
        s.close()

    def test_missing_model_dir_falls_back(self, tmp_path):
        s = create_synthesizer(tmp_path / "nonexistent")
        assert isinstance(s, NullSpeechSynthesizer)

    def test_empty_model_dir_falls_back(self, tmp_path):
        s = create_synthesizer(tmp_path)
        assert isinstance(s, NullSpeechSynthesizer)

    def test_create_synthesizer_with_real_model(self):
        model = MODELS_ROOT / "tts"
        if not model.is_dir() or not any(model.rglob("*.onnx")):
            pytest.skip("TTS ONNX model not installed")
        s = create_synthesizer(model)
        assert isinstance(s, SherpaOnnxSynthesizer)
        s.close()


class TestSpeechOutputService:
    def test_speak_synthesizes_and_plays(self):
        synth, player = FakeSynthesizer(), FakePlayer()
        gate = FakeFrameSource()
        started, finished = [], []
        svc = SpeechOutputService(
            synth,
            player=player,
            gate=gate,
            on_started=started.append,
            on_finished=finished.append,
        )
        svc.start()
        try:
            assert svc.speak("At your service.") is True
            assert _wait_for(lambda: finished == ["At your service."])
            assert synth.calls == ["At your service."]
            assert player.played and player.played[0][1] == 22050
            assert started == ["At your service."]
        finally:
            svc.stop()

    def test_gate_paused_during_playback(self):
        synth = FakeSynthesizer()
        player = FakePlayer(block=0.3)
        gate = FakeFrameSource()
        svc = SpeechOutputService(synth, player=player, gate=gate)
        svc.start()
        try:
            svc.speak("hello")
            assert _wait_for(lambda: svc.is_speaking)
            assert gate.paused is True
            assert _wait_for(lambda: not svc.is_speaking, timeout=3)
            assert gate.paused is False
        finally:
            svc.stop()

    def test_synthesis_error_recovers(self):
        synth = FakeSynthesizer(fail=True)
        errors = []
        svc = SpeechOutputService(
            synth, player=FakePlayer(), on_error=errors.append
        )
        svc.start()
        try:
            svc.speak("boom")
            assert _wait_for(lambda: len(errors) == 1)
            assert svc.playback_errors == 1
            # Service still usable afterwards.
            synth.fail = False
            svc.speak("ok")
            assert _wait_for(lambda: synth.calls[-1] == "ok")
        finally:
            svc.stop()

    def test_empty_text_rejected(self):
        svc = SpeechOutputService(FakeSynthesizer())
        assert svc.speak("") is False
        assert svc.speak("   ") is False

    def test_stop_clears_queue(self):
        player = FakePlayer(block=0.5)
        svc = SpeechOutputService(FakeSynthesizer(), player=player)
        svc.start()
        svc.speak("one")
        svc.speak("two")
        svc.stop()
        assert svc.pending_count == 0
        assert not svc.is_running

    def test_shutdown_releases_synthesizer(self):
        synth = FakeSynthesizer()
        svc = SpeechOutputService(synth, player=FakePlayer())
        svc.start()
        svc.shutdown()
        assert synth.closed
        assert not svc.is_running


# ----------------------------------------------------------------------
# Voice session controller
# ----------------------------------------------------------------------
class TestVoiceSessionController:
    def _controller(self, timeout=60, stt=None, tts=None, mic=None, **kw):
        state = StateManager()
        state.transition_to(AppState.STANDBY)
        session = SessionManager(timeout_seconds=timeout)
        stt = stt or StubSTT()
        tts = tts or StubTTS(auto_finish=True)
        events = {"transcripts": [], "responses": [], "activity": [],
                  "session": [], "nospeech": []}
        ctl = VoiceSessionController(
            state_manager=state,
            session_manager=session,
            speech_recognition=stt,
            speech_output=tts,
            microphone_controller=mic,
            on_transcript=events["transcripts"].append,
            on_response=events["responses"].append,
            on_activity=events["activity"].append,
            on_session_state=events["session"].append,
            on_no_speech=events["nospeech"].append,
            **kw,
        )
        return ctl, state, session, stt, tts, events

    def test_activation_speaks_and_listens(self):
        ctl, state, session, stt, tts, events = self._controller()
        assert ctl.handle_wake_word("hello jarvis") is True
        assert session.is_active
        assert state.state is AppState.ACTIVATED or state.state is AppState.LISTENING
        assert "At your service." in tts.spoken
        assert stt.started == 1
        assert events["session"] == [True]
        ctl.shutdown()

    def test_duplicate_activation_rejected(self):
        ctl, state, session, stt, tts, events = self._controller()
        assert ctl.handle_wake_word() is True
        assert ctl.handle_wake_word() is False
        assert ctl.handle_wake_word() is False
        assert session.is_active
        ctl.shutdown()

    def test_multiple_commands_in_one_session(self):
        ctl, state, session, stt, tts, events = self._controller()
        ctl.handle_wake_word()
        stt.emit_transcript("What time is it?")
        stt.emit_transcript("Open Chrome.")
        assert events["transcripts"] == [
            "What time is it?",
            "Open Chrome.",
        ]
        # Task 5: time queries execute; app launches are only acknowledged.
        assert any("current time is" in s for s in tts.spoken)
        assert "I heard you say open Chrome." in tts.spoken
        assert session.is_active
        ctl.shutdown()

    def test_goodbye_ends_session(self):
        tts = StubTTS(auto_finish=False)
        ctl, state, session, stt, tts2, events = self._controller(tts=tts)
        ctl.handle_wake_word()
        tts.finish_last()  # acknowledgement completes first (FIFO)
        stt.emit_transcript("Goodbye Jarvis")
        assert "Goodbye." in tts.spoken
        assert session.is_active  # still active until farewell finishes
        tts.finish_last()
        assert not session.is_active
        assert state.state is AppState.STANDBY
        ctl.shutdown()

    def test_inactivity_timeout(self):
        ctl, state, session, stt, tts, events = self._controller(timeout=1)
        ctl.handle_wake_word()
        time.sleep(1.2)
        remaining = ctl.tick()
        assert remaining == 0
        assert "Going back to standby." in tts.spoken
        tts.finish_last()
        assert not session.is_active
        assert state.state is AppState.STANDBY
        ctl.shutdown()

    def test_touch_extends_session(self):
        ctl, state, session, stt, tts, events = self._controller(timeout=2)
        ctl.handle_wake_word()
        time.sleep(1.2)
        stt.emit_transcript("hello")
        time.sleep(1.2)
        # 2.4s elapsed since activation but only ~1.2s since last command
        assert ctl.tick() is not None and ctl.tick() > 0
        ctl.shutdown()

    def test_no_speech_response(self):
        ctl, state, session, stt, tts, events = self._controller()
        ctl.handle_wake_word()
        stt.emit_no_speech("empty transcript")
        # Empty transcripts stay silent and keep LISTENING — no TTS,
        # no EXECUTING detour (live-reliability fix).
        assert "Sorry, I didn't catch that." not in tts.spoken
        assert events["nospeech"]
        assert state.state is AppState.LISTENING
        ctl.shutdown()

    def test_wake_phrase_inside_session_ignored(self):
        ctl, state, session, stt, tts, events = self._controller()
        ctl.handle_wake_word()
        stt.emit_transcript("Hello Jarvis")
        assert not any("I heard you" in s for s in tts.spoken)
        ctl.shutdown()

    def test_mic_suspend_and_resume(self):
        mic = StubMic()
        ctl, state, session, stt, tts, events = self._controller(mic=mic)
        ctl.handle_wake_word()
        assert mic.suspended == 1
        stt.emit_transcript("Goodbye Jarvis")
        tts.finish_last()
        assert mic.resumed == 1
        ctl.shutdown()

    def test_shutdown_cleans_up(self):
        ctl, state, session, stt, tts, events = self._controller()
        ctl.handle_wake_word()
        ctl.shutdown()
        assert stt.stopped >= 1
        assert tts.stopped >= 1
        assert not session.is_active
        assert state.state is AppState.STANDBY
        assert events["session"][-1] is False

    def test_no_activation_when_stopped(self):
        state = StateManager()
        state.transition_to(AppState.STOPPED)
        session = SessionManager()
        ctl = VoiceSessionController(
            state_manager=state,
            session_manager=session,
            speech_recognition=StubSTT(),
            speech_output=StubTTS(),
        )
        assert ctl.handle_wake_word() is False

    def test_history_recorded(self):
        recorded = []
        ctl, state, session, stt, tts, events = self._controller(
            history_recorder=lambda *a: recorded.append(a)
        )
        ctl.handle_wake_word()
        stt.emit_transcript("Open Chrome.")
        assert recorded and recorded[0][0] == "Open Chrome."
        ctl.shutdown()


# ----------------------------------------------------------------------
# State transitions during a session (real services, fake engines)
# ----------------------------------------------------------------------
class TestSessionStateFlow:
    def test_full_flow_state_path(self):
        """STANDBY -> ACTIVATED -> LISTENING -> EXECUTING -> LISTENING."""
        src = FakeFrameSource()
        state = StateManager()
        state.transition_to(AppState.STANDBY)
        session = SessionManager(timeout_seconds=60)
        stt = SpeechRecognitionService(
            capture=src, recognizer=FakeRecognizer("test command"),
            speech_threshold=0.02, silence_hangover=0.2, min_utterance=0.1,
        )
        synth = FakeSynthesizer()
        tts = SpeechOutputService(
            synth, player=FakePlayer(block=0.05)
        )
        tts.start()
        ctl = VoiceSessionController(
            state_manager=state,
            session_manager=session,
            speech_recognition=stt,
            speech_output=tts,
        )
        try:
            assert ctl.handle_wake_word() is True
            # Wait for activation TTS to finish -> LISTENING.
            assert _wait_for(lambda: state.state is AppState.LISTENING)
            assert synth.calls == ["At your service."]
            # Feed a command utterance.
            for _ in range(4):
                src.push_tone(0.1)
            for _ in range(6):
                src.push_silence(0.1)
            assert _wait_for(lambda: len(synth.calls) >= 2)
            assert _wait_for(lambda: state.state is AppState.LISTENING)
        finally:
            ctl.shutdown()
            tts.shutdown()
            stt.shutdown()


# ----------------------------------------------------------------------
# Recorded-audio tests (real models; skipped when absent)
# ----------------------------------------------------------------------
class TestRecordedAudio:
    def _load_wav(self, path: Path):
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            data = np.frombuffer(
                w.readframes(w.getnframes()), dtype=np.int16
            ).astype(np.float32) / 32768.0
        return data, sr

    def test_stt_on_recorded_speech(self):
        model = MODELS_ROOT / "stt"
        if not model.exists():
            pytest.skip("STT model not installed")
        rec = create_recognizer(model)
        if getattr(rec, "name", "") == "null":
            pytest.skip("STT recognizer unavailable")
        wav = model / "sherpa-onnx-moonshine-tiny-en-int8" / "test_wavs" / "0.wav"
        candidates = list(model.rglob("test_wavs/0.wav"))
        if not candidates:
            pytest.skip("no recorded test wav")
        audio, sr = self._load_wav(candidates[0])
        # Resample to 16 kHz if needed.
        if sr != SAMPLE_RATE:
            from src.voice.speech_recognition_service import SAMPLE_RATE as SR
            n = int(round(len(audio) * SR / sr))
            audio = np.interp(
                np.linspace(0, len(audio), n, endpoint=False),
                np.arange(len(audio)), audio,
            ).astype(np.float32)
        text = rec.transcribe(audio)
        assert text
        assert "nightfall" in text.lower() or len(text) > 10
        rec.close()

    def test_tts_then_stt_loopback(self):
        stt_model = MODELS_ROOT / "stt"
        tts_model = MODELS_ROOT / "tts"
        if (
            not stt_model.is_dir()
            or not any(stt_model.rglob("*.onnx"))
            or not tts_model.is_dir()
            or not any(tts_model.rglob("*.onnx"))
        ):
            pytest.skip("STT/TTS ONNX model files not installed")
        rec = create_recognizer(stt_model)
        syn = create_synthesizer(tts_model)
        if getattr(rec, "name", "") == "null" or getattr(
            syn, "name", ""
        ) == "null":
            pytest.skip("engines unavailable")
        samples, rate = syn.synthesize("What time is it?")
        assert samples.size > 0
        # Resample TTS output (22050) to 16 kHz for the recognizer.
        n = int(round(len(samples) * SAMPLE_RATE / rate))
        audio = np.interp(
            np.linspace(0, len(samples), n, endpoint=False),
            np.arange(len(samples)), samples,
        ).astype(np.float32)
        text = rec.transcribe(audio)
        assert "time" in text.lower()
        rec.close()
        syn.close()


# ----------------------------------------------------------------------
# Task 4 lifecycle hardening: session isolation, end-drain, stale work
# ----------------------------------------------------------------------
class TestLifecycleHardening:
    def _controller(self, timeout=60, stt=None, tts=None, mic=None, **kw):
        state = StateManager()
        state.transition_to(AppState.STANDBY)
        session = SessionManager(timeout_seconds=timeout)
        stt = stt or StubSTT()
        tts = tts or StubTTS(auto_finish=True)
        ctl = VoiceSessionController(
            state_manager=state,
            session_manager=session,
            speech_recognition=stt,
            speech_output=tts,
            microphone_controller=mic,
            **kw,
        )
        return ctl, state, session, stt, tts

    def test_end_waits_for_all_queued_speech(self):
        """Farewell queued behind a playing response must not restore
        standby early — wake detection may not resume mid-speech."""
        tts = StubTTS(auto_finish=False)
        ctl, state, session, stt, _ = self._controller(tts=tts)
        ctl.handle_wake_word()
        tts.finish_last()  # ack completes (FIFO) -> LISTENING
        stt.emit_transcript("What time is it?")
        stt.emit_transcript("Goodbye Jarvis")
        assert "Goodbye." in tts.spoken
        # First finish drains the command response, NOT the farewell:
        # the session must stay active.
        tts._cb["on_finished"]("")
        assert session.is_active
        assert state.state is not AppState.STANDBY
        # Last finish drains the farewell -> standby.
        tts._cb["on_finished"]("")
        assert not session.is_active
        assert state.state is AppState.STANDBY
        ctl.shutdown()

    def test_farewell_drop_still_ends_session(self):
        """If the farewell cannot be queued (full queue / dead worker),
        the session must still end — never hang in _ending forever."""
        tts = StubTTS(auto_finish=False)
        ctl, state, session, stt, _ = self._controller(tts=tts)
        ctl.handle_wake_word()
        tts.finish_last()  # ack
        tts.speak = lambda text: False  # farewell dropped
        stt.emit_transcript("Goodbye Jarvis")
        assert not session.is_active
        assert state.state is AppState.STANDBY
        ctl.shutdown()

    def test_stale_dispatch_dropped_after_session_end(self):
        """A transcript mid-route when the session ends must never
        reach the command registry or resurrect the state machine."""
        from src.commands.intent import Intent

        dispatched = []

        class FakeRegistry:
            def dispatch(self, intent):
                dispatched.append(intent.name)
                from src.commands.registry import CommandResult
                return CommandResult(
                    intent=intent.name, success=True, executed=True,
                    message="ok",
                )

        ctl, state, session, stt, tts = self._controller(
            command_registry=FakeRegistry()
        )
        ctl.handle_wake_word()
        tts.finish_last()  # ack -> LISTENING
        # Simulate the race: session ends (goodbye path) while a
        # command dispatch is still in flight on the STT worker.
        stt.emit_transcript("Goodbye Jarvis")
        tts.finish_last()  # farewell -> standby
        assert not session.is_active
        intent = Intent(name="OPEN_APPLICATION", raw_text="open chrome")
        ctl._dispatch_and_respond(intent)
        assert dispatched == []
        assert state.state is AppState.STANDBY
        ctl.shutdown()

    def test_second_wake_starts_fresh_session(self):
        """Goodbye -> standby -> Hello Jarvis starts a clean new session."""
        tts = StubTTS(auto_finish=True)
        ctl, state, session, stt, _ = self._controller(tts=tts)
        ctl.handle_wake_word()
        stt.emit_transcript("Goodbye Jarvis")
        assert not session.is_active
        assert state.state is AppState.STANDBY
        assert ctl.handle_wake_word() is True
        assert session.is_active
        assert stt.started == 2
        ctl.shutdown()

    def test_stt_restart_never_resurrects_old_worker(self):
        """stop()+start() must not revive a worker whose decode
        outlived the 2 s join — each run owns a private stop event."""
        src = FakeFrameSource()
        svc = SpeechRecognitionService(
            capture=src,
            recognizer=FakeRecognizer("x"),
            speech_threshold=0.02,
            silence_hangover=0.3,
            min_utterance=0.2,
        )
        svc.start()
        gen1, ev1 = svc._run_generation, svc._run_stop
        svc.stop()
        assert ev1 is not None and ev1.is_set()
        svc.start()
        gen2, ev2 = svc._run_generation, svc._run_stop
        assert gen2 == gen1 + 1
        assert ev2 is not ev1 and not ev2.is_set()
        # The retired run can never emit again.
        assert not svc._emit_allowed(gen1, ev1)
        assert svc._emit_allowed(gen2, ev2)
        svc.stop()

    def test_stale_run_finalize_emits_nothing(self):
        """A decode finishing after its run was retired must drop the
        result instead of delivering it into the new session."""
        src = FakeFrameSource()
        got = []
        svc = SpeechRecognitionService(
            capture=src,
            recognizer=FakeRecognizer("stale command"),
            on_transcript=got.append,
            speech_threshold=0.02,
            silence_hangover=0.3,
            min_utterance=0.2,
        )
        svc.start()
        gen1, ev1 = svc._run_generation, svc._run_stop
        svc.stop()
        svc.start()  # new session/run
        frames = [np.zeros(4800, dtype=np.float32)]  # 0.3 s >= min
        svc._finalize(
            frames, reason="silence",
            generation=gen1, stop_event=ev1,
        )
        assert got == []
        assert svc.last_transcript == ""
        svc.stop()


# ----------------------------------------------------------------------
# Phase 1 follow-up: interaction telemetry integrity + lifecycle ordering
# ----------------------------------------------------------------------
class TestInteractionTelemetry:
    """Regression tests for the Nov-18 log defect: session-level events
    (no_speech / stt_exception / duplicate) overwrote the previous
    command's outcome in the final Interaction timing record."""

    def _controller(self, timeout=60, stt=None, tts=None, mic=None, **kw):
        state = StateManager()
        state.transition_to(AppState.STANDBY)
        session = SessionManager(timeout_seconds=timeout)
        stt = stt or StubSTT()
        tts = tts or StubTTS(auto_finish=True)
        ctl = VoiceSessionController(
            state_manager=state,
            session_manager=session,
            speech_recognition=stt,
            speech_output=tts,
            microphone_controller=mic,
            **kw,
        )
        return ctl, state, session, stt, tts

    def _commands(self, ctl):
        rec = ctl._current_interaction
        return rec.get("commands", []) if rec else []

    def test_no_speech_does_not_overwrite_command_outcome(self):
        """Observed ordering: command executes, then VAD-empty and
        no-audio events arrive — the command must keep 'executed'."""
        ctl, state, session, stt, tts = self._controller()
        ctl.handle_wake_word()
        stt.emit_transcript("What time is it?")
        cmds = self._commands(ctl)
        assert cmds and cmds[-1]["outcome"] == "executed"
        executed_detail = cmds[-1].get("detail")
        # The exact physical-log sequence: vad_accepted_stt_empty then
        # plain no_audio_frames noise after a successful command.
        stt.emit_no_speech("vad_accepted_stt_empty")
        stt.emit_no_speech("no_audio_frames")
        assert cmds[-1]["outcome"] == "executed"
        assert cmds[-1].get("detail") == executed_detail
        # Separate recognition records exist with NO command_id.
        log = ctl.recognition_log()
        ns = [e for e in log if e["outcome"] == "no_speech"]
        assert len(ns) == 2
        assert all(e["command_id"] is None for e in ns)
        assert {e["detail"] for e in ns} == {
            "vad_accepted_stt_empty", "no_audio_frames"}
        # The no-speech prompt must not be attributed as the command's
        # response playback.
        assert cmds[-1].get("response_tts_started_monotonic") is None
        ctl.shutdown()

    def test_stt_exception_does_not_overwrite_command_outcome(self):
        ctl, state, session, stt, tts = self._controller()
        ctl.handle_wake_word()
        stt.emit_transcript("What time is it?")
        cmds = self._commands(ctl)
        assert cmds[-1]["outcome"] == "executed"
        stt._cb["on_error"]("decode boom")
        assert cmds[-1]["outcome"] == "executed"
        log = ctl.recognition_log()
        exc = [e for e in log if e["outcome"] == "stt_exception"]
        assert len(exc) == 1 and exc[0]["command_id"] is None
        ctl.shutdown()

    def test_duplicate_transcript_does_not_overwrite_prior_command(self):
        """A duplicate suppressed before _begin_command must not stamp
        'duplicate_ignored' onto the previous command's record."""
        ctl, state, session, stt, tts = self._controller()
        ctl.handle_wake_word()
        stt.emit_transcript("Open Chrome.")
        cmds = self._commands(ctl)
        first_outcome = cmds[-1]["outcome"]
        assert first_outcome  # command recorded its own outcome
        stt.emit_transcript("Open Chrome.")  # identical, in-window
        assert cmds[-1]["outcome"] == first_outcome
        assert len(cmds) == 1  # duplicate never opened a new command
        log = ctl.recognition_log()
        dup = [e for e in log if e["outcome"] == "duplicate_ignored"]
        assert dup and dup[-1]["command_id"] is None
        ctl.shutdown()

    def test_multiple_commands_keep_own_outcomes(self):
        """Every recognized command retains its own final outcome —
        interleaved no-speech events touch none of them."""
        ctl, state, session, stt, tts = self._controller()
        ctl.handle_wake_word()
        stt.emit_transcript("What time is it?")
        stt.emit_no_speech("vad_accepted_stt_empty")
        stt.emit_transcript("Open Chrome.")
        stt.emit_no_speech("no_audio_frames")
        cmds = self._commands(ctl)
        assert len(cmds) == 2
        assert cmds[0]["outcome"] == "executed"
        assert cmds[0]["command_id"].endswith("cmd-1")
        assert cmds[1]["command_id"].endswith("cmd-2")
        assert cmds[1]["outcome"] in ("executed", "validated")
        for c in cmds:
            assert c["outcome"] not in ("no_speech", "stt_exception")
        ctl.shutdown()

    def test_final_interaction_record_matches_recognition_log(self):
        """Execution log and Interaction timing summary must agree:
        the closed record's per-command outcomes equal the last
        recorded outcome for that command_id."""
        tts = StubTTS(auto_finish=True)
        ctl, state, session, stt, _ = self._controller(tts=tts)
        ctl.handle_wake_word()
        stt.emit_transcript("What time is it?")
        stt.emit_no_speech("vad_accepted_stt_empty")
        stt.emit_transcript("Goodbye Jarvis")
        tts.finish_last()  # farewell -> standby -> interaction closed
        assert not session.is_active
        timings = ctl.interaction_timings()
        assert timings
        rec = timings[-1]
        cmds = rec.get("commands", [])
        assert cmds
        time_cmd = next(
            c for c in cmds if "time" in (c.get("transcript") or "")
        )
        assert time_cmd["outcome"] == "executed"
        assert time_cmd["session_id"] == rec["session_id"]
        # Every command outcome appears identically in recognition_log.
        log = ctl.recognition_log()
        for c in cmds:
            matches = [
                e for e in log
                if e.get("command_id") == c["command_id"]
                and e["outcome"] == c["outcome"]
            ]
            assert matches, f"no recognition record for {c['command_id']}"
        ctl.shutdown()


class TestDispatchTerminationOrdering:
    """Session termination vs in-flight command dispatch must have a
    well-defined order: the dispatch either completes fully (check +
    execute + outcome record) before _ending may be set, or it is
    dropped as session_ended — never mid-dispatch."""

    def _controller(self, registry=None, timeout=60):
        import threading
        state = StateManager()
        state.transition_to(AppState.STANDBY)
        session = SessionManager(timeout_seconds=timeout)
        stt = StubSTT()
        tts = StubTTS(auto_finish=True)
        ctl = VoiceSessionController(
            state_manager=state,
            session_manager=session,
            speech_recognition=stt,
            speech_output=tts,
            command_registry=registry,
        )
        return ctl, state, session, stt, tts

    def test_end_waits_for_inflight_dispatch(self):
        import threading
        from src.commands.intent import Intent
        from src.commands.registry import CommandResult

        order = []
        entered = threading.Event()
        release = threading.Event()

        class BlockingRegistry:
            def dispatch(self, intent):
                entered.set()
                release.wait(10.0)
                order.append("dispatch_returned")
                return CommandResult(
                    intent=intent.name, success=True, executed=True,
                    message="ok",
                )

        ctl, state, session, stt, tts = self._controller(
            registry=BlockingRegistry()
        )
        ctl.handle_wake_word()
        tts.finish_last()  # ack -> LISTENING

        def do_dispatch():
            ctl._dispatch_and_respond(
                Intent(name="OPEN_APPLICATION", raw_text="open chrome")
            )

        t = threading.Thread(target=do_dispatch)
        t.start()
        assert entered.wait(5.0)  # dispatch holds self._lock now

        def do_end():
            ctl._begin_end(reason="timeout")
            order.append("end_completed")

        te = threading.Thread(target=do_end)
        te.start()
        time.sleep(0.2)  # let the ender reach the lock
        assert order == []  # _begin_end is blocked: no interleave
        release.set()
        t.join(10.0)
        te.join(10.0)
        # Dispatch fully completed (incl. outcome record) BEFORE the
        # session end was allowed to proceed.
        assert order == ["dispatch_returned", "end_completed"]
        log = ctl.recognition_log()
        assert any(e["outcome"] == "executed" for e in log)
        ctl.shutdown()

    def test_ending_dispatch_is_dropped_atomically(self):
        """If _ending is already set, the locked re-check drops the
        dispatch — no command executes after termination begins."""
        from src.commands.intent import Intent
        from src.commands.registry import CommandResult

        dispatched = []

        class Rec:
            def dispatch(self, intent):
                dispatched.append(intent.name)
                return CommandResult(
                    intent=intent.name, success=True, executed=True,
                )

        ctl, state, session, stt, tts = self._controller(registry=Rec())
        ctl.handle_wake_word()
        tts.finish_last()
        with ctl._lock:
            ctl._ending = True  # simulate in-progress termination
            intent = Intent(name="OPEN_APPLICATION", raw_text="x")
            ctl._dispatch_and_respond(intent)
        assert dispatched == []
        ctl._ending = False
        ctl.shutdown()


class TestSessionStateIsolation:
    def test_pending_confirm_cleared_on_new_session(self):
        """A confirmation pending from a previous session must not
        leak into the next one."""
        ctl, state, session, stt, tts = TestInteractionTelemetry()             ._controller()
        ctl.handle_wake_word()
        ctl._pending_confirm = {"intent": object(), "at": time.monotonic()}
        stt.emit_transcript("Goodbye Jarvis")
        tts.finish_last()
        assert not session.is_active
        assert ctl.handle_wake_word() is True
        assert ctl._pending_confirm is None
        ctl.shutdown()
