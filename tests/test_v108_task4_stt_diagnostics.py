"""V10.8 Task 4 — STT failure classification and diagnostics tests.

Covers the per-attempt taxonomy (capture / VAD / STT / intent /
entity / handler), structured attempt records, correlation of
recognition log entries, and session recovery after each failure
class. All deterministic — fake capture, recognizer and TTS.
"""

import logging
import time
import types

from src.commands.intent import Intent
from src.commands.registry import CommandResult
from src.core.session_manager import SessionManager
from src.core.state_manager import AppState, StateManager
from src.voice import recognition_taxonomy as tax
from src.voice.speech_recognition_service import SpeechRecognitionService
from src.voice.voice_session_controller import VoiceSessionController
from tests.test_task4 import (
    FakeFrameSource,
    FakeRecognizer,
    StubTTS,
    _wait_for,
)


# ----------------------------------------------------------------------
# Taxonomy unit tests
# ----------------------------------------------------------------------
class TestTaxonomy:
    def test_category_for_all_outcomes(self):
        cases = {
            "executed": "command_executed",
            "handler_failed": "handler_failed",
            "stt_exception": "stt_exception",
            "intent_unknown": "intent_unknown",
            "entity_missing": "entity_missing",
            "entity_ambiguous": "entity_ambiguous",
            "no_speech": None,  # resolved via detail
        }
        for outcome, expected in cases.items():
            assert tax.category_for_outcome(outcome) == expected, outcome

    def test_no_speech_reasons(self):
        assert tax.category_for_reason("vad_accepted_stt_empty") == (
            "stt_empty_transcript")
        assert tax.category_for_reason("audio_below_vad_threshold") == (
            "vad_rejected")
        assert tax.category_for_reason("capture_failed") == "capture_failed"
        assert tax.category_for_reason("no_audio_frames") == (
            "no_audio_received")
        assert tax.category_for_reason("utterance_too_short") is None

    def test_no_speech_outcome_uses_detail(self):
        assert tax.category_for_outcome(
            "no_speech", "vad_accepted_stt_empty") == "stt_empty_transcript"
        assert tax.category_for_outcome(
            "no_speech", "capture_failed") == "capture_failed"
        assert tax.category_for_outcome(
            "no_speech", "no_audio_frames") == "no_audio_received"
        assert tax.category_for_outcome("no_speech", "weird") is None

    def test_is_category(self):
        assert tax.is_category("stt_empty_transcript")
        assert not tax.is_category("utterance_too_short")
        assert not tax.is_category(None)


# ----------------------------------------------------------------------
# SpeechRecognitionService attempt records
# ----------------------------------------------------------------------
class _FakeCapture(FakeFrameSource):
    """FakeFrameSource + the stats/device surface the service reads."""

    device = types.SimpleNamespace(name="Test Mic", stable_id="mic-1")
    sample_rate = 16000
    input_channels = 4
    _stream = types.SimpleNamespace(active=True)

    def __init__(self, running=True, stream_active=True):
        super().__init__()
        self._running = running
        self._stream_active = stream_active
        self._frames_pushed = 0

    def push(self, samples):
        self._frames_pushed += 1
        super().push(samples)

    def stats(self):
        return {
            "device": self.device.name,
            "sample_rate": 16000,
            "running": self._running,
            "stream_active": self._stream_active,
            "frames_captured": self._frames_pushed,
            "frames_queued": self._queue.qsize(),
            "frames_dropped": 0,
        }


def _make_stt(text="", capture=None, **kw):
    src = capture or _FakeCapture()
    rec = FakeRecognizer(text)
    params = dict(
        speech_threshold=0.02,
        silence_hangover=0.3,
        min_utterance=0.2,
        frame_timeout=0.05,
    )
    params.update(kw)
    no_speech = []
    errors = []
    transcripts = []
    svc = SpeechRecognitionService(
        src, rec,
        on_transcript=transcripts.append,
        on_no_speech=no_speech.append,
        on_error=errors.append,
        **params,
    )
    # Keep starvation diagnostics out of the way unless a test
    # explicitly exercises them.
    svc._starvation_emit_after = 30.0
    return src, rec, svc, no_speech, errors, transcripts


def _speak(src, seconds=0.5):
    src.push_tone(seconds=seconds, level=0.5)
    src.push_silence(seconds=1.0)


class _FlakyRecognizer(FakeRecognizer):
    def __init__(self, text="ok", fail=False):
        super().__init__(text)
        self.fail = fail

    def transcribe(self, samples):
        self.calls.append(len(samples))
        if self.fail:
            raise RuntimeError("decoder exploded")
        return self.text


class _caplog:
    """Attach a list handler to the root logger for event assertions."""

    def __init__(self):
        self.records = []
        self.text = ""

    def __enter__(self):
        self._handler = _ListHandler(self.records)
        root = logging.getLogger()
        root.addHandler(self._handler)
        root.setLevel(logging.DEBUG)
        return self

    def __exit__(self, *a):
        logging.getLogger().removeHandler(self._handler)
        self.text = "\n".join(r.getMessage() for r in self.records)


class _ListHandler(logging.Handler):
    def __init__(self, records):
        super().__init__()
        self.records = records

    def emit(self, record):
        self.records.append(record)


class TestServiceAttempts:
    def test_stt_empty_transcript_attempt(self):
        src, rec, svc, no_speech, errors, transcripts = _make_stt("")
        with _caplog() as log:
            svc.start()
            _speak(src, 0.5)
            _wait_for(
                lambda: svc.last_attempt
                and svc.last_attempt.get("outcome")
                == "stt_empty_transcript", timeout=3.0)
            svc.stop()
        att = svc.last_attempt
        assert att["category"] == "stt_empty_transcript"
        assert att["outcome"] == "stt_empty_transcript"
        assert att["stt_invoked"] is True
        assert att["stt_duration_seconds"] is not None
        assert att["audio_seconds"] > 0
        assert att["attempt_id"].startswith("stt-utt-")
        assert att["engine"] == "fake-stt"
        assert "STT_EMPTY_TRANSCRIPT" in log.text
        assert "STT_TRANSCRIPT " not in log.text
        assert no_speech == ["vad_accepted_stt_empty"]
        assert errors == []
        assert transcripts == []
        stats = svc.stats()
        assert stats["last_attempt"] is att
        assert stats["outcome_counts"]["stt_empty_transcript"] == 1

    def test_stt_transcript_produced(self):
        src, rec, svc, _n, _er, transcripts = _make_stt("open notepad")
        with _caplog() as log:
            svc.start()
            _speak(src, 0.5)
            _wait_for(lambda: transcripts, timeout=3.0)
            svc.stop()
        att = svc.last_attempt
        assert att["outcome"] == "stt_transcript_produced"
        assert att["category"] == "stt_transcript_produced"
        assert att["transcript_chars"] == len("open notepad")
        assert att["stt_invoked"] is True
        assert "STT_TRANSCRIPT" in log.text
        assert transcripts == ["open notepad"]

    def test_stt_exception_attempt_and_recovery(self):
        rec = _FlakyRecognizer("ok", fail=True)
        src = _FakeCapture()
        errors = []
        svc = SpeechRecognitionService(
            src, rec, speech_threshold=0.02, silence_hangover=0.3,
            min_utterance=0.2, frame_timeout=0.05,
            on_error=errors.append,
        )
        svc._starvation_emit_after = 30.0
        with _caplog() as log:
            svc.start()
            _speak(src, 0.5)
            _wait_for(
                lambda: svc.last_attempt
                and svc.last_attempt.get("outcome") == "stt_exception",
                timeout=3.0)
            # Recover: next utterance decodes normally.
            rec.fail = False
            rec.text = "recovered"
            _speak(src, 0.5)
            _wait_for(
                lambda: svc.last_attempt
                and svc.last_attempt.get("outcome")
                == "stt_transcript_produced", timeout=3.0)
            svc.stop()
        att = svc.last_attempt
        assert att["outcome"] == "stt_transcript_produced"
        assert svc.inference_errors == 1
        assert errors and "decoder exploded" in errors[0]
        assert "STT_EXCEPTION" in log.text
        counts = svc.stats()["outcome_counts"]
        assert counts["stt_exception"] == 1
        assert counts["stt_transcript_produced"] >= 1
        # The failed attempt record kept its own error fields.
        # (last_attempt is the recovery attempt; check via counts only.)

    def test_stt_exception_record_fields(self):
        rec = _FlakyRecognizer(fail=True)
        src = _FakeCapture()
        svc = SpeechRecognitionService(
            src, rec, speech_threshold=0.02, silence_hangover=0.3,
            min_utterance=0.2, frame_timeout=0.05,
        )
        svc._starvation_emit_after = 30.0
        svc.start()
        _speak(src, 0.5)
        _wait_for(
            lambda: svc.last_attempt
            and svc.last_attempt.get("outcome") == "stt_exception",
            timeout=3.0)
        svc.stop()
        att = svc.last_attempt
        assert att["category"] == "stt_exception"
        assert att["error_type"] == "RuntimeError"
        assert "decoder exploded" in att["error"]
        assert att["stt_invoked"] is True
        assert att["stt_duration_seconds"] is not None
        assert att["audio_seconds"] > 0

    def test_utterance_too_short_not_stt(self):
        src, rec, svc, _n, _er, _t = _make_stt("hi", min_utterance=5.0)
        svc.start()
        _speak(src, 0.2)
        _wait_for(
            lambda: svc.last_attempt
            and svc.last_attempt.get("outcome") == "utterance_too_short",
            timeout=3.0)
        svc.stop()
        att = svc.last_attempt
        assert att["stt_invoked"] is False
        assert att["category"] is None
        assert rec.calls == []

    def test_vad_rejected_attempt(self):
        src = _FakeCapture()
        rec = FakeRecognizer("x")
        reasons = []
        svc = SpeechRecognitionService(
            src, rec, speech_threshold=0.02, silence_hangover=0.3,
            min_utterance=0.2, frame_timeout=0.05,
            on_no_speech=reasons.append,
        )
        svc._starvation_emit_after = 30.0
        svc.start()
        for _ in range(35):
            src.push_silence(seconds=0.1)
        _wait_for(
            lambda: svc.last_attempt
            and svc.last_attempt.get("outcome") == "vad_rejected",
            timeout=3.0)
        svc.stop()
        att = svc.last_attempt
        assert att["category"] == "vad_rejected"
        assert att["outcome"] == "vad_rejected"
        assert att["stt_invoked"] is False
        assert att["reason"] == "audio_below_vad_threshold"
        assert "audio_below_vad_threshold" in reasons
        assert rec.calls == []

    def test_capture_failed_vs_no_audio(self):
        dead = _FakeCapture(running=False, stream_active=False)
        dead.device = types.SimpleNamespace(name="dead-mic")
        rec = FakeRecognizer("x")
        reasons = []
        svc = SpeechRecognitionService(
            dead, rec, speech_threshold=0.02, silence_hangover=0.3,
            min_utterance=0.2, frame_timeout=0.05,
            on_no_speech=reasons.append,
        )
        svc._starvation_emit_after = 0.3
        svc.start()
        _wait_for(lambda: reasons, timeout=3.0)
        svc.stop()
        att = svc.last_attempt
        assert reasons == ["capture_failed"]
        assert att["outcome"] == "capture_failed"
        assert att["category"] == "capture_failed"
        assert att["stt_invoked"] is False
        assert att["reason"] == "capture_failed"
        assert att["rejected_frames"] == 0
        assert att["device"] == "dead-mic"
        assert att["sample_rate"] == 16000

        # Healthy stream but silent -> no_audio_received.
        src = _FakeCapture()
        rec2 = FakeRecognizer("x")
        reasons2 = []
        svc2 = SpeechRecognitionService(
            src, rec2, speech_threshold=0.02, silence_hangover=0.3,
            min_utterance=0.2, frame_timeout=0.05,
            on_no_speech=reasons2.append,
        )
        svc2._starvation_emit_after = 0.3
        svc2.start()
        _wait_for(lambda: reasons2, timeout=3.0)
        svc2.stop()
        assert reasons2 == ["no_audio_frames"]
        att2 = svc2.last_attempt
        assert att2["outcome"] == "no_audio_received"
        assert att2["category"] == "no_audio_received"
        assert att2["reason"] == "no_audio_frames"

    def test_vad_accepted_metadata(self):
        src, rec, svc, _n, _er, transcripts = _make_stt("hi")
        svc.start()
        _speak(src, 0.5)
        _wait_for(lambda: transcripts, timeout=3.0)
        svc.stop()
        att = svc.last_attempt
        assert att["vad_enabled"] is True
        assert att["vad_threshold_configured"] == 0.02
        assert att["vad_threshold_effective"] is not None
        assert att["voiced_frames_utterance"] >= 1
        assert att["rejected_frames_utterance"] >= 0
        assert att["rms_max"] is not None and att["rms_max"] > 0
        assert att["speech_started_monotonic"] is not None
        assert att["finalize_reason"] == "silence"
        # None = not yet calibrated (unavailable), distinct from 0.0.
        assert "noise_floor" in att
        assert att["device"] == "Test Mic"
        assert att["sample_rate"] == 16000
        assert att["engine"] == "fake-stt"
        assert att["audio_seconds"] > 0


# ----------------------------------------------------------------------
# Controller-level classification
# ----------------------------------------------------------------------
class _FakeRecognizer:
    def __init__(self, intent=None):
        self.intent = intent

    def recognize(self, text):
        if callable(self.intent):
            return self.intent(text)
        return self.intent


class _FakeRegistry:
    def __init__(self, result):
        self.result = result

    def dispatch(self, intent):
        return self.result


class _StubSTT:
    """Minimal STT fake for controller tests (no capture)."""

    def __init__(self):
        self.started = 0
        self.stopped = 0
        self._cb = {}
        self.last_attempt = {"attempt_id": "stt-test-1"}

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    @property
    def is_running(self):
        return self.started > self.stopped

    def pause(self):
        pass

    def resume(self):
        pass

    def emit_transcript(self, text):
        self._cb["on_transcript"](text)

    def emit_no_speech(self, reason):
        self._cb["on_no_speech"](reason)

    def emit_error(self, msg):
        self._cb["on_error"](msg)


def _controller(intent=None, result=None):
    sm = StateManager()
    sm.transition_to(AppState.STANDBY)
    sess = SessionManager(timeout_seconds=60)
    stt = _StubSTT()
    tts = StubTTS(auto_finish=True)
    ctl = VoiceSessionController(
        state_manager=sm,
        session_manager=sess,
        speech_recognition=stt,
        speech_output=tts,
        intent_recognizer=_FakeRecognizer(intent),
        command_registry=_FakeRegistry(result),
    )
    return ctl, stt, tts


def _wake(ctl):
    assert ctl.handle_wake_word() is True
    _wait_for(lambda: ctl._session.is_active, timeout=2.0)


class TestControllerClassification:
    def _last(self, ctl):
        return ctl.recognition_log()[-1]

    def test_no_speech_categories(self):
        ctl, stt, _tts = _controller()
        _wake(ctl)
        for reason, expected in [
            ("vad_accepted_stt_empty", "stt_empty_transcript"),
            ("audio_below_vad_threshold", "vad_rejected"),
            ("no_audio_frames", "no_audio_received"),
            ("capture_failed", "capture_failed"),
        ]:
            stt.emit_no_speech(reason)
            entry = self._last(ctl)
            assert entry["outcome"] == "no_speech"
            assert entry["detail"] == reason
            assert entry["category"] == expected, reason
            assert entry["stt_attempt"] == "stt-test-1"

    def test_stt_exception_outcome_and_recovery(self):
        ctl, stt, _tts = _controller()
        _wake(ctl)
        stt.emit_error("RuntimeError: decoder exploded")
        entry = self._last(ctl)
        assert entry["outcome"] == "stt_exception"
        assert entry["category"] == "stt_exception"
        assert "decoder exploded" in entry["detail"]
        # Session recovers: back to LISTENING, next attempt processed.
        _wait_for(lambda: ctl._state.state == AppState.LISTENING,
                  timeout=2.0)
        stt.emit_no_speech("vad_accepted_stt_empty")
        assert self._last(ctl)["category"] == "stt_empty_transcript"
        assert ctl._session.is_active

    def test_stt_exception_throttled_speech(self):
        ctl, stt, tts = _controller()
        _wake(ctl)
        stt.emit_error("err1")
        stt.emit_error("err2")
        stt.emit_error("err3")
        speaks = [s for s in tts.spoken if "trouble hearing" in s]
        assert len(speaks) == 1

    def test_intent_unknown(self):
        ctl, stt, _tts = _controller(
            intent=Intent("UNKNOWN", {}, raw_text="gibberish"))
        _wake(ctl)
        stt.emit_transcript("gibberish")
        entry = self._last(ctl)
        assert entry["outcome"] == "intent_unknown"
        assert entry["category"] == "intent_unknown"
        assert entry["text"] == "gibberish"

    def test_entity_missing(self):
        intent = Intent("OPEN_FILE", {"query": ""}, raw_text="open file")
        result = CommandResult(
            "OPEN_FILE", False, message="I need a file name.",
            error="missing_query")
        ctl, stt, _tts = _controller(intent=intent, result=result)
        _wake(ctl)
        stt.emit_transcript("open file")
        entry = self._last(ctl)
        assert entry["outcome"] == "entity_missing"
        assert entry["category"] == "entity_missing"
        assert entry["intent"] == "OPEN_FILE"

    def test_entity_ambiguous_flag(self):
        intent = Intent("OPEN_APPLICATION", {"app_name": "not"})
        result = CommandResult(
            "OPEN_APPLICATION", True, message="Which app?",
            ambiguous=True, data={"apps": []})
        ctl, stt, _tts = _controller(intent=intent, result=result)
        _wake(ctl)
        stt.emit_transcript("open notepad")
        entry = self._last(ctl)
        assert entry["outcome"] == "entity_ambiguous"
        assert entry["category"] == "entity_ambiguous"

    def test_entity_missing_via_clarify_tag(self):
        intent = Intent("CLOSE_APPLICATION", {"app_name": ""})
        result = CommandResult(
            "CLOSE_APPLICATION", True, message="Which app should I close?",
            executed=False,
            data={"clarify": {
                "entity_issue": "missing",
                "resolver": lambda t: None,
                "apply": lambda i, p: None,
                "message": "Which app should I close?",
            }})
        ctl, stt, _tts = _controller(intent=intent, result=result)
        _wake(ctl)
        stt.emit_transcript("close application")
        entry = self._last(ctl)
        assert entry["outcome"] == "entity_missing"
        assert entry["category"] == "entity_missing"
        assert entry["entity_issue"] == "missing"

    def test_handler_failed(self):
        intent = Intent("OPEN_APPLICATION", {"app_name": "x"})
        result = CommandResult(
            "OPEN_APPLICATION", False, message="Launch failed.",
            error="execution_error: boom")
        ctl, stt, _tts = _controller(intent=intent, result=result)
        _wake(ctl)
        stt.emit_transcript("open x")
        entry = self._last(ctl)
        assert entry["outcome"] == "handler_failed"
        assert entry["category"] == "handler_failed"

    def test_command_executed(self):
        intent = Intent("OPEN_APPLICATION", {"app_name": "x"})
        result = CommandResult(
            "OPEN_APPLICATION", True, message="Opened x.",
            executed=True)
        ctl, stt, _tts = _controller(intent=intent, result=result)
        _wake(ctl)
        stt.emit_transcript("open x")
        entry = self._last(ctl)
        assert entry["outcome"] == "executed"
        assert entry["category"] == "command_executed"

    def test_session_recovers_to_listening(self):
        ctl, stt, _tts = _controller(
            intent=Intent("UNKNOWN", {}, raw_text="gibberish"))
        _wake(ctl)
        stt.emit_transcript("gibberish")
        stt.emit_no_speech("vad_accepted_stt_empty")
        stt.emit_error("boom")
        _wait_for(lambda: ctl._state.state == AppState.LISTENING,
                  timeout=2.0)
        assert ctl._session.is_active
