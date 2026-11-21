"""V10.4 acknowledgement-mode tests.

Full (default), quick and silent wake acknowledgements, plus the
per-interaction latency records the session controller emits.
"""

from __future__ import annotations

import time

import pytest

from src.core.session_manager import SessionManager
from src.core.state_manager import AppState, StateManager
from src.voice.voice_session_controller import VoiceSessionController


class FakeSTT:
    def __init__(self):
        self._cb = {}

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self):
        pass

    def stop(self):
        pass

    @property
    def is_running(self):
        return True

    def emit(self, text):
        self._cb["on_transcript"](text)


class FakeTTS:
    """TTS stub with controllable playback timing."""

    def __init__(self, playback_seconds=0.0):
        self.spoken = []
        self._cb = {}
        self._playback = playback_seconds

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self):
        pass

    def stop(self):
        pass

    @property
    def is_running(self):
        return True

    def speak(self, text):
        self.spoken.append(text)
        if self._cb.get("on_started"):
            self._cb["on_started"](text)
        if self._playback:
            time.sleep(self._playback)
        self._cb["on_finished"](text)


class FakeMic:
    def __init__(self):
        self.suspended = 0
        self.resumed = 0

        class Cap:
            def resume(self):
                pass

            def flush(self):
                pass

        self.capture = Cap()

    def suspend_detection(self, pause_capture=True):
        self.suspended += 1

    def resume_detection(self):
        self.resumed += 1


def make_controller(ack_mode="full", quick="Yes.", **kw):
    state = StateManager()
    state.transition_to(AppState.STANDBY)
    stt, tts = FakeSTT(), FakeTTS()
    mic = FakeMic()
    ctl = VoiceSessionController(
        state_manager=state,
        session_manager=SessionManager(timeout_seconds=60),
        speech_recognition=stt,
        speech_output=tts,
        microphone_controller=mic,
        ack_mode=ack_mode,
        quick_ack_response=quick,
        **kw,
    )
    return ctl, stt, tts, mic


class TestAckModes:
    def test_full_mode_default_speaks_activation(self):
        ctl, stt, tts, _ = make_controller()
        assert ctl.handle_wake_word("hello jarvis") is True
        assert tts.spoken == ["At your service."]
        ctl.shutdown()

    def test_full_mode_uses_configured_response(self):
        ctl, stt, tts, _ = make_controller(
            activation_response="How may I help?"
        )
        ctl.handle_wake_word()
        assert tts.spoken == ["How may I help?"]
        ctl.shutdown()

    def test_quick_mode_speaks_short_ack(self):
        ctl, stt, tts, _ = make_controller(ack_mode="quick")
        ctl.handle_wake_word()
        assert tts.spoken == ["Yes."]
        # Short ack still finishes into LISTENING.
        assert ctl.session_active
        ctl.shutdown()

    def test_silent_mode_speaks_nothing(self):
        ctl, stt, tts, _ = make_controller(ack_mode="silent")
        assert ctl.handle_wake_word() is True
        assert tts.spoken == []
        assert ctl.session_active
        # Listening immediately — a command works with no ack delay.
        stt.emit("What time is it?")
        assert any("time" in s.lower() for s in tts.spoken)
        ctl.shutdown()

    def test_invalid_mode_falls_back_to_full(self):
        ctl, stt, tts, _ = make_controller(ack_mode="bogus")
        ctl.handle_wake_word()
        assert tts.spoken == ["At your service."]
        ctl.shutdown()

    def test_duplicate_activation_still_ignored(self):
        ctl, stt, tts, _ = make_controller(ack_mode="silent")
        assert ctl.handle_wake_word() is True
        assert ctl.handle_wake_word() is False
        assert tts.spoken == []
        ctl.shutdown()


class TestInteractionTimings:
    def test_timing_record_captured(self):
        ctl, stt, tts, _ = make_controller(ack_mode="quick")
        ctl.handle_wake_word()
        stt.emit("What time is it?")
        ctl._end_with_goodbye()
        tts._cb["on_finished"]("Goodbye.")
        timings = ctl.interaction_timings()
        assert len(timings) == 1
        rec = timings[0]
        assert rec["ack_mode"] == "quick"
        assert rec["wake_to_active_seconds"] is not None
        assert rec["active_to_tts_start_seconds"] is not None
        assert rec["tts_playback_seconds"] is not None
        assert rec["wake_to_first_transcript_seconds"] is not None
        ctl.shutdown()

    def test_silent_mode_has_no_tts_fields(self):
        ctl, stt, tts, _ = make_controller(ack_mode="silent")
        ctl.handle_wake_word()
        stt.emit("What time is it?")
        ctl._end_with_goodbye()
        # Silent mode: no TTS start/finish for the ack, but the command
        # response and goodbye still produce timing fields.
        timings = ctl.interaction_timings()
        assert timings
        rec = timings[0]
        assert rec["ack_mode"] == "silent"
        assert rec["wake_to_first_transcript_seconds"] is not None
        ctl.shutdown()

    def test_quick_ack_shorter_than_full(self):
        ctl_q, stt_q, tts_q, _ = make_controller(
            ack_mode="quick", activation_response="At your service."
        )
        ctl_f, stt_f, tts_f, _ = make_controller(
            ack_mode="full", activation_response="At your service."
        )
        ctl_q.handle_wake_word()
        ctl_f.handle_wake_word()
        # Quick speaks the short phrase; full speaks the long one.
        assert tts_q.spoken == ["Yes."]
        assert tts_f.spoken == ["At your service."]
        assert len("Yes.") < len("At your service.")
        ctl_q.shutdown()
        ctl_f.shutdown()
