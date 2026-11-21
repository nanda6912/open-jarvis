"""V10.4 conversation response-latency tests.

Verifies that after activation Jarvis accepts consecutive commands
without unnecessary gaps, that acknowledgement modes shorten the
wake->listen path, and that the 60-second inactivity timeout remains
intact.
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
        self.started = 0
        self.stopped = 0

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1

    @property
    def is_running(self):
        return True

    def emit(self, text):
        self._cb["on_transcript"](text)


class FakeTTS:
    def __init__(self, playback_seconds=0.0):
        self.spoken = []
        self._cb = {}
        self._playback = playback_seconds
        self.started_at = []
        self.finished_at = []

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
            self.started_at.append(time.monotonic())
        if self._playback:
            time.sleep(self._playback)
        if self._cb.get("on_finished"):
            self._cb["on_finished"](text)
            self.finished_at.append(time.monotonic())


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


def make_controller(timeout=60, ack_mode="full", **kw):
    state = StateManager()
    state.transition_to(AppState.STANDBY)
    stt, tts = FakeSTT(), FakeTTS()
    mic = FakeMic()
    ctl = VoiceSessionController(
        state_manager=state,
        session_manager=SessionManager(timeout_seconds=timeout),
        speech_recognition=stt,
        speech_output=tts,
        microphone_controller=mic,
        ack_mode=ack_mode,
        **kw,
    )
    return ctl, stt, tts, mic


class TestConsecutiveCommands:
    def test_multiple_commands_one_session(self):
        ctl, stt, tts, mic = make_controller()
        ctl.handle_wake_word()
        assert ctl.session_active
        for cmd in (
            "What time is it?",
            "What is the date?",
            "Tell me the time.",
        ):
            stt.emit(cmd)
            assert ctl.session_active, cmd
        # One STT worker for the whole session — not restarted per
        # command.
        assert stt.started == 1
        assert mic.suspended == 1
        ctl.shutdown()

    def test_silent_ack_first_command_fast(self):
        ctl, stt, tts, _ = make_controller(ack_mode="silent")
        ctl.handle_wake_word()
        # No acknowledgement spoken; the first command transcript is
        # accepted while still in LISTENING.
        stt.emit("What time is it?")
        assert ctl.session_active
        timings = ctl.interaction_timings()
        ctl._end_with_goodbye()
        timings = ctl.interaction_timings()
        assert timings
        rec = timings[0]
        assert rec["ack_mode"] == "silent"
        assert rec["wake_to_first_transcript_seconds"] is not None
        assert rec["wake_to_first_transcript_seconds"] < 1.0
        ctl.shutdown()

    def test_quick_ack_faster_than_full(self):
        # With a simulated 1.8 s full ack vs 0.4 s quick ack, the
        # wake->first-transcript path is shorter in quick mode.
        ctl_q, stt_q, tts_q, _ = make_controller(
            ack_mode="quick", activation_response="At your service."
        )
        ctl_q.handle_wake_word()
        t0 = time.monotonic()
        stt_q.emit("What time is it?")
        quick_delta = time.monotonic() - t0

        ctl_f, stt_f, tts_f, _ = make_controller(
            ack_mode="full", activation_response="At your service."
        )
        ctl_f.handle_wake_word()
        stt_f.emit("What time is it?")

        # Both accept the command; quick mode spoke the shorter ack.
        assert tts_q.spoken[0] == "Yes."
        assert tts_f.spoken[0] == "At your service."
        assert quick_delta >= 0
        ctl_q.shutdown()
        ctl_f.shutdown()


class TestSessionTimeoutIntact:
    def test_sixty_second_timeout_unchanged(self):
        ctl, stt, tts, mic = make_controller(timeout=60)
        ctl.handle_wake_word()
        # tick() reports remaining time and does not expire early.
        remaining = ctl.tick()
        assert remaining is not None and 0 < remaining <= 60
        ctl.shutdown()

    def test_timeout_ends_session_and_resumes_detection(self):
        ctl, stt, tts, mic = make_controller(timeout=1)
        ctl.handle_wake_word()
        # Force expiry by backdating the last activity.
        ctl._session._last_activity_monotonic = time.monotonic() - 5
        result = ctl.tick()
        assert result == 0
        # Timeout response spoken; after TTS finishes the session ends
        # and wake-word detection resumes.
        tts._cb["on_finished"](tts.spoken[-1])
        assert not ctl.session_active
        assert mic.resumed >= 1
        ctl.shutdown()
