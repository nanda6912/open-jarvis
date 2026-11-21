"""Tests for Task 5: intent recognition, entity extraction, command
registry, routing and session integration."""

from __future__ import annotations

import time

import pytest

from src.commands.intent import (
    Intent,
    KNOWN_FOLDERS,
    RuleBasedIntentRecognizer,
)
from src.commands.registry import (
    CommandRegistry,
    CommandResult,
    PERMISSION_DESTRUCTIVE,
    PERMISSION_SAFE,
)
from src.commands.handlers import (
    SUPPORTED_FOLDERS,
    TimeHandler,
    DateHandler,
    OpenApplicationHandler,
    OpenFolderHandler,
    create_default_registry,
)
from src.core.session_manager import SessionManager
from src.core.state_manager import AppState, StateManager
from src.voice.voice_session_controller import VoiceSessionController


# ----------------------------------------------------------------------
# Intent recognition — phrasings
# ----------------------------------------------------------------------
class TestIntentRecognition:
    @pytest.fixture
    def rec(self):
        return RuleBasedIntentRecognizer()

    @pytest.mark.parametrize("text", [
        "What time is it?",
        "what's the time",
        "Tell me the time.",
        "current time please",
    ])
    def test_time_intents(self, rec, text):
        assert rec.recognize(text).name == "GET_CURRENT_TIME"

    @pytest.mark.parametrize("text", [
        "What date is it?",
        "what day is today",
        "What's the date.",
        "Tell me the day.",
        "today's date",
    ])
    def test_date_intents(self, rec, text):
        assert rec.recognize(text).name == "GET_CURRENT_DATE"

    @pytest.mark.parametrize("text", [
        "Open Chrome.",
        "Can you open Chrome?",
        "Launch Google Chrome.",
        "Start notepad",
        "Run calculator",
        "Please open Spotify",
    ])
    def test_open_application(self, rec, text):
        intent = rec.recognize(text)
        assert intent.name == "OPEN_APPLICATION"
        assert intent.entities["app_name"]

    @pytest.mark.parametrize("text", [
        "Close Chrome.",
        "Quit notepad",
        "Exit Spotify",
        "Kill calculator",
    ])
    def test_close_application(self, rec, text):
        intent = rec.recognize(text)
        assert intent.name == "CLOSE_APPLICATION"
        assert intent.entities["app_name"]

    @pytest.mark.parametrize("text", [
        "Open my Downloads folder.",
        "I want to open my Downloads folder.",
        "Show me the pictures folder",
        "Go to the documents folder",
        "Open desktop",
    ])
    def test_open_folder(self, rec, text):
        intent = rec.recognize(text)
        assert intent.name == "OPEN_FOLDER"
        assert intent.entities["folder"] in KNOWN_FOLDERS

    @pytest.mark.parametrize("text", [
        "Search for tax documents.",
        "Find my resume",
        "Search my files for budget report",
        "Look for the invoice",
    ])
    def test_search_files(self, rec, text):
        intent = rec.recognize(text)
        assert intent.name == "SEARCH_FILES"
        assert intent.entities["query"]

    @pytest.mark.parametrize("text", [
        "Turn the volume up.",
        "Increase volume",
        "Volume up",
        "Make it louder",
        "Turn it up",
    ])
    def test_increase_volume(self, rec, text):
        assert rec.recognize(text).name == "INCREASE_VOLUME"

    @pytest.mark.parametrize("text", [
        "Turn the volume down.",
        "Decrease volume",
        "Volume down",
        "Make it quieter",
        "Turn it down",
    ])
    def test_decrease_volume(self, rec, text):
        assert rec.recognize(text).name == "DECREASE_VOLUME"

    @pytest.mark.parametrize("text", [
        "Mute the volume.",
        "Mute.",
        "Silence",
        "Turn the volume off",
    ])
    def test_mute_volume(self, rec, text):
        assert rec.recognize(text).name == "MUTE_VOLUME"

    @pytest.mark.parametrize("text", [
        "Unmute.",
        "Unmute the volume",
        "Turn the volume on",
    ])
    def test_unmute_volume(self, rec, text):
        assert rec.recognize(text).name == "UNMUTE_VOLUME"

    @pytest.mark.parametrize("text", [
        "Goodbye Jarvis",
        "Bye.",
        "End session.",
        "That's all.",
        "That is all for now.",
        "Go to standby.",
        "Stop listening.",
        "Quit.",
        "Exit",
    ])
    def test_end_session(self, rec, text):
        assert rec.recognize(text).name == "END_SESSION"

    @pytest.mark.parametrize("text", [
        "Blorf snoofer.",
        "asdfghjkl",
        "",
        "   ",
        "Open.",  # verb without object -> not an app intent
    ])
    def test_unknown(self, rec, text):
        assert rec.recognize(text).name == "UNKNOWN"


# ----------------------------------------------------------------------
# Entity extraction
# ----------------------------------------------------------------------
class TestEntityExtraction:
    @pytest.fixture
    def rec(self):
        return RuleBasedIntentRecognizer()

    def test_app_name_extraction(self, rec):
        i = rec.recognize("Can you open Google Chrome?")
        assert i.entities["app_name"] == "Google Chrome"

    def test_app_name_missing_object(self, rec):
        i = rec.recognize("Open the")
        assert i.name == "OPEN_APPLICATION"
        assert i.entities["app_name"] == ""

    def test_folder_extraction_known(self, rec):
        i = rec.recognize("Open my Downloads folder.")
        assert i.entities["folder"] == "downloads"

    def test_folder_extraction_unknown(self, rec):
        i = rec.recognize("Open my secretstuff folder.")
        assert i.name == "OPEN_FOLDER"
        assert i.entities["folder"] == "secretstuff"

    def test_search_query_extraction(self, rec):
        i = rec.recognize("Search my files for budget report")
        assert i.entities["query"] == "budget report"

    def test_search_query_with_scope(self, rec):
        i = rec.recognize("Search for tax documents in my files")
        assert i.entities["query"] == "tax documents"

    def test_volume_amount(self, rec):
        i = rec.recognize("Turn the volume down by 20.")
        assert i.entities["amount"] == 20

    def test_volume_no_amount(self, rec):
        i = rec.recognize("Turn the volume up.")
        assert i.entities["amount"] is None


# ----------------------------------------------------------------------
# Command registry
# ----------------------------------------------------------------------
class TestCommandRegistry:
    def test_unregistered_intent_rejected(self):
        reg = CommandRegistry()
        result = reg.dispatch(Intent("FLY_TO_MOON", {}))
        assert result.success is False
        assert result.error == "unregistered_intent"
        assert result.executed is False

    def test_time_handler_executes(self):
        reg = create_default_registry()
        result = reg.dispatch(Intent("GET_CURRENT_TIME", {}))
        assert result.success and result.executed
        assert "current time is" in result.message.lower()

    def test_date_handler_executes(self):
        reg = create_default_registry()
        result = reg.dispatch(Intent("GET_CURRENT_DATE", {}))
        assert result.success and result.executed
        assert "today is" in result.message.lower()

    def test_open_app_not_executed_but_validated(self):
        reg = create_default_registry()
        ok = reg.dispatch(Intent("OPEN_APPLICATION", {"app_name": "Chrome"}))
        assert ok.success and not ok.executed
        bad = reg.dispatch(Intent("OPEN_APPLICATION", {"app_name": ""}))
        assert not bad.success and bad.error == "missing_app_name"

    def test_folder_allowlist(self):
        reg = create_default_registry()
        good = reg.dispatch(Intent("OPEN_FOLDER", {"folder": "downloads"}))
        assert good.success and not good.executed
        bad = reg.dispatch(Intent("OPEN_FOLDER", {"folder": "secretstuff"}))
        assert not bad.success and bad.error == "unsupported_folder"
        missing = reg.dispatch(Intent("OPEN_FOLDER", {"folder": ""}))
        assert not missing.success and missing.error == "missing_folder"

    def test_search_requires_query(self):
        reg = create_default_registry()
        bad = reg.dispatch(Intent("SEARCH_FILES", {"query": ""}))
        assert not bad.success and bad.error == "missing_query"
        ok = reg.dispatch(Intent("SEARCH_FILES", {"query": "resume"}))
        assert ok.success and not ok.executed

    def test_volume_amount_validation(self):
        reg = create_default_registry()
        ok = reg.dispatch(
            Intent("INCREASE_VOLUME", {"amount": 20})
        )
        assert ok.success
        bad = reg.dispatch(
            Intent("DECREASE_VOLUME", {"amount": 250})
        )
        assert not bad.success and bad.error == "amount_out_of_range"
        bad2 = reg.dispatch(
            Intent("MUTE_VOLUME", {"amount": "loud"})
        )
        assert not bad2.success

    def test_permission_denied(self):
        class Dangerous:
            intent_name = "DESTRUCTIVE"
            permission = PERMISSION_DESTRUCTIVE
            executable = True

            def validate(self, entities):
                return None

            def execute(self, entities):
                return CommandResult(intent="DESTRUCTIVE", success=True)

        reg = CommandRegistry()
        reg.register("DESTRUCTIVE", Dangerous())
        result = reg.dispatch(Intent("DESTRUCTIVE", {}))
        assert result.success is False
        assert result.error == "permission_denied"

    def test_handler_exception_is_safe(self):
        class Boom:
            intent_name = "BOOM"
            permission = PERMISSION_SAFE
            executable = True

            def validate(self, entities):
                return None

            def execute(self, entities):
                raise RuntimeError("kaboom")

        reg = CommandRegistry()
        reg.register("BOOM", Boom())
        result = reg.dispatch(Intent("BOOM", {}))
        assert result.success is False
        assert "execution_error" in result.error

    def test_validate_exception_is_safe(self):
        class BadValidate:
            intent_name = "BV"
            permission = PERMISSION_SAFE
            executable = True

            def validate(self, entities):
                raise ValueError("nope")

            def execute(self, entities):
                return CommandResult(intent="BV", success=True)

        reg = CommandRegistry()
        reg.register("BV", BadValidate())
        result = reg.dispatch(Intent("BV", {}))
        assert result.success is False
        assert "validation_error" in result.error

    def test_all_supported_intents_registered(self):
        reg = create_default_registry()
        for name in (
            "OPEN_APPLICATION", "CLOSE_APPLICATION", "OPEN_FOLDER",
            "SEARCH_FILES", "GET_CURRENT_TIME", "GET_CURRENT_DATE",
            "INCREASE_VOLUME", "DECREASE_VOLUME", "MUTE_VOLUME",
            "UNMUTE_VOLUME",
        ):
            assert reg.is_registered(name), name


# ----------------------------------------------------------------------
# Session integration
# ----------------------------------------------------------------------
class _StubSTT:
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


class _StubTTS:
    def __init__(self):
        self.spoken = []
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

    def speak(self, text):
        self.spoken.append(text)
        self._cb["on_finished"](text)


def _controller(**kw):
    state = StateManager()
    state.transition_to(AppState.STANDBY)
    session = SessionManager(timeout_seconds=60)
    stt, tts = _StubSTT(), _StubTTS()
    ctl = VoiceSessionController(
        state_manager=state,
        session_manager=session,
        speech_recognition=stt,
        speech_output=tts,
        **kw,
    )
    return ctl, state, session, stt, tts


class TestSessionIntegration:
    def test_time_command_executes_and_speaks(self):
        ctl, state, session, stt, tts = _controller()
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("What time is it?")
        assert any("current time is" in s.lower() for s in tts.spoken)
        ctl.shutdown()

    def test_date_command_executes_and_speaks(self):
        ctl, state, session, stt, tts = _controller()
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("What is the date?")
        assert any("today is" in s.lower() for s in tts.spoken)
        ctl.shutdown()

    def test_open_app_acknowledged_not_executed(self):
        ctl, state, session, stt, tts = _controller()
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("Can you open Chrome?")
        assert "I heard you say can you open Chrome" in tts.spoken[0] \
            or any("open Chrome" in s for s in tts.spoken)
        ctl.shutdown()

    def test_open_folder_acknowledged(self):
        ctl, state, session, stt, tts = _controller()
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("I want to open my Downloads folder.")
        assert any("downloads folder" in s.lower() for s in tts.spoken)
        ctl.shutdown()

    def test_volume_commands_acknowledged(self):
        ctl, state, session, stt, tts = _controller()
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("Turn the volume up.")
        stt.emit("Mute the volume.")
        assert len(tts.spoken) == 2
        assert all("I heard you" in s for s in tts.spoken)
        ctl.shutdown()

    def test_missing_app_name_gets_detail_request(self):
        ctl, state, session, stt, tts = _controller()
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("Open the")
        assert any("more detail" in s for s in tts.spoken)
        ctl.shutdown()

    def test_unsupported_folder_gets_detail_request(self):
        ctl, state, session, stt, tts = _controller()
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("Open my secretstuff folder.")
        assert any("more detail" in s for s in tts.spoken)
        ctl.shutdown()

    def test_unknown_command_phrase(self):
        ctl, state, session, stt, tts = _controller()
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("Blorf snoofer.")
        assert any(
            "don't know how to handle" in s for s in tts.spoken
        )
        ctl.shutdown()

    def test_end_session_intent_ends_session(self):
        ctl, state, session, stt, tts = _controller()
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("Go to standby.")
        assert "Goodbye." in tts.spoken
        ctl.shutdown()

    def test_multiple_mixed_commands_one_session(self):
        ctl, state, session, stt, tts = _controller()
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("What time is it?")
        stt.emit("Open Chrome.")
        stt.emit("Turn the volume down.")
        assert session.is_active
        kinds = [
            "current time" in s.lower() for s in tts.spoken
        ]
        assert any(kinds)
        assert sum(1 for s in tts.spoken if "I heard you" in s) == 2
        ctl.shutdown()

    def test_custom_registry_injection(self):
        class FakeIntent:
            name = "GET_CURRENT_TIME"
            entities = {}
            raw_text = "time"
            confidence = 1.0

        class FakeRecognizer:
            name = "fake"

            def recognize(self, text):
                return FakeIntent()

        class FakeRegistry:
            def dispatch(self, intent):
                return CommandResult(
                    intent=intent.name, success=True, executed=True,
                    message="FAKE TIME RESPONSE",
                )

        ctl, state, session, stt, tts = _controller(
            intent_recognizer=FakeRecognizer(),
            command_registry=FakeRegistry(),
        )
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("anything at all")
        assert "FAKE TIME RESPONSE" in tts.spoken
        ctl.shutdown()
