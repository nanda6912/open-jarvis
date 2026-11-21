"""Tests for state management, sessions and application wiring."""

from __future__ import annotations

import time

import pytest

from src.core.application import VoiceOSApplication
from src.core.configuration import ConfigurationManager
from src.core.session_manager import SessionManager
from src.core.state_manager import (
    AppState,
    InvalidStateTransitionError,
    StateManager,
)
from src.storage.database import Database


@pytest.fixture
def config(tmp_path):
    """Provide a configuration manager pointed at a temp data dir."""
    return ConfigurationManager(config_path="does/not/exist.json")


class TestStateManager:
    """Tests for application state transitions."""

    def test_initial_state(self) -> None:
        assert StateManager().state is AppState.INITIALIZING

    def test_valid_transition(self) -> None:
        sm = StateManager()
        sm.transition_to(AppState.STANDBY)
        assert sm.state is AppState.STANDBY
        sm.transition_to(AppState.ACTIVATED)
        assert sm.state is AppState.ACTIVATED

    def test_same_state_is_noop(self) -> None:
        sm = StateManager()
        sm.transition_to(AppState.INITIALIZING)
        assert sm.state is AppState.INITIALIZING

    def test_invalid_transition_raises(self) -> None:
        sm = StateManager()
        with pytest.raises(InvalidStateTransitionError):
            sm.transition_to(AppState.LISTENING)

    def test_stopped_is_terminal(self) -> None:
        sm = StateManager()
        sm.transition_to(AppState.STOPPED)
        with pytest.raises(InvalidStateTransitionError):
            sm.transition_to(AppState.STANDBY)

    def test_error_recovery_to_standby(self) -> None:
        sm = StateManager()
        sm.transition_to(AppState.STANDBY)
        sm.transition_to(AppState.ERROR)
        sm.transition_to(AppState.STANDBY)
        assert sm.state is AppState.STANDBY

    def test_listener_called(self) -> None:
        events = []
        sm = StateManager(listener=lambda old, new: events.append((old, new)))
        sm.transition_to(AppState.STANDBY)
        assert events == [(AppState.INITIALIZING, AppState.STANDBY)]

    def test_set_listener(self) -> None:
        sm = StateManager()
        events = []
        sm.set_listener(lambda old, new: events.append(new))
        sm.transition_to(AppState.STANDBY)
        assert events == [AppState.STANDBY]

    def test_listener_exception_does_not_break_state(self) -> None:
        def bad_listener(old, new):
            raise RuntimeError("boom")

        sm = StateManager(listener=bad_listener)
        sm.transition_to(AppState.STANDBY)
        assert sm.state is AppState.STANDBY


class TestSessionManager:
    """Tests for session lifecycle."""

    def test_initially_inactive(self) -> None:
        sm = SessionManager(timeout_seconds=60)
        assert sm.is_active is False
        assert sm.started_at is None
        assert sm.is_expired() is False

    def test_activate_and_deactivate(self) -> None:
        sm = SessionManager(timeout_seconds=60)
        sm.activate()
        assert sm.is_active is True
        assert sm.started_at is not None
        sm.deactivate()
        assert sm.is_active is False
        assert sm.started_at is None

    def test_timeout_configuration(self) -> None:
        sm = SessionManager(timeout_seconds=5)
        assert sm.timeout_seconds == 5
        with pytest.raises(ValueError):
            SessionManager(timeout_seconds=0)
        with pytest.raises(ValueError):
            SessionManager(timeout_seconds=-1)

    def test_not_expired_within_timeout(self) -> None:
        sm = SessionManager(timeout_seconds=60)
        sm.activate()
        assert sm.is_expired() is False

    def test_expired_after_timeout(self) -> None:
        sm = SessionManager(timeout_seconds=1)
        sm.activate()
        time.sleep(1.1)
        assert sm.is_expired() is True

    def test_inactive_never_expires(self) -> None:
        sm = SessionManager(timeout_seconds=1)
        time.sleep(1.1)
        assert sm.is_expired() is False

    def test_elapsed_seconds(self) -> None:
        sm = SessionManager(timeout_seconds=60)
        assert sm.elapsed_seconds == 0.0
        sm.activate()
        time.sleep(0.05)
        assert sm.elapsed_seconds >= 0.04


class TestVoiceOSApplication:
    """Tests for application initialization and wiring."""

    def _make_app(self, tmp_path, monkeypatch) -> VoiceOSApplication:
        monkeypatch.setenv("VOICEOS_DATA_DIR", str(tmp_path))
        app = VoiceOSApplication(database_path=tmp_path / "app.db")
        return app

    def test_initialize_moves_to_standby(self, tmp_path, monkeypatch) -> None:
        app = self._make_app(tmp_path, monkeypatch)
        app.initialize()
        assert app.state_manager.state is AppState.STANDBY

    def test_initialize_creates_database(self, tmp_path, monkeypatch) -> None:
        app = self._make_app(tmp_path, monkeypatch)
        app.initialize()
        assert app.database.path.exists()
        assert set(app.database.table_names()) == {
            "applications", "user_settings", "command_history"
        }

    def test_repositories_available(self, tmp_path, monkeypatch) -> None:
        app = self._make_app(tmp_path, monkeypatch)
        app.initialize()
        app.settings_repository.set("k", "v")
        assert app.settings_repository.get("k") == "v"
        entry_id = app.history_repository.record("test command")
        assert entry_id > 0

    def test_shutdown_transitions_to_stopped(self, tmp_path, monkeypatch) -> None:
        app = self._make_app(tmp_path, monkeypatch)
        app.initialize()
        app.shutdown()
        assert app.state_manager.state is AppState.STOPPED

    def test_config_accessible(self, tmp_path, monkeypatch) -> None:
        app = self._make_app(tmp_path, monkeypatch)
        assert app.config.assistant_name == "Jarvis"
        assert app.config.version == "1.0.0"

    def test_session_timeout_from_config(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("VOICEOS_DATA_DIR", str(tmp_path))
        config = ConfigurationManager(config_path="does/not/exist.json")
        app = VoiceOSApplication(config=config, database_path=tmp_path / "x.db")
        assert app.session_manager.timeout_seconds == 60
