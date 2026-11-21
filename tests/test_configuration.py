"""Tests for the configuration manager."""

from __future__ import annotations

import json

import pytest

from src.core.configuration import (
    DEFAULT_SETTINGS,
    ConfigurationManager,
    get_app_data_dir,
)


class TestDefaultConfiguration:
    """Tests for default settings and accessors."""

    def test_defaults_match_spec(self) -> None:
        config = ConfigurationManager(config_path="does/not/exist.json")
        assert config.get("application.name") == "VoiceOS"
        assert config.get("application.assistant_name") == "Jarvis"
        assert config.get("application.version") == "1.0.0"
        assert config.get("voice.wake_word") == "Hello Jarvis"
        assert config.get("voice.activation_response") == "At your service"
        assert config.get("voice.exit_phrase") == "Goodbye Jarvis"
        assert config.get("voice.session_timeout_seconds") == 60
        assert config.get("voice.listening_enabled") is True
        assert config.get("startup.auto_start") is True
        assert config.get("startup.start_minimized") is True
        assert config.get("interface.theme") == "dark"
        assert config.get("interface.show_notifications") is True
        assert config.get("storage.database_name") == "voiceos.db"

    def test_convenience_properties(self) -> None:
        config = ConfigurationManager(config_path="does/not/exist.json")
        assert config.app_name == "VoiceOS"
        assert config.assistant_name == "Jarvis"
        assert config.version == "1.0.0"

    def test_missing_key_returns_default(self) -> None:
        config = ConfigurationManager(config_path="does/not/exist.json")
        assert config.get("nonexistent.key", "fallback") == "fallback"
        assert config.get("nonexistent.key") is None

    def test_settings_returns_copy(self) -> None:
        config = ConfigurationManager(config_path="does/not/exist.json")
        snapshot = config.settings
        snapshot["application"]["name"] = "Hacked"
        assert config.get("application.name") == "VoiceOS"


class TestConfigurationLoading:
    """Tests for loading settings from JSON files."""

    def test_loads_valid_overrides(self, tmp_path) -> None:
        config_file = tmp_path / "settings.json"
        config_file.write_text(
            json.dumps({"voice": {"session_timeout_seconds": 30}}),
            encoding="utf-8",
        )
        config = ConfigurationManager(config_path=config_file)
        assert config.get("voice.session_timeout_seconds") == 30
        # Untouched values keep defaults.
        assert config.get("application.name") == "VoiceOS"

    def test_missing_file_uses_defaults(self, tmp_path) -> None:
        config = ConfigurationManager(config_path=tmp_path / "missing.json")
        assert config.settings == DEFAULT_SETTINGS

    def test_malformed_json_uses_defaults(self, tmp_path) -> None:
        config_file = tmp_path / "bad.json"
        config_file.write_text("{not valid json", encoding="utf-8")
        config = ConfigurationManager(config_path=config_file)
        assert config.settings == DEFAULT_SETTINGS

    def test_non_object_json_uses_defaults(self, tmp_path) -> None:
        config_file = tmp_path / "list.json"
        config_file.write_text("[1, 2, 3]", encoding="utf-8")
        config = ConfigurationManager(config_path=config_file)
        assert config.settings == DEFAULT_SETTINGS

    def test_invalid_types_are_ignored(self, tmp_path) -> None:
        config_file = tmp_path / "types.json"
        config_file.write_text(
            json.dumps(
                {
                    "voice": {
                        "session_timeout_seconds": "sixty",  # wrong type
                        "listening_enabled": "yes",  # wrong type
                        "wake_word": "Hey Jarvis",  # valid
                    },
                    "application": {"version": 123},  # wrong type
                }
            ),
            encoding="utf-8",
        )
        config = ConfigurationManager(config_path=config_file)
        assert config.get("voice.session_timeout_seconds") == 60
        assert config.get("voice.listening_enabled") is True
        assert config.get("voice.wake_word") == "Hey Jarvis"
        assert config.get("application.version") == "1.0.0"

    def test_bool_rejected_for_int_field(self, tmp_path) -> None:
        config_file = tmp_path / "bool.json"
        config_file.write_text(
            json.dumps({"voice": {"session_timeout_seconds": True}}),
            encoding="utf-8",
        )
        config = ConfigurationManager(config_path=config_file)
        assert config.get("voice.session_timeout_seconds") == 60

    def test_unknown_sections_ignored(self, tmp_path) -> None:
        config_file = tmp_path / "extra.json"
        config_file.write_text(
            json.dumps({"unknown_section": {"foo": "bar"}}),
            encoding="utf-8",
        )
        config = ConfigurationManager(config_path=config_file)
        assert config.settings == DEFAULT_SETTINGS


class TestAppDataDir:
    """Tests for data directory resolution."""

    def test_env_override(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("VOICEOS_DATA_DIR", str(tmp_path / "custom"))
        assert get_app_data_dir() == tmp_path / "custom"
        assert (tmp_path / "custom").is_dir()

    def test_database_path_uses_data_dir(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("VOICEOS_DATA_DIR", str(tmp_path))
        config = ConfigurationManager(config_path="does/not/exist.json")
        assert config.database_path == tmp_path / "voiceos.db"

    def test_database_name_traversal_guarded(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.setenv("VOICEOS_DATA_DIR", str(tmp_path))
        config_file = tmp_path / "cfg.json"
        config_file.write_text(
            json.dumps({"storage": {"database_name": "..\\evil.db"}}),
            encoding="utf-8",
        )
        config = ConfigurationManager(config_path=config_file)
        assert config.database_path == tmp_path / "voiceos.db"
