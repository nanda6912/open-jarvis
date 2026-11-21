"""Application configuration management for VoiceOS.

Loads settings from a JSON file, validates them against the expected
schema, and provides a clean read-only access interface. Missing or
malformed configuration files fall back to built-in defaults.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SETTINGS: dict[str, Any] = {
    "application": {
        "name": "VoiceOS",
        "assistant_name": "Jarvis",
        "version": "1.0.0",
    },
    "voice": {
        "wake_word": "Hello Jarvis",
        "activation_response": "At your service",
        "exit_phrase": "Goodbye Jarvis",
        "session_timeout_seconds": 60,
        "listening_enabled": True,
        "microphone_enabled": True,
        "wake_word_enabled": True,
        "wake_word_threshold": 0.25,
        "stt_enabled": True,
        "stt_engine": "auto",
        "stt_model_dir": "",
        "stt_num_threads": 2,
        "stt_speech_threshold": 0.01,
        "tts_enabled": True,
        "tts_engine": "sherpa-onnx-vits",
        "tts_model_dir": "",
        "tts_num_threads": 2,
        "tts_speed": 1.0,
        "ack_mode": "full",
        "quick_ack_response": "Yes.",
    },
    "startup": {
        "auto_start": True,
        "start_minimized": True,
    },
    "interface": {
        "theme": "dark",
        "show_notifications": True,
    },
    "hud": {
        "enabled": True,
        "monitor": "auto",
        "size": 320,
        "position": "",
        "position_screen": "",
        "locked": True,
        "always_on_top": True,
        "fullscreen": True,
        "fps": 30,
        "appearance": {
            "preset": "balanced",
            "brightness": 65,
            "glow": 55,
            "opacity": 95,
            "ring": 75,
            "particle": 60,
            "waveform": 70,
            "speed": 100,
        },
    },
    "storage": {
        "database_name": "voiceos.db",
    },
}

# Minimal schema used for validation: section -> {key: expected type}.
_SETTINGS_SCHEMA: dict[str, dict[str, type]] = {
    "application": {
        "name": str,
        "assistant_name": str,
        "version": str,
    },
    "voice": {
        "wake_word": str,
        "activation_response": str,
        "exit_phrase": str,
        "session_timeout_seconds": int,
        "listening_enabled": bool,
        "microphone_enabled": bool,
        "wake_word_enabled": bool,
        "wake_word_threshold": float,
        "stt_enabled": bool,
        "stt_engine": str,
        "stt_model_dir": str,
        "stt_num_threads": int,
        "stt_speech_threshold": float,
        "tts_enabled": bool,
        "tts_engine": str,
        "tts_model_dir": str,
        "tts_num_threads": int,
        "tts_speed": float,
        "ack_mode": str,
        "quick_ack_response": str,
    },
    "startup": {
        "auto_start": bool,
        "start_minimized": bool,
    },
    "interface": {
        "theme": str,
        "show_notifications": bool,
    },
    "hud": {
        "enabled": bool,
        "monitor": str,
        "size": int,
        "position": str,
        "position_screen": str,
        "locked": bool,
        "always_on_top": bool,
        "fullscreen": bool,
        "fps": int,
        "appearance": dict,
    },
    "storage": {
        "database_name": str,
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default_settings.json"


def get_app_data_dir() -> Path:
    """Return the platform-appropriate local application data directory.

    Delegates to :func:`src.core.paths.runtime_dir`. On Windows this is
    ``%LOCALAPPDATA%/VoiceOS``. The result can be overridden with the
    ``VOICEOS_HOME`` environment variable (or the legacy
    ``VOICEOS_DATA_DIR`` alias), which is primarily useful for testing.
    """
    from src.core.paths import runtime_dir

    return runtime_dir()


class ConfigurationError(Exception):
    """Raised when configuration cannot be loaded or is invalid."""


def _load_dotenv() -> None:
    """Load ``.env`` from the project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(PROJECT_ROOT / ".env")


class ConfigurationManager:
    """Load, validate and expose VoiceOS settings.

    Settings are loaded from a JSON file when available. Missing files,
    malformed JSON and invalid values are handled gracefully: valid values
    from the file are kept, everything else falls back to defaults, and
    problems are logged.
    """

    def __init__(self, config_path: Path | str | None = None) -> None:
        """Create a configuration manager.

        Args:
            config_path: Path to the JSON settings file. Defaults to
                ``config/default_settings.json`` in the project root.
        """
        _load_dotenv()
        self._config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._settings: dict[str, Any] = {}
        self.load()

    # ------------------------------------------------------------------
    # Loading / validation
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load settings from disk and merge them with the defaults."""
        merged = deepcopy(DEFAULT_SETTINGS)

        if not self._config_path.exists():
            logger.info(
                "Configuration file %s not found; using defaults.",
                self._config_path,
            )
        else:
            try:
                raw = json.loads(self._config_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                logger.error(
                    "Failed to parse configuration file %s: %s. Using defaults.",
                    self._config_path,
                    exc,
                )
            else:
                if not isinstance(raw, dict):
                    logger.error(
                        "Configuration file %s must contain a JSON object. "
                        "Using defaults.",
                        self._config_path,
                    )
                else:
                    merged = self._merge_and_validate(raw, merged)

        self._settings = merged
        logger.debug("Configuration loaded from %s", self._config_path)

    def _merge_and_validate(
        self, raw: dict[str, Any], defaults: dict[str, Any]
    ) -> dict[str, Any]:
        """Merge user settings into defaults, keeping only valid values."""
        for section, expected_keys in _SETTINGS_SCHEMA.items():
            raw_section = raw.get(section)
            if not isinstance(raw_section, dict):
                if raw_section is not None:
                    logger.warning(
                        "Configuration section '%s' is not an object; "
                        "ignoring it.",
                        section,
                    )
                continue
            for key, expected_type in expected_keys.items():
                value = raw_section.get(key)
                if value is None:
                    continue
                # bool is a subclass of int; reject bools for int fields.
                if expected_type is int and isinstance(value, bool):
                    logger.warning(
                        "Invalid type for '%s.%s' (expected int); ignoring.",
                        section,
                        key,
                    )
                    continue
                if not isinstance(value, expected_type):
                    logger.warning(
                        "Invalid type for '%s.%s' (expected %s); ignoring.",
                        section,
                        key,
                        expected_type.__name__,
                    )
                    continue
                defaults[section][key] = value

        # Reject unknown top-level noise silently is fine; defaults cover it.
        return defaults

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get(self, key: str, default: Any = None) -> Any:
        """Get a setting using dot notation, e.g. ``get('voice.wake_word')``."""
        node: Any = self._settings
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def get_section(self, section: str) -> dict[str, Any]:
        """Return a copy of an entire settings section."""
        section_data = self._settings.get(section, {})
        return deepcopy(section_data) if isinstance(section_data, dict) else {}

    @property
    def settings(self) -> dict[str, Any]:
        """Return a deep copy of the full settings dictionary."""
        return deepcopy(self._settings)

    @property
    def app_name(self) -> str:
        """Return the application display name."""
        return str(self.get("application.name", "VoiceOS"))

    @property
    def assistant_name(self) -> str:
        """Return the assistant display name."""
        return str(self.get("application.assistant_name", "Jarvis"))

    @property
    def version(self) -> str:
        """Return the application version string."""
        return str(self.get("application.version", "0.0.0"))

    @property
    def database_path(self) -> Path:
        """Return the resolved SQLite database file path.

        ``VOICEOS_DB_PATH`` overrides the configured location entirely.
        """
        override = os.environ.get("VOICEOS_DB_PATH")
        if override:
            path = Path(override)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        data_dir = get_app_data_dir()
        name = str(self.get("storage.database_name", "voiceos.db"))
        # Guard against path traversal in the configured file name.
        if Path(name).name != name:
            logger.warning(
                "Invalid database_name %r; falling back to 'voiceos.db'.", name
            )
            name = "voiceos.db"
        return data_dir / name
