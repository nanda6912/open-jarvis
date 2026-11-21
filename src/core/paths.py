"""Central runtime path resolution for VoiceOS.

All user-writable runtime data lives outside the source tree so the
repository stays clean and the application works from any checkout
location. Paths are resolved on demand (they are only needed during
startup), never inside audio callbacks, inference or rendering loops.

Environment overrides (all optional)::

    VOICEOS_HOME              Runtime root directory.
                              Default: ``%LOCALAPPDATA%\\VoiceOS`` on
                              Windows, ``~/.local/share/voiceos`` else.
    VOICEOS_DATA_DIR          Legacy alias for ``VOICEOS_HOME``
                              (kept for backward compatibility).
    VOICEOS_KWS_MODEL_DIR     Wake-word (KWS) model directory.
    VOICEOS_STT_MODEL_DIR     Speech-to-text model directory.
    VOICEOS_TTS_MODEL_DIR     Text-to-speech model directory.
    VOICEOS_DB_PATH           Main SQLite database file.
    VOICEOS_FILE_INDEX_DB_PATH  File-index SQLite database file.
    VOICEOS_LOG_DIR           Log output directory.

Suggested runtime layout::

    <VOICEOS_HOME>/
        models/{kws,stt,tts}/
        logs/
        cache/
        voiceos.db
        file_index.db
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_DIR_NAME = "VoiceOS"


def _load_dotenv() -> None:
    """Load ``.env`` from the project root if python-dotenv is available.

    Existing process environment variables always win (dotenv default).
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


_load_dotenv()


def _env_value(*names: str) -> str | None:
    """Return the first non-empty environment variable among ``names``."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def runtime_dir() -> Path:
    """Return the VoiceOS runtime data directory (created if missing).

    Resolution order: ``VOICEOS_HOME`` -> ``VOICEOS_DATA_DIR`` ->
    platform default (``%LOCALAPPDATA%\\VoiceOS`` on Windows).
    """
    override = _env_value("VOICEOS_HOME", "VOICEOS_DATA_DIR")
    if override:
        path = Path(override)
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        path = (
            Path(base) / APP_DIR_NAME
            if base
            else Path.home() / "AppData" / "Local" / APP_DIR_NAME
        )
    else:
        path = Path.home() / ".local" / "share" / "voiceos"
    path.mkdir(parents=True, exist_ok=True)
    return path


def logs_dir() -> Path:
    """Return the log directory (``VOICEOS_LOG_DIR`` or ``<home>/logs``)."""
    override = _env_value("VOICEOS_LOG_DIR")
    path = Path(override) if override else runtime_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    """Return the shared models root (``<home>/models``)."""
    path = runtime_dir() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def kws_model_dir() -> Path:
    """Return the KWS model directory (``VOICEOS_KWS_MODEL_DIR`` override)."""
    override = _env_value("VOICEOS_KWS_MODEL_DIR")
    path = Path(override) if override else models_dir() / "kws"
    path.mkdir(parents=True, exist_ok=True)
    return path


def stt_model_dir() -> Path:
    """Return the STT model directory (``VOICEOS_STT_MODEL_DIR`` override)."""
    override = _env_value("VOICEOS_STT_MODEL_DIR")
    path = Path(override) if override else models_dir() / "stt"
    path.mkdir(parents=True, exist_ok=True)
    return path


def tts_model_dir() -> Path:
    """Return the TTS model directory (``VOICEOS_TTS_MODEL_DIR`` override)."""
    override = _env_value("VOICEOS_TTS_MODEL_DIR")
    path = Path(override) if override else models_dir() / "tts"
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path(default_name: str = "voiceos.db") -> Path:
    """Return the main SQLite database path.

    ``VOICEOS_DB_PATH`` overrides everything; otherwise the file lives at
    ``<runtime_dir>/<default_name>``.
    """
    override = _env_value("VOICEOS_DB_PATH")
    if override:
        path = Path(override)
    else:
        path = runtime_dir() / default_name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def file_index_db_path() -> Path:
    """Return the file-index SQLite database path."""
    override = _env_value("VOICEOS_FILE_INDEX_DB_PATH")
    if override:
        path = Path(override)
    else:
        path = runtime_dir() / "file_index.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def cache_dir() -> Path:
    """Return the runtime cache directory (created if missing)."""
    path = runtime_dir() / "cache"
    path.mkdir(parents=True, exist_ok=True)
    return path
