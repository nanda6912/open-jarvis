"""Application logging configuration for VoiceOS."""

from __future__ import annotations

import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


def setup_logging(
    log_dir: Path | str,
    level: int = logging.INFO,
    max_bytes: int = 1_000_000,
    backup_count: int = 5,
) -> Path:
    """Configure root logging with console and rotating file handlers.

    Args:
        log_dir: Directory for the rotating log file.
        level: Minimum log level.
        max_bytes: Maximum size of a single log file before rotation.
        backup_count: Number of rotated backup files to keep.

    Returns:
        Path to the active log file.
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = log_path / "voiceos.log"

    root = logging.getLogger()
    root.setLevel(level)

    # Avoid duplicate handlers if setup_logging is called twice.
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_DEFAULT_FORMAT)

    file_handler = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    logging.getLogger(__name__).debug("Logging initialized at %s", log_file)
    return log_file
