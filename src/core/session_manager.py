"""Session management for VoiceOS.

Tracks whether a voice session is active (i.e. the assistant has been
activated and has not yet timed out) and when it started.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """Manage the active/inactive lifecycle of a voice session.

    The session timeout is configurable (in seconds). This class only
    tracks session metadata; timers and wake-word detection are handled
    by later modules.
    """

    def __init__(self, timeout_seconds: int = 60) -> None:
        """Create a session manager.

        Args:
            timeout_seconds: Seconds of inactivity after which a session
                is considered expired. Must be positive.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds
        self._active = False
        self._started_at: Optional[datetime] = None
        self._started_monotonic: Optional[float] = None
        self._last_activity_monotonic: Optional[float] = None

    @property
    def is_active(self) -> bool:
        """Return whether a session is currently active."""
        return self._active

    @property
    def timeout_seconds(self) -> int:
        """Return the configured session timeout in seconds."""
        return self._timeout_seconds

    @property
    def started_at(self) -> Optional[datetime]:
        """Return the wall-clock session start time, or ``None``."""
        return self._started_at

    @property
    def elapsed_seconds(self) -> float:
        """Return seconds since activation, or ``0.0`` if inactive."""
        if self._started_monotonic is None:
            return 0.0
        return time.monotonic() - self._started_monotonic

    @property
    def remaining_seconds(self) -> float:
        """Seconds until the inactivity timeout, or 0.0 if inactive."""
        if not self._active or self._last_activity_monotonic is None:
            return 0.0
        remaining = self.timeout_seconds - (
            time.monotonic() - self._last_activity_monotonic
        )
        return max(0.0, remaining)

    def activate(self) -> None:
        """Mark the session as active and record the start time."""
        self._active = True
        self._started_at = datetime.now()
        self._started_monotonic = time.monotonic()
        self._last_activity_monotonic = self._started_monotonic
        logger.info("Voice session activated")

    def deactivate(self) -> None:
        """Mark the session as inactive and clear the start time."""
        was_active = self._active
        self._active = False
        self._started_at = None
        self._started_monotonic = None
        self._last_activity_monotonic = None
        if was_active:
            logger.info("Voice session deactivated")

    def touch(self) -> None:
        """Record user activity, restarting the inactivity countdown."""
        if self._active:
            self._last_activity_monotonic = time.monotonic()

    def is_expired(self) -> bool:
        """Return whether the session has been inactive past its timeout.

        Inactivity is measured from the last ``activate()``/``touch()``
        call, so each recognized command extends the session. An
        inactive session is never considered expired.
        """
        if not self._active or self._last_activity_monotonic is None:
            return False
        return (
            time.monotonic() - self._last_activity_monotonic
            >= self._timeout_seconds
        )
