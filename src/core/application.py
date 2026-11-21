"""Central application object for VoiceOS.

Wires together configuration, state, session and persistence without
depending on any UI toolkit, keeping the core independently testable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Optional

from src.core.configuration import ConfigurationManager, get_app_data_dir
from src.core.session_manager import SessionManager
from src.core.state_manager import AppState, StateManager
from src.storage.database import Database, DatabaseError
from src.storage.history_repository import HistoryRepository
from src.storage.settings_repository import SettingsRepository

logger = logging.getLogger(__name__)


class VoiceOSApplication:
    """Application facade coordinating core services.

    Args:
        config: Optional pre-built configuration manager.
        database_path: Optional explicit database path (mainly for tests).
        state_listener: Optional callback for state transitions.
    """

    def __init__(
        self,
        config: Optional[ConfigurationManager] = None,
        database_path: Optional[Path] = None,
        state_listener: Optional[Callable[[AppState, AppState], None]] = None,
    ) -> None:
        self.config = config or ConfigurationManager()
        self.state_manager = StateManager(listener=state_listener)
        timeout = int(self.config.get("voice.session_timeout_seconds", 60))
        self.session_manager = SessionManager(timeout_seconds=timeout)

        db_path = Path(database_path) if database_path else self.config.database_path
        self.database = Database(db_path)
        self.settings_repository = SettingsRepository(self.database)
        self.history_repository = HistoryRepository(self.database)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Initialize persistence and move to the STANDBY state.

        Raises:
            DatabaseError: If the database cannot be initialized.
        """
        self.database.initialize()
        self.state_manager.transition_to(AppState.STANDBY)
        logger.info(
            "%s %s initialized (assistant: %s)",
            self.config.app_name,
            self.config.version,
            self.config.assistant_name,
        )

    def shutdown(self) -> None:
        """Shut the application down cleanly."""
        if self.state_manager.state is not AppState.STOPPED:
            try:
                self.state_manager.transition_to(AppState.STOPPED)
            except Exception:  # noqa: BLE001 - shutdown must not fail
                logger.exception("Failed to transition to STOPPED")
        logger.info("%s shut down", self.config.app_name)

    def on_wake_word(self) -> bool:
        """Handle a detected wake word: activate session and state.

        Transitions STANDBY -> ACTIVATED and marks the voice session
        active. Returns True if the app is now activated.
        """
        if self.state_manager.state is AppState.STANDBY:
            self.state_manager.transition_to(AppState.ACTIVATED)
        if not self.session_manager.is_active:
            self.session_manager.activate()
        logger.info("Assistant activated by wake word")
        return self.state_manager.state is AppState.ACTIVATED

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def data_dir(self) -> Path:
        """Return the application data directory in use."""
        return get_app_data_dir()
