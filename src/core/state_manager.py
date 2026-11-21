"""Application state management for VoiceOS.

Defines the finite set of application states and enforces safe
transitions between them.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class AppState(Enum):
    """Possible states of the VoiceOS application."""

    INITIALIZING = "INITIALIZING"
    STANDBY = "STANDBY"
    ACTIVATED = "ACTIVATED"
    LISTENING = "LISTENING"
    PROCESSING = "PROCESSING"
    EXECUTING = "EXECUTING"
    CONFIRMATION = "CONFIRMATION"
    ERROR = "ERROR"
    STOPPED = "STOPPED"


# Allowed transitions. ERROR and STOPPED are reachable from anywhere;
# INITIALIZING may only transition forward into the runtime states.
_ALLOWED_TRANSITIONS: dict[AppState, frozenset[AppState]] = {
    AppState.INITIALIZING: frozenset(
        {AppState.STANDBY, AppState.ERROR, AppState.STOPPED}
    ),
    AppState.STANDBY: frozenset(
        {
            AppState.ACTIVATED,
            AppState.ERROR,
            AppState.STOPPED,
        }
    ),
    AppState.ACTIVATED: frozenset(
        {
            AppState.LISTENING,
            AppState.STANDBY,
            AppState.ERROR,
            AppState.STOPPED,
        }
    ),
    AppState.LISTENING: frozenset(
        {
            AppState.PROCESSING,
            AppState.ACTIVATED,
            AppState.STANDBY,
            AppState.ERROR,
            AppState.STOPPED,
        }
    ),
    AppState.PROCESSING: frozenset(
        {
            AppState.EXECUTING,
            AppState.CONFIRMATION,
            AppState.ACTIVATED,
            AppState.ERROR,
            AppState.STOPPED,
        }
    ),
    AppState.EXECUTING: frozenset(
        {
            AppState.ACTIVATED,
            AppState.LISTENING,
            AppState.STANDBY,
            AppState.ERROR,
            AppState.STOPPED,
        }
    ),
    AppState.CONFIRMATION: frozenset(
        {
            AppState.EXECUTING,
            AppState.ACTIVATED,
            AppState.STANDBY,
            AppState.ERROR,
            AppState.STOPPED,
        }
    ),
    AppState.ERROR: frozenset(
        {
            AppState.STANDBY,
            AppState.STOPPED,
        }
    ),
    AppState.STOPPED: frozenset(),
}


def allowed_transitions(state: AppState) -> frozenset:
    """Return the set of states reachable in one step from ``state``."""
    return _ALLOWED_TRANSITIONS[state]


class InvalidStateTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""


class StateManager:
    """Track and validate VoiceOS application state transitions.

    Optionally accepts a listener that is invoked after every successful
    state change, which the UI layer can use to update itself.
    """

    def __init__(
        self,
        initial_state: AppState = AppState.INITIALIZING,
        listener: Optional[Callable[[AppState, AppState], None]] = None,
    ) -> None:
        """Create a state manager.

        Args:
            initial_state: Starting state. Defaults to ``INITIALIZING``.
            listener: Optional callback ``(old_state, new_state)`` invoked
                after each successful transition.
        """
        self._state = initial_state
        self._listener = listener

    @property
    def state(self) -> AppState:
        """Return the current application state."""
        return self._state

    def set_listener(self, listener: Optional[Callable[[AppState, AppState], None]]) -> None:
        """Set or replace the state-change listener."""
        self._listener = listener

    def can_transition_to(self, new_state: AppState) -> bool:
        """Return whether a transition to ``new_state`` is allowed."""
        return new_state in _ALLOWED_TRANSITIONS[self._state]

    def transition_to(self, new_state: AppState) -> None:
        """Transition to ``new_state``, raising on illegal transitions.

        Args:
            new_state: The target state.

        Raises:
            InvalidStateTransitionError: If the transition is not allowed.
        """
        if new_state is self._state:
            return
        if not self.can_transition_to(new_state):
            raise InvalidStateTransitionError(
                f"Cannot transition from {self._state.name} to {new_state.name}"
            )
        old_state = self._state
        self._state = new_state
        logger.info("State transition: %s -> %s", old_state.name, new_state.name)
        if self._listener is not None:
            try:
                self._listener(old_state, new_state)
            except Exception:  # noqa: BLE001 - listeners must not break state
                logger.exception("State change listener failed")
