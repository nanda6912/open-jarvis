"""Allowlisted command registry for VoiceOS.

Every recognized intent is dispatched through this registry. A command
is only runnable when it has a registered handler that validates its
arguments, checks its declared permissions and returns a structured
:class:`CommandResult`. Speech is never executed as a shell command —
handlers perform only the explicit, reviewed action they implement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Protocol

from src.commands.intent import Intent

logger = logging.getLogger(__name__)

# Permission levels (declared by handlers, checked by the registry).
PERMISSION_SAFE = "safe"          # read-only informational
PERMISSION_SYSTEM = "system"      # OS-level but non-destructive
PERMISSION_DESTRUCTIVE = "destructive"

# Permissions the registry is currently willing to grant. Handlers that
# require anything else are refused before execution.
GRANTED_PERMISSIONS = frozenset({PERMISSION_SAFE, PERMISSION_SYSTEM})


@dataclass
class CommandResult:
    """Structured outcome of a dispatched command."""

    intent: str
    success: bool
    message: str = ""
    data: dict = field(default_factory=dict)
    error: Optional[str] = None
    executed: bool = False
    permission: str = PERMISSION_SAFE
    confirmation_required: bool = False
    ambiguous: bool = False
    options: tuple = ()


class CommandHandler(Protocol):
    """Interface every registered command handler must implement."""

    intent_name: str
    permission: str
    executable: bool

    def validate(self, entities: dict) -> Optional[str]:
        """Return an error string when arguments are invalid, else None."""
        ...

    def execute(self, entities: dict) -> CommandResult:
        """Perform the command and return a structured result."""
        ...


class CommandRegistry:
    """Central allowlist mapping intents to handlers.

    Dispatch order for every intent:
      1. handler must be registered (allowlist),
      2. handler's declared permission must be granted,
      3. ``validate()`` must accept the extracted entities,
      4. ``execute()`` runs inside a guarded boundary.
    """

    def __init__(self) -> None:
        self._handlers: dict[str, CommandHandler] = {}

    def register(self, intent_name: str, handler: CommandHandler) -> None:
        """Register ``handler`` for ``intent_name`` (idempotent)."""
        self._handlers[intent_name] = handler
        logger.debug("Registered command handler: %s", intent_name)

    def unregister(self, intent_name: str) -> None:
        self._handlers.pop(intent_name, None)

    def is_registered(self, intent_name: str) -> bool:
        return intent_name in self._handlers

    @property
    def registered_intents(self) -> frozenset:
        return frozenset(self._handlers)

    def dispatch(self, intent: Intent) -> CommandResult:
        """Route ``intent`` through validation, permissions and handler."""
        handler = self._handlers.get(intent.name)
        if handler is None:
            logger.info("No handler registered for intent %r", intent.name)
            return CommandResult(
                intent=intent.name,
                success=False,
                error="unregistered_intent",
                message="",
            )

        permission = getattr(handler, "permission", PERMISSION_SAFE)
        if permission not in GRANTED_PERMISSIONS:
            logger.warning(
                "Intent %r requires permission %r which is not granted",
                intent.name,
                permission,
            )
            return CommandResult(
                intent=intent.name,
                success=False,
                error="permission_denied",
                permission=permission,
            )

        try:
            error = handler.validate(intent.entities)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Validation crashed for %r", intent.name)
            return CommandResult(
                intent=intent.name,
                success=False,
                error=f"validation_error: {exc}",
                permission=permission,
            )
        if error:
            logger.info("Intent %r rejected: %s", intent.name, error)
            return CommandResult(
                intent=intent.name,
                success=False,
                error=error,
                permission=permission,
            )

        if not getattr(handler, "executable", True):
            # Recognized and validated, but this task does not execute it.
            return CommandResult(
                intent=intent.name,
                success=True,
                executed=False,
                permission=permission,
            )

        try:
            result = handler.execute(intent.entities)
            result.intent = intent.name
            result.permission = permission
            return result
        except Exception as exc:  # noqa: BLE001
            logger.exception("Handler %r failed", intent.name)
            return CommandResult(
                intent=intent.name,
                success=False,
                error=f"execution_error: {exc}",
                permission=permission,
            )
