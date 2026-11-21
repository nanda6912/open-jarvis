"""Shared/default command-handler primitives for VoiceOS.

This module provides baseline handlers, shared validation helpers and
the known-folder allowlist used by the specialized runtime handler
factories. It is not a complete description of which application
commands execute: the production registry also composes handlers from
the application, file, monitor, status, system and system-control
modules.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from src.commands.intent import KNOWN_FOLDERS
from src.commands.registry import (
    PERMISSION_SAFE,
    PERMISSION_SYSTEM,
    CommandResult,
)

logger = logging.getLogger(__name__)

# Folder names the OPEN_FOLDER handler accepts (allowlist).
SUPPORTED_FOLDERS = frozenset(KNOWN_FOLDERS)

_MAX_VOLUME = 100


class _BaseHandler:
    """Shared validation helpers."""

    permission = PERMISSION_SAFE
    executable = True
    intent_name = ""

    @staticmethod
    def _require(entities: dict, key: str) -> Optional[str]:
        value = entities.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            return f"missing_{key}"
        return None

    @staticmethod
    def _volume_amount(entities: dict):
        amount = entities.get("amount")
        if amount is None:
            return None
        try:
            amount = int(amount)
        except (TypeError, ValueError):
            return "invalid_amount"
        if amount < 0 or amount > _MAX_VOLUME:
            return "amount_out_of_range"
        return amount


class TimeHandler(_BaseHandler):
    """GET_CURRENT_TIME — safe informational intent."""

    intent_name = "GET_CURRENT_TIME"

    def validate(self, entities: dict) -> Optional[str]:
        return None

    def execute(self, entities: dict) -> CommandResult:
        now = datetime.now()
        hour = now.strftime("%I").lstrip("0") or "12"
        message = f"The current time is {hour}:{now.strftime('%M %p')}."
        return CommandResult(
            intent=self.intent_name,
            success=True,
            message=message,
            data={"hour": now.hour, "minute": now.minute},
            executed=True,
        )


class DateHandler(_BaseHandler):
    """GET_CURRENT_DATE — safe informational intent."""

    intent_name = "GET_CURRENT_DATE"

    def validate(self, entities: dict) -> Optional[str]:
        return None

    def execute(self, entities: dict) -> CommandResult:
        now = datetime.now()
        message = (
            f"Today is {now.strftime('%A, %B')} {now.day}, "
            f"{now.year}."
        )
        return CommandResult(
            intent=self.intent_name,
            success=True,
            message=message,
            data={"year": now.year, "month": now.month, "day": now.day},
            executed=True,
        )


class OpenApplicationHandler(_BaseHandler):
    """OPEN_APPLICATION — recognized and validated, not yet executed."""

    intent_name = "OPEN_APPLICATION"
    permission = PERMISSION_SYSTEM
    executable = False

    def validate(self, entities: dict) -> Optional[str]:
        return self._require(entities, "app_name")

    def execute(self, entities: dict) -> CommandResult:  # pragma: no cover
        raise NotImplementedError("execution lands in a later task")


class CloseApplicationHandler(_BaseHandler):
    """CLOSE_APPLICATION — recognized and validated, not yet executed."""

    intent_name = "CLOSE_APPLICATION"
    permission = PERMISSION_SYSTEM
    executable = False

    def validate(self, entities: dict) -> Optional[str]:
        return self._require(entities, "app_name")

    def execute(self, entities: dict) -> CommandResult:  # pragma: no cover
        raise NotImplementedError("execution lands in a later task")


class OpenFolderHandler(_BaseHandler):
    """OPEN_FOLDER — recognized and validated against an allowlist."""

    intent_name = "OPEN_FOLDER"
    permission = PERMISSION_SYSTEM
    executable = False

    def validate(self, entities: dict) -> Optional[str]:
        folder = entities.get("folder")
        if not folder or not str(folder).strip():
            return "missing_folder"
        if str(folder).strip().lower() not in SUPPORTED_FOLDERS:
            return "unsupported_folder"
        return None

    def execute(self, entities: dict) -> CommandResult:  # pragma: no cover
        raise NotImplementedError("execution lands in a later task")


class SearchFilesHandler(_BaseHandler):
    """SEARCH_FILES — recognized and validated, not yet executed."""

    intent_name = "SEARCH_FILES"
    permission = PERMISSION_SYSTEM
    executable = False

    def validate(self, entities: dict) -> Optional[str]:
        return self._require(entities, "query")

    def execute(self, entities: dict) -> CommandResult:  # pragma: no cover
        raise NotImplementedError("execution lands in a later task")


class SearchWebHandler(_BaseHandler):
    """SEARCH_WEB — recognized and validated, not executed here."""

    intent_name = "SEARCH_WEB"
    permission = PERMISSION_SYSTEM
    executable = False

    def validate(self, entities: dict) -> Optional[str]:
        return None

    def execute(self, entities: dict) -> CommandResult:  # pragma: no cover
        raise NotImplementedError("execution lands in a later task")


class _VolumeHandler(_BaseHandler):
    """Base for volume intents: validates optional amount 0-100."""

    permission = PERMISSION_SYSTEM
    executable = False

    def validate(self, entities: dict) -> Optional[str]:
        amount = entities.get("amount")
        if amount is None:
            return None
        check = self._volume_amount(entities)
        return check if isinstance(check, str) else None

    def execute(self, entities: dict) -> CommandResult:  # pragma: no cover
        raise NotImplementedError("execution lands in a later task")


class IncreaseVolumeHandler(_VolumeHandler):
    intent_name = "INCREASE_VOLUME"


class DecreaseVolumeHandler(_VolumeHandler):
    intent_name = "DECREASE_VOLUME"


class MuteVolumeHandler(_VolumeHandler):
    intent_name = "MUTE_VOLUME"


class UnmuteVolumeHandler(_VolumeHandler):
    intent_name = "UNMUTE_VOLUME"


class SetVolumeHandler(_VolumeHandler):
    """SET_VOLUME — recognized and validated, not executed here."""

    intent_name = "SET_VOLUME"

    def validate(self, entities: dict) -> Optional[str]:
        if entities.get("amount") is None:
            return "missing_amount"
        check = self._volume_amount(entities)
        return check if isinstance(check, str) else None


class GetVolumeHandler(_VolumeHandler):
    """GET_VOLUME — recognized, informational, not executed here."""

    intent_name = "GET_VOLUME"

    def validate(self, entities: dict) -> Optional[str]:
        return None


class LockWorkstationHandler(_BaseHandler):
    """LOCK_WORKSTATION — recognized and validated, not executed here."""

    intent_name = "LOCK_WORKSTATION"
    permission = PERMISSION_SYSTEM
    executable = False

    def validate(self, entities: dict) -> Optional[str]:
        return None

    def execute(self, entities: dict) -> CommandResult:  # pragma: no cover
        raise NotImplementedError("execution lands in a later task")


class UnsupportedActionHandler(_BaseHandler):
    """UNSUPPORTED_WINDOWS_ACTION — honest refusal, never executes.

    Windows power operations (sleep/shutdown/power off/restart) are
    deliberately unimplemented; this handler exists so the user hears
    that clearly instead of the session silently ending.
    """

    intent_name = "UNSUPPORTED_WINDOWS_ACTION"
    executable = True

    _MESSAGES = {
        "sleep": "put Windows to sleep",
        "shut_down": "shut down Windows",
        "power_off": "power off Windows",
        "restart": "restart Windows",
        "hibernate": "hibernate Windows",
        "sign_out": "sign out of Windows",
        "power": "change Windows power settings",
    }

    def validate(self, entities: dict) -> Optional[str]:
        return None

    def execute(self, entities: dict) -> CommandResult:
        action = str(entities.get("action") or "power")
        phrase = self._MESSAGES.get(action, self._MESSAGES["power"])
        return CommandResult(
            self.intent_name,
            False,
            message=f"Sorry, I can't {phrase} yet.",
            error="unsupported_action",
        )


class StatusStubHandler(_BaseHandler):
    """Recognized P2 awareness intents — validated, not executed here."""

    permission = PERMISSION_SYSTEM
    executable = False

    def __init__(self, intent_name: str) -> None:
        self.intent_name = intent_name

    def validate(self, entities: dict) -> Optional[str]:
        if self.intent_name == "SEND_NOTIFICATION":
            return self._require(entities, "message")
        return None

    def execute(self, entities: dict) -> CommandResult:  # pragma: no cover
        raise NotImplementedError("execution lands in a later task")


def create_default_registry() -> "CommandRegistry":
    """Build the registry with every supported intent registered."""
    from src.commands.registry import CommandRegistry

    registry = CommandRegistry()
    registry.register("GET_CURRENT_TIME", TimeHandler())
    registry.register("GET_CURRENT_DATE", DateHandler())
    registry.register("OPEN_APPLICATION", OpenApplicationHandler())
    registry.register("CLOSE_APPLICATION", CloseApplicationHandler())
    registry.register("OPEN_FOLDER", OpenFolderHandler())
    registry.register("SEARCH_FILES", SearchFilesHandler())
    registry.register("SEARCH_WEB", SearchWebHandler())
    registry.register("INCREASE_VOLUME", IncreaseVolumeHandler())
    registry.register("DECREASE_VOLUME", DecreaseVolumeHandler())
    registry.register("MUTE_VOLUME", MuteVolumeHandler())
    registry.register("UNMUTE_VOLUME", UnmuteVolumeHandler())
    registry.register("SET_VOLUME", SetVolumeHandler())
    registry.register("GET_VOLUME", GetVolumeHandler())
    registry.register("LOCK_WORKSTATION", LockWorkstationHandler())
    registry.register(
        "UNSUPPORTED_WINDOWS_ACTION", UnsupportedActionHandler()
    )
    for name in (
        "GET_BATTERY_STATUS", "GET_INTERNET_STATUS", "GET_WIFI_STATUS",
        "GET_ETHERNET_STATUS", "TAKE_SCREENSHOT", "SEND_NOTIFICATION",
    ):
        registry.register(name, StatusStubHandler(name))
    return registry
