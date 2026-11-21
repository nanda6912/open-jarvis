"""Windows system command handlers (P1: folders, volume, lock).

These handlers execute real, reviewed operations through the injected
services on ``_Svc`` — never through a shell and never with spoken text
as a command. LOCK_WORKSTATION reuses the confirmation lifecycle: the
first dispatch asks for confirmation, a fresh "yes" re-dispatches with
``_confirmed=True``.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.commands.handlers import SUPPORTED_FOLDERS, _BaseHandler
from src.commands.registry import CommandResult, PERMISSION_SYSTEM

logger = logging.getLogger(__name__)

_FOLDER_LABELS = {
    "desktop": "Desktop",
    "documents": "Documents",
    "downloads": "Downloads",
    "pictures": "Pictures",
    "photos": "Pictures",
    "music": "Music",
    "videos": "Videos",
    "movies": "Videos",
}

# Spoken volume step when the user gives no amount.
DEFAULT_VOLUME_STEP = 5


def _unavailable(intent_name: str, what: str) -> CommandResult:
    return CommandResult(
        intent_name,
        False,
        message=f"I couldn't reach {what}.",
        error="service_unavailable",
    )


class OpenKnownFolderHandler(_BaseHandler):
    """OPEN_FOLDER — open an allowlisted Windows known folder."""

    intent_name = "OPEN_FOLDER"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc):
        self._svc = svc

    def validate(self, entities: dict) -> Optional[str]:
        folder = entities.get("folder")
        if not folder or not str(folder).strip():
            return "missing_folder"
        if str(folder).strip().lower() not in SUPPORTED_FOLDERS:
            return "unsupported_folder"
        return None

    def execute(self, entities: dict) -> CommandResult:
        folder = str(entities.get("folder") or "").strip().lower()
        opener = getattr(self._svc, "folders", None)
        if opener is None:
            return _unavailable(self.intent_name, "the folder service")
        ok, detail = opener.open(folder)
        label = _FOLDER_LABELS.get(folder, folder.capitalize())
        if not ok:
            if detail == "folder_not_found":
                message = f"I couldn't find your {label} folder."
            elif detail == "unsupported_folder":
                message = f"I can't open {label}."
            else:
                message = f"I couldn't open your {label} folder."
            return CommandResult(
                self.intent_name, False, message=message,
                error=detail or "open_failed",
            )
        return CommandResult(
            self.intent_name,
            True,
            message=f"Opening your {label} folder.",
            data={"folder": folder},
            executed=True,
        )


class _VolumeSvcHandler(_BaseHandler):
    """Base for executable volume intents backed by svc.volume."""

    intent_name = ""
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc):
        self._svc = svc

    def _volume(self):
        volume = getattr(self._svc, "volume", None)
        if volume is None:
            return None
        return volume

    def _fail(self, error: str, message: str) -> CommandResult:
        return CommandResult(
            self.intent_name, False, message=message, error=error,
        )

    def _no_device(self) -> CommandResult:
        return self._fail(
            "no_audio_endpoint",
            "I couldn't find an audio output device.",
        )


class _AdjustVolumeHandler(_VolumeSvcHandler):
    """INCREASE_VOLUME / DECREASE_VOLUME — relative level change."""

    direction = 0

    def validate(self, entities: dict) -> Optional[str]:
        amount = entities.get("amount")
        if amount is None:
            return None
        check = _BaseHandler._volume_amount(entities)
        return check if isinstance(check, str) else None

    def execute(self, entities: dict) -> CommandResult:
        volume = self._volume()
        if volume is None:
            return _unavailable(self.intent_name, "the volume service")
        try:
            delta = int(entities.get("amount") or DEFAULT_VOLUME_STEP)
        except (TypeError, ValueError):
            delta = DEFAULT_VOLUME_STEP
        delta = max(1, min(delta, 100)) * self.direction
        level = volume.adjust(delta)
        if level is None:
            return self._no_device()
        muted = volume.is_muted()
        verb = "increased" if self.direction > 0 else "decreased"
        message = f"Volume {verb} to {level} percent."
        if muted:
            message += " The volume is still muted."
        return CommandResult(
            self.intent_name,
            True,
            message=message,
            data={"level": level, "muted": bool(muted)},
            executed=True,
        )


class IncreaseVolumeHandler(_AdjustVolumeHandler):
    intent_name = "INCREASE_VOLUME"
    direction = 1


class DecreaseVolumeHandler(_AdjustVolumeHandler):
    intent_name = "DECREASE_VOLUME"
    direction = -1


class _MuteHandler(_VolumeSvcHandler):
    muted = True
    past = "Muted."
    unpast = "Unmuted."

    def validate(self, entities: dict) -> Optional[str]:
        return None

    def execute(self, entities: dict) -> CommandResult:
        volume = self._volume()
        if volume is None:
            return _unavailable(self.intent_name, "the volume service")
        ok = volume.set_mute(self.muted)
        if not ok:
            return self._no_device()
        return CommandResult(
            self.intent_name,
            True,
            message=self.past if self.muted else self.unpast,
            data={"muted": self.muted},
            executed=True,
        )


class MuteVolumeHandler(_MuteHandler):
    intent_name = "MUTE_VOLUME"
    muted = True
    past = "Muted."
    unpast = "Unmuted."


class UnmuteVolumeHandler(_MuteHandler):
    intent_name = "UNMUTE_VOLUME"
    muted = False
    past = "Muted."
    unpast = "Unmuted."


class SetVolumeHandler(_VolumeSvcHandler):
    """SET_VOLUME — absolute level 0-100."""

    intent_name = "SET_VOLUME"

    def validate(self, entities: dict) -> Optional[str]:
        if entities.get("amount") is None:
            return "missing_amount"
        check = _BaseHandler._volume_amount(entities)
        return check if isinstance(check, str) else None

    def execute(self, entities: dict) -> CommandResult:
        volume = self._volume()
        if volume is None:
            return _unavailable(self.intent_name, "the volume service")
        try:
            target = int(entities["amount"])
        except (KeyError, TypeError, ValueError):
            return self._fail("invalid_amount", "Invalid volume level.")
        target = max(0, min(target, 100))
        level = volume.set_percent(target)
        if level is None:
            return self._no_device()
        return CommandResult(
            self.intent_name,
            True,
            message=f"Volume set to {level} percent.",
            data={"level": level},
            executed=True,
        )


class GetVolumeHandler(_VolumeSvcHandler):
    """GET_VOLUME — report the current output level."""

    intent_name = "GET_VOLUME"

    def validate(self, entities: dict) -> Optional[str]:
        return None

    def execute(self, entities: dict) -> CommandResult:
        volume = self._volume()
        if volume is None:
            return _unavailable(self.intent_name, "the volume service")
        level = volume.get_percent()
        if level is None:
            return self._no_device()
        if volume.is_muted():
            message = "Your volume is muted."
        else:
            message = f"Your volume is at {level} percent."
        return CommandResult(
            self.intent_name,
            True,
            message=message,
            data={"level": level, "muted": bool(volume.is_muted())},
            executed=True,
        )


class LockWorkstationHandler(_BaseHandler):
    """LOCK_WORKSTATION — LockWorkStation after explicit confirmation."""

    intent_name = "LOCK_WORKSTATION"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc):
        self._svc = svc

    def validate(self, entities: dict) -> Optional[str]:
        return None

    def execute(self, entities: dict) -> CommandResult:
        if not entities.get("_confirmed"):
            return CommandResult(
                self.intent_name,
                True,
                message="Should I lock your workstation? Say yes to confirm.",
                confirmation_required=True,
            )
        locker = getattr(self._svc, "locker", None)
        if locker is None:
            return _unavailable(self.intent_name, "the lock service")
        try:
            ok = bool(locker.lock())
        except Exception:  # noqa: BLE001
            logger.exception("LockWorkStation failed")
            ok = False
        if not ok:
            return CommandResult(
                self.intent_name,
                False,
                message="I couldn't lock the workstation.",
                error="lock_failed",
            )
        return CommandResult(
            self.intent_name,
            True,
            message="Locking your workstation.",
            executed=True,
        )
