"""Windows system-control command handlers (Milestone 1).

Power management (sleep / scheduled shutdown / scheduled restart /
cancel) and active-window desktop controls. All destructive operations
reuse the existing yes/no confirmation lifecycle: the first dispatch
returns ``confirmation_required``, a fresh "yes" re-dispatches the same
intent with ``_confirmed=True``. The confirmation is tied to the exact
action via ``_armed_action`` recorded on first dispatch.

Restart/shutdown are *scheduled*, never claimed as completed: the
handler returns a spoken acknowledgement carrying an ``after_speech``
callback; the session controller invokes it only after that
acknowledgement finishes playing, and only then does the 15-second
countdown begin. Sleep executes through the same deferred path so the
acknowledgement is always heard first.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.commands.handlers import _BaseHandler
from src.commands.registry import CommandResult, PERMISSION_SYSTEM
from src.controllers.power_manager import SCHEDULE_DELAY_SECONDS

logger = logging.getLogger(__name__)


def _unavailable(intent_name: str, what: str) -> CommandResult:
    return CommandResult(
        intent_name,
        False,
        message=f"I couldn't reach {what}.",
        error="service_unavailable",
    )


# ----------------------------------------------------------------------
# Power management
# ----------------------------------------------------------------------
class _PowerHandler(_BaseHandler):
    """Base for SLEEP/RESTART/SHUTDOWN — confirmation + deferred exec."""

    permission = PERMISSION_SYSTEM
    executable = True
    intent_name = ""
    action = ""            # "sleep" | "restart" | "shutdown"
    ask_message = ""
    decline_message = "Okay, I won't do that."

    def __init__(self, svc):
        self._svc = svc

    def _power(self):
        return getattr(self._svc, "power_ops", None)

    def validate(self, entities: dict) -> Optional[str]:
        return None

    def execute(self, entities: dict) -> CommandResult:
        power = self._power()
        if power is None:
            return _unavailable(self.intent_name, "the power service")
        # A live countdown already owns the machine's next power
        # transition — never stack a second request behind it.
        pending = power.pending_action()
        if pending is not None:
            return CommandResult(
                self.intent_name,
                False,
                message=f"A {pending} is already scheduled.",
                error="power_action_pending",
            )
        if not entities.get("_confirmed"):
            # Arm the confirmation: the stored intent carries this
            # marker so a "yes" can only ever confirm THIS action.
            entities["_armed_action"] = self.action
            logger.info(
                "Power action %s requested; awaiting confirmation",
                self.action,
            )
            return CommandResult(
                self.intent_name,
                True,
                message=self.ask_message,
                confirmation_required=True,
                data={"decline_message": self.decline_message},
            )
        # A confirmed dispatch must match the action that was armed —
        # a stale or mismatched confirmation never executes.
        if (
            entities.get("_confirmed") is not True
            or entities.get("_armed_action") != self.action
        ):
            logger.warning(
                "Rejected mismatched power confirmation for %s "
                "(armed=%r)",
                self.intent_name, entities.get("_armed_action"),
            )
            return CommandResult(
                self.intent_name,
                False,
                message="That confirmation doesn't match this request.",
                error="confirmation_mismatch",
            )
        return self._execute_confirmed(power, entities)

    def _execute_confirmed(self, power, entities: dict) -> CommandResult:
        raise NotImplementedError


class SleepComputerHandler(_PowerHandler):
    """SLEEP_COMPUTER — sleep after the acknowledgement finishes."""

    intent_name = "SLEEP_COMPUTER"
    action = "sleep"
    ask_message = (
        "Should I put your computer to sleep? Say yes to confirm."
    )
    decline_message = "Okay, I won't put it to sleep."

    def _execute_confirmed(self, power, entities: dict) -> CommandResult:
        pending = power.pending_action()
        if pending is not None:
            return CommandResult(
                self.intent_name,
                False,
                message=(
                    f"A {pending} is already scheduled. "
                    "Sleep would interrupt it."
                ),
                error="power_action_pending",
            )
        logger.info("Power action sleep confirmed")
        return CommandResult(
            self.intent_name,
            True,
            message="Putting your computer to sleep.",
            executed=True,
            data={"after_speech": lambda: self._sleep(power)},
        )

    @staticmethod
    def _sleep(power):
        """Deferred sleep; returns a follow-up message on failure."""
        if power.sleep():
            return None
        return "I couldn't put the computer to sleep."


class _ScheduledPowerHandler(_PowerHandler):
    """RESTART/SHUTDOWN — 15s cancellable countdown after the ack."""

    verb = ""           # "Restarting" / "Shutting down"
    noun = ""           # "restart" / "shutdown"

    def _execute_confirmed(self, power, entities: dict) -> CommandResult:
        pending = power.pending_action()
        if pending is not None:
            return CommandResult(
                self.intent_name,
                False,
                message=f"A {pending} is already scheduled.",
                error="power_action_pending",
            )
        logger.info("Power action %s confirmed", self.action)
        delay = SCHEDULE_DELAY_SECONDS
        return CommandResult(
            self.intent_name,
            True,
            message=(
                f"{self.verb} Windows in {delay} seconds. "
                f"Say 'cancel {self.noun}' to abort."
            ),
            executed=True,
            data={"after_speech": lambda: self._schedule(power)},
        )

    def _schedule(self, power):
        """Deferred scheduling; reports failure honestly via follow-up."""
        if power.schedule(self.action):
            return None
        return f"I couldn't schedule the {self.noun}."


class RestartComputerHandler(_ScheduledPowerHandler):
    intent_name = "RESTART_COMPUTER"
    action = "restart"
    verb = "Restarting"
    noun = "restart"
    ask_message = (
        "Should I restart your computer? Say yes to confirm."
    )
    decline_message = "Okay, I won't restart it."


class ShutdownComputerHandler(_ScheduledPowerHandler):
    intent_name = "SHUTDOWN_COMPUTER"
    action = "shutdown"
    verb = "Shutting down"
    noun = "shutdown"
    ask_message = (
        "Should I shut down your computer? Say yes to confirm."
    )
    decline_message = "Okay, I won't shut it down."


class CancelPowerActionHandler(_BaseHandler):
    """CANCEL_PENDING_POWER_ACTION — abort a live countdown only."""

    intent_name = "CANCEL_PENDING_POWER_ACTION"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc):
        self._svc = svc

    def validate(self, entities: dict) -> Optional[str]:
        return None

    def execute(self, entities: dict) -> CommandResult:
        power = getattr(self._svc, "power_ops", None)
        if power is None:
            return _unavailable(self.intent_name, "the power service")
        cancelled, action, error = power.cancel()
        if cancelled:
            logger.info("Pending power action %s cancelled by user",
                        action)
            return CommandResult(
                self.intent_name,
                True,
                message=f"Cancelled. The {action} has been aborted.",
                data={"cancelled": action},
                executed=True,
            )
        if error == "abort_failed":
            return CommandResult(
                self.intent_name,
                False,
                message=(
                    f"I couldn't cancel the {action}. "
                    "It may still proceed."
                ),
                error="abort_failed",
            )
        return CommandResult(
            self.intent_name,
            False,
            message="There's no power action scheduled to cancel.",
            error="nothing_pending",
        )


# ----------------------------------------------------------------------
# Desktop / window controls
# ----------------------------------------------------------------------
class _DesktopHandler(_BaseHandler):
    """Base for intents backed by svc.desktop (DesktopController)."""

    permission = PERMISSION_SYSTEM
    executable = True
    intent_name = ""

    def __init__(self, svc):
        self._svc = svc

    def _desktop(self):
        return getattr(self._svc, "desktop", None)

    def validate(self, entities: dict) -> Optional[str]:
        return None


class _ActiveWindowHandler(_DesktopHandler):
    """Minimize/restore the verified foreground window."""

    op = ""
    doing = ""
    done = ""

    def execute(self, entities: dict) -> CommandResult:
        desktop = self._desktop()
        if desktop is None:
            return _unavailable(self.intent_name, "the desktop service")
        ok, detail, title = getattr(desktop, self.op)()
        if not ok:
            if detail == "no_window":
                message = f"I couldn't find a window to {self.doing}."
            else:
                message = f"I couldn't {self.doing} the window."
            return CommandResult(
                self.intent_name, False, message=message,
                error=detail or "op_failed",
            )
        return CommandResult(
            self.intent_name,
            True,
            message=f"{self.done} {title}.",
            data={"window": title},
            executed=True,
        )


class MinimizeActiveWindowHandler(_ActiveWindowHandler):
    intent_name = "MINIMIZE_ACTIVE_WINDOW"
    op = "minimize_active"
    doing = "minimize"
    done = "Minimizing"


class RestoreActiveWindowHandler(_ActiveWindowHandler):
    intent_name = "RESTORE_ACTIVE_WINDOW"
    op = "restore_active"
    doing = "restore"
    done = "Restoring"


class CloseActiveWindowHandler(_DesktopHandler):
    """CLOSE_ACTIVE_WINDOW — WM_CLOSE the captured foreground hwnd.

    The hwnd is captured at request time and stored on the intent's
    entities, so the confirmed dispatch closes exactly the window the
    user saw — never whatever happens to be foreground later.
    """

    intent_name = "CLOSE_ACTIVE_WINDOW"

    def execute(self, entities: dict) -> CommandResult:
        desktop = self._desktop()
        if desktop is None:
            return _unavailable(self.intent_name, "the desktop service")
        if not entities.get("_confirmed"):
            w = desktop.active_window()
            if w is None or not getattr(w, "hwnd", 0):
                return CommandResult(
                    self.intent_name,
                    False,
                    message="I couldn't find a window to close.",
                    error="no_window",
                )
            entities["_hwnd"] = int(w.hwnd)
            entities["_window_title"] = getattr(w, "title", "") or ""
            label = entities["_window_title"] or "the active window"
            logger.info(
                "Close-active-window requested for hwnd %s (%r); "
                "awaiting confirmation",
                entities["_hwnd"], entities["_window_title"],
            )
            return CommandResult(
                self.intent_name,
                True,
                message=f"Should I close {label}? Say yes to confirm.",
                confirmation_required=True,
                data={"decline_message": "Okay, I won't close it."},
            )
        if entities.get("_confirmed") is not True or "_hwnd" not in entities:
            return CommandResult(
                self.intent_name,
                False,
                message="That confirmation doesn't match this request.",
                error="confirmation_mismatch",
            )
        hwnd = int(entities["_hwnd"])
        title = entities.get("_window_title") or "the window"
        ok, detail = desktop.close_window(hwnd)
        if not ok:
            if detail == "window_gone":
                message = f"{title} is no longer open."
            else:
                message = f"I couldn't close {title}."
            return CommandResult(
                self.intent_name, False, message=message,
                error=detail or "close_failed",
            )
        logger.info("Close-active-window executed for hwnd %s", hwnd)
        return CommandResult(
            self.intent_name,
            True,
            message=f"Closing {title}.",
            data={"hwnd": hwnd},
            executed=True,
        )


class _SwitchWindowHandler(_DesktopHandler):
    """SWITCH_NEXT_WINDOW / SWITCH_PREVIOUS_WINDOW (z-order)."""

    direction = ""
    doing = ""

    def execute(self, entities: dict) -> CommandResult:
        desktop = self._desktop()
        if desktop is None:
            return _unavailable(self.intent_name, "the desktop service")
        ok, detail, title = desktop.switch_window(self.direction)
        if not ok:
            if detail == "no_other_window":
                message = "There's no other window to switch to."
            elif detail == "no_active_window":
                message = "I couldn't identify the current window."
            else:
                message = "I couldn't switch windows."
            return CommandResult(
                self.intent_name, False, message=message,
                error=detail or "switch_failed",
            )
        return CommandResult(
            self.intent_name,
            True,
            message=f"Switching to {title}.",
            data={"window": title},
            executed=True,
        )


class SwitchNextWindowHandler(_SwitchWindowHandler):
    intent_name = "SWITCH_NEXT_WINDOW"
    direction = "next"


class SwitchPreviousWindowHandler(_SwitchWindowHandler):
    intent_name = "SWITCH_PREVIOUS_WINDOW"
    direction = "previous"


class _SwitchDesktopHandler(_DesktopHandler):
    """SWITCH_NEXT/PREVIOUS_VIRTUAL_DESKTOP (Win+Ctrl+Arrow)."""

    direction = ""
    doing = ""

    def execute(self, entities: dict) -> CommandResult:
        desktop = self._desktop()
        if desktop is None:
            return _unavailable(self.intent_name, "the desktop service")
        if not desktop.switch_desktop(self.direction):
            return CommandResult(
                self.intent_name,
                False,
                message=(
                    "I couldn't switch virtual desktops. "
                    "They may not be available on this system."
                ),
                error="desktop_switch_failed",
            )
        return CommandResult(
            self.intent_name,
            True,
            message=f"Switching to the {self.doing} desktop.",
            executed=True,
        )


class SwitchNextVirtualDesktopHandler(_SwitchDesktopHandler):
    intent_name = "SWITCH_NEXT_VIRTUAL_DESKTOP"
    direction = "next"
    doing = "next"


class SwitchPreviousVirtualDesktopHandler(_SwitchDesktopHandler):
    intent_name = "SWITCH_PREVIOUS_VIRTUAL_DESKTOP"
    direction = "previous"
    doing = "previous"


class _SnapWindowHandler(_DesktopHandler):
    """SNAP_ACTIVE_WINDOW_LEFT / SNAP_ACTIVE_WINDOW_RIGHT."""

    direction = ""
    doing = ""

    def execute(self, entities: dict) -> CommandResult:
        desktop = self._desktop()
        if desktop is None:
            return _unavailable(self.intent_name, "the desktop service")
        ok, detail, title = desktop.snap_active(self.direction)
        if not ok:
            if detail == "no_window":
                message = "I couldn't find a window to snap."
            else:
                message = f"I couldn't snap {title or 'the window'}."
            return CommandResult(
                self.intent_name, False, message=message,
                error=detail or "snap_failed",
            )
        return CommandResult(
            self.intent_name,
            True,
            message=f"Snapping {title} {self.doing}.",
            data={"window": title},
            executed=True,
        )


class SnapActiveWindowLeftHandler(_SnapWindowHandler):
    intent_name = "SNAP_ACTIVE_WINDOW_LEFT"
    direction = "left"
    doing = "left"


class SnapActiveWindowRightHandler(_SnapWindowHandler):
    intent_name = "SNAP_ACTIVE_WINDOW_RIGHT"
    direction = "right"
    doing = "right"


# ----------------------------------------------------------------------
def create_system_control_handlers(svc) -> dict:
    """All Milestone-1 system-control handlers keyed by intent name."""
    return {
        "SLEEP_COMPUTER": SleepComputerHandler(svc),
        "RESTART_COMPUTER": RestartComputerHandler(svc),
        "SHUTDOWN_COMPUTER": ShutdownComputerHandler(svc),
        "CANCEL_PENDING_POWER_ACTION": CancelPowerActionHandler(svc),
        "MINIMIZE_ACTIVE_WINDOW": MinimizeActiveWindowHandler(svc),
        "RESTORE_ACTIVE_WINDOW": RestoreActiveWindowHandler(svc),
        "CLOSE_ACTIVE_WINDOW": CloseActiveWindowHandler(svc),
        "SWITCH_NEXT_WINDOW": SwitchNextWindowHandler(svc),
        "SWITCH_PREVIOUS_WINDOW": SwitchPreviousWindowHandler(svc),
        "SWITCH_NEXT_VIRTUAL_DESKTOP": SwitchNextVirtualDesktopHandler(svc),
        "SWITCH_PREVIOUS_VIRTUAL_DESKTOP": (
            SwitchPreviousVirtualDesktopHandler(svc)
        ),
        "SNAP_ACTIVE_WINDOW_LEFT": SnapActiveWindowLeftHandler(svc),
        "SNAP_ACTIVE_WINDOW_RIGHT": SnapActiveWindowRightHandler(svc),
    }
