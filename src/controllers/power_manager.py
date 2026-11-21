"""Windows power management: sleep, scheduled shutdown/restart, cancel.

Safety model:

- ``sleep()`` invokes the Windows power API immediately (no countdown —
  sleep is trivially reversible with a key press).
- ``schedule("shutdown"|"restart")`` arms a fixed-argument
  ``shutdown.exe`` countdown (``/t 15``) — never a shell, never text
  derived from speech.
- ``cancel()`` only acts while an operation is genuinely pending and
  unexpired; it runs ``shutdown.exe /a``.

All OS interaction goes through injectable callables so tests never
sleep, restart or shut down the development machine. Every requested,
confirmed, scheduled, cancelled and executed transition is logged.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Fixed 15-second cancellation window for restart/shutdown. The
# countdown begins only when ``schedule()`` is invoked — the command
# layer defers that call until the confirmation acknowledgement has
# finished playing.
SCHEDULE_DELAY_SECONDS = 15

_SCHEDULE_FLAGS = {"shutdown": "/s", "restart": "/r"}
_SCHEDULE_COMMENTS = {
    "shutdown": "Jarvis scheduled shutdown - say 'cancel shutdown' to abort.",
    "restart": "Jarvis scheduled restart - say 'cancel restart' to abort.",
}
_SCHEDULE_TIMEOUT = 15.0


def _default_sleep() -> bool:
    """Sleep via the Win32 power API (powrprof.SetSuspendState).

    Returns False only when the API is unavailable or the call raises —
    a completed call is treated as sleep initiated (the API's own
    return value is logged for diagnostics).
    """
    import ctypes

    if not hasattr(ctypes, "windll"):
        logger.warning("SetSuspendState unavailable: not Windows")
        return False
    try:
        rc = ctypes.windll.powrprof.SetSuspendState(0, 0, 0)
        logger.info("SetSuspendState returned %r", rc)
        return True
    except Exception:  # noqa: BLE001
        logger.exception("SetSuspendState failed")
        return False


def _run_shutdown(args: list) -> bool:
    """Run ``shutdown.exe`` with a fixed argument list (no shell)."""
    try:
        proc = subprocess.run(
            ["shutdown.exe", *args],
            timeout=_SCHEDULE_TIMEOUT,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ok = proc.returncode == 0
        if not ok:
            logger.warning(
                "shutdown.exe %s exited with %s",
                args, proc.returncode,
            )
        return ok
    except Exception:  # noqa: BLE001
        logger.exception("shutdown.exe %s failed", args)
        return False


def _default_schedule(action: str) -> bool:
    """Arm the OS-level countdown: shutdown.exe /s|/r /t 15 /c <fixed>."""
    flag = _SCHEDULE_FLAGS.get(action)
    if flag is None:
        logger.warning("Refusing to schedule unknown power action %r",
                       action)
        return False
    return _run_shutdown([
        flag,
        "/t", str(SCHEDULE_DELAY_SECONDS),
        "/c", _SCHEDULE_COMMENTS[action],
    ])


def _default_abort() -> bool:
    """Abort a pending shutdown.exe countdown (shutdown.exe /a)."""
    return _run_shutdown(["/a"])


class PowerManager:
    """Tracks pending scheduled power actions and performs them safely.

    ``schedule()`` records ``{"action", "deadline"}`` only after the OS
    scheduling call succeeds, so a failed ``shutdown.exe`` never leaves
    phantom pending state. ``pending_action()`` lazily expires records
    whose countdown has elapsed.
    """

    def __init__(
        self,
        sleep_fn: Optional[Callable[[], bool]] = None,
        schedule_fn: Optional[Callable[[str], bool]] = None,
        abort_fn: Optional[Callable[[], bool]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._sleep_fn = sleep_fn or _default_sleep
        self._schedule_fn = schedule_fn or _default_schedule
        self._abort_fn = abort_fn or _default_abort
        self._clock = clock or time.monotonic
        self._pending: Optional[dict] = None
        self._lock = threading.Lock()

    # -- queries ---------------------------------------------------------
    def pending_action(self) -> Optional[str]:
        """The pending power action, or None when absent/expired."""
        with self._lock:
            p = self._pending
            if p is None:
                return None
            if self._clock() >= p["deadline"]:
                logger.info(
                    "Pending power action %r expired",
                    p.get("action"),
                )
                self._pending = None
                return None
            return p["action"]

    def pending_deadline(self) -> Optional[float]:
        with self._lock:
            p = self._pending
            if p is None or self._clock() >= p["deadline"]:
                return None
            return p["deadline"]

    # -- operations ------------------------------------------------------
    def sleep(self) -> bool:
        """Put the computer to sleep via the Windows power API."""
        logger.info("Power action requested: sleep")
        try:
            ok = bool(self._sleep_fn())
        except Exception:  # noqa: BLE001
            logger.exception("Sleep request failed")
            ok = False
        logger.info("Power action sleep %s", "executed" if ok else "failed")
        return ok

    def schedule(self, action: str) -> bool:
        """Arm a 15-second shutdown/restart countdown.

        Returns False when the action is unknown, another power action
        is already pending, or the OS scheduling call fails. Pending
        state is recorded only on success.
        """
        if action not in _SCHEDULE_FLAGS:
            logger.warning("Refusing unknown schedule action %r", action)
            return False
        with self._lock:
            p = self._pending
            if p is not None and self._clock() < p["deadline"]:
                logger.info(
                    "Schedule %r refused: power action already pending",
                    action,
                )
                return False
        logger.info("Power action requested: schedule %s", action)
        try:
            ok = bool(self._schedule_fn(action))
        except Exception:  # noqa: BLE001
            logger.exception("Schedule %s failed", action)
            ok = False
        if ok:
            with self._lock:
                self._pending = {
                    "action": action,
                    "deadline": self._clock() + SCHEDULE_DELAY_SECONDS,
                }
            logger.info(
                "Power action %s scheduled (%.0fs countdown)",
                action, SCHEDULE_DELAY_SECONDS,
            )
        else:
            logger.warning("Power action %s schedule failed", action)
        return ok

    def cancel(self):
        """Abort the pending countdown.

        Returns ``(cancelled, action, error)``: ``cancelled`` is True only
        when an unexpired pending action existed AND the OS abort
        succeeded. ``error`` is ``"nothing_pending"`` or
        ``"abort_failed"`` when not cancelled.
        """
        with self._lock:
            p = self._pending
            if p is None or self._clock() >= p["deadline"]:
                self._pending = None
                return False, None, "nothing_pending"
            action = p["action"]
        logger.info("Power action cancel requested for %s", action)
        try:
            ok = bool(self._abort_fn())
        except Exception:  # noqa: BLE001
            logger.exception("Power abort failed for %s", action)
            ok = False
        if ok:
            with self._lock:
                # Only clear if the same action is still pending — a
                # newer schedule must never be cancelled by a stale
                # abort result.
                if self._pending is p or (
                    self._pending is not None
                    and self._pending.get("action") == action
                ):
                    self._pending = None
            logger.info("Power action %s cancelled", action)
            return True, action, None
        logger.warning("Power abort failed for pending %s", action)
        return False, action, "abort_failed"
