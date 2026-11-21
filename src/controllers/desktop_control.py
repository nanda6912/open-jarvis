"""Desktop/window controls: active-window ops, z-order switching,
virtual-desktop switching and Aero-snap.

Every operation verifies its target first: window actions only ever
touch the verified foreground ``hwnd`` (or an hwnd captured earlier and
re-validated), and keyboard chords are fixed Win32 combos — never text
derived from speech. All Win32/keyboard access is injectable so tests
never touch real windows.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from src.controllers.windows_automation import WindowManager

logger = logging.getLogger(__name__)

# Virtual-key codes for the fixed chords Jarvis sends.
_VK = {
    "lwin": 0x5B,
    "rwin": 0x5C,
    "ctrl": 0x11,
    "left": 0x25,
    "right": 0x27,
    "up": 0x26,
    "down": 0x28,
}
_KEYUP = 0x0002


def _default_send_combo(combo: str) -> bool:
    """Send a fixed key chord (e.g. ``"win+ctrl+right"``) via keybd_event.

    Returns False on non-Windows hosts or when the API is missing.
    """
    import ctypes

    if not hasattr(ctypes, "windll"):
        return False
    try:
        user32 = ctypes.windll.user32
        vks = [_VK[part] for part in combo.split("+")]
    except (KeyError, AttributeError):
        logger.warning("Unknown key combo %r", combo)
        return False
    try:
        for vk in vks:
            user32.keybd_event(vk, 0, 0, None)
        for vk in reversed(vks):
            user32.keybd_event(vk, 0, _KEYUP, None)
        return True
    except Exception:  # noqa: BLE001
        logger.exception("keybd_event chord %r failed", combo)
        return False


class DesktopController:
    """Verified desktop/window operations on top of WindowManager."""

    def __init__(
        self,
        windows: Optional[WindowManager] = None,
        send_combo: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self._windows = windows if windows is not None else WindowManager()
        self._send_combo = send_combo or _default_send_combo

    # -- helpers ---------------------------------------------------------
    def active_window(self):
        """WindowInfo for the foreground window, or None."""
        try:
            return self._windows.foreground_window()
        except Exception:  # noqa: BLE001
            logger.exception("Foreground window lookup failed")
            return None

    def _active_or_fail(self):
        w = self.active_window()
        if w is None or not getattr(w, "hwnd", 0):
            return None
        return w

    # -- show desktop ----------------------------------------------------
    def show_desktop(self) -> bool:
        try:
            self._windows.show_desktop()
            return True
        except Exception:  # noqa: BLE001
            logger.exception("Show desktop failed")
            return False

    # -- active-window operations ----------------------------------------
    def minimize_active(self):
        """Minimize the verified foreground window."""
        w = self._active_or_fail()
        if w is None:
            return False, "no_window", ""
        try:
            self._windows.minimize(w.hwnd)
        except Exception:  # noqa: BLE001
            logger.exception("Minimize failed for hwnd %s", w.hwnd)
            return False, "op_failed", getattr(w, "title", "")
        return True, "ok", getattr(w, "title", "") or "the window"

    def restore_active(self):
        """Restore the verified foreground window (SW_RESTORE)."""
        w = self._active_or_fail()
        if w is None:
            return False, "no_window", ""
        try:
            self._windows.restore(w.hwnd)
        except Exception:  # noqa: BLE001
            logger.exception("Restore failed for hwnd %s", w.hwnd)
            return False, "op_failed", getattr(w, "title", "")
        return True, "ok", getattr(w, "title", "") or "the window"

    def close_window(self, hwnd: int):
        """WM_CLOSE a previously captured hwnd — only if it is still a
        live top-level window. Never closes by title/process guess."""
        try:
            wins = self._windows.list_windows()
        except Exception:  # noqa: BLE001
            logger.exception("Window enumeration failed for close")
            return False, "enum_failed"
        if not any(getattr(w, "hwnd", None) == hwnd for w in wins):
            return False, "window_gone"
        try:
            self._windows.close_window(hwnd)
        except Exception:  # noqa: BLE001
            logger.exception("WM_CLOSE failed for hwnd %s", hwnd)
            return False, "close_failed"
        return True, "ok"

    # -- z-order window switching -----------------------------------------
    def switch_window(self, direction: str):
        """Focus the next/previous window in z-order (wraps around).

        ``list_windows`` enumerates top-level windows in z-order, so the
        neighbour of the foreground window is the Alt+Tab neighbour.
        """
        direction = (direction or "").lower()
        if direction not in ("next", "previous"):
            return False, "bad_direction", ""
        try:
            wins = self._windows.list_windows()
        except Exception:  # noqa: BLE001
            logger.exception("Window enumeration failed for switch")
            return False, "enum_failed", ""
        if len(wins) < 2:
            return False, "no_other_window", ""
        fg_win = self.active_window()
        fg = getattr(fg_win, "hwnd", 0) if fg_win is not None else 0
        idx = next(
            (i for i, w in enumerate(wins) if w.hwnd == fg), None
        )
        if idx is None:
            return False, "no_active_window", ""
        step = 1 if direction == "next" else -1
        target = wins[(idx + step) % len(wins)]
        try:
            ok = bool(self._windows.focus(target.hwnd))
        except Exception:  # noqa: BLE001
            logger.exception("Focus failed for hwnd %s", target.hwnd)
            return False, "focus_failed", getattr(target, "title", "")
        if not ok:
            return False, "focus_failed", getattr(target, "title", "")
        return True, "ok", getattr(target, "title", "") or "the window"

    # -- virtual desktops --------------------------------------------------
    def switch_desktop(self, direction: str) -> bool:
        """Win+Ctrl+Left/Right — the shell handles virtual desktops."""
        direction = (direction or "").lower()
        arrow = {"next": "right", "previous": "left"}.get(direction)
        if arrow is None:
            return False
        try:
            return bool(self._send_combo(f"win+ctrl+{arrow}"))
        except Exception:  # noqa: BLE001
            logger.exception("Virtual desktop chord failed")
            return False

    # -- Aero snap ----------------------------------------------------------
    def snap_active(self, direction: str):
        """Win+Left/Right snap on the verified foreground window."""
        direction = (direction or "").lower()
        if direction not in ("left", "right"):
            return False, "bad_direction", ""
        w = self._active_or_fail()
        if w is None:
            return False, "no_window", ""
        try:
            ok = bool(self._send_combo(f"win+{direction}"))
        except Exception:  # noqa: BLE001
            logger.exception("Snap chord failed")
            return False, "combo_failed", getattr(w, "title", "")
        if not ok:
            return False, "combo_failed", getattr(w, "title", "")
        return True, "ok", getattr(w, "title", "") or "the window"
