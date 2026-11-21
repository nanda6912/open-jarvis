"""Windows toast/balloon notifications for VoiceOS.

Uses ``Shell_NotifyIconW`` with ``NIF_INFO`` on a private message-only
window — the classic balloon mechanism that Windows 10/11 renders as a
toast notification. No external packages, no network, no registration
APIs. Everything is rate-limited and validated; failures degrade to a
``False`` return so callers can stay silent or fall back.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from ctypes import wintypes
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Limits: balloon fields are fixed-size in NOTIFYICONDATAW.
MAX_TITLE = 63
MAX_MESSAGE = 255
MAX_TOTAL = 320

# Rate limiting keeps Jarvis from spamming the notification center.
DEFAULT_MIN_INTERVAL = 2.0      # seconds between notifications
DEFAULT_MAX_PER_MINUTE = 10

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitize_text(text: str, limit: int) -> str:
    """Strip control chars and clamp to ``limit`` chars."""
    t = _CONTROL_RE.sub(" ", text or "")
    t = " ".join(t.split())
    return t[:limit].strip()


class _BalloonApi:
    """Shell_NotifyIcon balloon delivery via a hidden message window."""

    NIM_ADD = 0
    NIM_DELETE = 2
    NIF_MESSAGE = 0x1
    NIF_ICON = 0x2
    NIF_TIP = 0x4
    NIF_INFO = 0x10
    NIIF_INFO = 0x1
    IDI_APPLICATION = 32512
    HWND_MESSAGE = -3

    def __init__(self) -> None:
        if not hasattr(__import__("ctypes"), "windll"):
            raise RuntimeError("Win32 unavailable")
        import ctypes

        self._ct = ctypes
        self._u = ctypes.windll.user32
        self._s = ctypes.windll.shell32
        self._hwnd = 0

        class NOTIFYICONDATAW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("hWnd", wintypes.HWND),
                ("uID", wintypes.UINT),
                ("uFlags", wintypes.UINT),
                ("uCallbackMessage", wintypes.UINT),
                ("hIcon", wintypes.HICON),
                ("szTip", wintypes.WCHAR * 128),
                ("dwState", wintypes.DWORD),
                ("dwStateMask", wintypes.DWORD),
                ("szInfo", wintypes.WCHAR * 256),
                ("uTimeoutOrVersion", wintypes.UINT),
                ("szInfoTitle", wintypes.WCHAR * 64),
                ("dwInfoFlags", wintypes.DWORD),
                ("guidItem", ctypes.c_ubyte * 16),
                ("hBalloonIcon", wintypes.HICON),
            ]

        self._nid_cls = NOTIFYICONDATAW
        self._u.CreateWindowExW.restype = wintypes.HWND
        self._u.LoadIconW.restype = wintypes.HICON
        # Second parameter is LPCWSTR but callers pass MAKEINTRESOURCE
        # ids (small ints) — accept a raw pointer-sized value.
        self._u.LoadIconW.argtypes = [
            wintypes.HINSTANCE, ctypes.c_void_p
        ]
        self._s.Shell_NotifyIconW.argtypes = [
            wintypes.DWORD, ctypes.POINTER(NOTIFYICONDATAW)
        ]
        self._s.Shell_NotifyIconW.restype = wintypes.BOOL

    def _ensure_window(self):
        if self._hwnd:
            return self._hwnd
        u = self._u
        hwnd = u.CreateWindowExW(
            0, "STATIC", "VoiceOSNotify", 0,
            0, 0, 0, 0,
            wintypes.HWND(self.HWND_MESSAGE), None, None, None,
        )
        if not hwnd:
            raise OSError("CreateWindowExW failed")
        self._hwnd = int(hwnd)
        return self._hwnd

    def _nid(self, flags: int, title: str = "", info: str = "") -> object:
        nid = self._nid_cls()
        nid.cbSize = self._ct.sizeof(self._nid_cls)
        nid.hWnd = self._hwnd
        nid.uID = 1
        nid.uFlags = flags
        if flags & self.NIF_ICON:
            nid.hIcon = self._u.LoadIconW(None, self.IDI_APPLICATION)
        if flags & self.NIF_TIP:
            nid.szTip = "VoiceOS"
        if flags & self.NIF_INFO:
            nid.szInfo = info[:255]
            nid.szInfoTitle = title[:63]
            nid.dwInfoFlags = self.NIIF_INFO
        return nid

    def show(self, title: str, message: str) -> bool:
        """Post a balloon/toast; schedules icon cleanup."""
        try:
            hwnd = self._ensure_window()
        except Exception:  # noqa: BLE001
            logger.exception("Notify window creation failed")
            return False
        nid = self._nid(
            self.NIF_MESSAGE | self.NIF_ICON | self.NIF_TIP | self.NIF_INFO,
            title=title, info=message,
        )
        try:
            ok = bool(self._s.Shell_NotifyIconW(self.NIM_ADD, nid))
        except Exception:  # noqa: BLE001
            logger.exception("Shell_NotifyIconW(NIM_ADD) failed")
            return False
        if ok:
            # Remove the icon shortly after; the toast itself persists.
            def _cleanup():
                try:
                    nid2 = self._nid_cls()
                    nid2.cbSize = self._ct.sizeof(self._nid_cls)
                    nid2.hWnd = hwnd
                    nid2.uID = 1
                    self._s.Shell_NotifyIconW(self.NIM_DELETE, nid2)
                except Exception:  # noqa: BLE001
                    pass

            threading.Timer(8.0, _cleanup).start()
        return ok


class NotificationController:
    """Validated, rate-limited notification entry point.

    Args:
        api: delivery backend (``.show(title, message) -> bool``);
            defaults to the Shell_NotifyIcon balloon API.
        enabled: master switch (config/setting can disable toasts).
        min_interval / max_per_minute: rate limits.
        clock: injectable monotonic clock for tests.
    """

    def __init__(
        self,
        api=None,
        enabled: bool = True,
        min_interval: float = DEFAULT_MIN_INTERVAL,
        max_per_minute: int = DEFAULT_MAX_PER_MINUTE,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        self._api = api
        self._api_failed = False
        self.enabled = enabled
        self._min_interval = min_interval
        self._max_per_minute = max_per_minute
        self._clock = clock or time.monotonic
        self._last_at = 0.0
        self._sent: list = []

    def _get_api(self):
        if self._api_failed:
            return None
        if self._api is None:
            try:
                self._api = _BalloonApi()
            except Exception:  # noqa: BLE001
                logger.debug("Balloon API unavailable", exc_info=True)
                self._api_failed = True
                return None
        return self._api

    def _throttled(self) -> bool:
        now = self._clock()
        self._sent = [t for t in self._sent if now - t < 60.0]
        if len(self._sent) >= self._max_per_minute:
            return True
        if self._last_at and now - self._last_at < self._min_interval:
            return True
        return False

    def notify(self, title: str, message: str) -> bool:
        """Deliver a notification. Returns True only on delivery."""
        if not self.enabled:
            logger.debug("Notifications disabled")
            return False
        title = sanitize_text(title, MAX_TITLE)
        message = sanitize_text(message, MAX_MESSAGE)
        if not title or not message:
            logger.debug("Notification rejected: empty text")
            return False
        if self._throttled():
            logger.debug("Notification throttled")
            return False
        api = self._get_api()
        if api is None:
            return False
        try:
            ok = bool(api.show(title, message))
        except Exception:  # noqa: BLE001
            logger.exception("Notification delivery failed")
            return False
        if ok:
            now = self._clock()
            self._last_at = now
            self._sent.append(now)
        return ok
