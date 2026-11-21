"""Dynamic multi-monitor discovery and resolution for VoiceOS.

``MonitorManager`` enumerates the displays Windows currently reports via
``EnumDisplayMonitors``/``GetMonitorInfoW`` and exposes stable,
descriptive information plus natural-language alias resolution
("left monitor", "monitor 2", "primary", "other", ...).

All Win32 access is isolated in :class:`_MonitorApi` (or an injected
fake) so the manager is fully testable off-Windows. Monitor handles are
never cached across calls: ``monitors()`` re-enumerates by default so
display changes (connect/disconnect/rearrange) are always reflected.
"""

from __future__ import annotations

import ctypes
import logging
import re
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)

MONITORINFOF_PRIMARY = 0x00000001


class MonitorError(Exception):
    """Raised for monitor enumeration/resolution failures."""


@dataclass(frozen=True)
class MonitorRect:
    """Axis-aligned rectangle in virtual screen coordinates."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def center_x(self) -> int:
        return (self.left + self.right) // 2

    @property
    def center_y(self) -> int:
        return (self.top + self.bottom) // 2


@dataclass(frozen=True)
class MonitorInfo:
    """A single connected display in the current configuration."""

    index: int               # 1-based logical number (1 = primary)
    device: str              # e.g. "\\\\.\\DISPLAY1"
    bounds: MonitorRect      # full monitor bounds
    work_area: MonitorRect   # bounds minus taskbar
    primary: bool
    handle: int = 0          # HMONITOR (opaque, session-scoped)

    @property
    def width(self) -> int:
        return self.bounds.width

    @property
    def height(self) -> int:
        return self.bounds.height

    @property
    def center_x(self) -> int:
        return self.bounds.center_x

    @property
    def center_y(self) -> int:
        return self.bounds.center_y

    def position_label(self) -> str:
        """Short spatial description, e.g. 'left', 'right', 'primary'."""
        if self.primary:
            return "primary"
        return "secondary"


# ----------------------------------------------------------------------
# Win32 monitor API (lazy — only usable on Windows)
# ----------------------------------------------------------------------
class _MonitorApi:
    """Thin ctypes wrapper over monitor-related user32 functions."""

    def __init__(self) -> None:
        windll = getattr(ctypes, "windll", None)
        if windll is None:  # pragma: no cover - non-Windows guard
            raise MonitorError("Win32 monitor API unavailable on this OS")
        self._u = windll.user32

        class MONITORINFOEXW(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.DWORD),
                ("rcMonitor", wintypes.RECT),
                ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD),
                ("szDevice", wintypes.WCHAR * 32),
            ]

        self._info_ex = MONITORINFOEXW
        monitor_cb = ctypes.WINFUNCTYPE(
            wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC,
            ctypes.POINTER(wintypes.RECT), wintypes.LPARAM,
        )
        self._monitor_cb = monitor_cb
        self._u.EnumDisplayMonitors.argtypes = [
            wintypes.HDC, ctypes.c_void_p, monitor_cb, wintypes.LPARAM
        ]
        self._u.GetMonitorInfoW.argtypes = [
            wintypes.HMONITOR, ctypes.c_void_p
        ]
        self._u.MonitorFromWindow.argtypes = [
            wintypes.HWND, wintypes.DWORD
        ]
        self._u.MonitorFromWindow.restype = wintypes.HMONITOR
        self._u.MonitorFromPoint.argtypes = [
            wintypes.POINT, wintypes.DWORD
        ]
        self._u.MonitorFromPoint.restype = wintypes.HMONITOR

    # -- raw calls ---------------------------------------------------------
    def enum_monitors(self) -> list[int]:
        """Return all HMONITOR handles in enumeration order."""
        handles: list[int] = []

        @self._monitor_cb
        def _cb(hmon, _hdc, _rect, _lp):
            handles.append(int(hmon))
            return True

        self._u.EnumDisplayMonitors(None, None, _cb, 0)
        return handles

    def monitor_info(self, hmon: int):
        """Return (MonitorRect bounds, MonitorRect work, primary, device)."""
        info = self._info_ex()
        info.cbSize = ctypes.sizeof(self._info_ex)
        ok = self._u.GetMonitorInfoW(hmon, ctypes.byref(info))
        if not ok:
            raise MonitorError(f"GetMonitorInfoW failed for {hmon}")
        bounds = MonitorRect(
            info.rcMonitor.left, info.rcMonitor.top,
            info.rcMonitor.right, info.rcMonitor.bottom,
        )
        work = MonitorRect(
            info.rcWork.left, info.rcWork.top,
            info.rcWork.right, info.rcWork.bottom,
        )
        primary = bool(info.dwFlags & MONITORINFOF_PRIMARY)
        return bounds, work, primary, info.szDevice

    def monitor_for_hwnd(self, hwnd: int) -> int:
        """HMONITOR containing ``hwnd`` (MONITOR_DEFAULTTONEAREST)."""
        return int(self._u.MonitorFromWindow(hwnd, 2) or 0)

    def monitor_for_point(self, x: int, y: int) -> int:
        """HMONITOR containing virtual-screen point (x, y)."""
        pt = wintypes.POINT(x, y)
        return int(self._u.MonitorFromPoint(pt, 2) or 0)


# ----------------------------------------------------------------------
# Resolution result
# ----------------------------------------------------------------------
@dataclass
class MonitorResolveResult:
    status: str  # "found" | "missing" | "ambiguous"
    monitor: Optional[MonitorInfo] = None
    options: tuple = ()


# ----------------------------------------------------------------------
# Manager
# ----------------------------------------------------------------------
class MonitorManager:
    """Discover connected displays and resolve spoken monitor aliases.

    Args:
        api: Win32 monitor API (injectable for tests).
        device_namer: Optional ``(device_name) -> friendly name`` hook;
            defaults to the raw device string.
    """

    def __init__(
        self,
        api: Optional[_MonitorApi] = None,
        device_namer: Optional[Callable[[str], str]] = None,
    ) -> None:
        self._api = api if api is not None else _MonitorApi()
        self._namer = device_namer or (lambda d: d)
        self._cache: list[MonitorInfo] = []

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def monitors(self, refresh: bool = True) -> list[MonitorInfo]:
        """Return all connected monitors in logical order.

        Logical order: Monitor 1 is the Windows primary display; the
        rest follow left->right, top->bottom. Re-enumerates on every
        call by default so hot-plug, resolution and arrangement
        changes are never served stale.
        """
        if not refresh and self._cache:
            return list(self._cache)
        try:
            handles = self._api.enum_monitors()
        except MonitorError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise MonitorError(f"Monitor enumeration failed: {exc}") from exc

        infos: list[MonitorInfo] = []
        for hmon in handles:
            try:
                bounds, work, primary, device = self._api.monitor_info(hmon)
            except Exception:  # noqa: BLE001
                logger.debug("monitor_info failed for %s", hmon,
                             exc_info=True)
                continue
            infos.append(MonitorInfo(
                index=0, device=device, bounds=bounds,
                work_area=work, primary=primary, handle=hmon,
            ))
        # Logical numbering (user convention, NOT Windows display ids
        # and NOT left-to-right position):
        #   Monitor 1 = the Windows PRIMARY display, always.
        #   Monitor 2..N = the remaining displays sorted left->right,
        #   then top->bottom (deterministic across enumeration order).
        # Spatial aliases ("left monitor", "right monitor") are still
        # resolved geometrically and are independent of this numbering.
        def _pos(m: MonitorInfo):
            return (m.bounds.left, m.bounds.top)

        by_pos = sorted(infos, key=_pos)
        primary = next((m for m in by_pos if m.primary), None)
        if primary is not None:
            ordered = [primary] + [m for m in by_pos if m is not primary]
        else:
            # No display flagged primary: keep pure position order so
            # numbering stays deterministic; primary() reports index 1.
            ordered = by_pos
        infos = [
            MonitorInfo(
                index=i, device=m.device, bounds=m.bounds,
                work_area=m.work_area, primary=m.primary, handle=m.handle,
            )
            for i, m in enumerate(ordered, start=1)
        ]
        self._cache = infos
        return list(infos)

    def count(self) -> int:
        return len(self.monitors())

    def primary(self) -> Optional[MonitorInfo]:
        for m in self.monitors():
            if m.primary:
                return m
        return self.monitors()[0] if self.monitors() else None

    def get(self, index: int) -> Optional[MonitorInfo]:
        """Return the monitor with 1-based logical ``index``."""
        for m in self.monitors():
            if m.index == index:
                return m
        return None

    def refresh(self) -> list[MonitorInfo]:
        return self.monitors(refresh=True)

    # ------------------------------------------------------------------
    # Spatial helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _leftmost(monitors: list[MonitorInfo]) -> MonitorInfo:
        return min(monitors, key=lambda m: (m.center_x, m.bounds.left))

    @staticmethod
    def _rightmost(monitors: list[MonitorInfo]) -> MonitorInfo:
        return max(monitors, key=lambda m: (m.center_x, m.bounds.right))

    @staticmethod
    def _topmost(monitors: list[MonitorInfo]) -> MonitorInfo:
        return min(monitors, key=lambda m: (m.center_y, m.bounds.top))

    @staticmethod
    def _bottommost(monitors: list[MonitorInfo]) -> MonitorInfo:
        return max(monitors, key=lambda m: (m.center_y, m.bounds.bottom))

    def describe(self, monitor: MonitorInfo) -> str:
        """Human-friendly label, e.g. 'monitor 2 (left, 1920x1080)'."""
        pos = ""
        all_m = self.monitors()
        if len(all_m) > 1:
            if monitor == self._leftmost(all_m) and monitor != self._rightmost(all_m):
                pos = "left, "
            elif monitor == self._rightmost(all_m) and monitor != self._leftmost(all_m):
                pos = "right, "
        return (
            f"monitor {monitor.index} ({pos}{monitor.width}x{monitor.height}"
            f"{', primary' if monitor.primary else ''})"
        )

    def spoken_label(self, monitor: MonitorInfo) -> str:
        """Short spoken reference, e.g. 'the left monitor'."""
        all_m = self.monitors()
        if len(all_m) > 1:
            if (monitor == self._leftmost(all_m)
                    and monitor != self._rightmost(all_m)):
                return "the left monitor"
            if (monitor == self._rightmost(all_m)
                    and monitor != self._leftmost(all_m)):
                return "the right monitor"
            if (monitor == self._topmost(all_m)
                    and monitor != self._bottommost(all_m)):
                return "the top monitor"
            if (monitor == self._bottommost(all_m)
                    and monitor != self._topmost(all_m)):
                return "the bottom monitor"
            if monitor.primary:
                return "the primary monitor"
        elif monitor.primary:
            return "the primary monitor"
        return f"monitor {monitor.index}"

    def summary(self) -> str:
        """One-line spoken summary of the current display setup."""
        mons = self.monitors()
        if not mons:
            return "I couldn't detect any monitors."
        if len(mons) == 1:
            m = mons[0]
            return (
                f"One monitor detected: {self.describe(m)}."
            )
        parts = []
        for m in mons:
            bits = []
            if m.primary:
                bits.append("primary")
            if len(mons) > 1:
                if m == self._leftmost(mons) and m != self._rightmost(mons):
                    bits.append("left")
                elif m == self._rightmost(mons) and m != self._leftmost(mons):
                    bits.append("right")
                elif m == self._topmost(mons) and m != self._bottommost(mons):
                    bits.append("top")
                elif m == self._bottommost(mons) and m != self._topmost(mons):
                    bits.append("bottom")
            parts.append(f"{self.describe(m)} ({', '.join(bits) or 'center'})")
        return f"{len(mons)} monitors detected: " + "; ".join(parts) + "."

    # ------------------------------------------------------------------
    # Alias resolution
    # ------------------------------------------------------------------
    _ORDINALS = {
        "first": 1, "1": 1, "one": 1,
        "second": 2, "2": 2, "two": 2, "too": 2, "to": 2,
        "third": 3, "3": 3, "three": 3, "tree": 3,
        "fourth": 4, "4": 4, "four": 4,
    }

    @staticmethod
    def _normalize_query(query: str) -> str:
        q = (query or "").lower().strip()
        q = re.sub(r"[^a-z0-9]+", " ", q)
        q = " ".join(q.split())
        # Drop generic display nouns and filler.
        q = re.sub(
            r"\b(monitors?|screens?|displays?|display|monitor|screen|"
            r"the|my|your|a|an|to|on|please)\b", " ", q
        )
        return " ".join(q.split())

    def resolve(
        self,
        query: str,
        relative_to: Optional[MonitorInfo] = None,
    ) -> MonitorResolveResult:
        """Resolve a spoken monitor phrase to a MonitorInfo.

        ``relative_to`` is the monitor the phrase is relative to (used
        for "other monitor"); when omitted the primary monitor is used.
        """
        mons = self.monitors()
        if not mons:
            return MonitorResolveResult("missing")

        q = self._normalize_query(query)
        if not q:
            # Bare "monitor"/"screen" with no qualifier is ambiguous.
            if len(mons) == 1:
                return MonitorResolveResult("found", mons[0])
            return MonitorResolveResult("ambiguous", options=tuple(mons))

        primary = self.primary()
        tokens = q.split()

        # --- explicit device name: "display 2", "\\\\.\\display2" --------
        m = re.search(r"\bdisplay\s*(\d+)\b", q)
        if m:
            idx = int(m.group(1))
            mon = self.get(idx)
            return (
                MonitorResolveResult("found", mon) if mon
                else MonitorResolveResult("missing")
            )

        # --- numbered / ordinal ------------------------------------------
        if len(tokens) == 1 and tokens[0] in self._ORDINALS:
            idx = self._ORDINALS[tokens[0]]
            mon = self.get(idx)
            return (
                MonitorResolveResult("found", mon) if mon
                else MonitorResolveResult("missing")
            )
        # "monitor number two" style already collapsed by normalize.

        # --- role aliases -------------------------------------------------
        if "primary" in tokens or "main" in tokens:
            return MonitorResolveResult("found", primary)
        if "secondary" in tokens or "second screen" in q:
            non_primary = [m for m in mons if not m.primary]
            if len(non_primary) == 1:
                return MonitorResolveResult("found", non_primary[0])
            if non_primary:
                return MonitorResolveResult(
                    "found", sorted(non_primary, key=lambda m: m.index)[0]
                )
            return MonitorResolveResult("missing")

        # --- spatial aliases ----------------------------------------------
        if "left" in tokens:
            return MonitorResolveResult("found", self._leftmost(mons))
        if "right" in tokens:
            return MonitorResolveResult("found", self._rightmost(mons))
        if "top" in tokens or "upper" in tokens:
            return MonitorResolveResult("found", self._topmost(mons))
        if "bottom" in tokens or "lower" in tokens:
            return MonitorResolveResult("found", self._bottommost(mons))

        # --- device-kind aliases ------------------------------------------
        # Windows does not expose panel type, so "laptop screen" cannot
        # be identified reliably — report missing so the caller asks
        # the user instead of guessing (primary != laptop guaranteed).
        if "laptop" in tokens or "notebook" in tokens or "built in" in q:
            return MonitorResolveResult("missing")
        if "external" in tokens or "second" in tokens and "monitor" in q:
            non_primary = [m for m in mons if not m.primary]
            if non_primary:
                return MonitorResolveResult(
                    "found", sorted(non_primary, key=lambda m: m.index)[0]
                )
            return MonitorResolveResult("missing")

        # --- relative -----------------------------------------------------
        if "other" in tokens or "another" in tokens:
            ref = relative_to or primary
            others = [m for m in mons if ref is None or m.index != ref.index]
            if not others:
                return MonitorResolveResult("missing")
            if len(others) == 1:
                return MonitorResolveResult("found", others[0])
            return MonitorResolveResult(
                "ambiguous", options=tuple(others)
            )

        # --- device-name match ("DISPLAY2", "\\\\.\\DISPLAY2") -------------
        dev_hit = [
            m for m in mons
            if q.replace(" ", "") in m.device.lower().replace(" ", "")
        ]
        if len(dev_hit) == 1:
            return MonitorResolveResult("found", dev_hit[0])

        return MonitorResolveResult("missing")

    # ------------------------------------------------------------------
    # Window -> monitor mapping
    # ------------------------------------------------------------------
    def monitor_for_hwnd(self, hwnd: int) -> Optional[MonitorInfo]:
        """Return the monitor currently containing ``hwnd``."""
        try:
            hmon = self._api.monitor_for_hwnd(hwnd)
        except Exception:  # noqa: BLE001
            logger.debug("monitor_for_hwnd failed", exc_info=True)
            return None
        if not hmon:
            return None
        for m in self.monitors():
            if m.handle == hmon:
                return m
        return None

    def monitor_for_point(self, x: int, y: int) -> Optional[MonitorInfo]:
        """Return the monitor containing virtual point (x, y)."""
        try:
            hmon = self._api.monitor_for_point(x, y)
        except Exception:  # noqa: BLE001
            logger.debug("monitor_for_point failed", exc_info=True)
            return None
        for m in self.monitors():
            if m.handle == hmon:
                return m
        return None
