"""Read-only system probes for the holographic HUD panels.

Everything here is best-effort and defensive: a probe failure returns a
placeholder string instead of raising, so the HUD can never take down
the voice pipeline. Services are injected so tests can run off-Windows.
"""

from __future__ import annotations

import ctypes
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class DesktopProbe:
    """Collects DESKTOP panel data: foreground app, window count,
    monitor layout and AppState. All Win32 access goes through the
    injected ``window_manager`` / ``monitor_manager`` (or fakes)."""

    def __init__(self, window_manager=None, monitor_manager=None,
                 state_manager=None) -> None:
        self._windows = window_manager
        self._monitors = monitor_manager
        self._state = state_manager

    def snapshot(self) -> dict:
        """Return a dict of display strings; never raises."""
        out = {
            "active_app": "—",
            "window_count": "—",
            "monitors": "—",
            "system_state": "—",
        }
        try:
            if self._state is not None:
                out["system_state"] = self._state.state.name
        except Exception:  # noqa: BLE001
            pass
        try:
            if self._windows is not None:
                fg = self._windows.foreground_window()
                if fg is not None:
                    name = getattr(fg, "process", "") or ""
                    title = (getattr(fg, "title", "") or "").strip()
                    label = name or title or "unknown"
                    out["active_app"] = (
                        f"{label} — {title[:40]}" if title and name
                        else label
                    )[:60]
                wins = self._windows.list_windows()
                out["window_count"] = str(len(wins))
        except Exception:  # noqa: BLE001
            logger.debug("Desktop window probe failed", exc_info=True)
        try:
            if self._monitors is not None:
                mons = self._monitors.monitors()
                if mons:
                    parts = []
                    for m in mons:
                        pos = "primary" if m.primary else "secondary"
                        parts.append(f"{m.index}:{pos}")
                    out["monitors"] = " · ".join(parts)
        except Exception:  # noqa: BLE001
            logger.debug("Monitor probe failed", exc_info=True)
        return out


class SystemProbe:
    """CPU% and RAM% via Win32 API (no extra dependencies)."""

    def __init__(self) -> None:
        self._last_cpu = None  # (idle, kernel, user)
        self._last_cpu_pct = 0.0
        self._last_ram = (0.0, 0.0)

    def _cpu_times(self):
        if not hasattr(ctypes, "windll"):
            return None
        class FILETIME(ctypes.Structure):
            _fields_ = [("dwLowDateTime", ctypes.c_uint32),
                        ("dwHighDateTime", ctypes.c_uint32)]

        idle = FILETIME()
        kernel = FILETIME()
        user = FILETIME()
        ok = ctypes.windll.kernel32.GetSystemTimes(
            ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
        )
        if not ok:
            return None

        def _ft(ft):
            return (ft.dwHighDateTime << 32) | ft.dwLowDateTime

        return _ft(idle), _ft(kernel), _ft(user)

    def _ram(self):
        if not hasattr(ctypes, "windll"):
            return None

        class MEMSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_uint32),
                ("dwMemoryLoad", ctypes.c_uint32),
                ("ullTotalPhys", ctypes.c_uint64),
                ("ullAvailPhys", ctypes.c_uint64),
                ("ullTotalPageFile", ctypes.c_uint64),
                ("ullAvailPageFile", ctypes.c_uint64),
                ("ullTotalVirtual", ctypes.c_uint64),
                ("ullAvailVirtual", ctypes.c_uint64),
                ("ullAvailExtendedVirtual", ctypes.c_uint64),
            ]

        st = MEMSTATUSEX()
        st.dwLength = ctypes.sizeof(MEMSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return None
        return st.ullTotalPhys, st.ullAvailPhys

    def snapshot(self) -> dict:
        """Return {'cpu': '42%', 'ram': '61%'} (best-effort)."""
        cpu_txt, ram_txt = "—", "—"
        try:
            times = self._cpu_times()
            if times is not None:
                idle, kernel, user = times
                total = kernel + user
                if self._last_cpu is not None:
                    d_total = total - self._last_cpu[1] - self._last_cpu[2]
                    d_idle = idle - self._last_cpu[0]
                    if d_total > 0:
                        self._last_cpu_pct = max(
                            0.0, min(100.0,
                                     100.0 * (1.0 - d_idle / d_total))
                        )
                self._last_cpu = times
                cpu_txt = f"{self._last_cpu_pct:.0f}%"
        except Exception:  # noqa: BLE001
            logger.debug("CPU probe failed", exc_info=True)
        try:
            ram = self._ram()
            if ram is not None:
                total, avail = ram
                self._last_ram = (total, avail)
                if total:
                    ram_txt = f"{100.0 * (1.0 - avail / total):.0f}%"
        except Exception:  # noqa: BLE001
            logger.debug("RAM probe failed", exc_info=True)
        return {"cpu": cpu_txt, "ram": ram_txt}


def normalize_level(raw: float, ceiling: float = 0.25) -> float:
    """Map a raw RMS value to a 0..1 display level with soft knee."""
    if raw <= 0.0:
        return 0.0
    x = min(1.0, raw / max(1e-6, ceiling))
    # Soft curve: quiet speech still visible, loud speech saturates.
    return min(1.0, x * (0.6 + 0.4 * x))
