"""Battery / power status via Win32 GetSystemPowerStatus.

Read-only. Reports only what Windows actually provides: AC line state,
battery presence, charge level, charge state and — only when Windows
supplies a sane estimate — remaining battery time. Unknown values are
reported as unknown, never invented.
"""

from __future__ import annotations

import ctypes
import logging
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# SYSTEM_POWER_STATUS (winbase.h)
AC_OFFLINE = 0
AC_ONLINE = 1
AC_UNKNOWN = 255

FLAG_HIGH = 0x1
FLAG_LOW = 0x2
FLAG_CRITICAL = 0x4
FLAG_CHARGING = 0x8
FLAG_NO_BATTERY = 0x80
FLAG_UNKNOWN = 0xFF

LIFE_UNKNOWN = 0xFFFFFFFF
# Estimates beyond a week are treated as garbage, not information.
_LIFE_MAX_SECONDS = 7 * 24 * 3600


class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", wintypes.BYTE),
        ("BatteryFlag", wintypes.BYTE),
        ("BatteryLifePercent", wintypes.BYTE),
        ("SystemStatusFlag", wintypes.BYTE),
        ("BatteryLifeTime", wintypes.DWORD),
        ("BatteryFullLifeTime", wintypes.DWORD),
    ]


@dataclass(frozen=True)
class PowerStatus:
    """Normalized snapshot of SYSTEM_POWER_STATUS."""

    ac_line: Optional[bool]       # True=on AC, False=on battery, None=unknown
    has_battery: Optional[bool]   # None when Windows can't tell
    percent: Optional[int]        # 0-100, None when unknown
    charging: Optional[bool]
    discharging: Optional[bool]
    life_seconds: Optional[int]   # remaining estimate, None if invalid
    full_life_seconds: Optional[int]

    @property
    def on_battery(self) -> Optional[bool]:
        if self.ac_line is None:
            return None
        return not self.ac_line


def _valid_life(value: int) -> Optional[int]:
    if value in (0, LIFE_UNKNOWN) or value > _LIFE_MAX_SECONDS:
        return None
    return int(value)


class PowerStatusController:
    """Query GetSystemPowerStatus; injectable for tests."""

    def __init__(self, status_fn=None) -> None:
        # status_fn() -> _SYSTEM_POWER_STATUS-like object or None.
        self._fn = status_fn or self._query

    def status(self) -> Optional[PowerStatus]:
        """Return a PowerStatus, or None when the API is unavailable."""
        try:
            raw = self._fn()
        except Exception:  # noqa: BLE001
            logger.exception("GetSystemPowerStatus failed")
            return None
        if raw is None:
            return None
        return self._normalize(raw)

    @staticmethod
    def _normalize(raw) -> PowerStatus:
        ac = int(raw.ACLineStatus)
        flag = int(raw.BatteryFlag)
        pct = int(raw.BatteryLifePercent)

        ac_line = (
            True if ac == AC_ONLINE
            else False if ac == AC_OFFLINE
            else None
        )
        if flag == FLAG_UNKNOWN:
            has_battery = None
            charging = None
        elif flag & FLAG_NO_BATTERY:
            has_battery = False
            charging = False
        else:
            has_battery = True
            charging = bool(flag & FLAG_CHARGING)
        percent = None if pct == 255 else max(0, min(int(pct), 100))

        discharging = (
            True if (ac_line is False and has_battery and not charging)
            else False if (ac_line is True or has_battery is False)
            else None
        )
        return PowerStatus(
            ac_line=ac_line,
            has_battery=has_battery,
            percent=percent,
            charging=charging,
            discharging=discharging,
            life_seconds=_valid_life(int(raw.BatteryLifeTime)),
            full_life_seconds=_valid_life(int(raw.BatteryFullLifeTime)),
        )

    @staticmethod
    def _query():
        if not hasattr(ctypes, "windll"):
            return None
        sps = _SYSTEM_POWER_STATUS()
        ok = ctypes.windll.kernel32.GetSystemPowerStatus(
            ctypes.byref(sps)
        )
        return sps if ok else None
