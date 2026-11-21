"""Boot timing diagnostics for VoiceOS startup.

Writes timestamped markers to ``boot_diag.log`` inside the runtime log
directory (``%LOCALAPPDATA%\\VoiceOS\\logs`` by default, overridable via
``VOICEOS_LOG_DIR``) so the delay between Windows process creation and
each startup stage can be measured. This module is intentionally
dependency-free and safe to import before logging or Qt are initialized.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FALLBACK_DIAG_LOG = PROJECT_ROOT / "logs" / "boot_diag.log"

_DIAG_LOG: Path | None = None


def _diag_log_path() -> Path:
    """Resolve the diagnostics log path once, with a safe fallback."""
    global _DIAG_LOG
    if _DIAG_LOG is None:
        try:
            from src.core.paths import logs_dir

            _DIAG_LOG = logs_dir() / "boot_diag.log"
        except Exception:  # noqa: BLE001 - diagnostics must never break startup
            _DIAG_LOG = FALLBACK_DIAG_LOG
    return _DIAG_LOG

_SCRIPT_START = time.perf_counter()
_PROCESS_CREATION_EPOCH: float | None = None


def _process_creation_time() -> float | None:
    """Return this process's creation time as a Unix epoch float.

    Uses ``GetProcessTimes`` on Windows; returns ``None`` elsewhere.
    """
    global _PROCESS_CREATION_EPOCH
    if _PROCESS_CREATION_EPOCH is not None:
        return _PROCESS_CREATION_EPOCH
    if os.name != "nt":
        return None
    try:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessTimes.argtypes = (
            [ctypes.c_void_p]
            + [ctypes.POINTER(ctypes.c_ulonglong)] * 4
        )
        creation, exit_t, kernel_t, user_t = (ctypes.c_ulonglong() for _ in range(4))
        ok = kernel32.GetProcessTimes(
            kernel32.GetCurrentProcess(),
            ctypes.byref(creation),
            ctypes.byref(exit_t),
            ctypes.byref(kernel_t),
            ctypes.byref(user_t),
        )
        if ok:
            # FILETIME is 100-ns intervals since 1601-01-01.
            _PROCESS_CREATION_EPOCH = creation.value / 10_000_000 - 116_444_73600
    except Exception:  # noqa: BLE001 - diagnostics must never break startup
        _PROCESS_CREATION_EPOCH = None
    return _PROCESS_CREATION_EPOCH


def mark(label: str) -> None:
    """Append a timestamped startup marker to the diagnostics log.

    Args:
        label: Short stage description, e.g. ``"QApplication created"``.
    """
    try:
        diag_log = _diag_log_path()
        diag_log.parent.mkdir(parents=True, exist_ok=True)
        elapsed = time.perf_counter() - _SCRIPT_START
        created = _process_creation_time()
        if created is not None:
            # Wall-clock delay between OS process creation and this marker.
            delta = time.time() - created
            line = (
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | script+{elapsed:8.3f}s "
                f"| proc+{delta:8.3f}s | {label}\n"
            )
        else:
            line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | script+{elapsed:8.3f}s | {label}\n"
        with diag_log.open("a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception:  # noqa: BLE001 - diagnostics must never break startup
        pass
