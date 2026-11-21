"""Single-instance protection for VoiceOS.

On Windows a named mutex is used: the operating system releases it
automatically when the owning process exits, so stale locks cannot
survive an unexpected termination. On other platforms a PID-based lock
file with stale detection is used as a fallback.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

from src.core.configuration import get_app_data_dir

logger = logging.getLogger(__name__)

DEFAULT_MUTEX_NAME = r"Local\VoiceOS.SingleInstance.v1"
ERROR_ALREADY_EXISTS = 183


def _pid_alive(pid: int) -> bool:
    """Return whether a process with ``pid`` appears to be running."""
    if pid <= 0:
        return False
    # Use os.name (not sys.platform) so tests can exercise the file-lock
    # fallback on Windows without hitting the unsafe os.kill(pid, 0)
    # code path, which would terminate the process on Windows.
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return False
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class InstanceGuard:
    """Ensure only one VoiceOS instance runs at a time.

    Args:
        name: Mutex name (Windows) used to detect a running instance.
        lock_path: Optional explicit lock-file path for the non-Windows
            fallback. Defaults to ``voiceos.instance.lock`` in the
            application data directory.
    """

    def __init__(
        self,
        name: str = DEFAULT_MUTEX_NAME,
        lock_path: Optional[Path] = None,
    ) -> None:
        self._name = name
        self._lock_path = (
            Path(lock_path) if lock_path else get_app_data_dir() / "voiceos.instance.lock"
        )
        self._mutex_handle = None
        self._file_lock_held = False
        self.acquired = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def acquire(self) -> bool:
        """Try to become the single running instance.

        Returns:
            True if the lock was acquired, False if another instance
            already holds it.
        """
        if sys.platform == "win32":
            return self._acquire_mutex()
        return self._acquire_file_lock()

    def release(self) -> None:
        """Release the instance lock. Safe to call when not acquired."""
        if self._mutex_handle is not None:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.ReleaseMutex(self._mutex_handle)
            kernel32.CloseHandle(self._mutex_handle)
            self._mutex_handle = None
        if self._file_lock_held:
            try:
                self._lock_path.unlink(missing_ok=True)
            except OSError:
                pass
            self._file_lock_held = False
        self.acquired = False
        logger.debug("Instance lock released")

    # ------------------------------------------------------------------
    # Windows: named mutex
    # ------------------------------------------------------------------
    def _acquire_mutex(self) -> bool:
        import ctypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel32.CreateMutexW(None, False, self._name)
        if not handle:
            logger.error("CreateMutexW failed (error %s)", ctypes.get_last_error())
            return False
        if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
            kernel32.CloseHandle(handle)
            logger.info("Another VoiceOS instance is already running")
            return False
        self._mutex_handle = handle
        self.acquired = True
        logger.info("Single-instance lock acquired (%s)", self._name)
        return True

    # ------------------------------------------------------------------
    # Fallback: PID lock file with stale detection
    # ------------------------------------------------------------------
    def _acquire_file_lock(self) -> bool:
        for _ in range(2):
            try:
                fd = os.open(
                    self._lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
            except FileExistsError:
                if self._lock_is_stale():
                    try:
                        self._lock_path.unlink()
                    except OSError:
                        return False
                    continue
                logger.info("Another VoiceOS instance is already running")
                return False
            except OSError as exc:
                logger.error("Cannot create instance lock %s: %s", self._lock_path, exc)
                return False

            try:
                os.write(fd, str(os.getpid()).encode("ascii"))
            finally:
                os.close(fd)
            self._file_lock_held = True
            self.acquired = True
            logger.info("Single-instance lock acquired (%s)", self._lock_path)
            return True
        return False

    def _lock_is_stale(self) -> bool:
        """Return True if the lock file belongs to a dead process."""
        try:
            pid = int(self._lock_path.read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            # Unreadable/corrupt lock file: treat as stale.
            return True
        return not _pid_alive(pid)

    def __enter__(self) -> "InstanceGuard":
        if not self.acquire():
            raise RuntimeError("Another VoiceOS instance is already running")
        return self

    def __exit__(self, *exc_info) -> None:
        self.release()
