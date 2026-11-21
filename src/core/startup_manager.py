"""Windows startup integration for VoiceOS.

Registers VoiceOS in the current user's Startup folder by creating a
shortcut (``.lnk``) to the application launcher. This requires no
administrator privileges and no registry changes. A ``.cmd`` fallback
is written if shortcut creation fails.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LINK_NAME = "VoiceOS.lnk"
FALLBACK_NAME = "VoiceOS.cmd"


class StartupError(Exception):
    """Raised when startup registration cannot be completed."""


def _default_pythonw() -> Path:
    """Return the windowless Python executable for this environment."""
    exe = Path(sys.executable)
    pythonw = exe.with_name("pythonw.exe")
    return pythonw if pythonw.exists() else exe


class StartupManager:
    """Manage the per-user Windows Startup folder entry.

    Args:
        startup_dir: Startup folder path. Defaults to the current user's
            ``...\\Start Menu\\Programs\\Startup`` directory.
        target: Executable the shortcut points to. Defaults to
            ``pythonw.exe`` next to the running interpreter.
        arguments: Arguments passed to the target.
        working_dir: Working directory for the launched process.
        link_creator: Callable used to create the ``.lnk`` file.
            Signature: ``(link_path, target, arguments, working_dir)``.
            Injectable for tests.
    """

    def __init__(
        self,
        startup_dir: Optional[Path | str] = None,
        target: Optional[Path | str] = None,
        arguments: Optional[str] = None,
        working_dir: Optional[Path | str] = None,
        link_creator: Optional[Callable[[Path, str, str, str], None]] = None,
    ) -> None:
        self._startup_dir = (
            Path(startup_dir) if startup_dir else self._default_startup_dir()
        )
        self._target = str(target) if target else str(_default_pythonw())
        self._arguments = (
            arguments if arguments is not None else f'"{PROJECT_ROOT / "run.py"}"'
        )
        self._working_dir = str(working_dir) if working_dir else str(PROJECT_ROOT)
        self._link_creator = link_creator or self._create_shortcut_powershell

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    @staticmethod
    def _default_startup_dir() -> Path:
        """Return the per-user Startup folder for the current Windows user."""
        appdata = os.environ.get("APPDATA")
        if not appdata:
            raise StartupError(
                "APPDATA environment variable is not set; "
                "cannot locate the Startup folder."
            )
        return (
            Path(appdata)
            / "Microsoft"
            / "Windows"
            / "Start Menu"
            / "Programs"
            / "Startup"
        )

    @property
    def shortcut_path(self) -> Path:
        """Path of the VoiceOS shortcut in the Startup folder."""
        return self._startup_dir / LINK_NAME

    @property
    def fallback_path(self) -> Path:
        """Path of the ``.cmd`` fallback launcher."""
        return self._startup_dir / FALLBACK_NAME

    @property
    def target(self) -> str:
        """Executable the startup entry launches."""
        return self._target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def is_registered(self) -> bool:
        """Return whether VoiceOS is currently registered for startup."""
        return self.shortcut_path.exists() or self.fallback_path.exists()

    def register(self) -> bool:
        """Register VoiceOS to start with Windows.

        Returns:
            True if a new entry was created, False if VoiceOS was
            already registered (no duplicate is created).

        Raises:
            StartupError: If the Startup folder is inaccessible and no
                entry could be created.
        """
        if self.is_registered():
            logger.info("VoiceOS is already registered for startup; skipping")
            return False

        try:
            self._startup_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StartupError(
                f"Cannot access Startup folder {self._startup_dir}: {exc}"
            ) from exc

        try:
            self._link_creator(
                self.shortcut_path, self._target, self._arguments, self._working_dir
            )
            logger.info(
                "Startup shortcut created: %s -> %s %s",
                self.shortcut_path,
                self._target,
                self._arguments,
            )
            return True
        except Exception as exc:  # noqa: BLE001 - try batch fallback
            logger.warning(
                "Shortcut creation failed (%s); writing .cmd fallback", exc
            )
            try:
                self._write_cmd_fallback()
            except OSError as fallback_exc:
                raise StartupError(
                    f"Startup registration failed: {exc}; "
                    f"fallback also failed: {fallback_exc}"
                ) from fallback_exc
            logger.info("Startup fallback created: %s", self.fallback_path)
            return True

    def unregister(self) -> bool:
        """Remove the VoiceOS startup entry.

        Returns:
            True if an entry was removed, False if none existed.
        """
        removed = False
        for path in (self.shortcut_path, self.fallback_path):
            try:
                if path.exists():
                    path.unlink()
                    removed = True
                    logger.info("Removed startup entry: %s", path)
            except OSError as exc:
                raise StartupError(f"Cannot remove startup entry {path}: {exc}") from exc
        if not removed:
            logger.info("VoiceOS was not registered for startup")
        return removed

    def set_enabled(self, enabled: bool) -> bool:
        """Enable or disable startup registration.

        Returns:
            True if the registration state changed.
        """
        if enabled:
            return self.register()
        return self.unregister()

    # ------------------------------------------------------------------
    # Shortcut creation
    # ------------------------------------------------------------------
    @staticmethod
    def _ps_quote(value: str) -> str:
        """Quote a string for use inside a PowerShell single-quoted literal."""
        return "'" + value.replace("'", "''") + "'"

    @classmethod
    def _create_shortcut_powershell(
        cls, link_path: Path, target: str, arguments: str, working_dir: str
    ) -> None:
        """Create a ``.lnk`` via the WScript.Shell PowerShell COM object."""
        script = (
            "$ws = New-Object -ComObject WScript.Shell; "
            f"$s = $ws.CreateShortcut({cls._ps_quote(str(link_path))}); "
            f"$s.TargetPath = {cls._ps_quote(target)}; "
            f"$s.Arguments = {cls._ps_quote(arguments)}; "
            f"$s.WorkingDirectory = {cls._ps_quote(working_dir)}; "
            "$s.Save()"
        )
        result = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise StartupError(
                f"PowerShell shortcut creation failed: {result.stderr.strip()}"
            )

    def _write_cmd_fallback(self) -> None:
        """Write a ``.cmd`` launcher as a fallback startup entry."""
        content = (
            "@echo off\r\n"
            f'start "" "{self._target}" {self._arguments}\r\n'
        )
        self.fallback_path.write_text(content, encoding="ascii")
