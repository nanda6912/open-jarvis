"""Allowlisted Windows known-folder resolution and opening.

Only the six reviewed known folders can ever be opened, and only after
their real Windows path (``SHGetKnownFolderPath``) has been resolved
and verified to exist on disk. Opening goes through ``os.startfile``
(ShellExecute on a directory) — spoken text is never used as a command
and no executable path is ever constructed from it.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Folder name -> (FOLDERID GUID for SHGetKnownFolderPath, fallback
# directory name under the user profile).
_KNOWN_FOLDERS = {
    "desktop": (
        "{754AC886-DF64-4CBA-86B5-F7FBF4FBCEF5}", "Desktop",
    ),
    "documents": (
        "{FDD39AD0-238F-46AF-ADB4-6C85480369C7}", "Documents",
    ),
    "downloads": (
        "{374DE290-123F-4565-9164-39C4925E467B}", "Downloads",
    ),
    "pictures": (
        "{33E28130-4E1E-4676-835A-98395C3BC3BB}", "Pictures",
    ),
    "photos": (
        "{33E28130-4E1E-4676-835A-98395C3BC3BB}", "Pictures",
    ),
    "music": (
        "{4BD8D571-6D19-48D3-BE99-4250C8E79C15}", "Music",
    ),
    "videos": (
        "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}", "Videos",
    ),
    "movies": (
        "{18989B1D-99B5-455B-841C-AB7C74E4DDFC}", "Videos",
    ),
}

# Spoken aliases that map onto a canonical allowlisted folder.
_ALIASES = {
    "photo": "pictures",
    "photos": "pictures",
    "movie": "videos",
    "movies": "videos",
}


def _guid_bytes(guid_text: str) -> bytes:
    """Convert '{XXXXXXXX-...}' text to the 16-byte GUID layout."""
    import uuid

    u = uuid.UUID(guid_text)
    # GUID struct: DWORD, WORD, WORD, 8 bytes — little-endian fields.
    return u.bytes_le


def _known_folder_path(guid_text: str) -> Optional[str]:
    """Call SHGetKnownFolderPath; return path or None."""
    import ctypes
    from ctypes import wintypes

    if not hasattr(ctypes, "windll"):
        return None
    try:
        shell32 = ctypes.windll.shell32
        ole32 = ctypes.windll.ole32
    except (AttributeError, OSError):
        return None

    class GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", wintypes.DWORD),
            ("Data2", wintypes.WORD),
            ("Data3", wintypes.WORD),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    raw = _guid_bytes(guid_text)
    guid = GUID()
    guid.Data1 = int.from_bytes(raw[0:4], "little")
    guid.Data2 = int.from_bytes(raw[4:6], "little")
    guid.Data3 = int.from_bytes(raw[6:8], "little")
    for i in range(8):
        guid.Data4[i] = raw[8 + i]

    ptr = ctypes.c_wchar_p()
    KF_FLAG_DEFAULT = 0
    hr = shell32.SHGetKnownFolderPath(
        ctypes.byref(guid), KF_FLAG_DEFAULT, None, ctypes.byref(ptr)
    )
    if hr != 0 or not ptr.value:
        ole32.CoTaskMemFree(ptr)
        return None
    path = ptr.value
    ole32.CoTaskMemFree(ptr)
    return path or None


def _fallback_paths(folder_name: str, env: dict) -> list[Path]:
    """Candidate profile-relative locations when the API fails."""
    info = _KNOWN_FOLDERS.get(folder_name)
    if info is None:
        return []
    _guid, dirname = info
    roots = []
    profile = env.get("USERPROFILE") or env.get("HOME") or ""
    if profile:
        roots.append(Path(profile))
        onedrive = env.get("OneDrive") or env.get("OneDriveConsumer")
        if onedrive:
            roots.append(Path(onedrive))
    out = []
    for root in roots:
        out.append(root / dirname)
        if dirname in ("Desktop", "Documents"):
            out.append(root / "OneDrive" / dirname)
    return out


class KnownFolderResolver:
    """Resolve allowlisted folder names to verified local paths."""

    def __init__(
        self,
        env: Optional[dict] = None,
        query_known: Optional[Callable[[str], Optional[str]]] = None,
        is_dir: Optional[Callable[[str], bool]] = None,
    ) -> None:
        self._env = env if env is not None else dict(os.environ)
        self._query_known = query_known or _known_folder_path
        self._is_dir = is_dir or os.path.isdir

    @staticmethod
    def normalize(name: str) -> str:
        n = (name or "").strip().lower()
        return _ALIASES.get(n, n)

    def resolve(self, folder_name: str) -> Optional[Path]:
        """Return the verified folder path, or None when unavailable."""
        name = self.normalize(folder_name)
        info = _KNOWN_FOLDERS.get(name)
        if info is None:
            return None
        guid_text, _dirname = info
        try:
            api_path = self._query_known(guid_text)
        except Exception:  # noqa: BLE001
            logger.debug("SHGetKnownFolderPath failed", exc_info=True)
            api_path = None
        candidates = []
        if api_path:
            candidates.append(Path(api_path))
        candidates.extend(_fallback_paths(name, self._env))
        for cand in candidates:
            try:
                if self._is_dir(str(cand)):
                    return cand
            except Exception:  # noqa: BLE001
                continue
        return None


class FolderOpener:
    """Open a validated known folder. Returns (ok, detail)."""

    def __init__(
        self,
        resolver: Optional[KnownFolderResolver] = None,
        open_fn: Optional[Callable[[str], object]] = None,
    ) -> None:
        self._resolver = resolver or KnownFolderResolver()
        # os.startfile uses ShellExecute on the directory — it never
        # runs a shell and never receives spoken text.
        self._open = open_fn or getattr(os, "startfile", None)

    def open(self, folder_name: str):
        name = KnownFolderResolver.normalize(folder_name)
        if name not in _KNOWN_FOLDERS:
            return False, "unsupported_folder"
        path = self._resolver.resolve(name)
        if path is None:
            logger.info("Known folder %r did not resolve", folder_name)
            return False, "folder_not_found"
        if self._open is None:
            return False, "unsupported_platform"
        try:
            self._open(str(path))
        except FileNotFoundError:
            return False, "folder_not_found"
        except PermissionError:
            return False, "permission_denied"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to open folder %s", path)
            return False, str(exc)
        logger.info("Opened folder %s", path)
        return True, "ok"
