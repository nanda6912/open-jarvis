"""Normalized running-application identity for VoiceOS.

Every window/process lookup in the app/window commands funnels through
:func:`identity_for_app` so discovery, launching, enumeration, window
matching, switching, focusing, minimizing, maximizing, moving and
closing all agree on what "the application" is.

An :class:`AppIdentity` carries the canonical id, display name,
executable path, candidate process names, package family name (PFN)
and AUMID. Package-identifier fragments (the publisher hash inside a
PFN, e.g. ``5319275a``) are NEVER used as user-facing names.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from src.controllers.application_discovery import AppInfo, _normalize

# Process names that host packaged-app windows without being the app.
_FRAMEWORK_STEMS = {"applicationframehost", "shellexperiencehost",
                    "startmenuexperiencehost", "searchhost", "sihost"}

# PFN/AppId fragments too generic to identify a process.
_GENERIC_TOKENS = {"app", "application", "exe", "windows", "microsoft"}


@dataclass(frozen=True)
class AppIdentity:
    """Canonical identity of an installed or running application."""

    app_id: str            # "exe:<stem>" or "aumid:<aumid>"
    name: str              # display name — never a package fragment
    exe_path: str
    process_names: tuple   # candidate exe stems (lowercase, no .exe)
    package_family: str    # PFN (lowercase) or ""
    aumid: str


def _squash(text: str) -> str:
    """Lowercase alphanumeric-only form ("Visual Studio Code" ->
    "visualstudiocode")."""
    return re.sub(r"[^a-z0-9]", "", _normalize(text))


def _pfn_product(pfn: str) -> str:
    """Product token inside a package family name.

    ``5319275A.WhatsAppDesktop_cv1g1gvanyjgm`` -> ``whatsappdesktop``.
    """
    tail = pfn.split(".", 1)[1] if "." in pfn else pfn
    return tail.split("_", 1)[0].lower()


def identity_for_app(app: AppInfo) -> AppIdentity:
    """Build the normalized identity for a discovered ``AppInfo``."""
    if getattr(app, "launch_kind", "exe") == "aumid":
        aumid = getattr(app, "aumid", "") or ""
        pfn = aumid.split("!", 1)[0] if "!" in aumid else ""
        candidates = set()
        product = _pfn_product(pfn)
        if product and product not in _GENERIC_TOKENS:
            candidates.add(product)
        squashed = _squash(app.name)
        if squashed and squashed not in _GENERIC_TOKENS:
            candidates.add(squashed)
        first = (_normalize(app.name).split() or [""])[0]
        if first and first not in _GENERIC_TOKENS:
            candidates.add(first)
        return AppIdentity(
            app_id=f"aumid:{aumid.lower()}",
            name=app.name,
            exe_path=app.exe_path,
            process_names=tuple(sorted(candidates)),
            package_family=pfn.lower(),
            aumid=aumid,
        )
    stem = Path(app.exe_path).stem.lower() if app.exe_path else ""
    candidates = {stem} if stem else set()
    squashed = _squash(app.name)
    if squashed and squashed not in _GENERIC_TOKENS:
        candidates.add(squashed)
    return AppIdentity(
        app_id=f"exe:{stem}",
        name=app.name,
        exe_path=app.exe_path,
        process_names=tuple(sorted(c for c in candidates if c)),
        package_family="",
        aumid="",
    )


def _window_stems(window) -> set:
    """Candidate process stems for ``window``.

    ``WhatsApp.Root.exe`` yields {"whatsapp.root", "whatsapp", "root",
    "whatsapproot"} so dotted exe names still match app identities.
    """
    proc = (getattr(window, "process", "") or "").lower()
    if not proc:
        return set()
    stem = proc.rsplit(".", 1)[0] if "." in proc else proc
    stems = {stem, stem.replace(".", "")}
    stems.update(s for s in stem.split(".") if s)
    return {s for s in stems if s}


def _title_matches_app(title: str, app: AppInfo) -> bool:
    """Conservative title match used only for framework-hosted windows.

    Every token of the app's display name must appear in the window
    title (``"WhatsApp"`` matches title ``"WhatsApp"``; a Chrome window
    titled ``"WhatsApp Web"`` also matches, which is acceptable — the
    window genuinely belongs to that app's UI in practice only via the
    framework-host check below).
    """
    name_norm = _normalize(app.name)
    if not name_norm:
        return False
    title_norm = _normalize(title)
    if not title_norm:
        return False
    if name_norm in title_norm:
        return True
    name_tokens = set(name_norm.split())
    title_tokens = set(title_norm.split())
    return bool(name_tokens) and name_tokens <= title_tokens


def window_matches_identity(window, app: AppInfo, ident: AppIdentity) -> bool:
    """``window_matches_app`` with a precomputed identity."""
    stems = _window_stems(window)
    if stems and stems & set(ident.process_names):
        return True
    pfn = (getattr(window, "package_family", "") or "").lower()
    if ident.package_family and pfn and pfn == ident.package_family:
        return True
    exe = (getattr(window, "exe_path", "") or "").lower()
    app_exe = (app.exe_path or "").lower()
    if exe and app_exe and exe == app_exe:
        return True
    if stems & _FRAMEWORK_STEMS and _title_matches_app(
        getattr(window, "title", "") or "", app
    ):
        return True
    return False


def window_matches_app(window, app: AppInfo) -> bool:
    """True when ``window`` belongs to ``app`` under the identity rules.

    Match order: owning-process name -> package family name -> exe
    path -> framework-host title fallback (ApplicationFrameHost and
    friends, which host packaged-app top-level windows).
    """
    return window_matches_identity(window, app, identity_for_app(app))


def identities_for(apps: Iterable[AppInfo]) -> list:
    """Precompute ``(app, AppIdentity)`` pairs for bulk matching."""
    return [(a, identity_for_app(a)) for a in apps]


def app_for_window(window, apps, identities=None) -> Optional[AppInfo]:
    """Best installed-app match for a running window (reverse lookup)."""
    pairs = identities if identities is not None else [
        (a, identity_for_app(a)) for a in apps
    ]
    for app, ident in pairs:
        if window_matches_identity(window, app, ident):
            return app
    return None


def windows_for_app(windows: Iterable, app: AppInfo) -> list:
    """Filter ``windows`` down to those belonging to ``app``."""
    ident = identity_for_app(app)
    return [w for w in windows if window_matches_identity(w, app, ident)]


def window_label(window, app: Optional[AppInfo] = None) -> str:
    """User-facing label for a window — never a package fragment."""
    if app is not None and app.name:
        return app.name
    stems = _window_stems(window)
    if stems & _FRAMEWORK_STEMS or not stems:
        title = (getattr(window, "title", "") or "").strip()
        return title or "application"
    proc = (getattr(window, "process", "") or "").lower()
    stem = proc.rsplit(".", 1)[0] if "." in proc else proc
    return stem or (getattr(window, "title", "") or "").strip() or "window"
