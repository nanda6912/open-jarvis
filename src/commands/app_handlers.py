"""Windows application/window command handlers (Task 6).

These handlers execute real, reviewed operations through the injected
services — never through a shell. CLOSE_APPLICATION requires explicit
user confirmation ("yes") before it signals windows.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.commands.intent import _extract_app_name, _strip_filler
from src.commands.registry import CommandResult, PERMISSION_SYSTEM
from src.controllers.application_discovery import AppInfo

logger = logging.getLogger(__name__)


class _Svc:
    """Bundle of automation services passed to handlers."""

    def __init__(
        self,
        discovery,
        launcher,
        monitor,
        windows,
        monitors=None,
        folders=None,
        fs=None,
        volume=None,
        locker=None,
        power=None,
        network=None,
        screenshot=None,
        notifier=None,
        power_ops=None,
        desktop=None,
    ):
        self.discovery = discovery
        self.launcher = launcher
        self.monitor = monitor
        self.windows = windows
        self.monitors = monitors
        self.folders = folders
        self.fs = fs
        self.volume = volume
        self.locker = locker
        self.power = power
        self.network = network
        self.screenshot = screenshot
        self.notifier = notifier
        self.power_ops = power_ops
        self.desktop = desktop


def _app_name(entities: dict) -> str:
    return str(entities.get("app_name") or entities.get("target") or "").strip()


def _lookup(svc: "_Svc", entities: dict, name: str):
    """Resolve ``name`` against the dynamic inventory.

    Returns a ``ResolveResult``; a pre-resolved ``_app_info`` entity
    (set by the disambiguation flow) short-circuits the lookup.
    """
    from src.controllers.application_discovery import ResolveResult

    preset = entities.get("_app_info")
    if preset is not None:
        return ResolveResult("found", app=preset)
    res = svc.discovery.resolve(name)
    # Conservative gate: a "found" result backed only by a weak match
    # (prefix/substring/fuzzy, score < 100) is never executed directly —
    # it becomes a single-option clarification so the user confirms the
    # intended application ("Open the winds" must not launch Windsurf).
    # Exception: every spoken token appears in the app name
    # ("adobe photoshop" -> "Adobe Photoshop 2024") — that is a strong,
    # unique match and executes normally.
    if (
        res.status == "found"
        and res.app is not None
        and res.score < 100
        and not _tokens_subset(name, res.app)
    ):
        return ResolveResult(
            "ambiguous", options=[res.app], score=res.score
        )
    return res


def _tokens_subset(query: str, app) -> bool:
    """True when all normalized query tokens appear in the app's
    name/exe tokens (order-independent subset match)."""
    from src.controllers.application_discovery import _normalize

    q = set(_normalize(query).split())
    if not q:
        return False
    a = set(_normalize(app.name).split()) | set(
        _normalize(app.exe_name).split()
    )
    return q <= a


def _app_string_target(app) -> str:
    """Best plain-string window target for ``app``.

    Never returns a package-identifier fragment: AUMID apps fall back
    to the display name (matched against window titles/processes).
    """
    if getattr(app, "launch_kind", "exe") == "aumid":
        return app.name
    return app.exe_name


def _app_windows(svc: "_Svc", app, include_hidden: bool = False) -> list:
    """Windows belonging to ``app`` (identity-aware).

    ``include_hidden`` also returns hidden top-level windows (tray /
    packaged-app windows) — used by launch readiness and restore flows.
    """
    fn = getattr(svc.windows, "windows_for_app", None)
    if fn is not None:
        try:
            return list(fn(app, include_hidden=include_hidden))
        except TypeError:  # pragma: no cover - older window managers
            try:
                return list(fn(app))
            except Exception:  # noqa: BLE001
                logger.exception("windows_for_app failed for %s",
                                 app.name)
                return []
        except Exception:  # noqa: BLE001
            logger.exception("windows_for_app failed for %s", app.name)
            return []
    return svc.windows.windows_for(_app_string_target(app))


def _resolve_to_app(svc: "_Svc", target: str):
    """Best-effort AppInfo for a spoken target (never raises)."""
    if not target:
        return None
    try:
        res = svc.discovery.resolve(target)
    except Exception:  # noqa: BLE001
        return None
    if res.status == "found" and res.app is not None:
        return res.app
    return None


def _windows_for_query(svc: "_Svc", preset, target: str) -> list:
    """Window list for a window operation.

    A pre-resolved ``_app_info`` wins; otherwise the spoken target is
    resolved through the inventory (identity matching) with a
    process/title string fallback for windows of non-inventory apps.
    """
    if preset is not None:
        return _app_windows(svc, preset)
    app = _resolve_to_app(svc, target)
    wins = _app_windows(svc, app) if app is not None else []
    if target:
        seen = {w.hwnd for w in wins}
        for w in svc.windows.windows_for(target):
            if w.hwnd not in seen:
                wins.append(w)
                seen.add(w.hwnd)
    return wins


def _ambiguous_result(intent_name: str, options) -> CommandResult:
    names = [a.name for a in options]
    if len(names) == 1:
        message = (
            f"I found {names[0]}. Did you mean that application?"
        )
    else:
        message = (
            "I found several applications: "
            + ", ".join(names[:5])
            + ". Which one did you mean?"
        )
    return CommandResult(
        intent_name,
        True,
        message=message,
        ambiguous=True,
        options=tuple(names),
        data={"apps": list(options)},
    )


class OpenAppHandler:
    intent_name = "OPEN_APPLICATION"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc: _Svc):
        self._svc = svc

    def validate(self, entities):
        return None if _app_name(entities) else "missing_app_name"

    def execute(self, entities):
        name = _app_name(entities)
        res = _lookup(self._svc, entities, name)
        if res.status == "ambiguous":
            return _ambiguous_result(self.intent_name, res.options)
        if res.status != "found" or res.app is None:
            return CommandResult(
                self.intent_name, False,
                message=f"I couldn't find {name}.",
                error="application_not_found",
            )
        app = res.app

        # Default-monitor rule: every application launch targets
        # logical Monitor 1 — defined by MonitorManager as the Windows
        # PRIMARY display (right-hand screen on this machine) — unless
        # the user named a destination. The
        # OPEN_APPLICATION_ON_MONITOR handler performs launch, window
        # matching, placement and verification.
        if getattr(self._svc, "monitors", None) is not None:
            try:
                from src.commands.monitor_handlers import (
                    OpenOnMonitorHandler,
                )

                ent = dict(entities)
                ent["_app_info"] = app
                mon1 = None
                try:
                    mon1 = self._svc.monitors.get(1)
                except Exception:  # noqa: BLE001
                    mon1 = None
                if mon1 is not None:
                    ent["_monitor_info"] = mon1
                    ent["monitor"] = mon1
                else:
                    ent["monitor"] = "primary"
                positioned = OpenOnMonitorHandler(self._svc).execute(ent)
                # Honest fallback: the app may have launched even when
                # positioning failed — never claim Monitor 1 placement
                # that wasn't verified.
                if positioned.error in (
                    None, "application_not_found",
                ) or positioned.executed:
                    return positioned
                # Non-executed failure (e.g. monitor resolution): fall
                # through to a plain launch so the app still opens —
                # but log it: an unpositioned launch lands wherever
                # Windows chooses, which is NOT Monitor 1.
                logger.warning(
                    "Monitor-aware launch did not execute for %s "
                    "(error=%s); falling back to unpositioned launch",
                    app.name, positioned.error,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Monitor-aware launch failed for %s", app.name
                )

        ok, detail = self._svc.launcher.launch(app)
        if not ok:
            return CommandResult(
                self.intent_name, False,
                message=f"I couldn't open {app.name}.",
                error=detail,
            )
        return CommandResult(
            self.intent_name, True,
            message=f"Opening {app.name}.",
            data={"app": app.name, "exe": app.exe_path},
            executed=True,
        )


class CloseAppHandler:
    intent_name = "CLOSE_APPLICATION"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc: _Svc):
        self._svc = svc

    def validate(self, entities):
        return None if _app_name(entities) else "missing_app_name"

    def execute(self, entities):
        name = _app_name(entities)
        res = _lookup(self._svc, entities, name)
        if res.status == "ambiguous":
            return _ambiguous_result(self.intent_name, res.options)
        if res.status != "found" or res.app is None:
            return CommandResult(
                self.intent_name, False,
                message=f"I couldn't find {name}.",
                error="application_not_found",
            )
        app = res.app
        if not entities.get("_confirmed"):
            return CommandResult(
                self.intent_name, True,
                message=f"Should I close {app.name}? Say yes to confirm.",
                confirmation_required=True,
            )
        count = self._svc.windows.close_windows(app)
        if count == 0:
            return CommandResult(
                self.intent_name, True,
                message=f"{app.name} doesn't appear to be running.",
                executed=True,
            )
        # WM_CLOSE is asynchronous: verify the windows actually went
        # away before reporting success (never force-kill processes).
        if self._windows_gone(app):
            return CommandResult(
                self.intent_name, True,
                message=f"Closing {app.name}.",
                data={"windows_closed": count},
                executed=True,
            )
        return CommandResult(
            self.intent_name, True,
            message=(
                f"I asked {app.name} to close, but it may still be "
                f"open. Check the window."
            ),
            data={"windows_closed": count, "verified": False},
            executed=True,
        )

    def _windows_gone(self, target, timeout: float = 2.0) -> bool:
        """Poll until no visible window matches ``target`` (or timeout)."""
        import time as _time

        windows = getattr(self._svc.windows, "windows_for", None)
        if windows is None:
            return True  # cannot verify; assume the close was posted
        deadline = _time.monotonic() + timeout
        while True:
            try:
                if isinstance(target, AppInfo):
                    remaining = len(_app_windows(self._svc, target))
                else:
                    remaining = len(windows(target))
            except Exception:  # noqa: BLE001
                remaining = 0
            if remaining == 0:
                return True
            if _time.monotonic() >= deadline:
                return False
            _time.sleep(0.1)


class SwitchToAppHandler:
    intent_name = "SWITCH_TO_APPLICATION"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc: _Svc):
        self._svc = svc

    def validate(self, entities):
        return None if _app_name(entities) else "missing_app_name"

    def execute(self, entities):
        name = _app_name(entities)
        res = _lookup(self._svc, entities, name)
        if res.status == "ambiguous":
            return _ambiguous_result(self.intent_name, res.options)
        app = res.app if res.status == "found" else None
        # Identity-aware target: AppInfo when the inventory knows the
        # app, plain spoken name otherwise (window may still exist).
        target = app if app is not None else name
        focus_fn = getattr(self._svc.windows, "focus_target", None)
        if focus_fn is not None:
            fres = focus_fn(target)
            status = getattr(fres, "status", "not_running")
            detail = getattr(fres, "title", "") or name
        else:  # pragma: no cover - legacy window managers
            ok, detail = self._svc.windows.focus_app(name)
            status = "ok" if ok else "not_running"
        label = app.name if app is not None else name
        if status == "not_running":
            return CommandResult(
                self.intent_name, False,
                message=f"{label} doesn't appear to be running.",
                error="not_running",
            )
        if status == "focus_failed":
            return CommandResult(
                self.intent_name, False,
                message=(
                    f"I found {label}, but I couldn't bring it to the "
                    f"front. It may be minimized or blocked by another "
                    f"window."
                ),
                error="focus_failed",
            )
        return CommandResult(
            self.intent_name, True,
            message=f"Switching to {label}.",
            data={"window": detail},
            executed=True,
        )


class FindAppHandler:
    intent_name = "FIND_APPLICATION"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc: _Svc):
        self._svc = svc

    def validate(self, entities):
        return None if _app_name(entities) else "missing_app_name"

    def execute(self, entities):
        name = _app_name(entities)
        res = _lookup(self._svc, entities, name)
        if res.status == "ambiguous":
            return _ambiguous_result(self.intent_name, res.options)
        if res.status == "missing":
            return CommandResult(
                self.intent_name, True,
                message=f"I couldn't find {name} installed.",
                executed=True,
            )
        app = res.app
        return CommandResult(
            self.intent_name, True,
            message=f"I found {app.name}.",
            data={"app": app.name, "exe": app.exe_path},
            executed=True,
        )


class ListRunningHandler:
    intent_name = "LIST_RUNNING_APPLICATIONS"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc: _Svc):
        self._svc = svc

    def validate(self, entities):
        return None

    def execute(self, entities):
        from src.controllers import app_identity

        try:
            apps = list(self._svc.discovery.discover())
        except Exception:  # noqa: BLE001
            apps = []
        identities = app_identity.identities_for(apps)
        names = []
        seen = set()
        try:
            wins = self._svc.windows.list_windows(include_hidden=True)
        except TypeError:  # pragma: no cover - older window managers
            wins = self._svc.windows.list_windows()
        for w in wins:
            app = app_identity.app_for_window(w, apps, identities)
            if app is None and not getattr(w, "visible", True):
                # Hidden utility windows (IME, notify icons, broadcast
                # helpers) with no app identity are not "applications".
                continue
            label = app_identity.window_label(w, app)
            key = label.lower()
            if label and key not in seen:
                seen.add(key)
                names.append(label)
        if not names:
            message = "No application windows are open."
        elif len(names) <= 3:
            message = "You currently have " + ", ".join(names) + " open."
        else:
            message = (
                "You currently have " + ", ".join(names[:-1])
                + f" and {names[-1]} open."
            )
        return CommandResult(
            self.intent_name, True, message=message,
            data={"apps": names}, executed=True,
        )


class _WindowOpHandler:
    """Minimize/maximize/restore/focus on windows matching a target."""

    permission = PERMISSION_SYSTEM
    executable = True
    intent_name = ""
    op = ""
    doing = ""
    done = ""

    # Verbs used in the "which application should I X?" clarification.
    clarify_verb = ""

    def __init__(self, svc: _Svc):
        self._svc = svc

    def validate(self, entities):
        return None

    def _target_clarify(self, verb: str):
        """Clarification payload for a missing window target."""
        def resolver(text: str):
            return self._svc.discovery.resolve(text)

        return {
            "resolver": resolver,
            "apply": lambda intent, app: (
                intent.entities.__setitem__("_app_info", app),
                intent.entities.__setitem__("target", app.name),
            ),
            "message": f"Which application should I {verb}?",
            "ambiguous_message": "Which application did you mean?",
            # V10.8 Task 4: missing window target entity.
            "entity_issue": "missing",
        }

    def execute(self, entities):
        if entities.get("needs_target"):
            return CommandResult(
                self.intent_name, True, executed=False,
                message=self._target_clarify(self.clarify_verb)["message"],
                data={"clarify": self._target_clarify(self.clarify_verb)},
            )
        preset = entities.get("_app_info")
        target = str(entities.get("target") or "").strip()
        if preset is not None:
            label = preset.name
        elif target:
            app = _resolve_to_app(self._svc, target)
            label = app.name if app is not None else target
        else:
            label = "the window"
        wins = _windows_for_query(self._svc, preset, target)
        if not wins and (preset is not None or target):
            return CommandResult(
                self.intent_name, False,
                message=f"{label} doesn't appear to be running.",
                error="not_running",
            )
        if not wins:
            hwnd = self._svc.windows._api.foreground_hwnd()
            if not hwnd:
                return CommandResult(
                    self.intent_name, False,
                    message="I couldn't find a window to act on.",
                    error="no_window",
                )
            wins = [type("W", (), {"hwnd": hwnd, "title": ""})()]
        count = 0
        for w in wins:
            try:
                res = getattr(self._svc.windows, self.op)(w.hwnd)
                if res is False:
                    continue
                count += 1
            except Exception:  # noqa: BLE001
                logger.debug("window op failed", exc_info=True)
        if count == 0 and wins and self.op == "focus":
            return CommandResult(
                self.intent_name, False,
                message=(
                    f"I found {label}, but I couldn't bring it to the "
                    f"front."
                ),
                error="focus_failed",
            )
        return CommandResult(
            self.intent_name, True,
            message=f"{self.done} {label}.",
            data={"windows": count},
            executed=True,
        )


class MinimizeWindowHandler(_WindowOpHandler):
    intent_name = "MINIMIZE_WINDOW"
    op = "minimize"
    done = "Minimizing"
    clarify_verb = "minimize"


class MaximizeWindowHandler(_WindowOpHandler):
    intent_name = "MAXIMIZE_WINDOW"
    op = "maximize"
    done = "Maximizing"
    clarify_verb = "maximize"


class RestoreWindowHandler(_WindowOpHandler):
    intent_name = "RESTORE_WINDOW"
    op = "restore"
    done = "Restoring"
    clarify_verb = "restore"


class FocusWindowHandler(_WindowOpHandler):
    intent_name = "FOCUS_WINDOW"
    op = "focus"
    done = "Focusing"


class ShowDesktopHandler:
    intent_name = "SHOW_DESKTOP"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc: _Svc):
        self._svc = svc

    def validate(self, entities):
        return None

    def execute(self, entities):
        self._svc.windows.show_desktop()
        return CommandResult(
            self.intent_name, True,
            message="Showing the desktop.",
            executed=True,
        )


class ListInstalledHandler:
    """LIST_INSTALLED_APPLICATIONS — inventory summary."""

    intent_name = "LIST_INSTALLED_APPLICATIONS"
    permission = PERMISSION_SYSTEM
    executable = True

    _MAX_SPOKEN = 8

    def __init__(self, svc: _Svc):
        self._svc = svc

    def validate(self, entities):
        return None

    def execute(self, entities):
        try:
            apps = self._svc.discovery.discover()
        except Exception:  # noqa: BLE001
            logger.exception("Installed-app listing failed")
            apps = []
        if not apps:
            return CommandResult(
                self.intent_name, False,
                message="I couldn't find any installed applications.",
                error="empty_inventory",
            )
        names = [a.name for a in apps]
        head = names[: self._MAX_SPOKEN]
        more = len(names) - len(head)
        message = (
            f"You have {len(names)} applications installed. "
            f"Some of them include " + ", ".join(head)
        )
        if more > 0:
            message += f", and {more} more"
        message += "."
        return CommandResult(
            self.intent_name, True, message=message,
            data={"apps": names, "count": len(names)}, executed=True,
        )


class RefreshAppsHandler:
    """REFRESH_APPLICATIONS — force discovery rescan."""

    intent_name = "REFRESH_APPLICATIONS"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc: _Svc):
        self._svc = svc

    def validate(self, entities):
        return None

    def execute(self, entities):
        try:
            apps = self._svc.discovery.refresh()
        except Exception:  # noqa: BLE001
            logger.exception("Application refresh failed")
            return CommandResult(
                self.intent_name, False,
                message="I couldn't refresh the application list.",
                error="refresh_failed",
            )
        return CommandResult(
            self.intent_name, True,
            message=f"Done. I found {len(apps)} installed applications.",
            data={"count": len(apps)}, executed=True,
        )


class RecentAppsHandler:
    """RECENT_APPLICATIONS — apps added since the previous scan."""

    intent_name = "RECENT_APPLICATIONS"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc: _Svc):
        self._svc = svc

    def validate(self, entities):
        return None

    def execute(self, entities):
        try:
            self._svc.discovery.discover()
            recent = self._svc.discovery.recently_added()
        except Exception:  # noqa: BLE001
            logger.exception("Recent-app lookup failed")
            recent = []
        if not recent:
            return CommandResult(
                self.intent_name, True,
                message="I don't see any newly installed applications.",
                executed=True,
            )
        names = [a.name for a in recent][:8]
        message = (
            f"{len(recent)} application{'s' if len(recent) != 1 else ''} "
            f"{'were' if len(recent) != 1 else 'was'} installed recently: "
            + ", ".join(names) + "."
        )
        return CommandResult(
            self.intent_name, True, message=message,
            data={"apps": [a.name for a in recent]}, executed=True,
        )


class SearchWebHandler:
    """SEARCH_WEB — open a web search in the default browser.

    General web requests ("Search YouTube", "look up X online") must
    never degrade into local file searches. The handler only builds a
    URL and hands it to the OS browser; an empty query opens the
    engine homepage. ``executable=False`` for tests that only check
    recognition; the real registry instance executes.
    """

    intent_name = "SEARCH_WEB"
    permission = PERMISSION_SYSTEM
    executable = True

    _HOME = {
        "youtube": "https://www.youtube.com/",
        "google": "https://www.google.com/",
        "wikipedia": "https://www.wikipedia.org/",
    }
    _SEARCH = {
        "youtube": "https://www.youtube.com/results?search_query={q}",
        "google": "https://www.google.com/search?q={q}",
        "wikipedia": "https://en.wikipedia.org/wiki/Special:Search?search={q}",
        "web": "https://www.google.com/search?q={q}",
    }

    def validate(self, entities):
        return None

    def execute(self, entities):
        import urllib.parse
        import webbrowser

        query = str(entities.get("query") or "").strip()
        engine = str(entities.get("engine") or "web").lower()
        if engine not in self._SEARCH:
            engine = "web"
        if not query:
            url = self._HOME.get(engine, self._HOME["google"])
        else:
            url = self._SEARCH[engine].format(
                q=urllib.parse.quote_plus(query)
            )
        try:
            opened = webbrowser.open(url, new=2)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Web search open failed")
            return CommandResult(
                self.intent_name, False,
                message="I couldn't open the web search.",
                error=str(exc),
            )
        if not opened:
            return CommandResult(
                self.intent_name, False,
                message="I couldn't open the web search.",
                error="browser_unavailable",
            )
        label = engine if engine != "web" else "the web"
        message = (
            f"Searching {label} for {query}."
            if query else f"Opening {label}."
        )
        return CommandResult(
            self.intent_name, True, message=message,
            data={"url": url, "query": query, "engine": engine},
            executed=True,
        )


def create_app_registry(svc: _Svc):
    """Registry with all Task 5+6 intents; app/window ops execute."""
    from src.commands.handlers import (
        DateHandler,
        DecreaseVolumeHandler,
        IncreaseVolumeHandler,
        MuteVolumeHandler,
        SearchFilesHandler,
        TimeHandler,
        UnmuteVolumeHandler,
        create_default_registry,
    )
    from src.commands.registry import CommandRegistry

    registry = CommandRegistry()
    # Reuse Task 5 safe handlers.
    base = create_default_registry()
    for name in (
        "GET_CURRENT_TIME", "GET_CURRENT_DATE",
        "UNSUPPORTED_WINDOWS_ACTION",
    ):
        registry.register(name, base._handlers[name])
    # Executable Windows handlers.
    registry.register("OPEN_APPLICATION", OpenAppHandler(svc))
    registry.register("CLOSE_APPLICATION", CloseAppHandler(svc))
    registry.register("SWITCH_TO_APPLICATION", SwitchToAppHandler(svc))
    registry.register("FIND_APPLICATION", FindAppHandler(svc))
    registry.register("LIST_RUNNING_APPLICATIONS", ListRunningHandler(svc))
    registry.register("MINIMIZE_WINDOW", MinimizeWindowHandler(svc))
    registry.register("MAXIMIZE_WINDOW", MaximizeWindowHandler(svc))
    registry.register("RESTORE_WINDOW", RestoreWindowHandler(svc))
    registry.register("FOCUS_WINDOW", FocusWindowHandler(svc))
    registry.register("SHOW_DESKTOP", ShowDesktopHandler(svc))
    # P1 system handlers: known folders, master volume, workstation
    # lock. Each degrades to an honest failure when its service is
    # unavailable rather than pretending to act.
    from src.commands.system_handlers import (
        DecreaseVolumeHandler,
        GetVolumeHandler,
        IncreaseVolumeHandler,
        LockWorkstationHandler,
        MuteVolumeHandler,
        OpenKnownFolderHandler,
        SetVolumeHandler,
        UnmuteVolumeHandler,
    )

    # V10.1 local file/folder discovery: OPEN_FOLDER/SEARCH_FILES and
    # the new OPEN_FILE/SEARCH_FOLDERS/SEARCH_LOCAL_ITEMS/OPEN_LOCAL_ITEM
    # intents all resolve through svc.fs (indexed local search). When
    # no fs service is injected the handlers degrade to honest
    # "service unavailable" failures rather than pretending to act.
    from src.commands.file_handlers import create_file_handlers

    for name, handler in create_file_handlers(svc).items():
        registry.register(name, handler)
    registry.register("INCREASE_VOLUME", IncreaseVolumeHandler(svc))
    registry.register("DECREASE_VOLUME", DecreaseVolumeHandler(svc))
    registry.register("MUTE_VOLUME", MuteVolumeHandler(svc))
    registry.register("UNMUTE_VOLUME", UnmuteVolumeHandler(svc))
    registry.register("SET_VOLUME", SetVolumeHandler(svc))
    registry.register("GET_VOLUME", GetVolumeHandler(svc))
    registry.register("LOCK_WORKSTATION", LockWorkstationHandler(svc))
    # P2 awareness handlers: battery, connectivity, screenshot, toast.
    from src.commands.status_handlers import (
        BatteryStatusHandler,
        EthernetStatusHandler,
        InternetStatusHandler,
        ScreenshotHandler,
        SendNotificationHandler,
        WifiStatusHandler,
    )

    registry.register("GET_BATTERY_STATUS", BatteryStatusHandler(svc))
    registry.register("GET_INTERNET_STATUS", InternetStatusHandler(svc))
    registry.register("GET_WIFI_STATUS", WifiStatusHandler(svc))
    registry.register("GET_ETHERNET_STATUS", EthernetStatusHandler(svc))
    registry.register("TAKE_SCREENSHOT", ScreenshotHandler(svc))
    registry.register("SEND_NOTIFICATION", SendNotificationHandler(svc))
    registry.register("LIST_INSTALLED_APPLICATIONS", ListInstalledHandler(svc))
    registry.register("REFRESH_APPLICATIONS", RefreshAppsHandler(svc))
    registry.register("RECENT_APPLICATIONS", RecentAppsHandler(svc))
    registry.register("SEARCH_WEB", SearchWebHandler())
    # Milestone 1: power management + desktop/window controls. Handlers
    # degrade to honest failures when their services are not injected.
    from src.commands.system_control_handlers import (
        create_system_control_handlers,
    )

    for name, handler in create_system_control_handlers(svc).items():
        registry.register(name, handler)
    # Monitor intents (Task 6 enhancement).
    if getattr(svc, "monitors", None) is not None:
        from src.commands.monitor_handlers import create_monitor_handlers

        for name, handler in create_monitor_handlers(svc).items():
            registry.register(name, handler)
    return registry
