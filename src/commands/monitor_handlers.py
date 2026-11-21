"""Monitor-related command handlers (Task 6 enhancement).

Read-only monitor queries plus window movement and launch-on-monitor.
All Win32 access goes through the injected services — handlers never
touch the shell and never guess a monitor.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from src.commands.app_handlers import (
    _Svc,
    _ambiguous_result,
    _app_name,
    _app_windows,
    _lookup,
    _resolve_to_app,
    _windows_for_query,
)
from src.commands.registry import CommandResult, PERMISSION_SYSTEM
from src.controllers.application_discovery import ResolveResult
from src.controllers.monitor_manager import MonitorInfo, MonitorResolveResult

logger = logging.getLogger(__name__)

_WINDOW_TIMEOUT = 12.0
_POLL_INTERVAL = 0.25
_REUSE_GRACE = 2.0


def _no_monitors() -> CommandResult:
    return CommandResult(
        "MONITOR", False,
        message="I couldn't detect any monitors.",
        error="no_monitors",
    )


def _monitor_clarify(svc: _Svc, spec: str):
    """Build a clarify payload that re-resolves a spoken monitor reply."""
    def resolver(text: str):
        return svc.monitors.resolve(text)

    def apply(intent, monitor: MonitorInfo) -> None:
        intent.entities["_monitor_info"] = monitor
        intent.entities["monitor"] = monitor

    return {
        "resolver": resolver,
        "apply": apply,
        "message": (
            f"I couldn't tell which monitor you meant by "
            f"'{spec}'. Which monitor should I use?"
        ),
        "ambiguous_message": "Which monitor did you mean?",
        # V10.8 Task 4: entity classification for diagnostics.
        "entity_issue": "ambiguous",
    }


def _resolve_monitor(svc: _Svc, entities: dict, relative_to=None):
    """Resolve the ``monitor`` entity to a MonitorInfo.

    Returns (MonitorInfo|None, CommandResult|None). A non-None result
    means resolution failed and should be returned to the caller.
    """
    preset = entities.get("_monitor_info")
    if isinstance(preset, MonitorInfo):
        return preset, None
    spec = entities.get("monitor")
    if isinstance(spec, MonitorInfo):
        return spec, None
    spec_text = str(spec or "").strip()
    lowered = spec_text.lower()
    if lowered in ("primary", "main"):
        mon = svc.monitors.primary()
        return (mon, None) if mon else (None, _no_monitors())
    if lowered in ("other", "another"):
        res = svc.monitors.resolve("other", relative_to=relative_to)
        if res.status == "found":
            return res.monitor, None
        if res.status == "ambiguous":
            return None, CommandResult(
                "MONITOR", True,
                message="Which monitor did you mean?",
                ambiguous=True,
                data={"apps": list(res.options)},
            )
        return None, CommandResult(
            "MONITOR", False,
            message="There's only one monitor connected.",
            error="single_monitor",
        )
    res = svc.monitors.resolve(spec_text, relative_to=relative_to)
    if res.status == "found":
        return res.monitor, None
    if res.status == "ambiguous":
        return None, CommandResult(
            "MONITOR", True,
            message="Which monitor did you mean?",
            ambiguous=True,
            data={"apps": list(res.options)},
        )
    return None, CommandResult(
        "MONITOR", False,
        message=(
            f"I couldn't identify a monitor matching "
            f"'{spec_text}'. Which monitor did you mean?"
        ),
        error="monitor_not_found",
    )


class ListMonitorsHandler:
    intent_name = "LIST_MONITORS"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc: _Svc):
        self._svc = svc

    def validate(self, entities):
        return None

    def execute(self, entities):
        if self._svc.monitors is None:
            return _no_monitors()
        try:
            mons = self._svc.monitors.monitors()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Monitor enumeration failed")
            return CommandResult(
                self.intent_name, False,
                message="I couldn't read the display configuration.",
                error=str(exc),
            )
        if not mons:
            return CommandResult(
                self.intent_name, True,
                message="No monitors are currently connected.",
                executed=True,
            )
        return CommandResult(
            self.intent_name, True,
            message=self._svc.monitors.summary(),
            data={"monitors": [m.device for m in mons]},
            executed=True,
        )


class PrimaryMonitorHandler:
    intent_name = "GET_PRIMARY_MONITOR"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc: _Svc):
        self._svc = svc

    def validate(self, entities):
        return None

    def execute(self, entities):
        if self._svc.monitors is None:
            return _no_monitors()
        primary = self._svc.monitors.primary()
        if primary is None:
            return CommandResult(
                self.intent_name, True,
                message="No monitors are currently connected.",
                executed=True,
            )
        others = [
            m for m in self._svc.monitors.monitors() if m.index != primary.index
        ]
        msg = (
            f"Your primary monitor is {self._svc.monitors.describe(primary)}."
        )
        if others:
            msg += (
                f" The other display is "
                f"{self._svc.monitors.describe(others[0])}."
            )
        return CommandResult(
            self.intent_name, True, message=msg,
            data={"monitor": primary.device}, executed=True,
        )


class _WindowTargetMixin:
    """Shared target/window resolution for monitor commands."""

    def _target_windows(self, svc: _Svc, entities: dict):
        """Return (windows, error_result_or_None, clarify_or_None)."""
        preset = entities.get("_app_info")
        target = str(
            entities.get("target") or entities.get("app_name") or ""
        ).strip()
        if preset is None and not target:
            fg = svc.windows.foreground_window()
            if fg is None:
                return [], CommandResult(
                    self.intent_name, False,
                    message="I couldn't find a window to act on.",
                    error="no_window",
                ), None
            return [fg], None, None
        wins = _windows_for_query(svc, preset, target)
        if not wins:
            # Hidden top-level windows (packaged apps minimized to
            # tray, e.g. WhatsApp) are still movable: resolve the app
            # identity and offer its hidden windows — they are
            # restored before the move.
            app = preset if preset is not None else _resolve_to_app(
                svc, target)
            if app is not None:
                hidden = [
                    w for w in _app_windows(svc, app, include_hidden=True)
                    if not getattr(w, "visible", True)
                ]
                evidenced = _title_evidence(hidden, app)
                if evidenced:
                    wins = evidenced
        if not wins:
            if preset is not None:
                label = preset.name
            else:
                app = _resolve_to_app(svc, target)
                label = app.name if app is not None else target
            return [], CommandResult(
                self.intent_name, False,
                message=f"{label} doesn't appear to be running.",
                error="not_running",
            ), None
        if len(wins) > 1:
            clarify = self._window_clarify(svc, wins)
            return [], None, clarify
        return wins, None, None

    def _window_clarify(self, svc: _Svc, wins):
        """Ambiguous-window clarification payload."""
        return _build_window_clarify(svc, wins)


def _window_title_score(win, app) -> int:
    """Title evidence that ``win`` is ``app``'s real window.

    3 = normalized title equals the app name; 2 = app name is a
    substring of the title; 1 = shared display-name token; 0 = none.
    Utility surfaces (``H.NotifyIcon_...``, ``GDI+ Window (...)``)
    score lower than the real window.
    """
    from src.controllers.application_discovery import _normalize

    name_norm = _normalize(app.name)
    title_norm = _normalize(getattr(win, "title", "") or "")
    if not name_norm or not title_norm:
        return 0
    if title_norm == name_norm:
        return 3
    if name_norm in title_norm:
        return 2
    name_tokens = set(name_norm.split())
    title_tokens = set(title_norm.split())
    return len(name_tokens & title_tokens)


def _title_evidence(wins, app):
    """Candidates with the strongest title evidence for ``app``.

    Returns only the top-scoring subset (all ties) so a real window
    outranks tray/utility surfaces sharing the process identity.
    """
    scored = [(_window_title_score(w, app), w) for w in wins]
    scored = [(s, w) for s, w in scored if s > 0]
    if not scored:
        return []
    top = max(s for s, _ in scored)
    return [w for s, w in scored if s == top]


def _ensure_restored(svc: _Svc, win) -> None:
    """Un-hide a hidden (e.g. tray) window so it can be moved.

    Packaged apps like WhatsApp keep a hidden top-level window while
    minimized to tray; ``SW_RESTORE`` makes it visible again. Safe to
    call on already-visible windows (no-op).
    """
    if getattr(win, "visible", True):
        return
    try:
        svc.windows.restore(win.hwnd)
        time.sleep(0.15)
    except Exception:  # noqa: BLE001
        logger.debug("restore failed for hwnd %s", win.hwnd,
                     exc_info=True)


def _build_window_clarify(svc: _Svc, wins):
    """Clarify payload for choosing among several windows."""
    titles = [f'"{w.title or w.process}"' for w in wins[:4]]

    def resolver(text: str):
        t = text.strip().lower()
        # Ordinal selection.
        ords = {"first": 0, "1": 0, "one": 0, "second": 1, "2": 1,
                "two": 1, "third": 2, "3": 2, "three": 2}
        stripped = t.strip(" .!?,")
        if stripped in ords and ords[stripped] < len(wins):
            return ResolveResult("found", app=wins[ords[stripped]])
        for token in t.replace(",", " ").split():
            token = token.strip(".!?")
            if token in ords and ords[token] < len(wins):
                return ResolveResult("found", app=wins[ords[token]])
        hits = [
            w for w in wins
            if t and (t in (w.title or "").lower()
                      or t in (w.process or "").lower())
        ]
        if len(hits) == 1:
            return ResolveResult("found", app=hits[0])
        return ResolveResult("missing")

    return {
        "resolver": resolver,
        "apply": lambda intent, win: intent.entities.__setitem__(
            "_window_info", win
        ),
        "message": (
            f"I found several windows: "
            + ", ".join(titles)
            + ". Which one should I move?"
        ),
        "ambiguous_message": "Which window did you mean?",
        # V10.8 Task 4: entity classification for diagnostics.
        "entity_issue": "ambiguous",
    }


class WindowMonitorHandler(_WindowTargetMixin):
    """GET_WINDOW_MONITOR — report which monitor a window is on."""

    intent_name = "GET_WINDOW_MONITOR"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc: _Svc):
        self._svc = svc

    def validate(self, entities):
        return None

    def execute(self, entities):
        if self._svc.monitors is None:
            return _no_monitors()
        preset_win = entities.get("_window_info")
        if preset_win is not None:
            wins, err, clarify = [preset_win], None, None
        else:
            wins, err, clarify = self._target_windows(self._svc, entities)
        if err is not None:
            return err
        if clarify is not None:
            return CommandResult(
                self.intent_name, True, executed=False,
                message=clarify["message"],
                data={"clarify": clarify},
            )
        w = wins[0]
        mon = self._svc.monitors.monitor_for_hwnd(w.hwnd)
        if mon is None:
            return CommandResult(
                self.intent_name, True,
                message="I couldn't determine which monitor that window is on.",
                executed=True,
            )
        label = (w.title or w.process or "The window").strip()
        return CommandResult(
            self.intent_name, True,
            message=f"{label} is on {self._svc.monitors.describe(mon)}.",
            data={"monitor": mon.device}, executed=True,
        )


class _MoveWindowHandler(_WindowTargetMixin):
    """MOVE_WINDOW_TO_* — move one window onto a target monitor."""

    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc: _Svc, monitor_spec: str):
        self._svc = svc
        self._monitor_spec = monitor_spec  # "", "primary", "other"
        self.intent_name = {
            "": "MOVE_WINDOW_TO_MONITOR",
            "primary": "MOVE_WINDOW_TO_PRIMARY_MONITOR",
            "other": "MOVE_WINDOW_TO_OTHER_MONITOR",
        }[monitor_spec]

    def validate(self, entities):
        return None

    def execute(self, entities):
        svc = self._svc
        if svc.monitors is None:
            return _no_monitors()

        # Pick the window first so "other monitor" can resolve relative
        # to where it currently lives.
        preset_win = entities.get("_window_info")
        if preset_win is not None:
            wins = [preset_win]
        else:
            wins, err, clarify = self._target_windows(svc, entities)
            if err is not None:
                return err
            if clarify is not None:
                return CommandResult(
                    self.intent_name, True, executed=False,
                    message=clarify["message"],
                    data={"clarify": clarify},
                )
        w = wins[0]
        _ensure_restored(svc, w)
        label = self._window_label(svc, w)
        current = svc.monitors.monitor_for_hwnd(w.hwnd)

        ent = dict(entities)
        if self._monitor_spec:
            ent["monitor"] = self._monitor_spec
        mon, merr = _resolve_monitor(svc, ent, relative_to=current)
        if merr is not None:
            if merr.ambiguous or merr.error == "monitor_not_found":
                # Re-ask which monitor, keeping the window target.
                spec = str(ent.get("monitor") or "")
                payload = _monitor_clarify(svc, spec)
                payload["apply"] = lambda intent, m: (
                    intent.entities.__setitem__("_monitor_info", m),
                    intent.entities.__setitem__("monitor", m),
                )
                return CommandResult(
                    self.intent_name, True, executed=False,
                    message=payload["message"],
                    data={"clarify": payload},
                )
            return merr
        if mon is None:
            return _no_monitors()
        if current is not None and mon.handle == current.handle:
            label = self._window_label(svc, w)
            return CommandResult(
                self.intent_name, True,
                message=f"{label} is already on {svc.monitors.describe(mon)}.",
                executed=True,
            )

        ok, _max = svc.windows.move_to_work_area(
            w.hwnd, mon.work_area
        )
        # Let the window manager (and the app, which may re-assert
        # its own maximized state) settle before verifying — a move
        # is only confirmed against the final geometry.
        time.sleep(0.3)
        verified = self._verify(svc, w.hwnd, mon)
        # Report the ACTUAL destination, confirmed from the window's
        # post-move position — never the requested one.
        actual = svc.monitors.monitor_for_hwnd(w.hwnd)
        dest = actual if actual is not None else mon
        spoken = svc.monitors.spoken_label(dest)
        label = self._window_label(svc, w)
        if ok and verified:
            return CommandResult(
                self.intent_name, True,
                message=f"Moved {label} to {spoken}.",
                data={"monitor": dest.device, "hwnd": w.hwnd},
                executed=True,
            )
        if ok:
            return CommandResult(
                self.intent_name, True,
                message=(
                    f"I moved {label}, but I couldn't verify it landed "
                    f"on {spoken}. Check the display arrangement."
                ),
                data={"monitor": mon.device}, executed=True,
            )
        return CommandResult(
            self.intent_name, False,
            message=f"I couldn't move {label} to {spoken}.",
            error="move_failed",
        )

    @staticmethod
    def _window_label(svc: _Svc, w) -> str:
        """Canonical app name when resolvable; window title/process else."""
        from src.controllers import app_identity

        try:
            apps = svc.discovery.discover()
            identities = app_identity.identities_for(apps)
            app = app_identity.app_for_window(w, apps, identities)
            if app is not None:
                return app.name
        except Exception:  # noqa: BLE001
            pass
        return app_identity.window_label(w)

    @staticmethod
    def _verify(svc: _Svc, hwnd: int, mon: MonitorInfo) -> bool:
        """Confirm the window's resulting bounds on the target monitor.

        Verified only when the window reports on ``mon`` AND its
        rectangle center sits inside that monitor's bounds (work area
        preferred). Unverifiable geometry falls back to the monitor
        handle check alone.
        """
        after = svc.monitors.monitor_for_hwnd(hwnd)
        monitor_ok = (
            after is not None
            and (after.handle == mon.handle or after.device == mon.device)
        )
        rect = svc.windows.window_rect(hwnd)
        if rect is None:
            if after is not None:
                return monitor_ok
            cx = cy = None
        else:
            cx = (rect[0] + rect[2]) // 2
            cy = (rect[1] + rect[3]) // 2
        if not monitor_ok and cx is not None:
            at = svc.monitors.monitor_for_point(cx, cy)
            monitor_ok = (
                at is not None and at.device == mon.device
            )
        if cx is None:
            return monitor_ok
        # Bounds check: the window's center must lie inside the target
        # monitor's bounds (not merely somewhere on the virtual
        # screen), and at least half of the window's area must overlap
        # that monitor — a sliver hanging off the edge is not a
        # verified placement.
        b = mon.bounds
        inside = b.left <= cx < b.right and b.top <= cy < b.bottom
        if not inside:
            return False
        if rect is not None:
            iw = max(0, min(rect[2], b.right) - max(rect[0], b.left))
            ih = max(0, min(rect[3], b.bottom) - max(rect[1], b.top))
            area = max(1, (rect[2] - rect[0]) * (rect[3] - rect[1]))
            if (iw * ih) / area < 0.5:
                return False
        return bool(monitor_ok)


class OpenOnMonitorHandler:
    """OPEN_APPLICATION_ON_MONITOR — launch an app onto a monitor."""

    intent_name = "OPEN_APPLICATION_ON_MONITOR"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(
        self,
        svc: _Svc,
        window_timeout: float = _WINDOW_TIMEOUT,
        poll_interval: float = _POLL_INTERVAL,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ):
        self._svc = svc
        self._timeout = window_timeout
        self._poll = poll_interval
        self._sleep = sleep
        self._monotonic = monotonic

    def validate(self, entities):
        if not _app_name(entities):
            return "missing_app_name"
        if not str(entities.get("monitor") or "").strip() \
                and not isinstance(entities.get("_monitor_info"), MonitorInfo):
            return "missing_monitor"
        return None

    def execute(self, entities):
        svc = self._svc
        if svc.monitors is None:
            return _no_monitors()
        name = _app_name(entities)

        res = _lookup(svc, entities, name)
        if res.status == "ambiguous":
            return _ambiguous_result(self.intent_name, res.options)
        if res.status != "found" or res.app is None:
            return CommandResult(
                self.intent_name, False,
                message=f"I couldn't find {name}.",
                error="application_not_found",
            )
        app = res.app

        mon, merr = _resolve_monitor(svc, entities)
        if merr is not None:
            if merr.ambiguous or (
                merr.error == "monitor_not_found"
            ):
                spec = str(entities.get("monitor") or "")
                payload = _monitor_clarify(svc, spec)
                return CommandResult(
                    self.intent_name, True, executed=False,
                    message=payload["message"],
                    data={"clarify": payload},
                )
            return merr
        if mon is None:
            return _no_monitors()

        # A window chosen via clarification skips launch entirely —
        # the app is already running with that window open.
        preset_win = entities.get("_window_info")
        if preset_win is not None:
            win = preset_win
        else:
            # Snapshot existing windows so a reused instance's *new*
            # window can be told apart from windows already open.
            before = {w.hwnd for w in _app_windows(svc, app)}

            ok, proc, detail = svc.launcher.launch_process(app)
            if not ok:
                return CommandResult(
                    self.intent_name, False,
                    message=f"I couldn't open {app.name}.",
                    error=detail,
                )
            pid = getattr(proc, "pid", None) if proc is not None else None

            win = self._await_window(app, pid, before)
        spoken = svc.monitors.spoken_label(mon)
        if win is None:
            # Reused instance with several existing windows: ask which
            # one instead of silently picking or failing outright.
            all_matches = _app_windows(svc, app, include_hidden=True)
            visible = [w for w in all_matches
                       if getattr(w, "visible", True)]
            leftover = (_title_evidence(visible, app) or visible
                        or _title_evidence(all_matches, app))
            if len(leftover) == 1:
                # A single unambiguous window survived discovery —
                # move it rather than reporting failure.
                win = leftover[0]
            elif len(leftover) > 1:
                payload = _build_window_clarify(svc, leftover)
                return CommandResult(
                    self.intent_name, True, executed=False,
                    message=payload["message"],
                    data={"clarify": payload},
                )
            else:
                return CommandResult(
                    self.intent_name, True,
                    message=(
                        f"{app.name} opened, but I couldn't find its "
                        f"window to move it to {spoken}. You can move "
                        f"it manually."
                    ),
                    data={"app": app.name, "monitor": mon.device},
                    executed=True,
                )
        if win is None:
            return CommandResult(
                self.intent_name, True,
                message=(
                    f"{app.name} opened, but I couldn't find its window "
                    f"to move it to {spoken}. You can move it manually."
                ),
                data={"app": app.name, "monitor": mon.device},
                executed=True,
            )

        _ensure_restored(svc, win)
        moved, _max = svc.windows.move_to_work_area(win.hwnd, mon.work_area)
        time.sleep(0.3)
        verified = _MoveWindowHandler._verify(svc, win.hwnd, mon)
        if moved and verified:
            return CommandResult(
                self.intent_name, True,
                message=f"Opened {app.name} on {spoken}.",
                data={"app": app.name, "monitor": mon.device,
                      "hwnd": win.hwnd},
                executed=True,
            )
        return CommandResult(
            self.intent_name, True,
            message=(
                f"{app.name} opened on {spoken}, but I couldn't verify "
                f"its final position."
            ),
            data={"app": app.name, "monitor": mon.device,
                  "hwnd": win.hwnd},
            executed=True,
        )

    def _await_window(self, app, pid, before):
        """Poll (bounded) for the launched app's top-level window.

        Waits for windows to appear asynchronously: first by the
        spawned pid, then by app identity (process name, package
        family, exe path, framework-host title) so packaged apps whose
        windows are hosted by ApplicationFrameHost are found too.
        Hidden top-level windows (packaged apps minimized to tray,
        e.g. WhatsApp) count as discovered — the caller restores them
        before moving.
        """
        t0 = self._monotonic()
        deadline = t0 + self._timeout
        grace = t0 + min(_REUSE_GRACE, self._timeout)
        attempts = 0
        last_sig = None
        logger.info(
            "Window discovery start: app=%s pid=%s before=%d "
            "timeout=%.1fs",
            app.name, pid, len(before), self._timeout,
        )
        while True:
            attempts += 1
            now = self._monotonic()
            if pid:
                wins = self._svc.windows.windows_for_pid(pid)
                fresh = [w for w in wins if w.hwnd not in before]
                if fresh:
                    self._log_found(app, attempts, t0, fresh, "pid")
                    return fresh[0]
                # No fresh pid windows: reused instance — never return
                # an arbitrary pre-existing window here.
            # Identity fallback: windows matching the app, including
            # hidden ones (packaged/tray apps). ``before`` was captured
            # from visible windows only, so hidden windows that existed
            # pre-launch still count as "not fresh" — correct, since a
            # reused instance's hidden window is the same window.
            matches = self._identity_windows(app)
            fresh = [w for w in matches if w.hwnd not in before]
            visible_fresh = [w for w in fresh
                             if getattr(w, "visible", True)]
            if visible_fresh:
                self._log_found(app, attempts, t0, visible_fresh,
                                "identity")
                return visible_fresh[0]
            if fresh:
                # Window exists but is hidden (packaged app in tray).
                # Give a visible window a few beats to appear, then
                # commit to the best hidden candidate — but ONLY one
                # whose title evidences the app's main window. Tray
                # utility windows ("H.NotifyIcon_...", IME surfaces)
                # share the process identity yet are not the app's
                # window; moving them would be wrong.
                best = self._best_hidden(fresh, app)
                for _ in range(3):
                    self._sleep(self._poll)
                    attempts += 1
                    vis = [
                        w for w in self._identity_windows(app)
                        if w.hwnd not in before
                        and getattr(w, "visible", True)
                    ]
                    if vis:
                        self._log_found(app, attempts, t0, vis,
                                        "identity")
                        return vis[0]
                if best is not None:
                    self._log_found(app, attempts, t0, [best],
                                    "identity-hidden")
                    return best
                # No title-evidenced hidden window: keep polling —
                # the real window may still be starting.
                fresh = []
            # Log transitions in candidate count for diagnostics.
            sig = (len(matches), sum(
                1 for w in matches if getattr(w, "visible", True)))
            if sig != last_sig:
                last_sig = sig
                logger.debug(
                    "Window discovery attempt %d for %s: %d match(es) "
                    "(%d visible) after %.1fs",
                    attempts, app.name, sig[0], sig[1],
                    now - t0,
                )
            # Reused instance: windows existed before launch and no new
            # window appeared within the grace period — resolve now
            # instead of burning the whole timeout. Visible windows
            # are preferred for ambiguity decisions; hidden-only apps
            # (tray) fall back to their hidden window.
            vis_matches = [w for w in matches
                           if getattr(w, "visible", True)]
            if before and now >= grace:
                pool = (_title_evidence(vis_matches, app)
                        or vis_matches
                        or _title_evidence(matches, app))
                if len(pool) == 1:
                    self._log_found(app, attempts, t0, pool,
                                    "reused-single")
                    return pool[0]
                if len(pool) > 1:
                    self._log_found(app, attempts, t0, pool,
                                    "reused-ambiguous")
                    return None
            if now >= deadline:
                pool = (_title_evidence(vis_matches, app)
                        or vis_matches
                        or _title_evidence(matches, app))
                if len(pool) == 1:
                    self._log_found(app, attempts, t0, pool,
                                    "timeout-single")
                    return pool[0]
                self._log_found(app, attempts, t0, pool,
                                "timeout-none")
                return None
            self._sleep(self._poll)

    def _identity_windows(self, app):
        """All top-level windows matching ``app``'s identity,
        including hidden ones."""
        fn = getattr(self._svc.windows, "windows_for_app", None)
        if fn is not None:
            try:
                return list(fn(app, include_hidden=True))
            except TypeError:  # pragma: no cover - older signatures
                try:
                    return list(fn(app))
                except Exception:  # noqa: BLE001
                    return []
            except Exception:  # noqa: BLE001
                logger.debug("windows_for_app failed", exc_info=True)
                return []
        return _app_windows(self._svc, app, include_hidden=True)

    @staticmethod
    def _best_hidden(fresh, app):
        """Pick the most likely real window among hidden candidates.

        A hidden window is only accepted when its title shares a token
        with the app's display name — this rejects tray utility
        windows (``H.NotifyIcon_...``) and IME surfaces that share the
        process identity but are not the application's window. Returns
        None when no candidate has title evidence (caller keeps
        polling / reports honestly rather than guessing).
        """
        evidenced = _title_evidence(fresh, app)
        return evidenced[0] if evidenced else None

    @staticmethod
    def _log_found(app, attempts, t0, wins, how):
        hwnds = [w.hwnd for w in wins[:6]]
        logger.info(
            "Window discovered for %s via %s after %d attempt(s), "
            "%.2fs: hwnds=%s",
            app.name, how, attempts, time.monotonic() - t0, hwnds,
        )


def create_monitor_handlers(svc: _Svc):
    """Return {intent_name: handler} for all monitor intents."""
    return {
        "LIST_MONITORS": ListMonitorsHandler(svc),
        "GET_PRIMARY_MONITOR": PrimaryMonitorHandler(svc),
        "GET_WINDOW_MONITOR": WindowMonitorHandler(svc),
        "MOVE_WINDOW_TO_MONITOR": _MoveWindowHandler(svc, ""),
        "MOVE_WINDOW_TO_PRIMARY_MONITOR": _MoveWindowHandler(svc, "primary"),
        "MOVE_WINDOW_TO_OTHER_MONITOR": _MoveWindowHandler(svc, "other"),
        "OPEN_APPLICATION_ON_MONITOR": OpenOnMonitorHandler(svc),
    }
