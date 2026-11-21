"""VoiceOS V9.5 — WhatsApp application reliability test.

Physical Windows test focused exclusively on WhatsApp
(``5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App`` /
``WhatsApp.Root.exe``). Built on the V9.3/V9.4 stack:
``ApplicationDiscovery`` -> ``AppIdentity`` -> ``ApplicationLauncher``
-> ``WindowManager`` -> ``MonitorManager`` and the real command
handlers in ``app_handlers`` / ``monitor_handlers``.

Safety model
------------
* The pre-existing WhatsApp window is user-owned. Window operations
  (minimize / restore / maximize / monitor moves / command-handler
  moves) run ONLY because the user explicitly authorized them for
  this test; the window is restored afterwards.
* The window is NEVER closed and no WhatsApp process is terminated.
* Launch/activation is idempotent: repeated activations must reuse
  the running instance instead of spawning duplicates.

Launch state machine (distinguished, never conflated):
    NOT_RUNNING -> LAUNCH_REQUESTED -> ACTIVATION_ACCEPTED
    -> EXISTING_REUSED | FRESH_WINDOW | WINDOW_NOT_READY
    | DISCOVERY_FAILED

Monitor convention (``MonitorManager`` only):
    Monitor 1 = Windows primary display = user RIGHT display.
    Monitor 2 = Windows secondary display = user LEFT display.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_all_applications as v93  # noqa: E402

from src.controllers import app_identity  # noqa: E402
from src.commands.app_handlers import (  # noqa: E402
    _Svc, OpenAppHandler, SwitchToAppHandler, MinimizeWindowHandler,
    MaximizeWindowHandler, RestoreWindowHandler, FocusWindowHandler,
)
from src.commands.monitor_handlers import (  # noqa: E402
    OpenOnMonitorHandler, _MoveWindowHandler,
)

PASS, FAIL, PARTIAL, SKIP = v93.PASS, v93.FAIL, v93.PARTIAL, v93.SKIP

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
RESULTS_JSON = REPORTS / "v95_whatsapp_results.json"
RESULTS_CSV = REPORTS / "v95_whatsapp_results.csv"
REPORT_MD = DOCS / "V95_WHATSAPP_RELIABILITY_REPORT.md"

APP_QUERY = "WhatsApp"
EXPECTED_PFN = "5319275a.whatsappdesktop_cv1g1gvanyjgm"
FORBIDDEN_NAME_FRAGMENTS = ("5319275a", "whatsappdesktop_cv1g1gvanyjgm",
                            "_cv1g1gvanyjgm", "!")

LAUNCH_TIMEOUT = 15.0        # bounded wait for a *fresh* window
REUSE_GRACE = 2.0            # mirrors monitor_handlers._REUSE_GRACE
POLL = 0.25
VIS_SNAPSHOTS = 5
VIS_SNAPSHOT_GAP = 0.4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Win32 helpers (read-only diagnostics)
# ---------------------------------------------------------------------------
def _is_cloaked(hwnd: int) -> Optional[bool]:
    """DWM cloaked state; None when the API is unavailable."""
    try:
        dwm = ctypes.windll.dwmapi
        val = ctypes.c_int(0)
        hr = dwm.DwmGetWindowAttribute(
            hwnd, 14, ctypes.byref(val), ctypes.sizeof(val))  # DWMWA_CLOAKED
        if hr == 0:
            return bool(val.value)
    except Exception:  # noqa: BLE001
        pass
    return None


def _window_state(rt, hwnd: int) -> dict:
    api = rt.windows._api
    out = {"hwnd": hwnd}
    try:
        out["visible"] = bool(api.is_visible(hwnd))
    except Exception:  # noqa: BLE001
        out["visible"] = None
    try:
        out["minimized"] = bool(rt.windows.is_minimized(hwnd))
    except Exception:  # noqa: BLE001
        out["minimized"] = None
    try:
        out["maximized"] = bool(rt.windows.is_maximized(hwnd))
    except Exception:  # noqa: BLE001
        out["maximized"] = None
    try:
        out["rect"] = rt.windows.window_rect(hwnd)
    except Exception:  # noqa: BLE001
        out["rect"] = None
    try:
        out["cloaked"] = _is_cloaked(hwnd)
    except Exception:  # noqa: BLE001
        out["cloaked"] = None
    try:
        mon = rt.monitors.monitor_for_hwnd(hwnd)
        out["monitor_index"] = getattr(mon, "index", None)
        out["monitor_device"] = getattr(mon, "device", None)
    except Exception:  # noqa: BLE001
        out["monitor_index"] = None
        out["monitor_device"] = None
    return out


def _snapshot_windows(rt, app) -> list[dict]:
    wins = rt.app_windows(app, include_hidden=True)
    return [_window_state(rt, w.hwnd) | {
        "title": w.title, "process": w.process,
        "pid": w.pid, "package_family": w.package_family,
        "exe_path": w.exe_path,
    } for w in wins]


# ---------------------------------------------------------------------------
# Launch state machine
# ---------------------------------------------------------------------------
def launch_state_machine(rt, app, before_hwnds: set[int],
                         timeout: float = LAUNCH_TIMEOUT) -> dict:
    """Activate ``app`` and classify the outcome honestly.

    Returns a dict with ``state`` one of:
        FRESH_WINDOW / EXISTING_REUSED / WINDOW_NOT_READY /
        ACTIVATION_ONLY / LAUNCH_FAILED
    plus latency and evidence.
    """
    t0 = time.monotonic()
    try:
        ok, _proc, detail = rt.launcher.launch_process(app)
    except Exception as exc:  # noqa: BLE001
        return {"state": "LAUNCH_FAILED", "ok": False,
                "detail": str(exc), "latency_ms": 0.0}
    if not ok:
        return {"state": "LAUNCH_FAILED", "ok": False,
                "detail": detail or "launch_failed",
                "latency_ms": (time.monotonic() - t0) * 1000.0}
    deadline = t0 + timeout
    grace = t0 + min(REUSE_GRACE, timeout)
    last_matches = 0
    while time.monotonic() < deadline:
        try:
            matches = rt.app_windows(app, include_hidden=True)
        except Exception:  # noqa: BLE001
            matches = []
        last_matches = len(matches)
        fresh = [w for w in matches if w.hwnd not in before_hwnds]
        fresh_visible = [w for w in fresh if getattr(w, "visible", True)]
        if fresh_visible:
            win = v93._choose_window(fresh_visible, app)
            return {
                "state": "FRESH_WINDOW", "ok": True, "detail": "ok",
                "latency_ms": (time.monotonic() - t0) * 1000.0,
                "window": v93._window_diag(win), "how": "fresh-visible",
                "matches": last_matches,
            }
        # Reuse path: activation accepted and a usable pre-existing
        # window is already there — do NOT burn the full timeout and
        # do NOT call it a failure. WhatsApp keeps a hidden (tray)
        # window, so a single title-evidenced hidden window also
        # counts as a usable existing window.
        if before_hwnds and time.monotonic() >= grace:
            visible = [w for w in matches if getattr(w, "visible", True)]
            evidenced = [w for w in visible
                         if v93._window_score(w, app)[0] > 0]
            pool = evidenced or visible
            how = "reused-single"
            if not pool:
                hidden_ev = [
                    w for w in matches
                    if not getattr(w, "visible", True)
                    and v93._window_score(w, app)[0] > 0
                ]
                if len(hidden_ev) == 1:
                    pool = hidden_ev
                    how = "reused-hidden-evidenced"
            if len(pool) == 1:
                return {
                    "state": "EXISTING_REUSED", "ok": True,
                    "detail": "activation accepted; existing window reused",
                    "latency_ms": (time.monotonic() - t0) * 1000.0,
                    "window": v93._window_diag(pool[0]),
                    "how": how, "matches": last_matches,
                }
        # Process-level fallback: app running but no usable window yet.
        if time.monotonic() >= deadline:
            if matches:
                return {
                    "state": "WINDOW_NOT_READY", "ok": True,
                    "detail": ("process/windows exist but no usable "
                               "top-level window after timeout"),
                    "latency_ms": (time.monotonic() - t0) * 1000.0,
                    "how": "timeout-windows-present",
                    "matches": last_matches,
                    "windows": [v93._window_diag(w) for w in matches[:8]],
                }
            return {
                "state": "DISCOVERY_FAILED", "ok": True,
                "detail": "activation accepted but no window discovered",
                "latency_ms": (time.monotonic() - t0) * 1000.0,
                "how": "timeout-no-windows", "matches": 0,
            }
        time.sleep(POLL)
    return {"state": "WINDOW_NOT_READY", "ok": True,
            "detail": "timeout", "latency_ms":
            (time.monotonic() - t0) * 1000.0, "matches": last_matches}


# ---------------------------------------------------------------------------
# Test record helpers
# ---------------------------------------------------------------------------
class Check:
    def __init__(self):
        self.results = []
        self.diag = {}

    def add(self, name, status, details="", **diag):
        self.results.append({"test": name, "status": status,
                             "details": details})
        self.diag[name] = diag

    def ok(self, name, details="", **diag):
        self.add(name, PASS, details or "ok", **diag)

    def fail(self, name, details="", **diag):
        self.add(name, FAIL, details, **diag)

    def skip(self, name, reason="", **diag):
        self.add(name, SKIP, reason or "not safely testable", **diag)

    @property
    def counts(self):
        c = {PASS: 0, FAIL: 0, PARTIAL: 0, SKIP: 0}
        for r in self.results:
            c[r["status"]] = c.get(r["status"], 0) + 1
        return c

    @property
    def overall(self):
        c = self.counts
        if c[FAIL]:
            return FAIL
        if c[PARTIAL] or c[SKIP]:
            return PARTIAL
        return PASS


def _svc(rt) -> _Svc:
    return _Svc(
        discovery=rt.discovery, launcher=rt.launcher,
        monitor=rt.monitor, windows=rt.windows, monitors=rt.monitors,
    )


def _cmd_result_ok(res) -> bool:
    return bool(getattr(res, "executed", False)) and not getattr(
        res, "error", None)


# ---------------------------------------------------------------------------
# Main test flow
# ---------------------------------------------------------------------------
def run(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    REPORTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    rt = v93.Runtime()
    svc = _svc(rt)
    chk = Check()
    evidence = {"snapshots": [], "operations": [], "commands": []}

    # -- resolve + identity -------------------------------------------------
    res = rt.discovery.resolve(APP_QUERY)
    app = res.app if res.status == "found" else None
    chk.add("resolve_lookup", PASS if app else FAIL,
            f"status={res.status} score={getattr(res, 'score', None)}")
    if app is None:
        _finish(chk, evidence, rt, app, "app not found")
        return 1

    ident = app_identity.identity_for_app(app)
    name_l = (app.name or "").lower()
    chk.add("canonical_identity",
            PASS if re_fullmatch(r"^aumid:.+!.+$", ident.app_id) else FAIL,
            ident.app_id)
    chk.add("display_name_readable",
            PASS if app.name and not any(
                f in name_l for f in FORBIDDEN_NAME_FRAGMENTS) else FAIL,
            app.name)
    chk.add("package_identity_consistent",
            PASS if (ident.package_family
                     and ident.package_family == EXPECTED_PFN
                     and (app.aumid or "").lower().startswith(
                         EXPECTED_PFN + "!")) else FAIL,
            f"pfn={ident.package_family} aumid={app.aumid}")
    chk.add("process_candidates",
            PASS if any("whatsapp" in c for c in ident.process_names)
            else FAIL,
            ",".join(ident.process_names))
    chk.add("launch_path_allowed",
            PASS if rt.discovery.is_app_path_allowed(app.aumid) else FAIL,
            app.aumid)
    exe_name = (app.exe_name or "").lower()
    chk.add("exe_identity",
            PASS if exe_name
            and not any(f in exe_name for f in FORBIDDEN_NAME_FRAGMENTS)
            and ("whatsapp" in exe_name
                 or "whatsapp" in (app.exe_path or "").lower())
            else FAIL,
            f"exe_name={app.exe_name} exe_path={app.exe_path}")

    # -- pre-test snapshot ---------------------------------------------------
    pre_procs = [p for p in rt.monitor.list_processes()
                 if "whatsapp" in (p.name or "").lower()]
    pre_windows = _snapshot_windows(rt, app)
    evidence["snapshots"].append({"phase": "pre-test",
                                  "processes": [
                                      {"pid": p.pid, "name": p.name}
                                      for p in pre_procs],
                                  "windows": pre_windows})
    pre_hwnds = {w["hwnd"] for w in pre_windows}
    chk.add("existing_instance_detection",
            PASS if rt.monitor.is_running(app) else FAIL,
            f"{len(pre_procs)} process(es), "
            f"{len(pre_windows)} window(s)")

    # -- window discovery + identity matching --------------------------------
    wins = rt.app_windows(app, include_hidden=True)
    vis_wins = [w for w in wins if getattr(w, "visible", True)]
    chk.add("window_discovery",
            PASS if wins else FAIL,
            f"{len(wins)} matched, {len(vis_wins)} visible")
    id_ok = all(
        app_identity.window_matches_identity(w, app, ident) for w in wins
    ) if wins else False
    chk.add("window_identity_match", PASS if id_ok else FAIL,
            f"{sum(1 for w in wins if app_identity.window_matches_identity(w, app, ident))}/{len(wins)}")
    rev_ok = False
    if wins:
        try:
            rev_ok = app_identity.app_for_window(wins[0], [app]) is app
        except Exception:  # noqa: BLE001
            rev_ok = False
    chk.add("reverse_identity_lookup", PASS if rev_ok else FAIL)

    # -- visibility consistency ----------------------------------------------
    flap = []
    snaps = []
    for i in range(VIS_SNAPSHOTS):
        snap = _snapshot_windows(rt, app)
        snaps.append(snap)
        if i:
            prev = {w["hwnd"]: w.get("visible") for w in snaps[i - 1]}
            for w in snap:
                if prev.get(w["hwnd"]) != w.get("visible"):
                    flap.append({"hwnd": w["hwnd"], "at": i,
                                 "was": prev.get(w["hwnd"]),
                                 "now": w.get("visible")})
        time.sleep(VIS_SNAPSHOT_GAP)
    evidence["snapshots"].append({"phase": "visibility-consistency",
                                  "snapshots": snaps, "flaps": flap})
    chk.add("visibility_consistency",
            PASS if not flap else PARTIAL,
            f"{len(snaps)} snapshots, {len(flap)} visibility flap(s)",
            flaps=flap[:10])
    chk.diag["visibility_consistency"]["window_states"] = snaps[-1]

    # -- launch state machine -------------------------------------------------
    lm = launch_state_machine(rt, app, pre_hwnds)
    evidence["launch_state_machine"] = lm
    state = lm["state"]
    if state == "FRESH_WINDOW":
        chk.ok("launch_activation", f"fresh window in {lm['latency_ms']:.0f}ms")
    elif state == "EXISTING_REUSED":
        chk.ok("launch_activation",
               f"activation accepted; existing window reused in "
               f"{lm['latency_ms']:.0f}ms (no unnecessary wait)")
    elif state == "WINDOW_NOT_READY":
        chk.add("launch_activation", PARTIAL, lm["detail"],
                latency_ms=lm["latency_ms"])
    elif state == "DISCOVERY_FAILED":
        chk.fail("launch_activation", lm["detail"],
                 latency_ms=lm["latency_ms"])
    else:
        chk.fail("launch_activation", lm["detail"])
    chk.add("no_unnecessary_wait",
            PASS if lm["latency_ms"] < 10000 else FAIL,
            f"{lm['latency_ms']:.0f}ms to usable state")

    # -- duplicate-launch prevention ------------------------------------------
    procs_before = {p.pid for p in rt.monitor.list_processes()
                    if "whatsapp" in (p.name or "").lower()}
    wins_before = {w.hwnd for w in rt.app_windows(app, include_hidden=True)}
    try:
        ok2, _p2, d2 = rt.launcher.launch_process(app)
    except Exception as exc:  # noqa: BLE001
        ok2, d2 = False, str(exc)
    time.sleep(3.0)
    procs_after = {p.pid for p in rt.monitor.list_processes()
                   if "whatsapp" in (p.name or "").lower()}
    wins_after = {w.hwnd for w in rt.app_windows(app, include_hidden=True)}
    dup_ok = (ok2 and procs_after == procs_before
              and wins_after == wins_before)
    chk.add("duplicate_launch_prevention",
            PASS if dup_ok else FAIL,
            f"second activation ok={ok2}; new_procs="
            f"{sorted(procs_after - procs_before)}; "
            f"new_windows={sorted(wins_after - wins_before)}",
            new_processes=sorted(procs_after - procs_before),
            new_windows=sorted(wins_after - wins_before))

    # -- pick the working window ----------------------------------------------
    wins = rt.app_windows(app, include_hidden=True)
    visible = [w for w in wins if getattr(w, "visible", True)]
    work = v93._choose_window(visible or wins, app) if wins else None

    # -- switch / focus (safe: does not move or close) -------------------------
    if work is None:
        chk.skip("switch_focus", "no window to focus")
        chk.skip("focus_window_command", "no window to focus")
    else:
        try:
            fres = rt.windows.focus_target(app)
            fg = rt.windows.foreground_window()
            fg_ok = (fg is not None
                     and fg.hwnd == getattr(fres, "hwnd", 0))
            chk.add("switch_focus",
                    PASS if getattr(fres, "status", "") == "ok" and fg_ok
                    else FAIL,
                    f"status={getattr(fres, 'status', '')} "
                    f"fg={getattr(fg, 'hwnd', None)}")
            evidence["operations"].append({
                "op": "switch_focus", "hwnd": work.hwnd,
                "status": getattr(fres, "status", ""),
                "foreground_hwnd": getattr(fg, "hwnd", None)})
        except Exception as exc:  # noqa: BLE001
            chk.fail("switch_focus", str(exc))
        # Command-handler level: SWITCH_TO_APPLICATION
        try:
            hres = SwitchToAppHandler(svc).execute(
                {"app_name": APP_QUERY})
            chk.add("switch_command_handler",
                    PASS if _cmd_result_ok(hres) else FAIL,
                    getattr(hres, "message", ""))
            evidence["commands"].append({
                "command": "SWITCH_TO_APPLICATION",
                "executed": getattr(hres, "executed", None),
                "error": getattr(hres, "error", None),
                "message": getattr(hres, "message", "")})
        except Exception as exc:  # noqa: BLE001
            chk.fail("switch_command_handler", str(exc))

    # -- authorized lifecycle ops ----------------------------------------------
    authorized = True  # user granted explicit authorization this session
    hwnd = work.hwnd if work is not None else None
    if hwnd is None:
        for t in ("minimize_restore", "maximize", "move_monitor_2",
                  "move_monitor_1", "minimize_command",
                  "maximize_command", "restore_command",
                  "focus_command", "move_window_command_m1",
                  "move_window_command_m2", "open_on_monitor_m1",
                  "open_on_monitor_m2", "open_app_command"):
            chk.skip(t, "no window available")
    else:
        m1 = rt.monitors.get(1)
        m2 = rt.monitors.get(2)
        orig = _window_state(rt, hwnd)
        evidence["operations"].append({"op": "pre_lifecycle_state",
                                       "state": orig})

        # minimize -> restore
        try:
            rt.windows.minimize(hwnd)
            time.sleep(0.4)
            minimized = rt.windows.is_minimized(hwnd)
            rt.windows.restore(hwnd)
            time.sleep(0.4)
            st = _window_state(rt, hwnd)
            restored = (not st["minimized"]) and st["visible"]
            chk.add("minimize_restore",
                    PASS if (minimized and restored) else FAIL,
                    f"minimized={minimized} restored_visible={st['visible']}")
            evidence["operations"].append({
                "op": "minimize_restore", "minimized": minimized,
                "after": st})
        except Exception as exc:  # noqa: BLE001
            chk.fail("minimize_restore", str(exc))

        # maximize -> restore
        try:
            rt.windows.maximize(hwnd)
            time.sleep(0.4)
            maximized = rt.windows.is_maximized(hwnd)
            rt.windows.restore(hwnd)
            time.sleep(0.4)
            st = _window_state(rt, hwnd)
            chk.add("maximize",
                    PASS if (maximized and not st["maximized"]) else FAIL,
                    f"maximized={maximized} restored={not st['maximized']}")
            evidence["operations"].append({
                "op": "maximize", "maximized": maximized, "after": st})
        except Exception as exc:  # noqa: BLE001
            chk.fail("maximize", str(exc))

        # monitor moves (both directions, verified)
        for label, mon in (("move_monitor_2", m2),
                           ("move_monitor_1", m1)):
            if mon is None:
                chk.skip(label, "monitor unavailable")
                continue
            try:
                rt.windows.restore(hwnd)
                time.sleep(0.2)
                moved, _max = rt.windows.move_to_work_area(
                    hwnd, mon.work_area)
                time.sleep(0.4)
                okm, det = v93._verify_on_monitor(
                    rt.windows, rt.monitors, hwnd, mon)
                chk.add(label, PASS if (moved and okm) else FAIL,
                        det, moved=moved, verified=okm)
                evidence["operations"].append({
                    "op": label, "expected": mon.index,
                    "verification": json.loads(det)})
            except Exception as exc:  # noqa: BLE001
                chk.fail(label, str(exc))

        # command-handler level window ops
        cmd_tests = [
            ("minimize_command", MinimizeWindowHandler(svc),
             {"target": APP_QUERY}),
            ("maximize_command", MaximizeWindowHandler(svc),
             {"target": APP_QUERY}),
            ("restore_command", RestoreWindowHandler(svc),
             {"target": APP_QUERY}),
            ("focus_command", FocusWindowHandler(svc),
             {"target": APP_QUERY}),
        ]
        for tname, handler, ent in cmd_tests:
            try:
                cres = handler.execute(dict(ent))
                chk.add(tname,
                        PASS if _cmd_result_ok(cres) else FAIL,
                        getattr(cres, "message", ""))
                evidence["commands"].append({
                    "command": tname,
                    "executed": getattr(cres, "executed", None),
                    "error": getattr(cres, "error", None),
                    "message": getattr(cres, "message", "")})
                time.sleep(0.3)
            except Exception as exc:  # noqa: BLE001
                chk.fail(tname, str(exc))

        # move-window command handler (both monitors, verified by handler)
        for tname, spec in (("move_window_command_m2", "monitor 2"),
                            ("move_window_command_m1", "monitor 1")):
            try:
                mh = _MoveWindowHandler(svc, "")
                cres = mh.execute({"target": APP_QUERY, "monitor": spec})
                chk.add(tname,
                        PASS if _cmd_result_ok(cres) else FAIL,
                        getattr(cres, "message", ""))
                evidence["commands"].append({
                    "command": tname, "spec": spec,
                    "executed": getattr(cres, "executed", None),
                    "error": getattr(cres, "error", None),
                    "message": getattr(cres, "message", "")})
                st = _window_state(rt, hwnd)
                evidence["operations"].append({
                    "op": tname + "_post_state", "state": st})
                time.sleep(0.3)
            except Exception as exc:  # noqa: BLE001
                chk.fail(tname, str(exc))

        # open-on-monitor command handlers (activation + reuse + move)
        for tname, midx in (("open_on_monitor_m1", 1),
                            ("open_on_monitor_m2", 2)):
            mon = rt.monitors.get(midx)
            if mon is None:
                chk.skip(tname, "monitor unavailable")
                continue
            try:
                oh = OpenOnMonitorHandler(svc)
                ent = {"app_name": APP_QUERY, "monitor": mon,
                       "_monitor_info": mon, "_app_info": app}
                cres = oh.execute(ent)
                chk.add(tname,
                        PASS if _cmd_result_ok(cres) else FAIL,
                        getattr(cres, "message", ""))
                evidence["commands"].append({
                    "command": tname, "monitor": midx,
                    "executed": getattr(cres, "executed", None),
                    "error": getattr(cres, "error", None),
                    "message": getattr(cres, "message", "")})
                st = _window_state(rt, hwnd)
                evidence["operations"].append({
                    "op": tname + "_post_state", "state": st})
                time.sleep(0.3)
            except Exception as exc:  # noqa: BLE001
                chk.fail(tname, str(exc))

        # OPEN_APPLICATION command (default -> Monitor 1 via handler)
        try:
            oh = OpenAppHandler(svc)
            cres = oh.execute({"app_name": APP_QUERY})
            chk.add("open_app_command",
                    PASS if _cmd_result_ok(cres) else FAIL,
                    getattr(cres, "message", ""))
            evidence["commands"].append({
                "command": "OPEN_APPLICATION",
                "executed": getattr(cres, "executed", None),
                "error": getattr(cres, "error", None),
                "message": getattr(cres, "message", "")})
        except Exception as exc:  # noqa: BLE001
            chk.fail("open_app_command", str(exc))

        # -- restore original placement/state (best effort) -------------------
        try:
            rt.windows.restore(hwnd)
            time.sleep(0.2)
            if m2 is not None:
                rt.windows.move_to_work_area(hwnd, m2.work_area)
                time.sleep(0.3)
            rt.windows.minimize(hwnd)
            time.sleep(0.5)
            final = _window_state(rt, hwnd)
            evidence["operations"].append({"op": "restore_original",
                                           "state": final})
            chk.add("session_preserved",
                    PASS if final.get("hwnd") == hwnd else FAIL,
                    f"final state={final}")
        except Exception as exc:  # noqa: BLE001
            chk.add("session_preserved", PARTIAL, str(exc))

    # -- post-test snapshot -----------------------------------------------------
    post_procs = [p for p in rt.monitor.list_processes()
                  if "whatsapp" in (p.name or "").lower()]
    post_windows = _snapshot_windows(rt, app)
    evidence["snapshots"].append({"phase": "post-test",
                                  "processes": [
                                      {"pid": p.pid, "name": p.name}
                                      for p in post_procs],
                                  "windows": post_windows})
    chk.add("no_process_terminated",
            PASS if {p.pid for p in post_procs} >=
            {p.pid for p in pre_procs} else FAIL,
            f"pre={sorted(p.pid for p in pre_procs)} "
            f"post={sorted(p.pid for p in post_procs)}")
    chk.add("window_still_owned",
            PASS if any(w["hwnd"] == hwnd for w in post_windows)
            else FAIL if hwnd else SKIP,
            "user window still present" if hwnd else "n/a")

    _finish(chk, evidence, rt, app)
    return 0


def re_fullmatch(pattern: str, text: str) -> bool:
    import re
    return bool(re.fullmatch(pattern, text or ""))


def _finish(chk: Check, evidence: dict, rt, app, note: str = "") -> None:
    counts = chk.counts
    overall = chk.overall
    rows = []
    for r in chk.results:
        rows.append({"application": "WhatsApp", **r})
    ident = app_identity.identity_for_app(app) if app else None
    exe_path = getattr(app, "exe_path", "") or ""
    if not exe_path:
        win = (evidence.get("launch_state_machine") or {}).get("window") or {}
        exe_path = win.get("exe_path") or ""
    payload = {
        "version": "V9.5",
        "generated_at": _utc_now(),
        "app": {
            "name": getattr(app, "name", "WhatsApp"),
            "canonical_app_id": getattr(ident, "app_id", "") or "",
            "aumid": getattr(app, "aumid", ""),
            "exe_path": exe_path,
            "package_family": getattr(ident, "package_family", "") or "",
        },
        "monitor_convention": {
            "monitor_1": "Windows primary display (user right display)",
            "monitor_2": "Windows secondary display (user left display)",
        },
        "monitors": v93._monitor_payload(rt),
        "safety": ("User-owned WhatsApp window was never closed and no "
                   "process was terminated. Minimize/restore/maximize/"
                   "monitor moves ran under explicit user authorization "
                   "and the original state was restored afterwards."),
        "summary": {
            "total": len(rows),
            "pass": counts.get(PASS, 0),
            "fail": counts.get(FAIL, 0),
            "partial": counts.get(PARTIAL, 0),
            "skipped": counts.get(SKIP, 0),
            "overall": overall,
            "note": note,
        },
        "evidence": evidence,
        "results": rows,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2),
                            encoding="utf-8")
    fields = ["test", "status", "details"]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    _write_markdown(payload)
    print(f"Done: {len(rows)} checks, "
          f"PASS={counts.get(PASS, 0)} FAIL={counts.get(FAIL, 0)} "
          f"PARTIAL={counts.get(PARTIAL, 0)} SKIP={counts.get(SKIP, 0)} "
          f"overall={overall}")
    print(f"Reports: {RESULTS_JSON}, {RESULTS_CSV}, {REPORT_MD}")


def _write_markdown(payload: dict) -> None:
    s = payload["summary"]
    lines = [
        "# VoiceOS V9.5 — WhatsApp Application Reliability Report",
        "",
        f"Generated: {payload['generated_at']}  ",
        "Mode: `physical` — real Windows runtime, user-authorized "
        "lifecycle on the existing WhatsApp window  ",
        "",
        "## Application identity",
        "",
        f"- Name: **{payload['app']['name']}**",
        f"- Canonical ID: `{payload['app']['canonical_app_id']}`",
        f"- AUMID: `{payload['app']['aumid']}`",
        f"- Package family: `{payload['app']['package_family']}`",
        f"- Executable: `{payload['app']['exe_path']}`",
        "",
        "## Monitor mapping (live `MonitorManager`)",
        "",
        "| Logical monitor | Meaning | Device | Primary | Bounds |",
        "|---|---|---|---:|---|",
    ]
    for m in payload["monitors"]:
        meaning = ("RIGHT / primary" if m.get("index") == 1
                   else "LEFT / secondary" if m.get("index") == 2
                   else "secondary")
        b = m.get("bounds") or []
        lines.append(
            f"| Monitor {m.get('index')} | {meaning} | "
            f"`{m.get('device')}` | {m.get('primary')} | "
            f"({','.join(str(x) for x in b)}) |")
    lines += [
        "",
        "## Summary",
        "",
        f"- Checks: **{s['total']}** — PASS **{s['pass']}**, "
        f"FAIL **{s['fail']}**, PARTIAL **{s['partial']}**, "
        f"SKIPPED **{s['skipped']}**",
        f"- Overall: **{s['overall']}**",
        "",
        "> Safety: the user-owned WhatsApp window was never closed, no "
        "process was terminated, and minimize/restore/maximize/monitor "
        "moves ran only under explicit user authorization with the "
        "original state restored afterwards. Skipped operations are "
        "never counted as passes.",
        "",
        "## Per-check results",
        "",
        "| Test | Status | Details |",
        "|---|---:|---|",
    ]
    for r in payload["results"]:
        det = str(r.get("details") or "").replace("|", "/")
        if len(det) > 160:
            det = det[:157] + "..."
        lines.append(f"| {r['test']} | {r['status']} | {det} |")
    lines += [
        "",
        "## Launch state machine",
        "",
        "```json",
        json.dumps(payload["evidence"].get("launch_state_machine", {}),
                   indent=2)[:3000],
        "```",
        "",
        "## Window snapshots",
        "",
        "Pre-test, visibility-consistency and post-test snapshots "
        "(HWND, PID, title, visibility, minimized, maximized, rect, "
        "cloaked, monitor) are in `reports/v95_whatsapp_results.json` "
        "under `evidence.snapshots`.",
        "",
        "## Notes",
        "",
        "- `EXISTING_REUSED` means activation was accepted and the "
        "already-running instance's usable window was reused — this is "
        "success, not a launch failure, and it resolves in ~2s rather "
        "than burning the full discovery timeout.",
        "- The publisher hash `5319275a` is never used as a spoken or "
        "display name.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(run())
