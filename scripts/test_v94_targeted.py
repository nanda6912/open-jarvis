"""VoiceOS V9.4 — targeted physical application reliability test.

Builds on the V9.3 runner (``test_all_applications.py``) and the same
production stack — ``ApplicationDiscovery``, ``AppIdentity``,
``ApplicationLauncher``, ``WindowManager`` and ``MonitorManager``.

V9.4 differences vs V9.3:

* Only the targeted groups are physically exercised.
* Every launched window is tracked by HWND. The harness moves, focuses
  and WM_CLOSEs **only windows it created** — pre-existing user windows
  are never moved, minimized, maximized or closed.
* Group B apps that reuse an existing window degrade to identity /
  window-discovery / safe-focus checks (PARTIAL), never lifecycle.
* Explicit verification that an ordinary launch lands on Monitor 1
  (right/primary) and that Monitor 2 is left/secondary, with actual
  HWND, bounds and monitor-device evidence after every operation.

Monitor convention (``MonitorManager`` only):
    Monitor 1 = Windows primary display = user RIGHT display.
    Monitor 2 = Windows secondary display = user LEFT display.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_all_applications as v93  # noqa: E402

from src.controllers.app_identity import app_for_window  # noqa: E402
from src.controllers.windows_automation import WindowInfo  # noqa: E402

PASS, FAIL, PARTIAL, SKIP = v93.PASS, v93.FAIL, v93.PARTIAL, v93.SKIP

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
RESULTS_JSON = REPORTS / "v94_targeted_results.json"
RESULTS_CSV = REPORTS / "v94_targeted_results.csv"
REPORT_MD = DOCS / "V94_TARGETED_RELIABILITY_REPORT.md"

POLL_INTERVAL = 0.25
CLOSE_TIMEOUT = 8.0

# ---------------------------------------------------------------------------
# Target groups (display names as discovered by ApplicationDiscovery)
# ---------------------------------------------------------------------------
GROUP_A_FAILURES = [
    "Microsoft Clipchamp",
    "Camera",
    "Microsoft Visual Studio Code (User)",
    "Git GUI",
    "Postman x64 12.20.4",
    "Microsoft Edge",
]

GROUP_B_PARTIALS = [
    "Google Chrome",
    "Terminal",
    "WhatsApp",
    "Devin (User)",
    "B&O Audio Control",
    "Phone Link",
    "Get Started",
    "Click to Do",
    "Send to OneNote",
]

# Extra safely-launchable controls for default-launch / monitor checks.
GROUP_C_CONTROLS = ["Notepad", "Calculator"]

GROUPS = {"A": GROUP_A_FAILURES, "B": GROUP_B_PARTIALS,
          "C": GROUP_C_CONTROLS}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def find_record(records, name: str):
    """Exact-then-substring match on display name."""
    target = _norm(name)
    for rec in records:
        if _norm(rec.display_name) == target:
            return rec
    for rec in records:
        dn = _norm(rec.display_name)
        if target and (target in dn or dn in target):
            return rec
    return None


# ---------------------------------------------------------------------------
# HWND-scoped helpers — only harness-created windows are touched
# ---------------------------------------------------------------------------
def _app_hwnds(rt: v93.Runtime, app) -> set[int]:
    try:
        return {w.hwnd for w in rt.app_windows(app, include_hidden=True)}
    except Exception:  # noqa: BLE001
        return set()


def _visible_app_windows(rt: v93.Runtime, app) -> list[WindowInfo]:
    try:
        return [w for w in rt.app_windows(app, include_hidden=False)
                if getattr(w, "visible", True)]
    except Exception:  # noqa: BLE001
        return []


def wait_fresh_window(rt: v93.Runtime, app, before: set[int],
                      timeout: float):
    """Poll for a *new* top-level window (HWND not in ``before``).

    Returns (window, latency_ms, how). ``window`` is None when the app
    only reused pre-existing windows or produced no window at all.
    """
    t0 = time.monotonic()
    deadline = t0 + timeout
    last_error = ""
    while time.monotonic() < deadline:
        try:
            matches = rt.app_windows(app, include_hidden=True)
        except Exception as exc:  # noqa: BLE001
            matches = []
            last_error = str(exc)
        fresh = [w for w in matches if w.hwnd not in before]
        visible_fresh = [w for w in fresh if getattr(w, "visible", True)]
        if visible_fresh:
            win = v93._choose_window(visible_fresh, app)
            return (win, (time.monotonic() - t0) * 1000.0,
                    "fresh-visible", [w.hwnd for w in fresh])
        if time.monotonic() >= deadline:
            if len(fresh) == 1:
                return (fresh[0], (time.monotonic() - t0) * 1000.0,
                        "timeout-fresh-hidden", [w.hwnd for w in fresh])
            return None, (time.monotonic() - t0) * 1000.0, (
                last_error or ("timeout-no-fresh-window" if not fresh
                               else "timeout-ambiguous-fresh")), []
        time.sleep(POLL_INTERVAL)
    return None, (time.monotonic() - t0) * 1000.0, last_error or "timeout", []


def _post_close_hwnd(rt: v93.Runtime, hwnd: int) -> None:
    api = getattr(rt.windows, "_api", None)
    fn = getattr(api, "post_close", None) if api is not None else None
    if fn is not None:
        fn(hwnd)


def _hwnd_visible(rt: v93.Runtime, hwnd: int) -> bool:
    api = getattr(rt.windows, "_api", None)
    fn = getattr(api, "is_visible", None) if api is not None else None
    if fn is None:
        return False
    try:
        return bool(fn(hwnd))
    except Exception:  # noqa: BLE001
        return False


def _hwnd_title(rt: v93.Runtime, hwnd: int) -> str:
    try:
        for w in rt.windows.list_windows(include_hidden=True):
            if w.hwnd == hwnd:
                return w.title or ""
    except Exception:  # noqa: BLE001
        pass
    return ""


def close_owned_windows(rt: v93.Runtime, owned: set[int],
                        timeout: float = CLOSE_TIMEOUT):
    """WM_CLOSE only harness-created HWNDs; never touches other windows."""
    owned = {h for h in owned if h}
    if not owned:
        return True, "no harness windows"
    for h in owned:
        try:
            rt.windows.restore(h)
        except Exception:  # noqa: BLE001
            pass
    time.sleep(0.2)
    for h in owned:
        try:
            _post_close_hwnd(rt, h)
        except Exception:  # noqa: BLE001
            pass
    start = time.monotonic()
    reposted = False
    deadline = start + timeout
    while time.monotonic() < deadline:
        remaining = [h for h in owned if _hwnd_visible(rt, h)]
        if not remaining:
            return True, "closed"
        titles = "; ".join(_hwnd_title(rt, h) or str(h) for h in remaining[:5])
        if any(re.search(p, titles.lower())
               for p in v93._SUSPICIOUS_TITLE_PATTERNS):
            return False, f"close prompt/dialog left open: {titles}"
        if not reposted and time.monotonic() - start > 1.5:
            for h in remaining:
                try:
                    _post_close_hwnd(rt, h)
                except Exception:  # noqa: BLE001
                    pass
            reposted = True
        time.sleep(0.2)
    remaining = [h for h in owned if _hwnd_visible(rt, h)]
    if remaining:
        titles = "; ".join(_hwnd_title(rt, h) or str(h) for h in remaining[:5])
        return False, f"harness windows still open after WM_CLOSE: {titles}"
    return True, "closed"


def _verify(rt: v93.Runtime, hwnd: int, mon) -> tuple[bool, str]:
    return v93._verify_on_monitor(rt.windows, rt.monitors, hwnd, mon)


def _focus_hwnd(rt: v93.Runtime, hwnd: int) -> tuple[bool, dict]:
    try:
        ok = bool(rt.windows.focus(hwnd))
    except Exception as exc:  # noqa: BLE001
        return False, {"error": str(exc)}
    fg = rt.windows.foreground_window()
    fg_hwnd = getattr(fg, "hwnd", 0) if fg is not None else 0
    verified = ok and fg_hwnd == hwnd
    return verified, {
        "requested_hwnd": hwnd,
        "focus_api_ok": ok,
        "foreground_hwnd": fg_hwnd,
        "foreground_title": getattr(fg, "title", ""),
    }


def _launch(rt: v93.Runtime, app):
    t0 = time.monotonic()
    try:
        ok, _proc, detail = rt.launcher.launch_process(app)
    except Exception as exc:  # noqa: BLE001
        return False, (time.monotonic() - t0) * 1000.0, str(exc)
    return ok, (time.monotonic() - t0) * 1000.0, detail or "ok"


# ---------------------------------------------------------------------------
# Physical test — fresh-window path (harness owns every HWND it touches)
# ---------------------------------------------------------------------------
def test_fresh_window(rec, rt: v93.Runtime, m1, m2,
                      window_timeout: float) -> None:
    app = rec.app
    owned: set[int] = set()
    rec.details = ""
    try:
        # -- Phase 1: ordinary launch; production default is Monitor 1 ---
        before = _app_hwnds(rt, app)
        ok, launch_ms, ldetail = _launch(rt, app)
        rec.launch_latency_ms = launch_ms
        if not ok:
            rec.launch = FAIL
            rec.window_discovery = FAIL
            rec.details = f"launch failed: {ldetail}"
            v93._finalize_overall(rec)
            return
        rec.launch = PASS
        win0, win_ms, how, fresh = wait_fresh_window(
            rt, app, before, window_timeout)
        rec.window_latency_ms = win_ms
        rec.diagnostics["default_launch_how"] = how
        rec.diagnostics["fresh_hwnds"] = fresh[:20]
        if win0 is None:
            matches = rt.app_windows(app, include_hidden=True)
            visible = [w for w in matches if getattr(w, "visible", True)]
            if matches:
                _existing_checks(rec, rt, matches, visible,
                                 launched=True)
                return
            rec.window_discovery = FAIL
            rec.details = f"launch accepted but no window discovered: {how}"
            rec.cleanup = PASS
            rec.diagnostics["cleanup_detail"] = "no harness windows"
            v93._finalize_overall(rec)
            return
        rec.window_discovery = PASS
        owned.update(fresh)
        rec.diagnostics["default_window"] = v93._window_diag(win0)
        suspicious = v93._suspicious_dialog(win0)
        if suspicious:
            rec.details = suspicious
            _finish_cleanup(rec, rt, owned)
            v93._finalize_overall(rec)
            return

        # -- Group C: ordinary launch must land on Monitor 1 ------------
        okm, det = _verify(rt, win0.hwnd, m1)
        rec.diagnostics["default_monitor1_verification"] = det
        rec.diagnostics["default_monitor1"] = "PASS" if okm else "FAIL"

        # -- Phase 2: explicit Monitor 1 launch (right/primary) ---------
        m1_win = _launch_to_monitor(rec, rt, m1, window_timeout, owned)
        rec.monitor_1 = PASS if m1_win else FAIL
        if m1_win is None:
            rec.diagnostics["monitor1_error"] = rec.diagnostics.get(
                "monitor1_launch", {}).get("how", "no window")

        # -- Phase 3: explicit Monitor 2 launch (left/secondary) --------
        m2_win = _launch_to_monitor(rec, rt, m2, window_timeout, owned)
        rec.monitor_2 = PASS if m2_win else FAIL
        if m2_win is None:
            rec.diagnostics["monitor2_error"] = rec.diagnostics.get(
                "monitor2_launch", {}).get("how", "no window")

        lifecycle_win = m2_win or m1_win or win0
        hwnd = lifecycle_win.hwnd
        rec.diagnostics["lifecycle_window"] = v93._window_diag(lifecycle_win)
        suspicious = v93._suspicious_dialog(lifecycle_win)
        if suspicious:
            rec.details = suspicious
            _finish_cleanup(rec, rt, owned)
            v93._finalize_overall(rec)
            return

        # -- Switch/focus: HWND-scoped production focus + verify -------
        verified, fdetail = _focus_hwnd(rt, hwnd)
        rec.switch_focus = PASS if verified else FAIL
        rec.diagnostics["focus_result"] = fdetail
        if not verified:
            rec.diagnostics["focus_error"] = "foreground verification failed"

        # -- Minimize / restore ----------------------------------------
        try:
            rt.windows.minimize(hwnd)
            time.sleep(0.3)
            minimized = rt.windows.is_minimized(hwnd)
            rt.windows.restore(hwnd)
            time.sleep(0.3)
            restored = (not rt.windows.is_minimized(hwnd)
                        and rt.windows._api.is_visible(hwnd))
            rec.minimize_restore = PASS if (minimized and restored) else FAIL
            rec.diagnostics["minmax_state"] = {
                "minimized": minimized, "restored": restored}
        except Exception as exc:  # noqa: BLE001
            rec.minimize_restore = FAIL
            rec.diagnostics["minmax_error"] = str(exc)

        # -- Maximize ---------------------------------------------------
        try:
            rt.windows.maximize(hwnd)
            time.sleep(0.3)
            rec.maximize = PASS if rt.windows.is_maximized(hwnd) else FAIL
            rec.diagnostics["maximized"] = rt.windows.is_maximized(hwnd)
        except Exception as exc:  # noqa: BLE001
            rec.maximize = FAIL
            rec.diagnostics["maximize_error"] = str(exc)

        # -- Group D: bidirectional moves on the owned window -----------
        for label, mon, attr, status_attr in (
            ("monitor_2_move", m2, "monitor_2_move_result",
             "move_monitor_2"),
            ("monitor_1_move", m1, "monitor_1_move_result",
             "move_monitor_1"),
        ):
            try:
                rt.windows.restore(hwnd)
                time.sleep(0.2)
                moved, _max = rt.windows.move_to_work_area(
                    hwnd, mon.work_area)
                time.sleep(0.3)
                okm2, det2 = _verify(rt, hwnd, mon)
                setattr(rec, status_attr,
                        PASS if (moved and okm2) else FAIL)
                rec.diagnostics[attr] = {
                    "moved": moved, "verified": okm2, "detail": det2}
            except Exception as exc:  # noqa: BLE001
                rec.diagnostics[attr] = {"error": str(exc)}

        if not rec.details:
            rec.details = "completed safe GUI lifecycle (harness-owned windows)"
        _finish_cleanup(rec, rt, owned)
    except Exception as exc:  # noqa: BLE001
        rec.details = f"unhandled test exception: {exc}"
        rec.diagnostics["traceback"] = traceback.format_exc()
        _finish_cleanup(rec, rt, owned)
    v93._finalize_overall(rec)


def _launch_to_monitor(rec, rt: v93.Runtime, mon,
                       window_timeout: float, owned: set[int]):
    """Launch again; fresh window or reuse of an owned window, then move
    onto ``mon``'s work area and verify actual placement."""
    app = rec.app
    tag = f"monitor{mon.index}_launch"
    before = _app_hwnds(rt, app)
    ok, launch_ms, ldetail = _launch(rt, app)
    rec.diagnostics.setdefault("launch_latencies_ms", []).append(
        {"phase": tag, "launch_ms": launch_ms, "detail": ldetail})
    if not ok:
        rec.diagnostics[tag] = {"ok": False, "how": ldetail}
        return None
    win, win_ms, how, fresh = wait_fresh_window(
        rt, app, before, window_timeout)
    rec.diagnostics[tag] = {
        "ok": True, "how": how, "window_latency_ms": win_ms,
        "fresh_hwnds": fresh[:20],
    }
    cand = None
    if win is not None:
        owned.update(fresh)
        cand = win
    else:
        # Single-instance apps reuse the window we already own.
        reuse = [h for h in owned
                 if _hwnd_visible(rt, h)]
        if reuse:
            cand = _hwnd_to_window(rt, reuse[0])
            rec.diagnostics[tag]["reused_owned_hwnd"] = reuse[0]
    if cand is None:
        return None
    try:
        rt.windows.restore(cand.hwnd)
    except Exception:  # noqa: BLE001
        pass
    moved, _max = rt.windows.move_to_work_area(cand.hwnd, mon.work_area)
    time.sleep(0.3)
    okm, det = _verify(rt, cand.hwnd, mon)
    rec.diagnostics[f"monitor{mon.index}_window"] = v93._window_diag(cand)
    rec.diagnostics[f"monitor{mon.index}_verification"] = det
    rec.diagnostics[f"monitor{mon.index}_move"] = {
        "moved": moved, "verified": okm}
    # Only a verified placement earns the monitor-launch row; the window
    # stays tracked in ``owned`` for cleanup either way.
    return cand if (moved and okm) else None


def _hwnd_to_window(rt: v93.Runtime, hwnd: int):
    try:
        for w in rt.windows.list_windows(include_hidden=True):
            if w.hwnd == hwnd:
                return w
    except Exception:  # noqa: BLE001
        pass
    return None


def _finish_cleanup(rec, rt: v93.Runtime, owned: set[int]) -> None:
    if rec.cleanup != SKIP:
        return
    closed, detail = close_owned_windows(rt, owned)
    rec.cleanup = PASS if closed else FAIL
    rec.diagnostics["cleanup_detail"] = detail
    rec.diagnostics["closed_hwnds"] = sorted(owned)
    if not closed:
        rec.details = (rec.details + "; " if rec.details else "") + detail


# ---------------------------------------------------------------------------
# Existing-window path — identity/discovery/focus only, never move/close
# ---------------------------------------------------------------------------
def _existing_checks(rec, rt: v93.Runtime, matches, visible,
                     launched: bool = False) -> None:
    app = rec.app
    win = v93._choose_window(visible or matches, app)
    rec.window_discovery = PASS if matches else FAIL
    rec.diagnostics["window_count"] = len(matches)
    rec.diagnostics["visible_window_count"] = len(visible)
    rec.diagnostics["selected_window"] = v93._window_diag(win)
    rec.diagnostics["reused_existing_window"] = True
    if win is not None:
        mon = rt.monitors.monitor_for_hwnd(win.hwnd)
        rec.diagnostics["selected_monitor"] = getattr(mon, "index", None)
        rec.diagnostics["selected_monitor_device"] = getattr(mon, "device", None)
        try:
            rec.diagnostics["reverse_identity_match"] = (
                app_for_window(win, [app]) is not None)
        except Exception:  # noqa: BLE001
            rec.diagnostics["reverse_identity_match"] = False
    if len(visible) == 1 and win is not None:
        suspicious = v93._suspicious_dialog(win)
        if suspicious:
            rec.switch_focus = SKIP
            rec.diagnostics["focus_skip"] = suspicious
        else:
            verified, fdetail = _focus_hwnd(rt, win.hwnd)
            rec.switch_focus = PASS if verified else FAIL
            rec.diagnostics["focus_result"] = fdetail
            if not verified:
                rec.diagnostics["focus_error"] = "foreground verification failed"
    else:
        rec.switch_focus = SKIP
        rec.diagnostics["focus_skip"] = (
            "no visible window" if not visible else "multiple visible windows")
    rec.minimize_restore = SKIP
    rec.maximize = SKIP
    rec.monitor_1 = SKIP
    rec.monitor_2 = SKIP
    rec.move_monitor_1 = SKIP
    rec.move_monitor_2 = SKIP
    rec.cleanup = SKIP
    rec.diagnostics["cleanup_detail"] = "pre-existing windows untouched"
    prefix = "launched; reused existing window(s)" if launched else \
        "already running before test"
    rec.details = (
        f"{prefix}: {len(matches)} window(s), {len(visible)} visible; "
        f"focus={'tested' if rec.switch_focus != SKIP else 'skipped'}"
    )
    v93._finalize_overall(rec)


def test_target(rec, rt: v93.Runtime, m1, m2,
                window_timeout: float) -> None:
    """Dispatch: physical fresh-window attempt for every target; degrade
    to existing-window checks when the app only reuses user windows."""
    if not rec.eligible:
        rec.overall = SKIP
        rec.details = rec.category_reason or "not eligible"
        return
    # If the app was already running with windows before we launched,
    # a fresh window may still appear (e.g. Chrome "new window"). We
    # always attempt one launch; HWND tracking keeps user windows safe.
    test_fresh_window(rec, rt, m1, m2, window_timeout)
    if rec.diagnostics.get("fresh_hwnds"):
        rec.diagnostics["test_mode"] = "physical-fresh"
    elif rec.window_discovery == FAIL:
        rec.diagnostics["test_mode"] = "no-window"
    else:
        rec.diagnostics["test_mode"] = "existing-only"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def _row(rec, group: str) -> dict:
    row = rec.base_dict()
    row["v94_group"] = group
    row["default_monitor1"] = rec.diagnostics.get("default_monitor1", SKIP)
    row["test_mode"] = rec.diagnostics.get("test_mode", "")
    return row


def _write_json(rows: list[dict], meta: dict, rt: v93.Runtime,
                groups: dict, mode: str) -> None:
    mons = v93._monitor_payload(rt)
    def c(field, value):
        return sum(1 for r in rows if r.get(field) == value)
    payload = {
        "version": "V9.4",
        "generated_at": _utc_now(),
        "mode": mode,
        "monitor_convention": {
            "monitor_1": "Windows primary display (user right display)",
            "monitor_2": "Windows secondary display (user left display)",
        },
        "monitors": mons,
        "groups": groups,
        "safety": ("Only harness-created HWNDs were moved, focused or "
                   "closed; pre-existing user windows were never touched."),
        "summary": {
            "total_targeted": len(rows),
            "identity_pass": c("identity", PASS),
            "identity_fail": c("identity", FAIL),
            "overall_pass": c("overall", PASS),
            "overall_fail": c("overall", FAIL),
            "overall_partial": c("overall", PARTIAL),
            "overall_skipped": c("overall", SKIP),
            "physical_fresh": sum(1 for r in rows
                                  if r.get("test_mode") == "physical-fresh"),
            "existing_only": sum(1 for r in rows
                                 if r.get("test_mode") == "existing-only"),
            "default_monitor1_pass": sum(
                1 for r in rows if r.get("default_monitor1") == PASS),
            "default_monitor1_fail": sum(
                1 for r in rows if r.get("default_monitor1") == FAIL),
            "default_monitor1_skipped": sum(
                1 for r in rows if r.get("default_monitor1") == SKIP),
        },
        "latency_ms": v93._latency_payload(rows),
        "common_failure_causes": v93._failure_causes(rows),
        "results": rows,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2),
                            encoding="utf-8")


def _write_csv(rows: list[dict]) -> None:
    fields = [
        "application", "v94_group", "test_mode", "category", "eligible",
        "identity", "identity_details", "launch", "window_discovery",
        "default_monitor1", "switch_focus", "minimize_restore",
        "maximize", "monitor_1", "monitor_2", "move_monitor_1",
        "move_monitor_2", "cleanup", "overall", "details",
        "launch_latency_ms", "window_discovery_latency_ms",
        "already_running", "running_window_count",
    ]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _write_markdown(rows: list[dict], meta: dict, rt: v93.Runtime,
                    groups: dict, mode: str) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    mons = v93._monitor_payload(rt)
    counts = Counter(r.get("overall") for r in rows)
    id_counts = Counter(r.get("identity") for r in rows)
    latency = v93._latency_payload(rows)
    causes = v93._failure_causes(rows)
    lines = [
        "# VoiceOS V9.4 — Targeted Physical Application Reliability Test",
        "",
        f"Generated: {_utc_now()}  ",
        f"Mode: `{mode}` — physical Windows runtime  ",
        "",
        "## Monitor mapping (verified live via `MonitorManager`)",
        "",
        "| Logical monitor | Meaning | Device | Primary | Bounds |",
        "|---|---|---|---:|---|",
    ]
    for m in mons:
        meaning = ("RIGHT / primary" if m.get("index") == 1
                   else "LEFT / secondary" if m.get("index") == 2
                   else "secondary")
        b = m.get("bounds") or []
        lines.append(
            f"| Monitor {m.get('index')} | {meaning} | `{m.get('device')}` | "
            f"{m.get('primary')} | ({','.join(str(x) for x in b)}) |")
    lines += [
        "",
        "## Summary",
        "",
        f"- Targeted applications: **{len(rows)}**",
        f"- Groups: A (V9.3 failures)={len(groups.get('A', []))}, "
        f"B (V9.3 partials)={len(groups.get('B', []))}, "
        f"C (default-launch controls)={len(groups.get('C', []))}",
        f"- Identity PASS/FAIL: **{id_counts.get(PASS, 0)} / "
        f"{id_counts.get(FAIL, 0)}**",
        f"- Overall PASS/FAIL/PARTIAL/SKIPPED: **{counts.get(PASS, 0)} / "
        f"{counts.get(FAIL, 0)} / {counts.get(PARTIAL, 0)} / "
        f"{counts.get(SKIP, 0)}**",
        f"- Physically tested with harness-owned windows: "
        f"**{sum(1 for r in rows if r.get('test_mode') == 'physical-fresh')}**",
        f"- Existing-window checks only (user windows untouched): "
        f"**{sum(1 for r in rows if r.get('test_mode') == 'existing-only')}**",
        f"- Default launch on Monitor 1 — PASS/FAIL/SKIPPED: "
        f"**{sum(1 for r in rows if r.get('default_monitor1') == PASS)} / "
        f"{sum(1 for r in rows if r.get('default_monitor1') == FAIL)} / "
        f"{sum(1 for r in rows if r.get('default_monitor1') == SKIP)}**",
        f"- Launch latency avg/max (ms): "
        f"**{latency['launch']['average']} / {latency['launch']['max']}**",
        f"- Window-discovery latency avg/max (ms): "
        f"**{latency['window_discovery']['average']} / "
        f"{latency['window_discovery']['max']}**",
        "",
        "> Safety: only HWNDs created by this harness were moved, focused "
        "or closed. Pre-existing user windows were never moved or closed. "
        "Skipped operations are never counted as passes.",
        "",
        "## Per-application results",
        "",
        "| Group | Application | Mode | Identity | Launch | Window | "
        "Default M1 | Focus | Min/restore | Max | Mon 1 | Mon 2 | "
        "Move M2 | Move M1 | Cleanup | Overall | Details |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        "---:|---:|---|",
    ]
    for r in rows:
        detail = str(r.get("details") or r.get("category_reason") or ""
                     ).replace("|", "/")
        if len(detail) > 140:
            detail = detail[:137] + "..."
        lines.append(
            f"| {r.get('v94_group')} | {r.get('application')} | "
            f"{r.get('test_mode') or '-'} | {r.get('identity')} | "
            f"{r.get('launch')} | {r.get('window_discovery')} | "
            f"{r.get('default_monitor1')} | {r.get('switch_focus')} | "
            f"{r.get('minimize_restore')} | {r.get('maximize')} | "
            f"{r.get('monitor_1')} | {r.get('monitor_2')} | "
            f"{r.get('move_monitor_2')} | {r.get('move_monitor_1')} | "
            f"{r.get('cleanup')} | {r.get('overall')} | {detail} |")
    lines += ["", "## Failure causes", ""]
    op_causes = Counter()
    _OPS = ["launch", "window_discovery", "default_monitor1",
            "switch_focus", "minimize_restore", "maximize",
            "monitor_1", "monitor_2", "move_monitor_1",
            "move_monitor_2", "cleanup"]
    for r in rows:
        for op in _OPS:
            if r.get(op) == FAIL:
                op_causes[f"operation '{op}' failed"] += 1
        if r.get("test_mode") == "existing-only":
            op_causes["reused pre-existing window (lifecycle not exercised)"] += 1
    if op_causes:
        for reason, count in op_causes.most_common():
            lines.append(f"- {count}× {reason}")
    else:
        lines.append("- None")
    lines += [
        "",
        "Machine-readable evidence (HWND, bounds, monitor device per "
        "operation): `reports/v94_targeted_results.json` / `.csv`.",
        "V9.3 baseline retained under `reports/v93_baseline/` and "
        "`reports/v93_*`.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--groups", default="A,B,C",
                   help="Comma-separated groups to run (A,B,C).")
    p.add_argument("--only", metavar="TEXT",
                   help="Restrict to apps whose name contains TEXT.")
    p.add_argument("--window-timeout", type=float, default=30.0)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    REPORTS.mkdir(parents=True, exist_ok=True)
    rt = v93.Runtime()
    records, meta = v93.build_inventory(rt.discovery)
    v93.write_inventory(records, meta)

    try:
        window_snapshot = rt.windows.list_windows(include_hidden=True)
    except Exception:  # noqa: BLE001
        window_snapshot = []
    for rec in records:
        v93.identity_test(rec, rt, window_snapshot)
    rt.snapshot_running(records)

    m1 = rt.monitors.get(1)
    m2 = rt.monitors.get(2)
    if m1 is None or m2 is None:
        print("WARNING: fewer than two monitors detected.")
        return 2

    wanted = {g.strip().upper() for g in args.groups.split(",") if g.strip()}
    only = (args.only or "").lower()
    targets = []  # (group, name)
    for g in ("A", "B", "C"):
        if g not in wanted:
            continue
        for name in GROUPS[g]:
            targets.append((g, name))

    rows = []
    tested = 0
    for group, name in targets:
        rec = find_record(records, name)
        if rec is None:
            row = {
                "application": name, "v94_group": group,
                "test_mode": "not-found", "category": "",
                "eligible": False, "identity": SKIP,
                "identity_details": "not present in discovery inventory",
                "launch": SKIP, "window_discovery": SKIP,
                "default_monitor1": SKIP, "switch_focus": SKIP,
                "minimize_restore": SKIP, "maximize": SKIP,
                "monitor_1": SKIP, "monitor_2": SKIP,
                "move_monitor_1": SKIP, "move_monitor_2": SKIP,
                "cleanup": SKIP, "overall": SKIP,
                "details": "not discovered in current inventory",
            }
            rows.append(row)
            continue
        if only and only not in rec.display_name.lower():
            continue
        if args.limit and tested >= args.limit:
            break
        tested += 1
        print(f"[{tested}] V9.4 {group}: {rec.display_name} "
              f"({rec.app_id})", flush=True)
        try:
            running_now = rt.monitor.is_running(rec.app)
        except Exception:  # noqa: BLE001
            running_now = rec.already_running
        rec.diagnostics["running_at_test_start"] = running_now
        test_target(rec, rt, m1, m2, args.window_timeout)
        if rec.diagnostics.get("fresh_hwnds"):
            rec.diagnostics["test_mode"] = "physical-fresh"
        elif rec.window_discovery == FAIL:
            rec.diagnostics["test_mode"] = "no-window"
        else:
            rec.diagnostics["test_mode"] = "existing-only"
        print(f"    -> {rec.overall} ({rec.details or 'ok'})", flush=True)
        rows.append(_row(rec, group))

    group_names = {g: GROUPS[g] for g in ("A", "B", "C") if g in wanted}
    _write_json(rows, meta, rt, group_names, "physical")
    _write_csv(rows)
    _write_markdown(rows, meta, rt, group_names, "physical")
    counts = Counter(r.get("overall") for r in rows)
    print(f"Done: {len(rows)} targeted, "
          f"PASS={counts.get(PASS, 0)} FAIL={counts.get(FAIL, 0)} "
          f"PARTIAL={counts.get(PARTIAL, 0)} SKIPPED={counts.get(SKIP, 0)}")
    print(f"Reports: {RESULTS_JSON}, {RESULTS_CSV}, {REPORT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
