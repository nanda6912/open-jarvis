"""VoiceOS V9.5.1 — WhatsApp state verification and acceptance.

Follow-up to ``test_v95_whatsapp.py``. Fixes two reporting gaps:

1. Maximize is now verified with the authoritative Win32 ``IsZoomed``
   API plus the actual window rectangle captured *while maximized*
   (V9.5 only recorded the post-restore state).
2. Original window state is captured as a full baseline (HWND, PID,
   visibility, minimized, maximized, zoomed, iconic, rect, normal
   rect, monitor device, cloaked, foreground) and restored in a
   ``finally`` block, then diffed field-by-field against the baseline.

Safety: identical to V9.5 — the user-owned WhatsApp window is never
closed, no process is terminated, lifecycle ops ran under explicit
user authorization for this task.

Deliverables:
    docs/V951_WHATSAPP_STATE_VERIFICATION.md
    reports/v951_whatsapp_state_results.json
    reports/v951_whatsapp_state_results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_all_applications as v93  # noqa: E402
import test_v95_whatsapp as v95  # noqa: E402

from src.controllers import app_identity  # noqa: E402
from src.commands.app_handlers import (  # noqa: E402
    _Svc, OpenAppHandler, SwitchToAppHandler, MinimizeWindowHandler,
    MaximizeWindowHandler, RestoreWindowHandler, FocusWindowHandler,
)
from src.commands.monitor_handlers import (  # noqa: E402
    OpenOnMonitorHandler, _MoveWindowHandler,
)

PASS, FAIL, PARTIAL, SKIP = v93.PASS, v93.FAIL, v93.PARTIAL, v93.SKIP
Check = v95.Check

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
RESULTS_JSON = REPORTS / "v951_whatsapp_state_results.json"
RESULTS_CSV = REPORTS / "v951_whatsapp_state_results.csv"
REPORT_MD = DOCS / "V951_WHATSAPP_STATE_VERIFICATION.md"

APP_QUERY = "WhatsApp"
POLL = 0.2
STATE_TIMEOUT = 4.0
RECT_TOLERANCE = 4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# State capture / polling / restoration
# ---------------------------------------------------------------------------
def capture_state(rt, hwnd: int) -> dict:
    """Full window-state snapshot used as the restore baseline."""
    st = v95._window_state(rt, hwnd)
    api = rt.windows._api
    try:
        st["is_zoomed"] = bool(rt.windows.is_zoomed(hwnd))
    except Exception:  # noqa: BLE001
        st["is_zoomed"] = None
    try:
        st["is_iconic"] = bool(rt.windows.is_iconic(hwnd))
    except Exception:  # noqa: BLE001
        st["is_iconic"] = None
    try:
        st["normal_rect"] = rt.windows.normal_rect(hwnd)
    except Exception:  # noqa: BLE001
        st["normal_rect"] = None
    st["foreground"] = (rt.windows.foreground_window() is not None
                        and rt.windows.foreground_window().hwnd == hwnd)
    return st


def poll_state(rt, hwnd: int, pred, timeout: float = STATE_TIMEOUT):
    """Poll ``capture_state`` until ``pred(state)`` or timeout."""
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < timeout:
        last = capture_state(rt, hwnd)
        try:
            if pred(last):
                return True, last, (time.monotonic() - t0) * 1000.0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(POLL)
    return False, last, (time.monotonic() - t0) * 1000.0


def _rect_close(a, b, tol: int = RECT_TOLERANCE) -> bool:
    if a is None or b is None:
        return a is b
    return all(abs(int(x) - int(y)) <= tol for x, y in zip(a, b))


def diff_states(baseline: dict, final: dict) -> dict:
    """Field-by-field comparison of baseline vs final state."""
    diffs = {}
    for key in ("visible", "minimized", "maximized", "is_zoomed",
                "is_iconic", "monitor_index", "monitor_device",
                "cloaked"):
        if baseline.get(key) != final.get(key):
            diffs[key] = {"baseline": baseline.get(key),
                          "final": final.get(key)}
    if not _rect_close(baseline.get("rect"), final.get("rect")):
        diffs["rect"] = {"baseline": baseline.get("rect"),
                         "final": final.get("rect")}
    return diffs


def restore_baseline(rt, hwnd: int, baseline: dict) -> dict:
    """Best-effort restore of the captured baseline state.

    Order: un-minimize/un-maximize -> restore geometry -> apply the
    original show state. Never closes the window or kills the process.
    Returns the post-restore capture.
    """
    try:
        rt.windows.restore(hwnd)
        time.sleep(0.3)
    except Exception:  # noqa: BLE001
        pass
    # Geometry: prefer the baseline rect; when the baseline was iconic
    # its rect is the (-32000,-32000) parked position, so use the
    # recorded normal rect instead.
    target_rect = baseline.get("rect")
    if baseline.get("is_iconic") or baseline.get("minimized"):
        target_rect = baseline.get("normal_rect") or target_rect
    if target_rect:
        try:
            rt.windows.set_rect(hwnd, target_rect)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.2)
    # Show state: minimized -> SW_MINIMIZE (WhatsApp re-hides to tray),
    # maximized -> SW_MAXIMIZE, otherwise normal/restore.
    try:
        if baseline.get("maximized") or baseline.get("is_zoomed"):
            rt.windows.maximize(hwnd)
        elif baseline.get("minimized") or baseline.get("is_iconic"):
            rt.windows.minimize(hwnd)
        else:
            rt.windows.restore(hwnd)
    except Exception:  # noqa: BLE001
        pass
    time.sleep(0.6)
    # Second geometry pass: some apps (e.g. WhatsApp) re-snap their own
    # size after a restore/show transition, undoing the first SetWindowPos.
    if target_rect and not (
            baseline.get("maximized") or baseline.get("is_zoomed")
            or baseline.get("minimized") or baseline.get("is_iconic")):
        try:
            cur = capture_state(rt, hwnd)
            if not _rect_close(cur.get("rect"), target_rect):
                rt.windows.set_rect(hwnd, target_rect)
                time.sleep(0.4)
        except Exception:  # noqa: BLE001
            pass
    return capture_state(rt, hwnd)


# ---------------------------------------------------------------------------
# Verified lifecycle ops
# ---------------------------------------------------------------------------
def test_maximize(rt, hwnd: int) -> tuple[str, str, dict]:
    """SW_MAXIMIZE -> poll IsZoomed; PASS only on observed zoom."""
    rt.windows.maximize(hwnd)
    ok, st, ms = poll_state(rt, hwnd, lambda s: s.get("is_zoomed"))
    ev = {"is_zoomed": st.get("is_zoomed") if st else None,
          "show_cmd_maximized": st.get("maximized") if st else None,
          "rect": st.get("rect") if st else None,
          "monitor_index": st.get("monitor_index") if st else None,
          "wait_ms": round(ms, 1)}
    if not ok or st is None:
        return FAIL, f"IsZoomed never observed within {STATE_TIMEOUT}s", ev
    return PASS, (f"IsZoomed=True after {ms:.0f}ms; "
                  f"rect={st.get('rect')} monitor={st.get('monitor_index')}"), ev


def test_minimize_restore(rt, hwnd: int) -> tuple[str, str, dict]:
    rt.windows.minimize(hwnd)
    ok_min, st_min, ms_min = poll_state(
        rt, hwnd, lambda s: s.get("is_iconic"))
    rt.windows.restore(hwnd)
    ok_res, st_res, ms_res = poll_state(
        rt, hwnd,
        lambda s: (not s.get("is_iconic"))
        and (not s.get("is_zoomed"))
        and bool(s.get("visible")))
    ev = {"minimized_after_ms": round(ms_min, 1),
          "is_iconic": st_min.get("is_iconic") if st_min else None,
          "restored_after_ms": round(ms_res, 1),
          "final": st_res}
    if ok_min and ok_res:
        return PASS, (f"IsIconic observed ({ms_min:.0f}ms), restored "
                      f"visible ({ms_res:.0f}ms)"), ev
    return FAIL, (f"min iconic={ok_min} restored={ok_res}"), ev


def test_maximize_restore(rt, hwnd: int) -> tuple[str, str, dict]:
    rt.windows.maximize(hwnd)
    ok_max, st_max, ms_max = poll_state(rt, hwnd,
                                        lambda s: s.get("is_zoomed"))
    rt.windows.restore(hwnd)
    ok_res, st_res, ms_res = poll_state(
        rt, hwnd,
        lambda s: (not s.get("is_iconic"))
        and (not s.get("is_zoomed"))
        and bool(s.get("visible")))
    ev = {"zoomed_after_ms": round(ms_max, 1),
          "is_zoomed": st_max.get("is_zoomed") if st_max else None,
          "maximized_rect": st_max.get("rect") if st_max else None,
          "restored_after_ms": round(ms_res, 1),
          "final": st_res}
    if ok_max and ok_res:
        return PASS, (f"IsZoomed observed ({ms_max:.0f}ms), restored "
                      f"({ms_res:.0f}ms)"), ev
    return FAIL, f"zoom={ok_max} restored={ok_res}", ev


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(argv=None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    REPORTS.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)

    rt = v93.Runtime()
    svc = v95._svc(rt)
    chk = Check()
    evidence = {"snapshots": [], "operations": [], "commands": []}

    res = rt.discovery.resolve(APP_QUERY)
    app = res.app if res.status == "found" else None
    chk.add("resolve_lookup", PASS if app else FAIL,
            f"status={res.status} score={getattr(res, 'score', None)}")
    if app is None:
        _finish(chk, evidence, rt, None, "app not found")
        return 1

    ident = app_identity.identity_for_app(app)
    chk.add("canonical_identity",
            PASS if ident.app_id.startswith("aumid:") else FAIL,
            ident.app_id)
    chk.add("display_name_readable",
            PASS if app.name == "WhatsApp" else FAIL, app.name)
    chk.add("exe_identity",
            PASS if (app.exe_name or "") == "whatsappdesktop"
            else FAIL, f"exe_name={app.exe_name}")

    pre_procs = [p for p in rt.monitor.list_processes()
                 if "whatsapp" in (p.name or "").lower()]
    pre_windows = v95._snapshot_windows(rt, app)
    pre_hwnds = {w["hwnd"] for w in pre_windows}
    evidence["snapshots"].append({"phase": "pre-test",
                                  "processes": [
                                      {"pid": p.pid, "name": p.name}
                                      for p in pre_procs],
                                  "windows": pre_windows})
    chk.add("existing_instance_detection",
            PASS if rt.monitor.is_running(app) else FAIL,
            f"{len(pre_procs)} process(es), "
            f"{len(pre_windows)} window(s)")

    wins = rt.app_windows(app, include_hidden=True)
    chk.add("window_discovery",
            PASS if wins else FAIL,
            f"{len(wins)} matched, "
            f"{sum(1 for w in wins if getattr(w, 'visible', True))} visible")
    chk.add("window_identity_match",
            PASS if wins and all(
                app_identity.window_matches_identity(w, app, ident)
                for w in wins) else FAIL,
            f"{sum(1 for w in wins if app_identity.window_matches_identity(w, app, ident))}/{len(wins)}")

    # Launch state machine (reuse V9.5 implementation).
    lm = v95.launch_state_machine(rt, app, pre_hwnds)
    evidence["launch_state_machine"] = lm
    state = lm["state"]
    chk.add("launch_activation",
            PASS if state in ("FRESH_WINDOW", "EXISTING_REUSED") else
            (PARTIAL if state == "WINDOW_NOT_READY" else FAIL),
            f"{state} in {lm.get('latency_ms', 0):.0f}ms "
            f"({lm.get('how', '')})")
    chk.add("no_unnecessary_wait",
            PASS if lm.get("latency_ms", 9e9) < 10000 else FAIL,
            f"{lm.get('latency_ms', 0):.0f}ms")

    # Visibility consistency.
    flap = []
    snaps = []
    for i in range(v95.VIS_SNAPSHOTS):
        snap = v95._snapshot_windows(rt, app)
        snaps.append(snap)
        if i:
            prev = {w["hwnd"]: w.get("visible") for w in snaps[i - 1]}
            for w in snap:
                if prev.get(w["hwnd"]) != w.get("visible"):
                    flap.append({"hwnd": w["hwnd"], "at": i,
                                 "was": prev.get(w["hwnd"]),
                                 "now": w.get("visible")})
        time.sleep(v95.VIS_SNAPSHOT_GAP)
    evidence["snapshots"].append({"phase": "visibility-consistency",
                                  "snapshots": snaps, "flaps": flap})
    chk.add("visibility_consistency",
            PASS if not flap else PARTIAL,
            f"{len(snaps)} snapshots, {len(flap)} flap(s)")

    wins = rt.app_windows(app, include_hidden=True)
    visible = [w for w in wins if getattr(w, "visible", True)]
    work = v93._choose_window(visible or wins, app) if wins else None
    hwnd = work.hwnd if work is not None else None

    if hwnd is None:
        for t in ("switch_focus", "minimize_restore", "maximize_verify",
                  "move_monitor_2", "move_monitor_1", "command_handlers",
                  "restore_original"):
            chk.skip(t, "no window available")
        _finish(chk, evidence, rt, app)
        return 0

    # ---- baseline + try/finally restoration -------------------------------
    baseline = capture_state(rt, hwnd)
    baseline["hwnd"] = hwnd
    baseline["pid"] = work.pid
    evidence["baseline"] = baseline
    chk.add("baseline_captured",
            PASS if baseline.get("rect") and baseline.get("monitor_index")
            else FAIL,
            json.dumps({k: baseline.get(k) for k in (
                "visible", "minimized", "maximized", "is_zoomed",
                "is_iconic", "rect", "monitor_index", "monitor_device")},
                default=str))

    try:
        # switch / focus
        try:
            fres = rt.windows.focus_target(app)
            fg = rt.windows.foreground_window()
            fg_ok = fg is not None and fg.hwnd == hwnd
            chk.add("switch_focus",
                    PASS if getattr(fres, "status", "") == "ok" and fg_ok
                    else FAIL,
                    f"status={getattr(fres, 'status', '')} "
                    f"fg={getattr(fg, 'hwnd', None)}")
            evidence["operations"].append({
                "op": "switch_focus", "hwnd": hwnd,
                "status": getattr(fres, "status", ""),
                "foreground_hwnd": getattr(fg, "hwnd", None)})
        except Exception as exc:  # noqa: BLE001
            chk.fail("switch_focus", str(exc))

        # verified minimize/restore
        st, det, ev = test_minimize_restore(rt, hwnd)
        chk.add("minimize_restore", st, det, **ev)
        evidence["operations"].append({"op": "minimize_restore", **ev})

        # verified maximize/restore (IsZoomed + maximized rect)
        st, det, ev = test_maximize_restore(rt, hwnd)
        chk.add("maximize_verify", st, det, **ev)
        evidence["operations"].append({"op": "maximize_verify", **ev})

        # monitor moves both directions (verified)
        m1, m2 = rt.monitors.get(1), rt.monitors.get(2)
        for label, mon in (("move_monitor_2", m2),
                           ("move_monitor_1", m1)):
            if mon is None:
                chk.skip(label, "monitor unavailable")
                continue
            try:
                rt.windows.restore(hwnd)
                time.sleep(0.2)
                moved, _ = rt.windows.move_to_work_area(
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

        # production command handlers
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
                        PASS if v95._cmd_result_ok(cres) else FAIL,
                        getattr(cres, "message", ""))
                evidence["commands"].append({
                    "command": tname,
                    "executed": getattr(cres, "executed", None),
                    "error": getattr(cres, "error", None),
                    "message": getattr(cres, "message", "")})
                time.sleep(0.3)
            except Exception as exc:  # noqa: BLE001
                chk.fail(tname, str(exc))

        for tname, spec in (("move_window_command_m2", "monitor 2"),
                            ("move_window_command_m1", "monitor 1")):
            try:
                mh = _MoveWindowHandler(svc, "")
                cres = mh.execute({"target": APP_QUERY, "monitor": spec})
                chk.add(tname,
                        PASS if v95._cmd_result_ok(cres) else FAIL,
                        getattr(cres, "message", ""))
                evidence["commands"].append({
                    "command": tname, "spec": spec,
                    "executed": getattr(cres, "executed", None),
                    "error": getattr(cres, "error", None),
                    "message": getattr(cres, "message", "")})
                time.sleep(0.3)
            except Exception as exc:  # noqa: BLE001
                chk.fail(tname, str(exc))

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
                        PASS if v95._cmd_result_ok(cres) else FAIL,
                        getattr(cres, "message", ""))
                evidence["commands"].append({
                    "command": tname, "monitor": midx,
                    "executed": getattr(cres, "executed", None),
                    "error": getattr(cres, "error", None),
                    "message": getattr(cres, "message", "")})
                time.sleep(0.3)
            except Exception as exc:  # noqa: BLE001
                chk.fail(tname, str(exc))

        try:
            cres = OpenAppHandler(svc).execute({"app_name": APP_QUERY})
            chk.add("open_app_command",
                    PASS if v95._cmd_result_ok(cres) else FAIL,
                    getattr(cres, "message", ""))
            evidence["commands"].append({
                "command": "OPEN_APPLICATION",
                "executed": getattr(cres, "executed", None),
                "error": getattr(cres, "error", None),
                "message": getattr(cres, "message", "")})
        except Exception as exc:  # noqa: BLE001
            chk.fail("open_app_command", str(exc))

        try:
            hres = SwitchToAppHandler(svc).execute(
                {"app_name": APP_QUERY})
            chk.add("switch_command_handler",
                    PASS if v95._cmd_result_ok(hres) else FAIL,
                    getattr(hres, "message", ""))
        except Exception as exc:  # noqa: BLE001
            chk.fail("switch_command_handler", str(exc))
    finally:
        # ---- restore original state (always attempted) -------------------
        try:
            final = restore_baseline(rt, hwnd, baseline)
        except Exception as exc:  # noqa: BLE001
            final = {"error": str(exc)}
        diffs = diff_states(baseline, final) if "error" not in final else {
            "restore_error": final["error"]}
        evidence["final_state"] = final
        evidence["state_diff"] = diffs
        unrestored = [k for k in diffs]
        chk.add("restore_original",
                PASS if not unrestored else PARTIAL,
                "state restored" if not unrestored
                else f"unrestored fields: {unrestored}",
                diff=diffs)
        evidence["operations"].append({"op": "restore_original",
                                       "diff": diffs})

    post_procs = [p for p in rt.monitor.list_processes()
                  if "whatsapp" in (p.name or "").lower()]
    post_windows = v95._snapshot_windows(rt, app)
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
            else FAIL,
            "user window still present")

    _finish(chk, evidence, rt, app)
    return 0


def _finish(chk: Check, evidence: dict, rt, app, note: str = "") -> None:
    counts = chk.counts
    overall = chk.overall
    rows = [{"application": "WhatsApp", **r} for r in chk.results]
    ident = app_identity.identity_for_app(app) if app else None
    lm = evidence.get("launch_state_machine") or {}
    win = lm.get("window") or {}
    payload = {
        "version": "V9.5.1",
        "generated_at": _utc_now(),
        "app": {
            "name": getattr(app, "name", "WhatsApp"),
            "canonical_app_id": getattr(ident, "app_id", "") or "",
            "aumid": getattr(app, "aumid", ""),
            "exe_path": getattr(app, "exe_path", "") or win.get("exe_path") or "",
            "exe_name": getattr(app, "exe_name", ""),
            "package_family": getattr(ident, "package_family", "") or "",
        },
        "monitor_convention": {
            "monitor_1": "Windows primary display (user right display)",
            "monitor_2": "Windows secondary display (user left display)",
        },
        "monitors": v93._monitor_payload(rt),
        "safety": ("User-owned WhatsApp window never closed; no process "
                   "terminated. Lifecycle ops under explicit user "
                   "authorization; baseline state captured and restored "
                   "in a finally block with a field-level diff."),
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
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["test", "status", "details"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in
                        ("test", "status", "details")})
    _write_markdown(payload)
    print(f"Done: {len(rows)} checks, "
          f"PASS={counts.get(PASS, 0)} FAIL={counts.get(FAIL, 0)} "
          f"PARTIAL={counts.get(PARTIAL, 0)} SKIP={counts.get(SKIP, 0)} "
          f"overall={overall}")
    print(f"Reports: {RESULTS_JSON}, {RESULTS_CSV}, {REPORT_MD}")


def _write_markdown(p: dict) -> None:
    s = p["summary"]
    ev = p["evidence"]
    base = ev.get("baseline") or {}
    fin = ev.get("final_state") or {}
    diff = ev.get("state_diff") or {}
    lm = ev.get("launch_state_machine") or {}
    lines = [
        "# VoiceOS V9.5.1 — WhatsApp State Verification",
        "",
        f"Generated: {p['generated_at']}  ",
        "Mode: `physical` — real Windows runtime; lifecycle on the "
        "user-owned window under explicit authorization  ",
        "",
        "## Application",
        "",
        f"- Name: **{p['app']['name']}** — `{p['app']['canonical_app_id']}`",
        f"- AUMID: `{p['app']['aumid']}`",
        f"- Package family: `{p['app']['package_family']}`",
        f"- exe_name: `{p['app']['exe_name']}` (publisher-hash fix "
        "retained from V9.5)",
        "",
        "## Monitor mapping",
        "",
        "| Logical | Meaning | Device | Primary | Bounds |",
        "|---|---|---|---|---:|",
    ]
    for m in p["monitors"]:
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
        f"SKIPPED **{s['skipped']}** — overall **{s['overall']}**",
        "",
        "## Baseline vs final state",
        "",
        "| Field | Baseline | Final |",
        "|---|---|---|",
    ]
    keys = ["visible", "minimized", "maximized", "is_zoomed",
            "is_iconic", "rect", "normal_rect", "monitor_index",
            "monitor_device", "cloaked", "foreground"]
    for k in keys:
        lines.append(f"| {k} | `{base.get(k)}` | `{fin.get(k)}` |")
    lines += [
        "",
        f"**State diff:** `{json.dumps(diff)}`",
        "",
        "## Maximize verification",
        "",
        "```json",
        json.dumps(next((o for o in ev.get("operations", [])
                         if o.get("op") == "maximize_verify"), {}),
                   indent=2)[:2500],
        "```",
        "",
        "## Launch state machine",
        "",
        "```json",
        json.dumps(lm, indent=2)[:2500],
        "```",
        "",
        "## Per-check results",
        "",
        "| Test | Status | Details |",
        "|---|---:|---|",
    ]
    for r in p["results"]:
        det = str(r.get("details") or "").replace("|", "/")
        if len(det) > 150:
            det = det[:147] + "..."
        lines.append(f"| {r['test']} | {r['status']} | {det} |")
    lines += [
        "",
        "> Maximize now requires observed `IsZoomed` plus the actual "
        "maximized rectangle; minimize/restore require observed "
        "`IsIconic` and a visible non-iconic restored state. Original "
        "state restoration runs in `finally` and is diffed "
        "field-by-field; unrestorable fields are reported, never "
        "hidden.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(run())
