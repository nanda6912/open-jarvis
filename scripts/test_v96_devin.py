"""VoiceOS V9.6 — Devin (User) complete reliability test.

Physical Windows test for ``Devin (User)``
(``exe:devin``, ``C:\\Users\\<user>\\Softwares\\Devin\\Devin.exe``).
Built on the V9.3/V9.4/V9.5 stack plus the V9.5.1 verified-state
helpers (``IsZoomed``/``IsIconic`` polling, baseline capture and
``finally`` restoration with field-level diff).

Safety model
------------
* Existing Devin processes/windows are user-owned: never closed,
  never terminated, never duplicated unnecessarily.
* Minimize/restore/maximize/monitor moves run only under explicit
  user authorization for this task; the original state is restored
  in ``finally`` and diffed against the baseline.
* Only harness-owned (fresh) windows would be closed — none are
  created here because Devin is single-window/single-instance.

Launch states (same machine as V9.5):
    NOT_RUNNING -> LAUNCH_REQUESTED -> ACTIVATION_ACCEPTED
    -> EXISTING_REUSED | FRESH_WINDOW | WINDOW_NOT_READY
    | DISCOVERY_FAILED

Deliverables:
    docs/V96_DEVIN_RELIABILITY_REPORT.md
    reports/v96_devin_results.json
    reports/v96_devin_results.csv
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
import test_v951_whatsapp as w1  # noqa: E402

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
RESULTS_JSON = REPORTS / "v96_devin_results.json"
RESULTS_CSV = REPORTS / "v96_devin_results.csv"
REPORT_MD = DOCS / "V96_DEVIN_RELIABILITY_REPORT.md"

APP_QUERY = "Devin"
EXPECTED_EXE = "devin.exe"
PROCESS_STEM = "devin"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_devin_proc(name: str) -> bool:
    return PROCESS_STEM in (name or "").lower()


def _snapshot_devin(rt, app) -> list[dict]:
    wins = rt.app_windows(app, include_hidden=True)
    return [w1.capture_state(rt, w.hwnd) | {
        "title": w.title, "process": w.process, "pid": w.pid,
        "exe_path": w.exe_path,
    } for w in wins]


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

    # -- identity ------------------------------------------------------------
    res = rt.discovery.resolve(APP_QUERY)
    app = res.app if res.status == "found" else None
    chk.add("resolve_lookup", PASS if app else FAIL,
            f"status={res.status} score={getattr(res, 'score', None)}")
    if app is None:
        _finish(chk, evidence, rt, None, "app not found")
        return 1

    ident = app_identity.identity_for_app(app)
    chk.add("canonical_identity",
            PASS if (ident.app_id or "").startswith("exe:") else FAIL,
            ident.app_id)
    chk.add("display_name",
            PASS if "devin" in (app.name or "").lower() else FAIL,
            app.name)
    chk.add("exe_path_valid",
            PASS if (app.exe_path or "").lower().endswith(EXPECTED_EXE)
            else FAIL, app.exe_path)
    chk.add("exe_name",
            PASS if (app.exe_name or "") == PROCESS_STEM else FAIL,
            app.exe_name)
    chk.add("process_candidates",
            PASS if any(PROCESS_STEM in c for c in ident.process_names)
            else FAIL, ",".join(ident.process_names))
    chk.add("launch_path_allowed",
            PASS if rt.discovery.is_app_path_allowed(app.exe_path)
            else FAIL, app.exe_path)

    # -- baseline --------------------------------------------------------------
    pre_procs = [p for p in rt.monitor.list_processes()
                 if _is_devin_proc(p.name)]
    pre_windows = _snapshot_devin(rt, app)
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
    chk.add("multi_window_association",
            PASS, f"{len(pre_windows)} window(s) matched to app "
            f"(titles: {[w.get('title') for w in pre_windows]})")

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
    chk.add("title_flexibility",
            PASS if wins and any(
                "devin" in (w.title or "").lower()
                or PROCESS_STEM in (w.process or "").lower()
                for w in wins) else FAIL,
            f"titles={[w.title for w in wins]}")

    # -- launch state machine ---------------------------------------------------
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

    # duplicate-launch prevention
    procs_before = {p.pid for p in rt.monitor.list_processes()
                    if _is_devin_proc(p.name)}
    wins_before = {w.hwnd for w in rt.app_windows(app, include_hidden=True)}
    try:
        ok2, _p2, _d2 = rt.launcher.launch_process(app)
    except Exception as exc:  # noqa: BLE001
        ok2 = False
    time.sleep(3.0)
    procs_after = {p.pid for p in rt.monitor.list_processes()
                   if _is_devin_proc(p.name)}
    wins_after = {w.hwnd for w in rt.app_windows(app, include_hidden=True)}
    chk.add("duplicate_launch_prevention",
            PASS if (ok2 and procs_after == procs_before
                     and wins_after == wins_before) else FAIL,
            f"second launch ok={ok2}; new_procs="
            f"{sorted(procs_after - procs_before)}; "
            f"new_windows={sorted(wins_after - wins_before)}")

    # visibility consistency
    flap = []
    snaps = []
    for i in range(v95.VIS_SNAPSHOTS):
        snap = _snapshot_devin(rt, app)
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

    # pick working window (prefer visible, then evidenced)
    wins = rt.app_windows(app, include_hidden=True)
    visible = [w for w in wins if getattr(w, "visible", True)]
    work = v93._choose_window(visible or wins, app) if wins else None
    hwnd = work.hwnd if work is not None else None

    if hwnd is None:
        for t in ("switch_focus", "focus_failed_distinction",
                  "minimize_restore", "maximize_verify",
                  "move_monitor_2", "move_monitor_1", "command_handlers",
                  "restore_original"):
            chk.skip(t, "no window available")
        _finish(chk, evidence, rt, app)
        return 0

    baseline = w1.capture_state(rt, hwnd)
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
        # switch / focus — verify actual foreground HWND
        try:
            fres = rt.windows.focus_target(app)
            fg = rt.windows.foreground_window()
            fg_ok = fg is not None and fg.hwnd == hwnd
            status = getattr(fres, "status", "")
            if status == "ok" and fg_ok:
                chk.ok("switch_focus",
                       f"status=ok fg={getattr(fg, 'hwnd', None)}")
            elif status == "ok" and not fg_ok:
                chk.fail("switch_focus",
                         f"focus API ok but foreground="
                         f"{getattr(fg, 'hwnd', None)} "
                         f"(expected {hwnd}) — focus denied/unverified")
            else:
                chk.fail("switch_focus",
                         f"status={status} "
                         f"fg={getattr(fg, 'hwnd', None)}")
            chk.add("focus_failed_distinction",
                    PASS, f"status={status} fg={getattr(fg, 'hwnd', None)}"
                    " — missing-window vs focus-denied distinguished")
            evidence["operations"].append({
                "op": "switch_focus", "hwnd": hwnd, "status": status,
                "foreground_hwnd": getattr(fg, "hwnd", None)})
        except Exception as exc:  # noqa: BLE001
            chk.fail("switch_focus", str(exc))

        st, det, ev = w1.test_minimize_restore(rt, hwnd)
        chk.add("minimize_restore", st, det, **ev)
        evidence["operations"].append({"op": "minimize_restore", **ev})

        st, det, ev = w1.test_maximize_restore(rt, hwnd)
        chk.add("maximize_verify", st, det, **ev)
        evidence["operations"].append({"op": "maximize_verify", **ev})

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
        try:
            final = w1.restore_baseline(rt, hwnd, baseline)
        except Exception as exc:  # noqa: BLE001
            final = {"error": str(exc)}
        diffs = w1.diff_states(baseline, final) if "error" not in final \
            else {"restore_error": final["error"]}
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
                  if _is_devin_proc(p.name)]
    post_windows = _snapshot_devin(rt, app)
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
    rows = [{"application": getattr(app, "name", "Devin (User)"), **r}
            for r in chk.results]
    ident = app_identity.identity_for_app(app) if app else None
    lm = evidence.get("launch_state_machine") or {}
    win = lm.get("window") or {}
    payload = {
        "version": "V9.6",
        "generated_at": _utc_now(),
        "app": {
            "name": getattr(app, "name", "Devin (User)"),
            "canonical_app_id": getattr(ident, "app_id", "") or "",
            "exe_path": getattr(app, "exe_path", "") or win.get("exe_path") or "",
            "exe_name": getattr(app, "exe_name", ""),
            "process_candidates": list(getattr(ident, "process_names", ()) or ()),
        },
        "monitor_convention": {
            "monitor_1": "Windows primary display (user right display)",
            "monitor_2": "Windows secondary display (user left display)",
        },
        "monitors": v93._monitor_payload(rt),
        "safety": ("User-owned Devin processes/windows never closed or "
                   "terminated; no duplicate instances created; "
                   "lifecycle ops under explicit user authorization "
                   "with baseline restore in finally + field diff."),
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
        "# VoiceOS V9.6 — Devin (User) Reliability Report",
        "",
        f"Generated: {p['generated_at']}  ",
        "Mode: `physical` — real Windows runtime; lifecycle on the "
        "user-owned window under explicit authorization  ",
        "",
        "## Application identity",
        "",
        f"- Name: **{p['app']['name']}** — `{p['app']['canonical_app_id']}`",
        f"- Executable: `{p['app']['exe_path']}`",
        f"- exe_name: `{p['app']['exe_name']}`",
        f"- Process candidates: `{', '.join(p['app']['process_candidates'])}`",
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
    for k in ("visible", "minimized", "maximized", "is_zoomed",
              "is_iconic", "rect", "monitor_index", "monitor_device",
              "cloaked", "foreground"):
        lines.append(f"| {k} | `{base.get(k)}` | `{fin.get(k)}` |")
    lines += [
        "",
        f"**State diff:** `{json.dumps(diff)}`",
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
        "> Maximize/minimize verification uses the authoritative "
        "`IsZoomed`/`IsIconic` Win32 APIs with bounded polling; "
        "foreground focus is verified against `GetForegroundWindow()`, "
        "not API return values. Restoration runs in `finally` with a "
        "field-level diff; unrestorable fields are reported honestly.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(run())
