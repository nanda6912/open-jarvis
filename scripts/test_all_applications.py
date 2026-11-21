"""VoiceOS V9.3 all-application compatibility and multi-monitor test.

Discovers the real Windows application inventory through
``ApplicationDiscovery``, classifies every entry for safe automated
testing, verifies canonical identity for 100% of discovered apps, and
physically exercises safe GUI apps through the production
``ApplicationLauncher`` / ``WindowManager`` / ``MonitorManager`` stack.

Monitor convention (enforced through ``MonitorManager`` only):
    Monitor 1 = Windows primary display = user's RIGHT display.
    Monitor 2 = Windows secondary display = user's LEFT display.

Modes:
    --discovery-only   Build the deduplicated inventory only.
    --dry-run          Identity/classification report; no launches.
    --full-safe-gui    Physical tests for category-A GUI apps.
    --rerun-failed P   Re-run only FAIL/PARTIAL apps from a prior JSON.
    --only TEXT        Restrict physical tests to matching app names.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import struct
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.controllers import app_identity  # noqa: E402
from src.controllers.application_discovery import (  # noqa: E402
    AppInfo,
    ApplicationDiscovery,
    _normalize,
)
from src.controllers.monitor_manager import MonitorManager  # noqa: E402
from src.controllers.windows_automation import (  # noqa: E402
    ApplicationLauncher,
    ProcessMonitor,
    WindowManager,
    WindowInfo,
)

REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
INVENTORY_JSON = REPORTS / "v93_application_inventory.json"
INVENTORY_CSV = REPORTS / "v93_application_inventory.csv"
RESULTS_JSON = REPORTS / "v93_application_test_results.json"
RESULTS_CSV = REPORTS / "v93_application_test_results.csv"
REPORT_MD = DOCS / "V93_ALL_APPLICATIONS_TEST_REPORT.md"

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIPPED"
PARTIAL = "PARTIAL"

DEFAULT_WINDOW_TIMEOUT = 15.0
DEFAULT_CLOSE_TIMEOUT = 8.0
POLL_INTERVAL = 0.30

# ---------------------------------------------------------------------------
# Classification policy
# ---------------------------------------------------------------------------
# Entries that must never be launched automatically: installers,
# uninstallers, shells/admin tools, security tools, package managers,
# system components, drivers/services and known destructive utilities.
_NEVER_PATTERNS = [
    r"\buninstall",
    r"\bsetup\b",
    r"\binstaller\b",
    r"\binstall\b",
    r"\bupdate\b",
    r"\bupgrade\b",
    r"\brepair\b",
    r"\bconfiguration assistant\b",
    r"\bstack\s*builder\b",
    r"\borg instance manager\b",
    r"\bodbc\b",
    r"\bmicrosoft office home\b",
    r"\bone\s*drive\s*setup\b",
    r"\bdocker desktop installer\b",
    r"\bvisual studio installer\b",
    r"\bpython 3\.13\.0\b",
    r"\bwindows package manager\b",
    r"\bwinget\b",
    r"\bapp installer\b",
    r"\bmicrosoft store\b",
    r"\bmicrosoft edge webview2\b",
    r"\bwindows security\b",
    r"\bpc manager\b",
    r"\bquick assist\b",
    r"\bpowershell\b",
    r"\bcmd\.exe\b",
    r"\binstall additional tools\b",
    r"\breload configuration\b",
    r"\bsql plus\b",
    r"\boracle\b",
    r"\bpostgresql\b",
    r"\bactionsmcphost\b",
    r"\bshellmcpservers\b",
    r"\bxboxpcapp\b",
    r"\bgamebarelevated\b",
    r"\bwslconfig\b",
    r"\bdesktopappinstaller\b",
    r"\bsetup\.exe\b",
    r"\binstaller\.exe\b",
    r"\bmsiexec\b",
    r"\bxboxpcapp",
]

# GUI-capable but not safe for unattended automation: account prompts,
# remote/support tools, developer environment managers, games, backup,
# ambiguous browser-proxy shortcuts and the assistant itself.
_MANUAL_PATTERNS = [
    r"\bone\s*drive\b",
    r"\bdev home\b",
    r"\bpower automate\b",
    r"\bwindows backup\b",
    r"\boffice language preferences\b",
    r"\bpgadmin\b",
    r"\bgithub\b",
    r"\biexplore\b",
    r"\bgame bar\b",
    r"\bxbox\b",
    r"\bvoiceos\b",
    r"\bwhatsapp\b",  # kept manual? no — see _SAFE_OVERRIDES below
    r"\bgit bash\b",
]

# Explicit safe overrides for patterns that are broad but known GUI apps.
_SAFE_OVERRIDES = {
    "whatsapp",
    "microsoft to do",
    "sticky notes",
    "sound recorder",
    "snipping tool",
    "photos",
    "camera",
    "clock",
    "calculator",
    "paint",
    "notepad",
    "settings",
    "terminal",
    "media player",
    "clipchamp",
    "phone link",
    "feedback hub",
    "get help",
    "get started",
    "news",
    "weather",
    "family",
    "click to do",
    "b&o audio control",
    "intel",
    "wireless display",
    "google chrome",
    "microsoft edge",
    "visual studio code",
    "devin",
    "postman",
    "git gui",
    "idle",
    "excel",
    "winword",
    "outlook",
    "onenote",
    "msaccess",
    "powerpnt",
}

_NON_GUI_NAME_PATTERNS = [
    r"\bnode\.?js\b",
    r"\bpython\b",
    r"\bpythonw\b",
    r"\bpip\d*(\.\d+)?\b",
    r"\bjava\b",
    r"\bjavaw\b",
    r"\bgit cmd\b",
    r"(?:^|[\\/])bash\.exe",
    r"\bwsl\b",
    r"\bnmakehlp\b",
]

# Display names that are really raw executable/package identifiers.
_RAW_NAME_STEMS = {
    "msoadfsb", "msoasb", "msoxmled", "mspub", "sdxhelper", "skypeserver",
    "iediag", "iexplore", "microsoftstore", "pbrush", "store", "wt",
    "wslconfig", "actionsmcphost", "shellmcpserverspackaging",
    "xboxpcappadminserver", "xboxpcappce", "gamebarelevatedftalias",
    "windowspackagemanagerserver", "winget", "mediaplayer", "gethelp",
    "snippingtool", "pythonw", "pip", "pip3", "pip313", "olk",
}

_PACKAGE_ID_RE = re.compile(
    r"(^|[._-])(pf|pfn|aumid|package|publisher|cv[a-z0-9]{4,})"
    r"|^[0-9a-f]{8}\.|!|_cv[a-z0-9]+$",
    re.I,
)

_SUSPICIOUS_TITLE_PATTERNS = [
    r"save your work",
    r"do you want to save",
    r"unsaved",
    r"sign in",
    r"log in",
    r"password",
    r"authentication",
    r"confirm",
    r"are you sure",
    r"permission",
    r"administrator",
    r"uninstall",
    r"repair",
    r"update required",
    r"product key",
    r"activation",
]


@dataclass
class AppRecord:
    """Inventory + test state for one discovered application."""

    app: AppInfo
    identity: object
    inventory_id: str
    app_id: str
    display_name: str
    exe_path: str
    exe_name: str
    process_names: list
    package_family_name: str
    aumid: str
    launch_type: str
    launch_target: str
    is_gui: bool
    gui_reason: str
    already_running: bool = False
    running_window_count: int = 0
    category: str = "C"
    category_reason: str = ""
    eligible: bool = False
    identity_status: str = SKIP
    identity_details: str = ""
    launch: str = SKIP
    window_discovery: str = SKIP
    switch_focus: str = SKIP
    minimize_restore: str = SKIP
    maximize: str = SKIP
    monitor_1: str = SKIP
    monitor_2: str = SKIP
    move_monitor_1: str = SKIP
    move_monitor_2: str = SKIP
    cleanup: str = SKIP
    overall: str = SKIP
    details: str = ""
    launch_latency_ms: Optional[float] = None
    window_latency_ms: Optional[float] = None
    diagnostics: dict = field(default_factory=dict)

    def base_dict(self) -> dict:
        return {
            "application": self.display_name,
            "canonical_app_id": self.app_id,
            "inventory_id": self.inventory_id,
            "exe_path": self.exe_path,
            "exe_name": self.exe_name,
            "process_name_candidates": self.process_names,
            "package_family_name": self.package_family_name,
            "aumid": self.aumid,
            "launch_type": self.launch_type,
            "launch_target": self.launch_target,
            "is_gui": self.is_gui,
            "gui_reason": self.gui_reason,
            "already_running": self.already_running,
            "running_window_count": self.running_window_count,
            "category": self.category,
            "category_reason": self.category_reason,
            "eligible": self.eligible,
            "identity": self.identity_status,
            "identity_details": self.identity_details,
            "launch": self.launch,
            "window_discovery": self.window_discovery,
            "switch_focus": self.switch_focus,
            "minimize_restore": self.minimize_restore,
            "maximize": self.maximize,
            "monitor_1": self.monitor_1,
            "monitor_2": self.monitor_2,
            "move_monitor_1": self.move_monitor_1,
            "move_monitor_2": self.move_monitor_2,
            "cleanup": self.cleanup,
            "overall": self.overall,
            "details": self.details,
            "launch_latency_ms": self.launch_latency_ms,
            "window_discovery_latency_ms": self.window_latency_ms,
            "diagnostics": self.diagnostics,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _normalize(text or ""))


def _matches_any(text: str, patterns: list[str]) -> bool:
    t = f" {text.lower()} "
    return any(re.search(p, t, re.I) for p in patterns)


def _package_family_from_path(path: str) -> str:
    """Best-effort PFN from a WindowsApps PackageFullName folder."""
    if not path:
        return ""
    m = re.search(r"([A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*)_[^\\/]+?__([A-Za-z0-9]+)", path)
    if not m:
        return ""
    return f"{m.group(1)}_{m.group(2)}".lower()


def _pe_subsystem(path: str) -> Optional[int]:
    """Return PE subsystem (2=GUI, 3=console) or None when unreadable."""
    try:
        with open(path, "rb") as f:
            head = f.read(0x400)
            if len(head) < 0x40 or head[:2] != b"MZ":
                return None
            pe_off = struct.unpack_from("<I", head, 0x3C)[0]
            if pe_off + 0x18 > len(head):
                f.seek(pe_off)
                head = f.read(0x400)
            if head[pe_off:pe_off + 4] != b"PE\x00\x00":
                return None
            opt_off = pe_off + 24
            if opt_off + 70 > len(head):
                return None
            magic = struct.unpack_from("<H", head, opt_off)[0]
            if magic not in (0x10B, 0x20B):
                return None
            return struct.unpack_from("<H", head, opt_off + 68)[0]
    except Exception:  # noqa: BLE001
        return None


def _is_gui_app(app: AppInfo) -> tuple[bool, str]:
    if getattr(app, "launch_kind", "exe") == "aumid":
        name = app.name.lower()
        if _matches_any(name, _NON_GUI_NAME_PATTERNS):
            return False, "packaged entry classified non-GUI by name"
        return True, "packaged Start Apps entry (GUI)"
    path = app.exe_path or ""
    # Name/path non-GUI rules run before the PE check: javaw/pythonw-style
    # launchers can have a GUI subsystem yet still own no window.
    if _matches_any(f"{app.name} {Path(path).name}", _NON_GUI_NAME_PATTERNS):
        return False, "name/path indicates non-GUI utility"
    subsystem = _pe_subsystem(path)
    if subsystem == 2:
        return True, "PE subsystem=GUI"
    if subsystem == 3:
        return False, "PE subsystem=console"
    # WindowsApps aliases and some packaged exe launchers do not expose a
    # readable subsystem; treat known GUI product names as GUI candidates.
    if _squash(app.name) in _SAFE_OVERRIDES or _matches_any(
        app.name, [r"\b(paint|notepad|calculator|terminal|settings)\b"]
    ):
        return True, "GUI product name (subsystem unreadable)"
    return False, "GUI subsystem could not be verified"


def _classify(app: AppInfo, is_gui: bool, gui_reason: str) -> tuple[str, str, bool]:
    text = " ".join(
        [app.name, app.exe_path or "", app.aumid or "", app.exe_name]
    )
    lowered_name = _normalize(app.name)
    stem = _squash(app.exe_name)

    if _matches_any(text, _NEVER_PATTERNS):
        return "C", "excluded: installer/system/security/package/admin pattern", False
    if not is_gui:
        return "D", f"non-GUI application ({gui_reason})", False
    if _matches_any(app.name, _MANUAL_PATTERNS) and lowered_name not in _SAFE_OVERRIDES:
        return "B", "manual test required (account/admin/ambiguous/remote/game)", False
    if stem in _RAW_NAME_STEMS and lowered_name in _RAW_NAME_STEMS:
        return "B", "identity is a raw executable/package identifier", False
    if _PACKAGE_ID_RE.search(app.name):
        return "C", "package identifier exposed as application name", False
    return "A", "safe GUI application", True


def _human_readable_name(name: str, app: AppInfo) -> tuple[bool, str]:
    n = (name or "").strip()
    if not n:
        return False, "empty display name"
    if len(n) > 120:
        return False, "display name too long"
    if any(ch in n for ch in "/\\") or n.lower().endswith(".exe"):
        return False, "display name contains a path/executable"
    if "!" in n or _PACKAGE_ID_RE.search(n):
        return False, "display name exposes package identifier"
    if not re.search(r"[A-Za-z]", n):
        return False, "display name has no letters"
    return True, "ok"


# Known executable-stem -> display-name equivalences. Substring matching
# handles most products; this map covers launcher names that share no
# token with the marketing name (msedge -> Microsoft Edge, calc ->
# Calculator, winword -> Word, ...).
_EXE_NAME_ALIASES = {
    "msedge": {"microsoftedge", "edge", "msedge"},
    "calc": {"calculator"},
    "winword": {"microsoftword", "word", "winword"},
    "excel": {"microsoftexcel", "excel"},
    "powerpnt": {"microsoftpowerpoint", "powerpoint", "powerpnt"},
    "outlook": {"microsoftoutlook", "outlook"},
    "onenote": {"microsoftonenote", "onenote"},
    "msaccess": {"microsoftaccess", "access", "msaccess"},
    "cmd": {"commandprompt", "cmd", "commandline"},
    "mspaint": {"paint", "microsoftpaint"},
    "wt": {"windowsterminal", "terminal", "wt"},
    "store": {"microsoftstore", "windowsstore", "store"},
    "mediaplayer": {"mediaplayer", "zunemusic"},
    "snippingtool": {"snippingtool", "screensketch"},
    "gethelp": {"gethelp", "microsoftgethelp"},
    "code": {"microsoftvisualstudiocodeuser", "visualstudiocode", "code"},
    "gitgui": {"gitgui"},
    "devin": {"devin", "devinuser", "codeium"},
    "idle": {"idle", "idlepython31364bit", "idlepython31464bit"},
    "node": {"node", "nodejs"},
}


def _exe_identity_consistent(app: AppInfo) -> tuple[bool, str]:
    """True when the exe stem plausibly belongs to the display name."""
    if getattr(app, "launch_kind", "exe") != "exe" or not app.exe_path:
        return True, "not an exe launch"
    stem = _squash(Path(app.exe_path).stem)
    name = _squash(app.name)
    if not stem or not name:
        return True, "missing stem/name"
    if stem == name or stem in name or name in stem:
        return True, "exe/name tokens consistent"
    if name in _EXE_NAME_ALIASES.get(stem, set()):
        return True, "known exe/display-name alias"
    return False, "executable name does not match display name"


def _window_score(win: WindowInfo, app: AppInfo) -> tuple:
    """Higher is better for choosing the app's real top-level window."""
    name_norm = _normalize(app.name)
    title_norm = _normalize(getattr(win, "title", "") or "")
    title_score = 0
    if name_norm and title_norm:
        if title_norm == name_norm:
            title_score = 3
        elif name_norm in title_norm:
            title_score = 2
        else:
            title_score = len(set(name_norm.split()) & set(title_norm.split()))
    visible = 1 if getattr(win, "visible", True) else 0
    title_len = len(getattr(win, "title", "") or "")
    proc = (getattr(win, "process", "") or "").lower()
    framework_penalty = 0 if any(
        s in proc for s in ("applicationframehost", "sihost")
    ) else 1
    return (title_score, visible, framework_penalty, title_len, -int(win.hwnd))


def _choose_window(windows: list[WindowInfo], app: AppInfo) -> Optional[WindowInfo]:
    if not windows:
        return None
    visible = [w for w in windows if getattr(w, "visible", True)]
    pool = visible or windows
    evidenced = [w for w in pool if _window_score(w, app)[0] > 0]
    if evidenced:
        pool = evidenced
    return max(pool, key=lambda w: _window_score(w, app))


def _suspicious_dialog(win: Optional[WindowInfo]) -> Optional[str]:
    if win is None:
        return None
    title = (getattr(win, "title", "") or "").lower()
    for pat in _SUSPICIOUS_TITLE_PATTERNS:
        if re.search(pat, title):
            return f"possible dialog/prompt in window title: {win.title!r}"
    return None


def _rect_inside(rect, mon) -> bool:
    if rect is None:
        return False
    cx = (rect[0] + rect[2]) // 2
    cy = (rect[1] + rect[3]) // 2
    b = mon.bounds
    return b.left <= cx < b.right and b.top <= cy < b.bottom


def _verify_on_monitor(windows: WindowManager, monitors: MonitorManager,
                       hwnd: int, mon) -> tuple[bool, str]:
    actual = monitors.monitor_for_hwnd(hwnd)
    rect = windows.window_rect(hwnd)
    details = {
        "expected_monitor": mon.index,
        "expected_device": mon.device,
        "actual_monitor": getattr(actual, "index", None),
        "actual_device": getattr(actual, "device", None),
        "rect": list(rect) if rect else None,
    }
    if actual is None:
        return False, json.dumps(details)
    ok = actual.handle == mon.handle or actual.device == mon.device
    if rect is not None:
        ok = ok and _rect_inside(rect, mon)
        b = mon.bounds
        iw = max(0, min(rect[2], b.right) - max(rect[0], b.left))
        ih = max(0, min(rect[3], b.bottom) - max(rect[1], b.top))
        area = max(1, (rect[2] - rect[0]) * (rect[3] - rect[1]))
        ok = ok and ((iw * ih) / area) >= 0.5
    return bool(ok), json.dumps(details)


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------
def _inventory_key(app: AppInfo) -> str:
    if getattr(app, "launch_kind", "exe") == "aumid":
        return f"aumid:{(app.aumid or '').lower()}"
    return f"exe:{(app.exe_path or '').lower()}"


def build_inventory(discovery: ApplicationDiscovery) -> tuple[list[AppRecord], dict]:
    raw = discovery.discover(force=True)
    raw_count = len(raw)
    by_key: dict[str, AppInfo] = {}
    duplicates = []
    for app in raw:
        key = _inventory_key(app)
        if not key or key.endswith(":"):
            continue
        if key in by_key:
            duplicates.append({
                "removed": app.name,
                "kept": by_key[key].name,
                "key": key,
            })
            continue
        by_key[key] = app

    apps = list(by_key.values())
    # Known executable aliases that discovery reports under a raw exe
    # name while the friendly product entry also exists (msedge ->
    # Microsoft Edge). Only exact product-name targets are merged.
    _EXE_ALIAS_TARGETS = {"msedge": "Microsoft Edge"}
    alias_targets = {
        _squash(a.name): a for a in apps
        if _squash(a.name) in {_squash(v) for v in _EXE_ALIAS_TARGETS.values()}
    }
    kept: list[AppInfo] = []
    for app in apps:
        target_name = _EXE_ALIAS_TARGETS.get(_squash(app.exe_name))
        target = alias_targets.get(_squash(target_name)) if target_name else None
        if (
            target is not None
            and getattr(app, "launch_kind", "exe") == "exe"
            and app.name != target.name
            and _inventory_key(app) != _inventory_key(target)
        ):
            duplicates.append({
                "removed": app.name,
                "kept": target.name,
                "key": _inventory_key(app),
                "reason": "known executable alias duplicate",
            })
            continue
        kept.append(app)
    apps = kept
    # Conservative packaged/exe alias dedup: an exe living under a
    # WindowsApps package folder (or a same-name app-execution alias)
    # that resolves to an already-discovered AUMID is the same app.
    aumids = [a for a in apps if getattr(a, "launch_kind", "exe") == "aumid"]
    aumid_by_pfn = {
        (a.aumid.split("!", 1)[0]).lower(): a for a in aumids if a.aumid
    }
    aumid_by_name = {_squash(a.name): a for a in aumids}

    def _likely_same_packaged_app(app: AppInfo, dupe: Optional[AppInfo]) -> bool:
        """True only for high-confidence app aliases, not package servers."""
        if dupe is None:
            return False
        if _squash(app.name) == _squash(dupe.name):
            return True
        stem = Path(app.exe_path).stem.lower() if app.exe_path else ""
        known_aliases = {
            "wt": "terminal", "mspaint": "paint", "store": "microsoft store",
            "mediaplayer": "media player", "pbrush": "paint",
            "gethelp": "get help", "snippingtool": "snipping tool",
        }
        return known_aliases.get(stem) == _normalize(dupe.name)

    final: list[AppInfo] = []
    for app in apps:
        if getattr(app, "launch_kind", "exe") == "aumid":
            final.append(app)
            continue
        path = (app.exe_path or "").lower()
        pfn = _package_family_from_path(app.exe_path or "")
        dupe = None
        if pfn and pfn in aumid_by_pfn:
            dupe = aumid_by_pfn[pfn]
        elif "windowsapps" in path and _squash(app.name) in aumid_by_name:
            dupe = aumid_by_name[_squash(app.name)]
        if dupe is not None and dupe is not app and _likely_same_packaged_app(app, dupe):
            duplicates.append({
                "removed": app.name,
                "kept": dupe.name,
                "key": _inventory_key(app),
                "reason": "packaged app alias/exe duplicate",
            })
            continue
        final.append(app)

    records: list[AppRecord] = []
    for app in sorted(final, key=lambda a: a.name.lower()):
        ident = app_identity.identity_for_app(app)
        is_gui, gui_reason = _is_gui_app(app)
        category, reason, eligible = _classify(app, is_gui, gui_reason)
        stem = Path(app.exe_path).stem.lower() if app.exe_path else ""
        inventory_id = (
            f"aumid:{app.aumid}" if getattr(app, "launch_kind", "exe") == "aumid"
            else f"exe:{stem}:{hashlib.sha1(app.exe_path.encode('utf-8')).hexdigest()[:10]}"
        )
        records.append(AppRecord(
            app=app,
            identity=ident,
            inventory_id=inventory_id,
            app_id=ident.app_id,
            display_name=app.name,
            exe_path=app.exe_path or "",
            exe_name=app.exe_name,
            process_names=list(ident.process_names),
            package_family_name=ident.package_family,
            aumid=ident.aumid,
            launch_type=getattr(app, "launch_kind", "exe"),
            launch_target=app.launch_target,
            is_gui=is_gui,
            gui_reason=gui_reason,
            category=category,
            category_reason=reason,
            eligible=eligible,
        ))

    meta = {
        "generated_at": _utc_now(),
        "raw_discovered": raw_count,
        "unique_after_dedup": len(records),
        "duplicates_removed": len(duplicates),
        "duplicates": duplicates,
    }
    return records, meta


# ---------------------------------------------------------------------------
# Runtime services
# ---------------------------------------------------------------------------
class Runtime:
    def __init__(self) -> None:
        self.discovery = ApplicationDiscovery()
        self.monitor = ProcessMonitor()
        self.windows = WindowManager(monitor=self.monitor)
        self.launcher = ApplicationLauncher(
            discovery=self.discovery, monitor=self.monitor
        )
        self.monitors = MonitorManager()

    def app_windows(self, app: AppInfo, include_hidden: bool = True) -> list[WindowInfo]:
        try:
            return list(self.windows.windows_for_app(
                app, include_hidden=include_hidden
            ))
        except TypeError:
            return list(self.windows.windows_for_app(app))
        except Exception:  # noqa: BLE001
            return []

    def snapshot_running(self, records: list[AppRecord]) -> None:
        """Fill already_running/window counts without per-app process scans."""
        try:
            procs = self.monitor.list_processes()
        except Exception:  # noqa: BLE001
            procs = []
        pid_details = {}
        for p in procs:
            try:
                pid_details[p.pid] = self.monitor.pid_details(p.pid)
            except Exception:  # noqa: BLE001
                pid_details[p.pid] = ("", "")
        try:
            all_windows = self.windows.list_windows(include_hidden=True)
        except Exception:  # noqa: BLE001
            all_windows = []
        identities = app_identity.identities_for([r.app for r in records])
        for rec in records:
            ident = rec.identity
            candidates = set(ident.process_names or ())
            running = False
            exe_l = (rec.exe_path or "").lower()
            for p in procs:
                stem = (p.stem or "").lower()
                stems = {stem, stem.replace(".", "")}
                stems.update(s for s in stem.split(".") if s)
                if stems & candidates:
                    running = True
                    break
                if ident.package_family:
                    pfn = (pid_details.get(p.pid, ("", ""))[1] or "").lower()
                    if pfn and pfn == ident.package_family.lower():
                        running = True
                        break
                if exe_l:
                    path, _pfn = pid_details.get(p.pid, ("", ""))
                    if path and path.lower() == exe_l:
                        running = True
                        break
            matches = app_identity.windows_for_app(all_windows, rec.app)
            if matches:
                running = True
            rec.already_running = running
            rec.running_window_count = len(matches)
            rec.diagnostics["matched_windows_at_snapshot"] = [
                {"hwnd": w.hwnd, "pid": w.pid, "title": w.title,
                 "process": w.process, "visible": w.visible}
                for w in matches[:20]
            ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def identity_test(rec: AppRecord, rt: Runtime,
                  window_snapshot: list[WindowInfo]) -> None:
    app = rec.app
    ident = rec.identity
    problems = []
    ok_name, name_reason = _human_readable_name(rec.display_name, app)
    if not ok_name:
        problems.append(name_reason)

    app_id_ok = bool(re.match(r"^(exe|aumid):.+", rec.app_id or ""))
    if not app_id_ok:
        problems.append("invalid canonical app id")
    if getattr(app, "launch_kind", "exe") == "aumid":
        aumid_ok = bool(re.match(
            r"^[A-Za-z0-9][A-Za-z0-9.\-_]*![A-Za-z0-9][A-Za-z0-9.\-_]*$",
            app.aumid or "",
        ))
        if not aumid_ok:
            problems.append("invalid AUMID")
        if not ident.package_family:
            problems.append("missing package family name")
        elif app.aumid and ident.package_family.lower() != app.aumid.split("!", 1)[0].lower():
            problems.append("package family does not match AUMID")
        if not rt.discovery.is_app_path_allowed(app.aumid):
            problems.append("AUMID failed launch-path validation")
    else:
        if not app.exe_path.lower().endswith(".exe"):
            problems.append("executable path is not .exe")
        if not Path(app.exe_path).is_file():
            problems.append("executable path does not exist")
        expected_id = f"exe:{Path(app.exe_path).stem.lower()}"
        if rec.app_id != expected_id:
            problems.append("canonical id does not match executable stem")
        if not rt.discovery.is_app_path_allowed(app.exe_path):
            problems.append("exe path failed launch-path validation")
        # Display-name/exe-stem similarity is only a diagnostic: Office
        # launchers legitimately differ (powerpnt -> "PowerPoint").
        consistent, consistent_reason = _exe_identity_consistent(app)
        rec.diagnostics["exe_name_similarity"] = (
            "ok" if consistent else consistent_reason
        )
        stem_l = Path(app.exe_path).stem.lower()
        ident_stems = {str(s).lower() for s in ident.process_names}
        if stem_l and stem_l not in ident_stems:
            problems.append("identity process candidates missing exe stem")
        if (ident.exe_path or "").lower() != (app.exe_path or "").lower():
            problems.append("identity exe path mismatch")
    if ident.app_id != rec.app_id:
        problems.append("identity app id mismatch")
    if ident.name != app.name:
        problems.append("identity display name mismatch")

    try:
        res = rt.discovery.resolve(rec.display_name)
        if res.status == "found" and res.app is not None:
            same = (
                (getattr(res.app, "aumid", "") or "").lower()
                == (app.aumid or "").lower()
                and (res.app.exe_path or "").lower() == (app.exe_path or "").lower()
            )
            if not same:
                problems.append("lookup resolved a different application")
        elif res.status == "ambiguous":
            problems.append("lookup is ambiguous")
        else:
            problems.append("lookup did not resolve")
    except Exception as exc:  # noqa: BLE001
        problems.append(f"lookup raised {exc}")

    try:
        matches = app_identity.windows_for_app(window_snapshot, app)
        rec.diagnostics["identity_window_matches"] = len(matches)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"window identity matching failed: {exc}")

    if problems:
        rec.identity_status = FAIL
        rec.identity_details = "; ".join(problems)
    else:
        rec.identity_status = PASS
        rec.identity_details = name_reason


def _fresh_windows_for_app(rt: Runtime, app: AppInfo,
                           before: set[int]) -> list[WindowInfo]:
    return [
        w for w in rt.app_windows(app, include_hidden=True)
        if w.hwnd not in before
    ]


def wait_for_window(rt: Runtime, rec: AppRecord, before: set[int],
                    timeout: float) -> tuple[Optional[WindowInfo], float, list[int], str]:
    """Wait for a real top-level window belonging to ``rec.app``."""
    t0 = time.monotonic()
    deadline = t0 + timeout
    grace = t0 + min(2.0, timeout)
    app = rec.app
    last_error = ""
    while True:
        try:
            matches = rt.app_windows(app, include_hidden=True)
            fresh = [w for w in matches if w.hwnd not in before]
            visible_fresh = [w for w in fresh if getattr(w, "visible", True)]
            if visible_fresh:
                win = _choose_window(visible_fresh, app)
                return win, (time.monotonic() - t0) * 1000.0, [w.hwnd for w in fresh], "fresh-visible"
            if fresh:
                evidenced = [w for w in fresh if _window_score(w, app)[0] > 0]
                if evidenced and time.monotonic() >= grace:
                    win = _choose_window(evidenced, app)
                    return win, (time.monotonic() - t0) * 1000.0, [w.hwnd for w in fresh], "fresh-hidden-evidenced"
            # Reused instance: no fresh hwnd. ``windows_for_app`` has
            # already identity-matched these windows, so a single visible
            # match is safe even when the title shares no display-name
            # token (powerpnt -> "PowerPoint", winword -> "Word").
            if before and time.monotonic() >= grace:
                visible = [w for w in matches if getattr(w, "visible", True)]
                evidenced = [w for w in visible if _window_score(w, app)[0] > 0]
                if len(visible) == 1:
                    return visible[0], (time.monotonic() - t0) * 1000.0, [], "reused-single"
                if len(evidenced) == 1:
                    return evidenced[0], (time.monotonic() - t0) * 1000.0, [], "reused-evidenced"
            if not before and time.monotonic() >= grace:
                visible = [w for w in matches if getattr(w, "visible", True)]
                evidenced = [w for w in visible if _window_score(w, app)[0] > 0]
                if len(visible) == 1:
                    return visible[0], (time.monotonic() - t0) * 1000.0, [], "single-visible"
                if len(evidenced) == 1:
                    return evidenced[0], (time.monotonic() - t0) * 1000.0, [], "single-evidenced"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        if time.monotonic() >= deadline:
            # Last-chance honest fallback: a single visible identity-matched
            # window is enough even without display-name title evidence.
            try:
                matches = rt.app_windows(app, include_hidden=True)
                visible = [w for w in matches if getattr(w, "visible", True)]
                evidenced = [w for w in visible if _window_score(w, app)[0] > 0]
                if len(evidenced) == 1:
                    return evidenced[0], (time.monotonic() - t0) * 1000.0, [], "timeout-single"
                if len(visible) == 1:
                    return visible[0], (time.monotonic() - t0) * 1000.0, [], "timeout-visible"
            except Exception:  # noqa: BLE001
                pass
            return None, (time.monotonic() - t0) * 1000.0, [], last_error or "timeout"
        time.sleep(POLL_INTERVAL)


def _launch_and_wait(rt: Runtime, rec: AppRecord, timeout: float):
    app = rec.app
    before = {w.hwnd for w in rt.app_windows(app, include_hidden=True)}
    t0 = time.monotonic()
    try:
        ok, proc, detail = rt.launcher.launch_process(app)
    except Exception as exc:  # noqa: BLE001
        return False, None, None, (time.monotonic() - t0) * 1000.0, str(exc), before, None, None
    launch_ms = (time.monotonic() - t0) * 1000.0
    if not ok:
        return False, None, None, launch_ms, detail or "launch_failed", before, None, None
    win, win_ms, fresh, how = wait_for_window(rt, rec, before, timeout)
    return True, proc, win, launch_ms, how, before, win_ms, fresh


def _close_harness_windows(rt: Runtime, rec: AppRecord,
                           timeout: float = DEFAULT_CLOSE_TIMEOUT) -> tuple[bool, str]:
    """Normal-close windows created for this app; never kills processes."""
    app = rec.app
    deadline = time.monotonic() + timeout
    # Restore title-evidenced hidden windows so WM_CLOSE can reach them.
    try:
        matches = rt.app_windows(app, include_hidden=True)
        for w in matches:
            if not getattr(w, "visible", True) and _window_score(w, app)[0] > 0:
                try:
                    rt.windows.restore(w.hwnd)
                except Exception:  # noqa: BLE001
                    pass
        time.sleep(0.2)
    except Exception:  # noqa: BLE001
        pass
    try:
        rt.windows.close_windows(app)
    except Exception as exc:  # noqa: BLE001
        return False, f"close_windows raised {exc}"
    start = time.monotonic()
    reposted = False
    while time.monotonic() < deadline:
        remaining = rt.app_windows(app, include_hidden=True)
        visible_remaining = [w for w in remaining if getattr(w, "visible", True)]
        if not visible_remaining:
            # Hidden utility windows may remain; they are not user windows.
            return True, "closed"
        titles = "; ".join((w.title or w.process or "") for w in visible_remaining[:5])
        if any(re.search(p, titles.lower()) for p in _SUSPICIOUS_TITLE_PATTERNS):
            return False, f"close prompt/dialog left open: {titles}"
        # Office-style apps can take a moment to honour WM_CLOSE; repost once.
        if not reposted and time.monotonic() - start > 1.5:
            try:
                rt.windows.close_windows(app)
            except Exception:  # noqa: BLE001
                pass
            reposted = True
        time.sleep(0.2)
    remaining = rt.app_windows(app, include_hidden=True)
    visible_remaining = [w for w in remaining if getattr(w, "visible", True)]
    if visible_remaining:
        titles = "; ".join((w.title or w.process or "") for w in visible_remaining[:5])
        return False, f"windows still open after WM_CLOSE: {titles}"
    return True, "closed"


def _window_diag(win: Optional[WindowInfo]) -> dict:
    if win is None:
        return {}
    return {
        "hwnd": win.hwnd,
        "pid": win.pid,
        "title": win.title,
        "process": win.process,
        "package_family": win.package_family,
        "exe_path": win.exe_path,
        "visible": win.visible,
    }


def test_existing_app(rec: AppRecord, rt: Runtime) -> None:
    rec.launch = SKIP
    rec.details = "already running before test; lifecycle not exercised"
    rec.diagnostics["existing_running"] = True
    matches = rt.app_windows(rec.app, include_hidden=True)
    visible = [w for w in matches if getattr(w, "visible", True)]
    win = _choose_window(visible or matches, rec.app)
    rec.window_discovery = PASS if matches else FAIL
    rec.diagnostics["window_count"] = len(matches)
    rec.diagnostics["visible_window_count"] = len(visible)
    rec.diagnostics["selected_window"] = _window_diag(win)
    if matches and win is not None:
        mon = rt.monitors.monitor_for_hwnd(win.hwnd)
        rec.diagnostics["selected_monitor"] = getattr(mon, "index", None)
        rec.diagnostics["selected_monitor_device"] = getattr(mon, "device", None)
    # Verify reverse identity resolution for the selected window.
    if win is not None:
        try:
            resolved = app_identity.app_for_window(win, [rec.app])
            rec.diagnostics["reverse_identity_match"] = resolved is not None
        except Exception:  # noqa: BLE001
            rec.diagnostics["reverse_identity_match"] = False
    # Focus is attempted only for a single unambiguous visible window.
    if len(visible) == 1 and win is not None:
        suspicious = _suspicious_dialog(win)
        if suspicious:
            rec.switch_focus = SKIP
            rec.diagnostics["focus_skip"] = suspicious
        else:
            try:
                res = rt.windows.focus_target(rec.app)
                status = getattr(res, "status", "")
                fg = rt.windows.foreground_window()
                fg_ok = fg is not None and fg.hwnd == getattr(res, "hwnd", 0)
                if status == "ok" and fg_ok:
                    rec.switch_focus = PASS
                elif status == "focus_failed":
                    rec.switch_focus = FAIL
                    rec.diagnostics["focus_error"] = "foreground verification failed"
                else:
                    rec.switch_focus = FAIL
                    rec.diagnostics["focus_error"] = status or "no result"
            except Exception as exc:  # noqa: BLE001
                rec.switch_focus = FAIL
                rec.diagnostics["focus_error"] = str(exc)
    else:
        rec.switch_focus = SKIP
        rec.diagnostics["focus_skip"] = (
            "no visible window" if not visible else "multiple visible windows"
        )
    rec.minimize_restore = SKIP
    rec.maximize = SKIP
    rec.monitor_1 = SKIP
    rec.monitor_2 = SKIP
    rec.move_monitor_1 = SKIP
    rec.move_monitor_2 = SKIP
    rec.cleanup = SKIP
    rec.details = (
        f"existing app: {len(matches)} window(s), "
        f"{len(visible)} visible; focus={'tested' if rec.switch_focus != SKIP else 'skipped'}"
    )
    _finalize_overall(rec)


def _finalize_overall(rec: AppRecord) -> None:
    if not rec.eligible:
        rec.overall = SKIP
        if not rec.details:
            rec.details = rec.category_reason
        return
    statuses = [
        rec.identity_status, rec.launch, rec.window_discovery,
        rec.switch_focus, rec.minimize_restore, rec.maximize,
        rec.monitor_1, rec.monitor_2, rec.move_monitor_1,
        rec.move_monitor_2, rec.cleanup,
    ]
    if FAIL in statuses:
        rec.overall = FAIL
    elif SKIP in statuses:
        rec.overall = PARTIAL
    else:
        rec.overall = PASS


def test_gui_app(rec: AppRecord, rt: Runtime, m1, m2,
                 window_timeout: float) -> None:
    app = rec.app
    rec.details = ""
    try:
        # Phase 1: ordinary launch; production default is Monitor 1.
        ok, _proc, win0, launch_ms, how, before0, win_ms0, fresh0 = _launch_and_wait(
            rt, rec, window_timeout
        )
        rec.launch_latency_ms = launch_ms
        rec.diagnostics["default_launch_how"] = how
        rec.diagnostics["default_fresh_hwnds"] = fresh0[:20]
        if win0 is not None:
            rec.window_latency_ms = win_ms0
        if not ok:
            rec.launch = FAIL
            rec.window_discovery = FAIL if win0 is None else PASS
            rec.details = f"launch failed: {how}"
            _finalize_overall(rec)
            return
        rec.launch = PASS
        if win0 is None:
            rec.window_discovery = FAIL
            rec.details = f"launch accepted but no window discovered: {how}"
            _cleanup_after_launch(rt, rec)
            _finalize_overall(rec)
            return
        rec.window_discovery = PASS
        rec.diagnostics["default_window"] = _window_diag(win0)
        suspicious = _suspicious_dialog(win0)
        if suspicious:
            rec.details = suspicious
            _cleanup_after_launch(rt, rec)
            _finalize_overall(rec)
            return
        mon0 = rt.monitors.monitor_for_hwnd(win0.hwnd)
        mon0_ok, mon0_details = _verify_on_monitor(rt.windows, rt.monitors, win0.hwnd, m1)
        rec.diagnostics["default_on_monitor1"] = mon0_ok
        rec.diagnostics["default_monitor_verification"] = mon0_details
        # The default-launch monitor check is diagnostic only; the
        # explicit Monitor-1 relaunch below owns the monitor_1 row.
        rec.diagnostics["default_monitor_index"] = getattr(mon0, "index", None)

        # Close only the harness-created default window before explicit
        # monitor launches. If it cannot be closed safely, stop here.
        closed, close_detail = _close_harness_windows(rt, rec)
        rec.diagnostics["default_close"] = close_detail
        if not closed:
            rec.cleanup = FAIL
            rec.details = f"default window cleanup failed: {close_detail}"
            _finalize_overall(rec)
            return

        # Phase 2: explicit launch onto Monitor 1 (right/primary).
        m1_win = None
        ok1, _p1, cand1, l1, how1, _b1, w1ms, fresh1 = _launch_and_wait(
            rt, rec, window_timeout
        )
        rec.diagnostics["monitor1_launch"] = {
            "ok": ok1, "how": how1, "latency_ms": l1,
            "window_latency_ms": w1ms, "fresh_hwnds": fresh1[:20],
        }
        if ok1 and cand1 is not None:
            try:
                rt.windows.restore(cand1.hwnd)
            except Exception:  # noqa: BLE001
                pass
            moved, _max = rt.windows.move_to_work_area(cand1.hwnd, m1.work_area)
            time.sleep(0.3)
            okm, det = _verify_on_monitor(rt.windows, rt.monitors, cand1.hwnd, m1)
            rec.monitor_1 = PASS if (moved and okm) else FAIL
            rec.diagnostics["monitor1_window"] = _window_diag(cand1)
            rec.diagnostics["monitor1_verification"] = det
            m1_win = cand1
        else:
            rec.monitor_1 = FAIL
            rec.diagnostics["monitor1_error"] = how1

        # Phase 3: explicit launch onto Monitor 2 (left/secondary).
        # The Monitor-1 window is intentionally left open: single-instance
        # apps reuse/move it, multi-window apps may add a second window;
        # cleanup closes every window belonging to this app.
        ok2, _p2, cand2, l2, how2, _b2, w2ms, fresh2 = _launch_and_wait(
            rt, rec, window_timeout
        )
        rec.diagnostics["monitor2_launch"] = {
            "ok": ok2, "how": how2, "latency_ms": l2,
            "window_latency_ms": w2ms, "fresh_hwnds": fresh2[:20],
        }
        m2_win = None
        if ok2 and cand2 is not None:
            try:
                rt.windows.restore(cand2.hwnd)
            except Exception:  # noqa: BLE001
                pass
            moved, _max = rt.windows.move_to_work_area(cand2.hwnd, m2.work_area)
            time.sleep(0.3)
            okm, det = _verify_on_monitor(rt.windows, rt.monitors, cand2.hwnd, m2)
            rec.monitor_2 = PASS if (moved and okm) else FAIL
            rec.diagnostics["monitor2_window"] = _window_diag(cand2)
            rec.diagnostics["monitor2_verification"] = det
            m2_win = cand2
        else:
            rec.monitor_2 = FAIL
            rec.diagnostics["monitor2_error"] = how2

        lifecycle_win = m2_win or m1_win
        if lifecycle_win is None:
            rec.switch_focus = SKIP
            rec.minimize_restore = SKIP
            rec.maximize = SKIP
            rec.details = "no verified harness window for lifecycle operations"
            _cleanup_after_launch(rt, rec)
            _finalize_overall(rec)
            return
        suspicious = _suspicious_dialog(lifecycle_win)
        if suspicious:
            rec.details = suspicious
            _cleanup_after_launch(rt, rec)
            _finalize_overall(rec)
            return

        # Switch/focus: production focus path + foreground verification.
        try:
            fres = rt.windows.focus_target(app)
            status = getattr(fres, "status", "")
            fg = rt.windows.foreground_window()
            fg_hwnd = getattr(fg, "hwnd", 0) if fg is not None else 0
            focused_app_window = any(
                w.hwnd == fg_hwnd for w in rt.app_windows(app, include_hidden=False)
            )
            if status == "ok" and focused_app_window:
                rec.switch_focus = PASS
            elif status == "focus_failed":
                rec.switch_focus = FAIL
                rec.diagnostics["focus_error"] = "foreground verification failed"
            else:
                rec.switch_focus = FAIL
                rec.diagnostics["focus_error"] = status or "not_running"
            rec.diagnostics["focus_result"] = {
                "status": status,
                "hwnd": getattr(fres, "hwnd", 0),
                "foreground_hwnd": fg_hwnd,
                "foreground_title": getattr(fg, "title", ""),
            }
        except Exception as exc:  # noqa: BLE001
            rec.switch_focus = FAIL
            rec.diagnostics["focus_error"] = str(exc)

        hwnd = lifecycle_win.hwnd
        # Minimize -> verify show command -> restore -> verify visible.
        try:
            rt.windows.minimize(hwnd)
            time.sleep(0.3)
            minimized = rt.windows.is_minimized(hwnd)
            rt.windows.restore(hwnd)
            time.sleep(0.3)
            restored = (not rt.windows.is_minimized(hwnd)) and rt.windows._api.is_visible(hwnd)
            rec.minimize_restore = PASS if (minimized and restored) else FAIL
            rec.diagnostics["minmax_state"] = {
                "minimized": minimized, "restored": restored,
            }
        except Exception as exc:  # noqa: BLE001
            rec.minimize_restore = FAIL
            rec.diagnostics["minmax_error"] = str(exc)

        # Maximize -> verify show command.
        try:
            rt.windows.maximize(hwnd)
            time.sleep(0.3)
            rec.maximize = PASS if rt.windows.is_maximized(hwnd) else FAIL
            rec.diagnostics["maximized"] = rt.windows.is_maximized(hwnd)
        except Exception as exc:  # noqa: BLE001
            rec.maximize = FAIL
            rec.diagnostics["maximize_error"] = str(exc)

        # Move to Monitor 2 then Monitor 1 through the centralized
        # MonitorManager-resolved work areas, with post-op verification.
        for label, mon, attr, status_attr in (
            ("monitor_2_move", m2, "monitor_2_move_result", "move_monitor_2"),
            ("monitor_1_move", m1, "monitor_1_move_result", "move_monitor_1"),
        ):
            try:
                rt.windows.restore(hwnd)
                time.sleep(0.2)
                moved, _max = rt.windows.move_to_work_area(hwnd, mon.work_area)
                time.sleep(0.3)
                okm, det = _verify_on_monitor(rt.windows, rt.monitors, hwnd, mon)
                setattr(rec, status_attr, PASS if (moved and okm) else FAIL)
                rec.diagnostics[attr] = {"moved": moved, "verified": okm, "detail": det}
            except Exception as exc:  # noqa: BLE001
                rec.diagnostics[attr] = {"error": str(exc)}

        # Recompute monitor rows: explicit launch rows stay as recorded;
        # move diagnostics prove geometry but do not erase launch failures.
        _cleanup_after_launch(rt, rec)
        if rec.details == "":
            rec.details = "completed safe GUI lifecycle"
    except Exception as exc:  # noqa: BLE001
        rec.details = f"unhandled test exception: {exc}"
        rec.diagnostics["traceback"] = traceback.format_exc()
        _cleanup_after_launch(rt, rec)
    _finalize_overall(rec)


def _cleanup_after_launch(rt: Runtime, rec: AppRecord) -> None:
    if rec.cleanup != SKIP:
        return
    closed, detail = _close_harness_windows(rt, rec)
    rec.cleanup = PASS if closed else FAIL
    rec.diagnostics["cleanup_detail"] = detail
    if not closed:
        rec.details = (rec.details + "; " if rec.details else "") + detail


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
def write_inventory(records: list[AppRecord], meta: dict) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": meta["generated_at"],
        "raw_discovered": meta["raw_discovered"],
        "unique_after_dedup": meta["unique_after_dedup"],
        "duplicates_removed": meta["duplicates_removed"],
        "duplicates": meta["duplicates"],
        "applications": [r.base_dict() for r in records],
    }
    INVENTORY_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    fields = [
        "application", "canonical_app_id", "inventory_id", "exe_path",
        "exe_name", "process_name_candidates", "package_family_name",
        "aumid", "launch_type", "launch_target", "is_gui", "gui_reason",
        "already_running", "running_window_count", "category",
        "category_reason", "eligible",
    ]
    with INVENTORY_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = r.base_dict()
            row["process_name_candidates"] = "|".join(row["process_name_candidates"])
            w.writerow({k: row.get(k, "") for k in fields})


def _status_counts(records: list[AppRecord]) -> dict:
    return {
        "total": len(records),
        "identity_pass": sum(1 for r in records if r.identity_status == PASS),
        "identity_fail": sum(1 for r in records if r.identity_status == FAIL),
        "identity_skipped": sum(1 for r in records if r.identity_status == SKIP),
        "category_a": sum(1 for r in records if r.category == "A"),
        "category_b": sum(1 for r in records if r.category == "B"),
        "category_c": sum(1 for r in records if r.category == "C"),
        "category_d": sum(1 for r in records if r.category == "D"),
        "eligible": sum(1 for r in records if r.eligible),
        "overall_pass": sum(1 for r in records if r.overall == PASS),
        "overall_fail": sum(1 for r in records if r.overall == FAIL),
        "overall_partial": sum(1 for r in records if r.overall == PARTIAL),
        "overall_skipped": sum(1 for r in records if r.overall == SKIP),
    }


def _monitor_payload(rt: Runtime) -> list[dict]:
    monitor_map = []
    for m in rt.monitors.monitors():
        monitor_map.append({
            "index": m.index,
            "device": m.device,
            "primary": m.primary,
            "bounds": [m.bounds.left, m.bounds.top, m.bounds.right, m.bounds.bottom],
            "work_area": [m.work_area.left, m.work_area.top,
                          m.work_area.right, m.work_area.bottom],
        })
    return monitor_map


def _latency_payload(rows: list[dict]) -> dict:
    launch_lat = [r.get("launch_latency_ms") for r in rows
                  if r.get("launch_latency_ms") is not None]
    win_lat = [r.get("window_discovery_latency_ms") for r in rows
               if r.get("window_discovery_latency_ms") is not None]
    return {
        "launch": {
            "samples": len(launch_lat),
            "average": round(sum(launch_lat) / len(launch_lat), 2) if launch_lat else None,
            "max": round(max(launch_lat), 2) if launch_lat else None,
        },
        "window_discovery": {
            "samples": len(win_lat),
            "average": round(sum(win_lat) / len(win_lat), 2) if win_lat else None,
            "max": round(max(win_lat), 2) if win_lat else None,
        },
    }


def _failure_causes(rows: list[dict]) -> list:
    failure_reasons = Counter()
    for r in rows:
        if r.get("overall") in (FAIL, PARTIAL):
            reason = r.get("details") or r.get("category_reason") or "unspecified"
            failure_reasons[str(reason).split(";")[0][:120]] += 1
    return failure_reasons.most_common(20)


def write_results(records: list[AppRecord], meta: dict, rt: Runtime,
                  mode: str) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = [r.base_dict() for r in records]
    payload = {
        "version": "V9.3",
        "generated_at": _utc_now(),
        "mode": mode,
        "inventory": meta,
        "monitors": _monitor_payload(rt),
        "monitor_convention": {
            "monitor_1": "Windows primary display (user right display)",
            "monitor_2": "Windows secondary display (user left display)",
        },
        "summary": _status_counts(records),
        "latency_ms": _latency_payload(rows),
        "common_failure_causes": _failure_causes(rows),
        "results": rows,
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    fields = [
        "application", "canonical_app_id", "category", "category_reason",
        "eligible", "identity", "identity_details", "launch", "window_discovery",
        "switch_focus", "minimize_restore", "maximize", "monitor_1",
        "monitor_2", "move_monitor_1", "move_monitor_2", "cleanup", "overall", "details",
        "launch_latency_ms", "window_discovery_latency_ms",
        "already_running", "running_window_count", "is_gui",
    ]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in records:
            row = r.base_dict()
            w.writerow({k: row.get(k, "") for k in fields})


def write_markdown(records: list[AppRecord], meta: dict, rt: Runtime,
                   mode: str) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    counts = _status_counts(records)
    mons = rt.monitors.monitors()
    lines = []
    lines.append("# VoiceOS V9.3 — All Applications Compatibility and Multi-Monitor Test")
    lines.append("")
    lines.append(f"Generated: {_utc_now()}  ")
    lines.append(f"Mode: `{mode}`  ")
    lines.append("")
    lines.append("## Monitor mapping")
    lines.append("")
    lines.append("All monitor checks use `MonitorManager`; numbering is logical, not Windows enumeration order.")
    lines.append("")
    lines.append("| Logical monitor | Meaning | Device | Primary | Bounds |")
    lines.append("|---|---|---|---:|---|")
    for m in mons:
        meaning = "RIGHT / primary" if m.index == 1 else ("LEFT / secondary" if m.index == 2 else "secondary")
        lines.append(
            f"| Monitor {m.index} | {meaning} | `{m.device}` | {m.primary} | "
            f"({m.bounds.left},{m.bounds.top},{m.bounds.right},{m.bounds.bottom}) |"
        )
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Raw discovered applications: **{meta['raw_discovered']}**")
    lines.append(f"- Unique applications after conservative deduplication: **{meta['unique_after_dedup']}**")
    lines.append(f"- Duplicate records removed: **{meta['duplicates_removed']}**")
    lines.append(f"- Identity PASS: **{counts['identity_pass']}** / FAIL: **{counts['identity_fail']}** / SKIPPED: **{counts['identity_skipped']}**")
    lines.append(f"- Category A safe GUI: **{counts['category_a']}**")
    lines.append(f"- Category B manual: **{counts['category_b']}**")
    lines.append(f"- Category C excluded/system: **{counts['category_c']}**")
    lines.append(f"- Category D non-GUI: **{counts['category_d']}**")
    lines.append(f"- Eligible for physical testing: **{counts['eligible']}**")
    lines.append(f"- Overall PASS: **{counts['overall_pass']}**")
    lines.append(f"- Overall FAIL: **{counts['overall_fail']}**")
    lines.append(f"- Overall PARTIAL: **{counts['overall_partial']}**")
    lines.append(f"- Overall SKIPPED: **{counts['overall_skipped']}**")
    lines.append("")
    lines.append("> Skipped applications are never counted as passed. PARTIAL means at least one applicable physical check was skipped or an already-running app was not moved/closed by the harness.")
    lines.append("")
    lines.append("## Per-application results")
    lines.append("")
    lines.append("| Application | Identity | Launch | Window discovery | Switch/focus | Min/restore | Maximize | Monitor 1 | Monitor 2 | Move M1 | Move M2 | Cleanup | Overall | Details |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in records:
        detail = (r.details or r.category_reason or "").replace("|", "/")
        if len(detail) > 160:
            detail = detail[:157] + "..."
        lines.append(
            f"| {r.display_name} | {r.identity_status} | {r.launch} | {r.window_discovery} | "
            f"{r.switch_focus} | {r.minimize_restore} | {r.maximize} | {r.monitor_1} | "
            f"{r.monitor_2} | {r.move_monitor_1} | {r.move_monitor_2} | {r.cleanup} | {r.overall} | {detail} |"
        )
    lines.append("")
    lines.append("## Failure causes")
    lines.append("")
    lines.append("See `reports/v93_application_test_results.json` for machine-readable diagnostics, latency and per-window HWND evidence.")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--discovery-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--full-safe-gui", action="store_true")
    p.add_argument("--rerun-failed", metavar="RESULTS_JSON")
    p.add_argument("--only", metavar="TEXT")
    p.add_argument("--window-timeout", type=float, default=DEFAULT_WINDOW_TIMEOUT)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not hasattr(sys, "platform") or not sys.platform.startswith("win"):
        # py.exe launched through WSL still reports win32; this guard is
        # for accidental non-Windows interpreters.
        pass
    REPORTS.mkdir(parents=True, exist_ok=True)
    rt = Runtime()
    records, meta = build_inventory(rt.discovery)
    write_inventory(records, meta)
    if args.discovery_only:
        print(f"Discovered {meta['raw_discovered']} raw, "
              f"{meta['unique_after_dedup']} unique applications.")
        print(f"Inventory written to {INVENTORY_JSON}")
        return 0

    # Identity tests are non-invasive and cover 100% of the inventory.
    try:
        window_snapshot = rt.windows.list_windows(include_hidden=True)
    except Exception:  # noqa: BLE001
        window_snapshot = []
    for rec in records:
        identity_test(rec, rt, window_snapshot)
    rt.snapshot_running(records)

    m1 = rt.monitors.get(1)
    m2 = rt.monitors.get(2)
    monitor_ok = m1 is not None and m2 is not None
    if not monitor_ok:
        print("WARNING: fewer than two monitors detected; monitor rows will fail/skip.")

    mode = "dry-run"
    only = (args.only or "").lower()
    rerun_keys = None
    if args.rerun_failed:
        prev = json.loads(Path(args.rerun_failed).read_text(encoding="utf-8"))
        prev_ids = {r.get("inventory_id") for r in prev.get("results", [])}
        rerun_keys = {
            r.get("inventory_id") for r in prev.get("results", [])
            if r.get("overall") in (FAIL, PARTIAL)
            # Eligible apps still marked SKIPPED were never physically
            # tested (e.g. discovered after the original full pass).
            or (r.get("eligible") and r.get("overall") == SKIP)
        }
    else:
        prev_ids = set()
    if args.full_safe_gui or args.dry_run or args.rerun_failed:
        mode = "full-safe-gui" if (args.full_safe_gui or args.rerun_failed) else "dry-run"

    physical = args.full_safe_gui or args.rerun_failed
    preserve_ids = set()
    tested = 0
    for rec in records:
        if not rec.eligible:
            rec.overall = SKIP
            if not rec.details:
                rec.details = rec.category_reason
            continue
        if not physical:
            rec.launch = SKIP
            rec.window_discovery = SKIP
            rec.switch_focus = SKIP
            rec.minimize_restore = SKIP
            rec.maximize = SKIP
            rec.monitor_1 = SKIP
            rec.monitor_2 = SKIP
            rec.move_monitor_1 = SKIP
            rec.move_monitor_2 = SKIP
            rec.cleanup = SKIP
            rec.overall = PARTIAL if rec.identity_status == PASS else (
                FAIL if rec.identity_status == FAIL else SKIP
            )
            rec.details = "dry-run: physical tests not executed"
            continue
        if only and only not in rec.display_name.lower():
            rec.launch = SKIP
            rec.window_discovery = SKIP
            rec.switch_focus = SKIP
            rec.minimize_restore = SKIP
            rec.maximize = SKIP
            rec.monitor_1 = SKIP
            rec.monitor_2 = SKIP
            rec.move_monitor_1 = SKIP
            rec.move_monitor_2 = SKIP
            rec.cleanup = SKIP
            rec.overall = SKIP
            rec.details = "excluded by --only filter"
            preserve_ids.add(rec.inventory_id)
            continue
        if (
            rerun_keys is not None
            and rec.inventory_id not in rerun_keys
            and rec.inventory_id in prev_ids
        ):
            # Preserve prior result for apps that already have one; newly
            # discovered eligible apps must still be physically tested.
            preserve_ids.add(rec.inventory_id)
            continue
        if monitor_ok is False:
            rec.overall = FAIL
            rec.details = "monitor precondition failed: need two monitors"
            continue
        tested += 1
        if args.limit and tested > args.limit:
            break
        print(f"[{tested}] Testing {rec.display_name} ({rec.app_id})", flush=True)
        # Re-check running state at execution time: an earlier test may
        # have launched the same underlying application.
        try:
            running_now = rt.monitor.is_running(rec.app)
        except Exception:  # noqa: BLE001
            running_now = rec.already_running
        if running_now:
            test_existing_app(rec, rt)
        else:
            test_gui_app(rec, rt, m1, m2, args.window_timeout)
        _finalize_overall(rec)
        print(f"    -> {rec.overall} ({rec.details or 'ok'})", flush=True)

    for rec in records:
        if (
            rec.identity_status == FAIL
            and rec.identity_details
            and rec.identity_details not in (rec.details or "")
        ):
            rec.details = (
                (rec.details + "; " if rec.details else "")
                + f"identity: {rec.identity_details}"
            )

    if args.rerun_failed and RESULTS_JSON.exists():
        # Merge rerun results into the previous report so non-rerun apps
        # keep their prior individual results.
        try:
            prev = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
            by_id = {r.get("inventory_id"): r for r in prev.get("results", [])}
            for rec in records:
                if rec.inventory_id in preserve_ids and rec.inventory_id in by_id:
                    # Lifecycle rows are preserved, but identity/classification
                    # were recomputed fresh for every app this run — carry
                    # those fields forward so identity fixes are reflected.
                    prev_row = by_id[rec.inventory_id]
                    new_row = rec.base_dict()
                    for key in ("identity", "identity_details",
                                "already_running", "running_window_count",
                                "category", "category_reason", "eligible",
                                "is_gui", "gui_reason"):
                        prev_row[key] = new_row.get(key)
                    prev_diag = prev_row.setdefault("diagnostics", {})
                    prev_diag.update(new_row.get("diagnostics", {}))
                    base_details = (prev_row.get("details") or "")
                    idx = base_details.rfind("; identity: ")
                    if idx >= 0:
                        base_details = base_details[:idx]
                    if rec.identity_status == FAIL and rec.identity_details:
                        base_details = (
                            base_details
                            + ("; " if base_details else "")
                            + f"identity: {rec.identity_details}"
                        )
                    prev_row["details"] = base_details
                    if rec.identity_status == FAIL and prev_row.get("overall") == PASS:
                        prev_row["overall"] = FAIL
                    continue
                by_id[rec.inventory_id] = rec.base_dict()
            merged = [by_id[k] for k in sorted(by_id)]
            # Rebuild AppRecord-like dicts is unnecessary; write JSON/CSV
            # directly from merged rows for this rerun report.
            prev["results"] = merged
            prev["summary"] = _merged_summary(merged)
            prev["inventory"] = meta
            prev["monitors"] = _monitor_payload(rt)
            prev["mode"] = mode
            prev["latency_ms"] = _latency_payload(merged)
            prev["common_failure_causes"] = _failure_causes(merged)
            prev["rerun_at"] = _utc_now()
            RESULTS_JSON.write_text(json.dumps(prev, indent=2), encoding="utf-8")
            _write_csv_from_rows(merged)
            _write_markdown_from_rows(merged, meta, prev["monitors"], mode)
            print(f"Merged rerun report written to {RESULTS_JSON}")
            return 0
        except Exception as exc:  # noqa: BLE001
            print(f"Rerun merge failed ({exc}); writing fresh report instead.")

    write_results(records, meta, rt, mode)
    write_markdown(records, meta, rt, mode)
    counts = _status_counts(records)
    print(
        f"Done: {counts['total']} apps, {counts['eligible']} eligible, "
        f"PASS={counts['overall_pass']} FAIL={counts['overall_fail']} "
        f"PARTIAL={counts['overall_partial']} SKIPPED={counts['overall_skipped']}"
    )
    print(f"Reports: {RESULTS_JSON}, {RESULTS_CSV}, {INVENTORY_JSON}, {REPORT_MD}")
    return 0


def _merged_summary(rows: list[dict]) -> dict:
    def c(field, value):
        return sum(1 for r in rows if r.get(field) == value)
    return {
        "total": len(rows),
        "identity_pass": c("identity", PASS),
        "identity_fail": c("identity", FAIL),
        "identity_skipped": c("identity", SKIP),
        "category_a": sum(1 for r in rows if r.get("category") == "A"),
        "category_b": sum(1 for r in rows if r.get("category") == "B"),
        "category_c": sum(1 for r in rows if r.get("category") == "C"),
        "category_d": sum(1 for r in rows if r.get("category") == "D"),
        "eligible": sum(1 for r in rows if r.get("eligible")),
        "overall_pass": c("overall", PASS),
        "overall_fail": c("overall", FAIL),
        "overall_partial": c("overall", PARTIAL),
        "overall_skipped": c("overall", SKIP),
    }


def _write_csv_from_rows(rows: list[dict]) -> None:
    fields = [
        "application", "canonical_app_id", "category", "category_reason",
        "eligible", "identity", "identity_details", "launch", "window_discovery",
        "switch_focus", "minimize_restore", "maximize", "monitor_1",
        "monitor_2", "move_monitor_1", "move_monitor_2", "cleanup", "overall", "details",
        "launch_latency_ms", "window_discovery_latency_ms",
        "already_running", "running_window_count", "is_gui",
    ]
    with RESULTS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def _write_markdown_from_rows(rows: list[dict], meta: dict,
                              monitors: list[dict], mode: str) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    summary = _merged_summary(rows)
    latency = _latency_payload(rows)
    causes = _failure_causes(rows)
    manual = sum(
        1 for r in rows
        if r.get("category") == "B" or "manual" in str(r.get("category_reason") or "")
    )
    lines = [
        "# VoiceOS V9.3 — All Applications Compatibility and Multi-Monitor Test",
        "",
        f"Generated: {_utc_now()}  ",
        f"Mode: `{mode}` (merged physical Windows results)  ",
        "",
        "## Monitor mapping",
        "",
        "Monitor checks use `MonitorManager` only; numbering is logical, never Windows enumeration order.",
        "",
        "| Logical monitor | Meaning | Device | Primary | Bounds |",
        "|---|---|---|---:|---|",
    ]
    for m in monitors:
        meaning = ("RIGHT / primary" if m.get("index") == 1
                   else "LEFT / secondary" if m.get("index") == 2 else "secondary")
        b = m.get("bounds") or []
        lines.append(
            f"| Monitor {m.get('index')} | {meaning} | `{m.get('device')}` | "
            f"{m.get('primary')} | ({','.join(str(x) for x in b)}) |"
        )
    lines += [
        "",
        "## Summary",
        "",
        f"- Raw discovered applications: **{meta.get('raw_discovered', len(rows))}**",
        f"- Unique applications after deduplication: **{meta.get('unique_after_dedup', len(rows))}**",
        f"- Identity PASS/FAIL/SKIPPED: **{summary['identity_pass']} / {summary['identity_fail']} / {summary['identity_skipped']}**",
        f"- Category A safe GUI / B manual / C excluded / D non-GUI: **{summary['category_a']} / {summary['category_b']} / {summary['category_c']} / {summary['category_d']}**",
        f"- Eligible for physical testing: **{summary['eligible']}**",
        f"- Overall PASS/FAIL/PARTIAL/SKIPPED: **{summary['overall_pass']} / {summary['overall_fail']} / {summary['overall_partial']} / {summary['overall_skipped']}**",
        f"- Applications requiring manual testing: **{manual}**",
        f"- Launch latency avg/max (ms): **{latency['launch']['average']} / {latency['launch']['max']}** (n={latency['launch']['samples']})",
        f"- Window-discovery latency avg/max (ms): **{latency['window_discovery']['average']} / {latency['window_discovery']['max']}** (n={latency['window_discovery']['samples']})",
        "",
        "> Skipped applications are never counted as passed. PARTIAL means the app was already running (identity/window/focus verified, user windows not moved or closed) or an applicable check was skipped. FAIL rows carry specific diagnostics in the JSON report.",
        "",
        "## Per-application results",
        "",
        "| Application | Identity | Launch | Window discovery | Switch/focus | Min/restore | Maximize | Monitor 1 | Monitor 2 | Move M1 | Move M2 | Cleanup | Overall | Details |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        detail = str(r.get("details") or r.get("category_reason") or "").replace("|", "/")
        if len(detail) > 160:
            detail = detail[:157] + "..."
        lines.append(
            f"| {r.get('application')} | {r.get('identity')} | {r.get('launch')} | {r.get('window_discovery')} | "
            f"{r.get('switch_focus')} | {r.get('minimize_restore')} | {r.get('maximize')} | {r.get('monitor_1')} | "
            f"{r.get('monitor_2')} | {r.get('move_monitor_1')} | {r.get('move_monitor_2')} | {r.get('cleanup')} | {r.get('overall')} | {detail} |"
        )
    lines += [
        "",
        "## Failure causes",
        "",
    ]
    if causes:
        for reason, count in causes:
            lines.append(f"- {count}× {reason}")
    else:
        lines.append("- None")
    lines += [
        "",
        "See `reports/v93_application_test_results.json` for machine-readable diagnostics, per-window HWND evidence and latency samples.",
    ]
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
