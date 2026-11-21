"""Dynamic installed-application discovery for VoiceOS.

The inventory is built from the user's actual Windows installation:

- ``App Paths`` registry keys (HKLM + HKCU, 32/64-bit views)
- Uninstall registry entries (DisplayName + DisplayIcon)
- User and system Start Menu ``.lnk`` shortcuts (resolved to exe targets)
- Per-user application directories (``%LOCALAPPDATA%\\Programs`` etc.)
- Well-known install locations as additional candidates

Results are validated (existing ``.exe`` files only) and cached with a
TTL. Matching is scored and case/punctuation-insensitive; ambiguous
matches are reported, never silently chosen. Discovery is read-only.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_UNINSTALL_KEYS = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall",
    r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall",
)
_APP_PATHS_KEYS = (
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths",
    r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\App Paths",
)

_START_MENU_SUBDIR = r"Microsoft\Windows\Start Menu\Programs"
_MAX_SHORTCUTS = 600
_MAX_LNK_BYTES = 1_000_000

# Packaged-app launch id: PackageFamilyName!AppId (e.g. PFN!App).
_AUMID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.\-_]*![A-Za-z0-9][A-Za-z0-9.\-_]*$")


@dataclass(frozen=True)
class AppInfo:
    """A discovered, validated application."""

    name: str
    exe_path: str
    source: str = "unknown"
    publisher: str = ""
    version: str = ""
    category: str = ""
    aliases: tuple = ()
    launch_kind: str = "exe"          # "exe" | "aumid"
    aumid: str = ""                   # PackageFamily!AppId when packaged
    last_seen: float = 0.0

    @property
    def exe_name(self) -> str:
        if self.launch_kind == "aumid":
            return _aumid_exe_name(self.aumid)
        return Path(self.exe_path).stem.lower()

    @property
    def launch_target(self) -> str:
        """What the launcher actually opens."""
        if self.launch_kind == "aumid":
            return f"shell:AppsFolder\\{self.aumid}"
        return self.exe_path


@dataclass
class ResolveResult:
    """Outcome of matching a spoken name against the inventory."""

    status: str  # "found" | "ambiguous" | "missing"
    app: Optional[AppInfo] = None
    options: list = field(default_factory=list)
    score: int = 0  # best match score (100 = exact/alias hit)


# PFN segments that carry no app identity (release channels etc.).
_GENERIC_PFN_TOKENS = {
    "stable", "beta", "alpha", "preview", "release", "production",
    "prod", "dev", "canary",
}


def _aumid_exe_name(aumid: str) -> str:
    """Best-effort executable stem for a packaged (AUMID) app.

    PFN shape: ``<PublisherHash>.<AppName>_<Hash>`` (sometimes with
    extra dotted segments, e.g. ``Microsoft.MicrosoftEdge.Stable_x``).
    The old behaviour returned the first dotted segment — the publisher
    hash — which is never a usable exe/process name. Prefer the last
    non-generic segment with its ``_<hash>`` suffix stripped.
    """
    pfn = (aumid or "").split("!")[0]
    segs = [s for s in pfn.split(".") if s]
    for seg in reversed(segs[1:] or segs):
        stem = re.sub(r"_[A-Za-z0-9]+$", "", seg).lower()
        if stem and stem not in _GENERIC_PFN_TOKENS:
            return stem
    return (segs[-1] if segs else "").lower()


def _env_path(env: dict, *names: str) -> str:
    lowered = {k.lower(): v for k, v in env.items()}
    for n in names:
        v = env.get(n) or lowered.get(n.lower())
        if v:
            return v
    return ""


# Optional spoken aliases -> canonical app names. Kept separate from the
# matching engine; only used when the raw query finds nothing.
_ALIASES = {
    "vs code": "visual studio code",
    "vs-code": "visual studio code",
    "vscode": "visual studio code",
    "edge": "microsoft edge",
    "chrome browser": "google chrome",
    "file explorer": "file explorer",
    "windows explorer": "file explorer",
    "calc": "calculator",
    "v s code": "visual studio code",
    "vs code": "visual studio code",
    "note pad": "notepad",
    "note pad plus": "notepad",
    "visual studio": "visual studio code",
    "chrome browser": "google chrome",
    "google chrome browser": "google chrome",
    "microsoft edge browser": "microsoft edge",
    "file manager": "file explorer",
    "windows explorer": "file explorer",
    "command prompt": "command prompt",
    "powershell": "powershell",
    "task manager": "task manager",
    "settings": "settings",
    "control panel": "control panel",
    "intellij": "intellij idea",
    "intellij idea": "intellij idea",
    "idea": "intellij idea",
    "pycharm": "pycharm",
    "webstorm": "webstorm",
    "windows terminal": "windows terminal",
    "window terminal": "windows terminal",
    "terminal": "windows terminal",
    "word": "microsoft word",
    "excel": "microsoft excel",
    "powerpoint": "microsoft powerpoint",
    "outlook": "microsoft outlook",
    "teams": "microsoft teams",
    "discord": "discord",
    "vlc": "vlc media player",
    "vlc player": "vlc media player",
    "media player": "vlc media player",
    "paint": "paint",
    "paint 3d": "paint 3d",
    "snipping tool": "snipping tool",
    "taskmgr": "task manager",
    "file manager": "file explorer",
    "my computer": "file explorer",
    "this pc": "file explorer",
    # STT pronunciation/spelling variants observed in physical-mic logs.
    "devine": "devin",
    "devinn": "devin",
    "devon": "devin",
    "whats app": "whatsapp",
    "what s app": "whatsapp",
    "whatsap": "whatsapp",
}


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation/extensions, collapse whitespace."""
    t = (text or "").lower()
    t = re.sub(r"\.exe\b", "", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


_VOWELS = frozenset("aeiouy")


def _skeleton(text: str) -> str:
    """Consonant skeleton for STT-robust name comparison.

    Vowel substitutions are the most common speech-recognition error
    ("divin" for "devin", "chrome" for "chrom"). Reducing a name to
    its first letter plus consonants makes those variants collapse
    to the same key while staying specific enough to avoid most
    collisions. Multi-word names are concatenated first.
    """
    t = _normalize(text).replace(" ", "")
    if not t:
        return ""
    return t[0] + "".join(c for c in t[1:] if c not in _VOWELS)


def _score(query: str, app: AppInfo) -> int:
    """Match score 0-100 for ``query`` against ``app``."""
    q = _normalize(query)
    if not q:
        return 0
    # Verified per-app aliases are authoritative.
    for alias in getattr(app, "aliases", ()) or ():
        if _normalize(alias) == q:
            return 100
    name = _normalize(app.name)
    stem = _normalize(app.exe_name)
    if name == q or stem == q:
        return 100
    if name.startswith(q) or stem.startswith(q):
        return 80
    if q in name or q in stem:
        return 60
    # All query tokens present in the name -> partial credit.
    qt = set(q.split())
    nt = set(name.split()) | set(stem.split())
    if qt and qt <= nt:
        return 55
    # Reverse subset: every app-name token appears in the query
    # ("visual studio code" -> app "code"). Weak score on purpose — the
    # caller turns it into a clarification, never a silent launch.
    at = set(name.split()) | set(stem.split())
    if at and at <= qt:
        return 55
    # Consonant-skeleton match: STT vowel errors ("divin" -> "devin",
    # "grown" -> "groan") collapse to the same skeleton. Requires at
    # least 3 skeleton chars so tiny names can't collide trivially.
    # Scored at the "confident" boundary: a single skeleton match is
    # trusted, two skeleton-equal apps still become a clarification.
    qs = _skeleton(q)
    if len(qs) >= 3 and (qs == _skeleton(name) or qs == _skeleton(stem)):
        return 80
    # Fuzzy fallback for STT spelling variations ("spotifly",
    # "note pad", "chrom"). Only confident ratios score.
    best = max(
        difflib.SequenceMatcher(None, q, name).ratio(),
        difflib.SequenceMatcher(None, q, stem).ratio(),
    )
    if best >= 0.86:
        return 62
    return 0


class ApplicationDiscovery:
    """Discover and resolve installed Windows applications.

    Args:
        env: Environment mapping (defaults to ``os.environ``); injectable
            for tests.
        file_exists: Predicate replacing ``Path.is_file`` (tests).
        registry_reader: ``(key_path) -> {subkey: {value: data}}`` reader
            (tests); defaults to ``winreg``.
        lnk_resolver: ``(lnk_path) -> exe_path`` shortcut resolver
            (tests); defaults to the built-in binary parser.
        dir_lister: ``(dir) -> iterable of file paths`` (tests).
        cache_ttl: Seconds before the cache is considered stale.
        extra_candidates: Additional ``{name: [paths]}`` seed candidates.
        extra_dirs: Additional directories scanned like ``Programs``.
        start_apps_lister: ``() -> [(name, app_id)]`` packaged-app
            source (tests); defaults to ``Get-StartApps``.
        command_runner: ``(argv) -> stdout`` for packaged discovery.
    """

    def __init__(
        self,
        env: Optional[dict] = None,
        file_exists: Optional[Callable[[str], bool]] = None,
        registry_reader: Optional[Callable[[str], dict]] = None,
        lnk_resolver: Optional[Callable[[str], str]] = None,
        dir_lister: Optional[Callable[[str], list]] = None,
        cache_ttl: float = 300.0,
        extra_candidates: Optional[dict] = None,
        extra_dirs: Optional[list] = None,
        start_apps_lister: Optional[Callable[[], list]] = None,
        command_runner: Optional[Callable[[list], str]] = None,
    ) -> None:
        self._env = env if env is not None else dict(os.environ)
        self._file_exists = file_exists or (lambda p: Path(p).is_file())
        self._registry_reader = registry_reader or self._winreg_read
        self._lnk_resolver = lnk_resolver or self._resolve_lnk
        self._dir_lister = dir_lister or self._list_files
        self._cache_ttl = cache_ttl
        self._extra = extra_candidates or {}
        self._extra_dirs = list(extra_dirs or [])
        self._start_apps_lister = start_apps_lister
        self._command_runner = command_runner or self._run_command
        self._cache: list[AppInfo] = []
        self._cached_at = 0.0
        self._previous: dict = {}   # last completed snapshot (path->name)
        self._recent: list = []     # AppInfo added since previous scan
        self._refreshing = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def discover(self, force: bool = False) -> list[AppInfo]:
        """Return all discovered applications (cached)."""
        if (
            not force
            and self._cache
            and time.monotonic() - self._cached_at < self._cache_ttl
        ):
            return list(self._cache)
        apps: dict[str, AppInfo] = {}
        now = time.time()
        sources = (
            self._seed_candidates()
            + self._app_paths()
            + self._uninstall_entries()
            + self._start_menu_shortcuts()
            + self._program_dir_exes()
            + self._packaged_apps()
        )
        for info in sources:
            key = self._app_key(info)
            if not key:
                continue
            existing = apps.get(key)
            if existing is None or self._prefer(info, existing):
                apps[key] = AppInfo(
                    name=info.name, exe_path=info.exe_path,
                    source=info.source, publisher=info.publisher,
                    version=info.version, category=info.category,
                    aliases=info.aliases, launch_kind=info.launch_kind,
                    aumid=info.aumid, last_seen=now,
                )
        # Dedup same-product entries across sources: a packaged AUMID
        # and an exe for "Notepad" are one app, not two. Prefer the
        # exe launch kind; otherwise keep the higher-ranked source.
        fresh = sorted(apps.values(), key=lambda a: a.name.lower())
        by_name: dict = {}
        for a in fresh:
            key = _normalize(a.name) or _normalize(a.exe_name)
            cur = by_name.get(key)
            if cur is None:
                by_name[key] = a
                continue
            def rank(x):
                r = {"uninstall": 4, "app_paths": 3, "shortcut": 2,
                     "packaged": 3, "program_dir": 1, "custom_dir": 1,
                     "seed": 0}
                return (0 if x.launch_kind == "exe" else -1,
                        r.get(x.source, 0))
            if rank(a) > rank(cur):
                by_name[key] = a
        fresh = sorted(by_name.values(), key=lambda a: a.name.lower())
        # Track additions/removals between completed scans.
        new_keys = set(apps)
        # First completed scan establishes the baseline: nothing is
        # "recent" until a second snapshot exists to diff against.
        added = (
            [a for a in fresh if self._app_key(a) not in self._previous]
            if self._previous else []
        )
        self._recent = added
        self._previous = new_keys
        self._cache = fresh
        self._cached_at = time.monotonic()
        logger.info("Discovered %d applications", len(self._cache))
        return list(self._cache)

    @staticmethod
    def _app_key(info: AppInfo) -> str:
        if info.launch_kind == "aumid":
            return f"aumid:{info.aumid.lower()}"
        return info.exe_path.lower()

    @staticmethod
    def _prefer(new: AppInfo, old: AppInfo) -> bool:
        """Prefer richer/more authoritative records on key collision."""
        rank = {"uninstall": 4, "app_paths": 3, "shortcut": 2,
                "program_dir": 1, "packaged": 3, "seed": 0,
                "custom_dir": 1}
        return rank.get(new.source, 0) > rank.get(old.source, 0)

    def find_all(self, query: str) -> list[tuple[AppInfo, int]]:
        """All inventory matches for ``query`` with scores, best first."""
        if not query or not query.strip():
            return []
        scored = [
            (a, _score(query, a)) for a in self.discover()
        ]
        return sorted(
            (x for x in scored if x[1] >= 50),
            key=lambda x: -x[1],
        )

    def resolve(self, query: str) -> ResolveResult:
        """Match ``query`` to the inventory.

        - exactly one clear winner -> ``found``
        - several plausible matches -> ``ambiguous`` (never guesses)
        - nothing -> ``missing``
        """
        matches = self.find_all(query)
        alias = _ALIASES.get(_normalize(query))
        if alias:
            alias_matches = self.find_all(alias)
            # A verified alias that lands an exact/strong hit outranks
            # weak fuzzy near-misses of the raw query ("Devine" must
            # resolve to Devin, not stall on a 62-score guess).
            if (
                alias_matches
                and alias_matches[0][1] >= 80
                and (not matches or matches[0][1] < 80)
            ):
                matches = alias_matches
        if not matches and self._maybe_refresh_on_miss():
            # On-demand refresh: the app may have been installed since
            # the last scan. Retry matching once against fresh data.
            matches = self.find_all(query)
            if not matches:
                alias = _ALIASES.get(_normalize(query))
                if alias:
                    matches = self.find_all(alias)
        if not matches:
            return ResolveResult("missing")
        top_score = matches[0][1]
        if len(matches) == 1:
            # A single candidate is only trusted on a strong match;
            # weaker hits become a clarification, never a launch.
            if top_score >= 80:
                return ResolveResult(
                    "found", app=matches[0][0], score=top_score
                )
            return ResolveResult(
                "ambiguous", options=[matches[0][0]], score=top_score
            )
        second = matches[1][1]
        # A dominant top score is still a reliable single match.
        if top_score >= 80 and top_score - second >= 20:
            return ResolveResult(
                "found", app=matches[0][0], score=top_score
            )
        plausible = [a for a, s in matches if s >= 55][:5]
        return ResolveResult(
            "ambiguous", options=plausible, score=top_score
        )

    def find(self, query: str) -> Optional[AppInfo]:
        """Best single match, or None (compatibility helper)."""
        result = self.resolve(query)
        return result.app if result.status == "found" else None

    def is_app_path_allowed(self, exe_path: str) -> bool:
        """Validate an executable path before launch/close."""
        if not exe_path:
            return False
        # Packaged launch identifiers are validated by shape, not path.
        if "!" in exe_path:
            return bool(_AUMID_RE.match(exe_path))
        return self._valid_exe(exe_path)

    def refresh(self) -> list[AppInfo]:
        """Force a full re-discovery (explicit refresh command)."""
        return self.discover(force=True)

    def recently_added(self) -> list[AppInfo]:
        """Apps discovered since the previous completed scan."""
        return list(self._recent)

    def invalidate(self) -> None:
        """Mark the cache stale; next discover() rescans."""
        self._cached_at = 0.0

    def _maybe_refresh_on_miss(self) -> bool:
        """Refresh once when a lookup misses and cache may be stale."""
        if self._refreshing:
            return False
        # Only pay the rescan cost when the cache is at least half stale
        # or was populated before this feature's recents tracking ran.
        age = time.monotonic() - self._cached_at
        if age < self._cache_ttl * 0.5 and self._cache:
            return False
        try:
            self._refreshing = True
            self.discover(force=True)
        except Exception:  # noqa: BLE001
            logger.debug("On-demand discovery refresh failed",
                         exc_info=True)
            return False
        finally:
            self._refreshing = False
        return True

    # ------------------------------------------------------------------
    # Lifecycle helpers: registry watcher + background refresh
    # ------------------------------------------------------------------
    def start_registry_watcher(self, on_invalidate=None, api=None):
        """Watch uninstall/App Paths keys; invalidate cache on change.

        Returns a stop callable. Uses ``advapi32.RegNotifyChangeKeyValue``
        via ctypes on a daemon thread — ``winreg`` does not expose that
        API. Best-effort: when the native watcher cannot start, the
        periodic background refresh remains the fallback and the
        failure is logged clearly. Never raises into voice services.
        """
        if api is None:
            try:
                api = _WinRegNotifyApi()
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Registry watcher unavailable (%s); relying on "
                    "periodic background refresh", exc,
                )
                return lambda: None
        watcher = _RegistryWatcher(self, api, on_invalidate)
        return watcher.start()

    def start_background_refresh(self, interval: float = 600.0):
        """Daemon thread re-running discover() every ``interval`` s.

        Returns a stop callable. Never blocks the GUI or microphone.
        """
        import threading

        stop = threading.Event()

        def _loop():
            while not stop.is_set():
                if stop.wait(interval):
                    break
                try:
                    self.discover(force=True)
                except Exception:  # noqa: BLE001
                    logger.debug("Background discovery refresh failed",
                                 exc_info=True)

        t = threading.Thread(target=_loop, daemon=True,
                             name="voiceos-discovery")
        t.start()
        return stop.set

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    def _seed_candidates(self) -> list[AppInfo]:
        """Well-known install locations (supplementary, not authoritative)."""
        pf = _env_path(self._env, "ProgramFiles", "ProgramW6432")
        pfx = _env_path(self._env, "ProgramFiles(x86)", "ProgramW6432")
        lad = _env_path(self._env, "LOCALAPPDATA")
        windir = _env_path(self._env, "SystemRoot", "windir") or r"C:\Windows"
        seeds = {
            "Google Chrome": [
                rf"{pf}\Google\Chrome\Application\chrome.exe",
                rf"{pfx}\Google\Chrome\Application\chrome.exe",
                rf"{lad}\Google\Chrome\Application\chrome.exe",
            ],
            "Microsoft Edge": [
                rf"{pfx}\Microsoft\Edge\Application\msedge.exe",
                rf"{pf}\Microsoft\Edge\Application\msedge.exe",
            ],
            "Notepad": [
                rf"{windir}\System32\notepad.exe",
                rf"{windir}\notepad.exe",
            ],
            "Calculator": [rf"{windir}\System32\calc.exe"],
            "File Explorer": [rf"{windir}\explorer.exe"],
            "Visual Studio Code": [
                rf"{lad}\Programs\Microsoft VS Code\Code.exe",
                rf"{pf}\Microsoft VS Code\Code.exe",
                rf"{pfx}\Microsoft VS Code\Code.exe",
            ],
        }
        seeds.update(self._extra)
        out = []
        for name, candidates in seeds.items():
            for path in candidates:
                if path and self._valid_exe(path):
                    out.append(AppInfo(name=name, exe_path=path, source="seed"))
                    break
        return out

    def _app_paths(self) -> list[AppInfo]:
        """HKLM/HKCU App Paths entries (exe stem -> exe path)."""
        out = []
        for key in _APP_PATHS_KEYS:
            try:
                subs = self._registry_reader(key)
            except Exception:  # noqa: BLE001
                continue
            for sub, values in subs.items():
                default = values.get("", "") or next(
                    (v for k, v in values.items() if not k), ""
                )
                path = str(default).split(",")[0].strip().strip('"')
                if self._valid_exe(path):
                    out.append(
                        AppInfo(
                            name=Path(sub).stem,
                            exe_path=path,
                            source="app_paths",
                        )
                    )
        return out

    def _uninstall_entries(self) -> list[AppInfo]:
        """Uninstall registry DisplayName/DisplayIcon entries."""
        out = []
        for key in _UNINSTALL_KEYS:
            try:
                subs = self._registry_reader(key)
            except Exception:  # noqa: BLE001
                continue
            for _sub, values in subs.items():
                name = values.get("DisplayName")
                icon = str(values.get("DisplayIcon", ""))
                path = icon.split(",")[0].strip().strip('"')
                if not name:
                    continue
                # Quiet/system components are not desktop apps.
                if str(name).startswith("(") or "redistributable" in str(
                    name
                ).lower():
                    continue
                if not self._valid_exe(path):
                    # DisplayIcon often points at .ico/.cpl — fall back
                    # to a bounded probe of the registered install dir.
                    path = self._exe_in_install_location(
                        str(values.get("InstallLocation", "") or ""),
                        str(name),
                    )
                if not path:
                    continue
                out.append(
                    AppInfo(
                        name=str(name),
                        exe_path=path,
                        source="uninstall",
                        publisher=str(values.get("Publisher", "") or ""),
                        version=str(values.get("DisplayVersion", "") or ""),
                        category=str(values.get("SystemComponent", "") or ""),
                    )
                )
        return out

    def _exe_in_install_location(self, location: str, name: str) -> str:
        """Find a launchable exe inside a registered install dir.

        Bounded: top-level *.exe only (max 25). Prefers an exe whose
        stem matches the product name; common GUI launchers
        (``*w.exe``) win ties. Returns "" when nothing is convincing —
        a wrong guess is worse than no entry.
        """
        location = (location or "").strip().strip('"')
        if not location or not os.path.isdir(location):
            return ""
        candidates = [location]
        bin_dir = os.path.join(location, "bin")
        if os.path.isdir(bin_dir):
            candidates.append(bin_dir)
        exes = []
        for base in candidates:
            try:
                exes.extend(
                    os.path.join(base, f)
                    for f in os.listdir(base)
                    if f.lower().endswith(".exe")
                )
            except OSError:
                continue
            if len(exes) > 25:
                break
        exes = [e for e in exes if self._valid_exe(e)][:25]
        if not exes:
            return ""
        if len(exes) == 1:
            return exes[0]
        q = _normalize(name).split()
        first = q[0] if q else ""
        stem_matches = [
            e for e in exes
            if first and first in _normalize(Path(e).stem)
        ]
        if len(stem_matches) == 1:
            return stem_matches[0]
        gui = [e for e in exes if Path(e).stem.lower().endswith("w")]
        if len(gui) == 1:
            return gui[0]
        return ""

    def _shortcut_dirs(self) -> list[str]:
        appdata = _env_path(self._env, "APPDATA")
        programdata = _env_path(self._env, "ProgramData")
        dirs = []
        if appdata:
            dirs.append(os.path.join(appdata, _START_MENU_SUBDIR))
        if programdata:
            dirs.append(os.path.join(programdata, _START_MENU_SUBDIR))
        return dirs

    def _start_menu_shortcuts(self) -> list[AppInfo]:
        """Resolve Start Menu .lnk files to their exe targets."""
        out = []
        seen = 0
        for base in self._shortcut_dirs():
            try:
                files = self._dir_lister(base)
            except Exception:  # noqa: BLE001
                continue
            for lnk in files:
                if seen >= _MAX_SHORTCUTS:
                    break
                if not str(lnk).lower().endswith(".lnk"):
                    continue
                seen += 1
                try:
                    target = self._lnk_resolver(str(lnk))
                except Exception:  # noqa: BLE001
                    target = ""
                if target and self._valid_exe(target):
                    name = Path(lnk).stem
                    out.append(
                        AppInfo(name=name, exe_path=target, source="shortcut")
                    )
        return out

    def _program_dir_exes(self) -> list[AppInfo]:
        """Scan per-user app dirs (%LOCALAPPDATA%\\Programs\\*\\*.exe)."""
        out = []
        lad = _env_path(self._env, "LOCALAPPDATA")
        roots = []
        if lad:
            roots.append(os.path.join(lad, "Programs"))
        # Store app-execution aliases live directly under WindowsApps.
        if lad:
            roots.append(os.path.join(lad, "Microsoft", "WindowsApps"))
        roots.extend(self._extra_dirs)
        for programs in roots:
            out.extend(self._scan_app_dir(programs))
        return out

    def _scan_app_dir(self, programs: str) -> list[AppInfo]:
        out = []
        try:
            entries = self._dir_lister(programs)
        except Exception:  # noqa: BLE001
            return out
        for entry in entries:
            # entry may be a dir; look one level deep for .exe files
            if not str(entry).lower().endswith(".exe"):
                try:
                    for sub in self._dir_lister(str(entry)):
                        if str(sub).lower().endswith(".exe") and self._valid_exe(str(sub)):
                            out.append(
                                AppInfo(
                                    name=Path(sub).stem,
                                    exe_path=str(sub),
                                    source="program_dir",
                                )
                            )
                except Exception:  # noqa: BLE001
                    continue
            elif self._valid_exe(str(entry)):
                out.append(
                    AppInfo(
                        name=Path(entry).stem,
                        exe_path=str(entry),
                        source="program_dir",
                    )
                )
        return out

    # ------------------------------------------------------------------
    # Packaged (MSIX/Store) applications via Get-StartApps
    # ------------------------------------------------------------------
    def _packaged_apps(self) -> list[AppInfo]:
        """Discover packaged apps and their launch identifiers.

        ``Get-StartApps`` is a fixed read-only PowerShell cmdlet; its
        output is name + AppID where AppID is either an AUMID
        (``PFN!App``) or an exe path. Arbitrary spoken names never reach
        this code — only Windows-reported entries are used.
        """
        lister = self._start_apps_lister
        if lister is None:
            try:
                lister = self._default_start_apps
            except Exception:  # noqa: BLE001
                return []
        try:
            entries = lister() or []
        except Exception:  # noqa: BLE001
            logger.debug("Get-StartApps failed", exc_info=True)
            return []
        out = []
        for name, app_id in entries:
            name = str(name or "").strip()
            app_id = str(app_id or "").strip()
            if not name or not app_id:
                continue
            if "!" in app_id and _AUMID_RE.match(app_id):
                out.append(AppInfo(
                    name=name, exe_path="", source="packaged",
                    launch_kind="aumid", aumid=app_id,
                ))
            elif app_id.lower().endswith(".exe"):
                path = app_id.split(",")[0].strip().strip('"')
                if self._valid_exe(path):
                    out.append(AppInfo(
                        name=name, exe_path=path, source="packaged",
                    ))
        return out

    def _default_start_apps(self):
        """Run Get-StartApps and parse 'Name','AppID' rows."""
        import csv
        import io

        raw = self._command_runner(
            [
                "powershell", "-NoProfile", "-Command",
                "Get-StartApps | ConvertTo-Csv -NoTypeInformation",
            ]
        )
        rows = []
        for row in csv.reader(io.StringIO(raw or "")):
            if len(row) >= 2:
                name, app_id = row[0].strip().strip('"'), row[1].strip()
                if name and name.lower() != "name":
                    rows.append((name.strip('"'), app_id.strip('"')))
        return rows

    @staticmethod
    def _run_command(argv: list) -> str:
        import subprocess

        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=30,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return proc.stdout or ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _valid_exe(self, path: str) -> bool:
        if not path or not isinstance(path, str):
            return False
        if Path(path).suffix.lower() != ".exe":
            return False
        try:
            return bool(self._file_exists(str(Path(path))))
        except Exception:  # noqa: BLE001
            return False

    @staticmethod
    def _list_files(directory: str) -> list:
        """Default recursive file listing (bounded)."""
        root = Path(directory)
        if not root.is_dir():
            return []
        out = []
        for p in root.rglob("*"):
            if p.is_file():
                out.append(str(p))
            if len(out) > 2000:
                break
        return out

    @staticmethod
    def _resolve_lnk(lnk_path: str) -> str:
        """Extract the exe target from a .lnk binary.

        .lnk files store the target as UTF-16LE (and sometimes ANSI)
        strings; we scan for absolute paths ending in .exe and return
        the first that exists on disk.
        """
        try:
            data = Path(lnk_path).read_bytes()[:_MAX_LNK_BYTES]
        except OSError:
            return ""
        candidates = []
        # UTF-16LE decode of the whole blob.
        text = data.decode("utf-16le", errors="ignore")
        for m in re.finditer(r"[A-Za-z]:\\[^\x00-\x1f\"]+?\.exe", text, re.I):
            candidates.append(m.group(0))
        # ASCII/ANSI fallback.
        text_a = data.decode("mbcs", errors="ignore")
        for m in re.finditer(r"[A-Za-z]:\\[^\x00-\x1f\"]+?\.exe", text_a, re.I):
            candidates.append(m.group(0))
        for cand in candidates:
            cand = cand.strip().rstrip("\x00")
            if Path(cand).is_file():
                return cand
        return ""

    @staticmethod
    def _winreg_read(key_path: str) -> dict:
        """Default registry reader using winreg (Windows only)."""
        import winreg  # type: ignore

        result = {}
        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            try:
                with winreg.OpenKey(hive, key_path) as root:
                    i = 0
                    while True:
                        try:
                            sub = winreg.EnumKey(root, i)
                        except OSError:
                            break
                        i += 1
                        values = {}
                        try:
                            with winreg.OpenKey(root, sub) as sk:
                                j = 0
                                while True:
                                    try:
                                        vn, vd, vt = winreg.EnumValue(sk, j)
                                    except OSError:
                                        break
                                    j += 1
                                    if vt in (winreg.REG_SZ, winreg.REG_EXPAND_SZ):
                                        values[vn] = vd
                        except OSError:
                            pass
                        result[sub] = values
            except OSError:
                continue
        return result


# ----------------------------------------------------------------------
# Registry change watcher (advapi32 via ctypes — winreg has no
# RegNotifyChangeKeyValue binding)
# ----------------------------------------------------------------------
_HKLM = 0x80000002
_HKCU = 0x80000001
_KEY_NOTIFY = 0x0010
_ERROR_SUCCESS = 0
_REG_NOTIFY_CHANGE_NAME = 0x00000001
_REG_NOTIFY_CHANGE_LAST_SET = 0x00000004
_WAIT_OBJECT_0 = 0x00000000
_WAIT_TIMEOUT = 0x00000102

_WATCH_KEYS = (
    (_HKLM, _UNINSTALL_KEYS[0]),
    (_HKLM, _UNINSTALL_KEYS[1]),
    (_HKLM, _APP_PATHS_KEYS[0]),
    (_HKLM, _APP_PATHS_KEYS[1]),
    (_HKCU, _UNINSTALL_KEYS[0]),
    (_HKCU, _APP_PATHS_KEYS[0]),
)


class _WinRegNotifyApi:
    """Thin advapi32/kernel32 wrapper; raises when unavailable."""

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        if not hasattr(ctypes, "windll"):
            raise RuntimeError("Win32 unavailable")
        self._ct = ctypes
        self._wint = wintypes
        self._adv = ctypes.windll.advapi32
        self._k32 = ctypes.windll.kernel32
        self._adv.RegOpenKeyExW.argtypes = [
            wintypes.HKEY, wintypes.LPCWSTR, wintypes.DWORD,
            wintypes.DWORD, ctypes.POINTER(wintypes.HKEY),
        ]
        self._adv.RegOpenKeyExW.restype = wintypes.LONG
        self._adv.RegNotifyChangeKeyValue.argtypes = [
            wintypes.HKEY, wintypes.BOOL, wintypes.DWORD,
            wintypes.HANDLE,
        ]
        self._adv.RegNotifyChangeKeyValue.restype = wintypes.LONG
        self._adv.RegCloseKey.argtypes = [wintypes.HKEY]
        self._k32.CreateEventW.restype = wintypes.HANDLE
        self._k32.CreateEventW.argtypes = [
            wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL,
            wintypes.LPCWSTR,
        ]
        self._k32.WaitForSingleObject.argtypes = [
            wintypes.HANDLE, wintypes.DWORD,
        ]
        self._k32.WaitForSingleObject.restype = wintypes.DWORD
        self._k32.CloseHandle.argtypes = [wintypes.HANDLE]

    def open_key(self, hive: int, path: str):
        hkey = self._wint.HKEY()
        rc = self._adv.RegOpenKeyExW(
            self._wint.HKEY(hive), path, 0, _KEY_NOTIFY,
            self._ct.byref(hkey),
        )
        if rc != _ERROR_SUCCESS or not hkey.value:
            raise OSError(f"RegOpenKeyExW({path}) failed: {rc}")
        return hkey

    def create_event(self):
        # Auto-reset event (bManualReset=FALSE).
        return self._k32.CreateEventW(None, False, False, None)

    def notify(self, hkey, event) -> int:
        return int(self._adv.RegNotifyChangeKeyValue(
            hkey, True,
            _REG_NOTIFY_CHANGE_NAME | _REG_NOTIFY_CHANGE_LAST_SET,
            event,
        ))

    def wait(self, event, timeout_ms: int) -> int:
        return int(self._k32.WaitForSingleObject(event, timeout_ms))

    def close_key(self, hkey) -> None:
        try:
            self._adv.RegCloseKey(hkey)
        except Exception:  # noqa: BLE001
            pass

    def close_handle(self, handle) -> None:
        try:
            self._k32.CloseHandle(handle)
        except Exception:  # noqa: BLE001
            pass


class _RegistryWatcher:
    """Daemon thread re-arming RegNotifyChangeKeyValue on watch keys."""

    def __init__(self, discovery, api, on_invalidate=None) -> None:
        self._discovery = discovery
        self._api = api
        self._on_invalidate = on_invalidate

    def start(self):
        import threading

        stop = threading.Event()
        thread = threading.Thread(
            target=self._run, args=(stop,),
            name="voiceos-regwatch", daemon=True,
        )
        thread.start()
        return stop.set

    def _run(self, stop) -> None:
        api = self._api
        keys = []
        for hive, path in _WATCH_KEYS:
            try:
                keys.append(api.open_key(hive, path))
            except OSError as exc:
                logger.debug("Registry watch key unavailable %s: %s",
                             path, exc)
        if not keys:
            logger.warning(
                "Registry watcher: no watch keys could be opened; "
                "relying on periodic background refresh"
            )
            return
        event = api.create_event()
        if not event:
            logger.warning(
                "Registry watcher: CreateEvent failed; relying on "
                "periodic background refresh"
            )
            for k in keys:
                api.close_key(k)
            return
        try:
            armed = self._arm(api, keys, event)
            while not stop.is_set():
                rc = api.wait(event, 500)
                if rc == _WAIT_OBJECT_0:
                    self._discovery.invalidate()
                    if self._on_invalidate is not None:
                        try:
                            self._on_invalidate()
                        except Exception:  # noqa: BLE001
                            pass
                    armed = self._arm(api, keys, event)
                    if not armed:
                        logger.warning(
                            "Registry watcher: notify re-arm failed; "
                            "stopping watcher (periodic refresh remains)"
                        )
                        break
                elif rc != _WAIT_TIMEOUT:
                    logger.warning(
                        "Registry watcher: wait failed (%s); stopping",
                        rc,
                    )
                    break
        except Exception:  # noqa: BLE001
            logger.exception(
                "Registry watcher thread failed; periodic background "
                "refresh remains active"
            )
        finally:
            for k in keys:
                api.close_key(k)
            api.close_handle(event)
            logger.info("Registry watcher stopped")

    @staticmethod
    def _arm(api, keys, event) -> bool:
        ok = True
        for k in keys:
            try:
                rc = api.notify(k, event)
            except Exception:  # noqa: BLE001
                rc = -1
            if rc != _ERROR_SUCCESS:
                ok = False
        return ok
