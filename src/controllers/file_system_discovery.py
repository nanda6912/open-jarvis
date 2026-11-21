"""Local file and folder discovery for VoiceOS (V10.1).

Provides an indexed, privacy-preserving search over the user's
accessible local storage:

- :class:`FileItem` — metadata for one file or folder.
- :class:`FileIndexStore` — persistent SQLite index (filenames and
  metadata only; file *contents* are never read or stored).
- :class:`FileSystemIndexer` — background worker that walks priority
  roots (Windows Known Folders, OneDrive, user profile, other fixed
  drives), skips system/cache/reparse locations and refreshes
  incrementally.
- :class:`FileMatcher` — normalization, token and extension matching
  with conservative ranking (weak matches are never auto-opened).
- :class:`FileContext` — short-lived memory of the last search results
  so follow-ups like "open it" / "the second one" resolve safely.
- :class:`FileSystemService` — facade used by command handlers:
  folder/file resolution, safe opening (Explorer for folders, Shell
  file associations for files, dangerous-type gate) and search.

Everything stays local: no network access, no file contents, no
modification of user files. Opening uses ``os.startfile``
(ShellExecute) on validated absolute paths only — spoken text is never
used as a command.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

_KIND_DIR = "dir"
_KIND_FILE = "file"

# Windows FILE_ATTRIBUTE_REPARSE_POINT — junctions/symlinks are indexed
# as items but never descended into (loop + escape safety).
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400
_DRIVE_FIXED = 3

# Directory names never descended into during indexing. The directory
# itself is still indexed so "open AppData folder" style requests can
# still resolve, but caches, VCS internals, virtualenvs and OS install
# trees are skipped.
_EXCLUDED_DIR_NAMES = frozenset({
    "windows", "program files", "program files (x86)", "programdata",
    "appdata", "$recycle.bin", "system volume information",
    "node_modules", ".git", ".svn", ".hg", "__pycache__",
    ".venv", "venv", "env", ".tox", ".mypy_cache", ".pytest_cache",
    ".idea", ".vs", "packages", "vendor",
})

# Extensions that are executable or script-like. They are never opened
# as ordinary documents without explicit user confirmation.
_DANGEROUS_EXTENSIONS = frozenset({
    ".exe", ".bat", ".cmd", ".com", ".msi", ".msp", ".mst", ".scr",
    ".pif", ".ps1", ".psm1", ".psd1", ".vbs", ".vbe", ".js", ".jse",
    ".wsf", ".wsh", ".hta", ".cpl", ".dll", ".sys", ".lnk", ".reg",
    ".inf", ".jar", ".sh", ".app", ".deb", ".rpm", ".run", ".gadget",
})

# Spoken/file extensions that count as a type filter in queries.
_KNOWN_EXTENSIONS = frozenset({
    "pdf", "doc", "docx", "xls", "xlsx", "ppt", "pptx", "txt", "csv",
    "rtf", "odt", "ods", "png", "jpg", "jpeg", "gif", "bmp", "webp",
    "tif", "tiff", "svg", "mp3", "wav", "flac", "ogg", "mp4", "mkv",
    "avi", "mov", "wmv", "webm", "zip", "rar", "7z", "tar", "gz",
    "java", "py", "js", "ts", "jsx", "tsx", "html", "htm", "css",
    "json", "xml", "yaml", "yml", "md", "c", "cpp", "h", "cs", "go",
    "rs", "rb", "php", "sql", "log",
})

_ORDINALS = {
    "first": 0, "1": 0, "one": 0,
    "second": 1, "2": 1, "two": 1,
    "third": 2, "3": 2, "three": 2,
    "fourth": 3, "4": 3, "four": 3,
    "fifth": 4, "5": 4, "five": 4,
    "sixth": 5, "6": 5, "six": 5,
}

_PUNCT_RE = re.compile(r"[^a-z0-9]+")


def normalize_name(text: str) -> str:
    """Lowercase, punctuation->space, collapse whitespace."""
    return " ".join(_PUNCT_RE.sub(" ", (text or "").lower()).split())


def _stem(name: str) -> str:
    """Filename without its final extension."""
    p = Path(name)
    return p.stem if p.suffix else name


def _extension(name: str) -> str:
    """Lowercase extension including the dot ('' when none)."""
    return Path(name).suffix.lower()


def _is_reparse(stat_result) -> bool:
    return bool(
        getattr(stat_result, "st_file_attributes", 0)
        & _FILE_ATTRIBUTE_REPARSE_POINT
    )


# ----------------------------------------------------------------------
# FileItem + resolve result
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class FileItem:
    """One discovered file or folder (metadata only, never content)."""

    name: str
    path: str
    kind: str                     # "file" | "dir"
    extension: str = ""
    parent: str = ""
    size: int = 0
    mtime: float = 0.0
    root: str = ""

    @property
    def is_dir(self) -> bool:
        return self.kind == _KIND_DIR

    @property
    def stem(self) -> str:
        return _stem(self.name)

    @property
    def norm_name(self) -> str:
        return normalize_name(self.name)

    @property
    def norm_stem(self) -> str:
        return normalize_name(_stem(self.name))

    @property
    def label(self) -> str:
        kind = "folder" if self.is_dir else "file"
        return f"{self.name} ({kind})"


@dataclass
class ItemResolveResult:
    """Outcome of matching a spoken query against the index."""

    status: str                   # "found" | "ambiguous" | "missing"
    item: Optional[FileItem] = None
    options: list = field(default_factory=list)   # list[FileItem]
    score: int = 0


# ----------------------------------------------------------------------
# SQLite index store
# ----------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS fs_items (
    path TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    kind TEXT NOT NULL,
    extension TEXT NOT NULL DEFAULT '',
    parent TEXT NOT NULL DEFAULT '',
    size INTEGER NOT NULL DEFAULT 0,
    mtime REAL NOT NULL DEFAULT 0,
    root TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_fs_items_name ON fs_items(name);
CREATE INDEX IF NOT EXISTS idx_fs_items_kind ON fs_items(kind);
CREATE TABLE IF NOT EXISTS fs_state (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class FileIndexStore:
    """Persistent SQLite store for the file/folder index.

    Only filenames and metadata are stored — never file contents.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def initialize(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    @staticmethod
    def item_to_row(item: FileItem):
        return (
            item.path, item.name, item.kind, item.extension,
            item.parent, item.size, item.mtime, item.root,
        )

    @staticmethod
    def row_to_item(row: sqlite3.Row) -> FileItem:
        return FileItem(
            name=row["name"], path=row["path"], kind=row["kind"],
            extension=row["extension"], parent=row["parent"],
            size=row["size"], mtime=row["mtime"], root=row["root"],
        )

    def upsert(self, items: Iterable[FileItem]) -> int:
        rows = [self.item_to_row(i) for i in items]
        if not rows:
            return 0
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO fs_items(path, name, kind, extension, "
                "parent, size, mtime, root) VALUES (?,?,?,?,?,?,?,?) "
                "ON CONFLICT(path) DO UPDATE SET "
                "name=excluded.name, kind=excluded.kind, "
                "extension=excluded.extension, parent=excluded.parent, "
                "size=excluded.size, mtime=excluded.mtime, "
                "root=excluded.root",
                rows,
            )
            conn.commit()
        return len(rows)

    def delete_paths(self, paths: Iterable[str]) -> int:
        paths = list(paths)
        if not paths:
            return 0
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM fs_items WHERE path = ?",
                [(p,) for p in paths],
            )
            conn.commit()
        return len(paths)

    def delete_under(self, roots: Iterable[str]) -> int:
        """Delete every row whose path sits under one of ``roots``."""
        prefixes = [r.rstrip("\\/") + os.sep for r in roots]
        if not prefixes:
            return 0
        clause = " OR ".join("path LIKE ?" for _ in prefixes)
        params = [p + "%" for p in prefixes]
        with self._connect() as conn:
            cur = conn.execute(
                f"DELETE FROM fs_items WHERE {clause}", params
            )
            conn.commit()
            return cur.rowcount

    def all_items(self) -> list[FileItem]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM fs_items"
            ).fetchall()
        return [self.row_to_item(r) for r in rows]

    def count(self) -> int:
        with self._connect() as conn:
            return int(
                conn.execute("SELECT COUNT(*) c FROM fs_items")
                .fetchone()["c"]
            )

    def get_state(self, key: str, default: Optional[str] = None):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM fs_state WHERE key = ?", (key,)
            ).fetchone()
        return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO fs_state(key, value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            conn.commit()

    def _connect(self):
        from contextlib import contextmanager

        @contextmanager
        def _cm():
            conn = sqlite3.connect(str(self._path))
            try:
                conn.row_factory = sqlite3.Row
                yield conn
            finally:
                conn.close()

        return _cm()


# ----------------------------------------------------------------------
# Filesystem indexer
# ----------------------------------------------------------------------

class FileSystemIndexer:
    """Walk priority roots and maintain the SQLite index.

    Args:
        store: :class:`FileIndexStore` (created in ``data/`` by default).
        env: environment mapping (injectable for tests).
        extra_roots: additional directories to index (tests).
        known_folder_resolver: ``name -> Optional[path]`` used for the
            six standard user folders (defaults to the real Windows
            Known Folder API via :mod:`folder_opener`).
        max_depth: default per-root descent limit.
        drive_depth: descent limit for non-profile fixed-drive roots.
        profile_depth: descent limit for the user-profile root.
        max_items: hard cap per scan (safety valve).
        refresh_interval: seconds between background refreshes.
        dir_lister / stat_fn / is_dir: injectable filesystem hooks for
            tests.
    """

    def __init__(
        self,
        store: Optional[FileIndexStore] = None,
        env: Optional[dict] = None,
        extra_roots: Optional[Iterable[str]] = None,
        known_folder_resolver: Optional[Callable[[str], Optional[str]]] = None,
        max_depth: int = 10,
        drive_depth: int = 4,
        profile_depth: int = 5,
        onedrive_depth: int = 5,
        max_items: int = 250_000,
        refresh_interval: float = 900.0,
        excluded_names: Optional[Iterable[str]] = None,
    ) -> None:
        self._store = store
        self._env = env if env is not None else dict(os.environ)
        self._extra_roots = [str(r) for r in (extra_roots or [])]
        self._known_resolver = known_folder_resolver
        self._max_depth = max_depth
        self._drive_depth = drive_depth
        self._profile_depth = profile_depth
        self._onedrive_depth = onedrive_depth
        self._max_items = max_items
        self._refresh_interval = refresh_interval
        self._excluded = (
            frozenset(n.lower() for n in excluded_names)
            if excluded_names is not None
            else _EXCLUDED_DIR_NAMES
        )
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._scan_lock = threading.Lock()
        self._last_scan_at = 0.0
        self._last_scan_stats: dict = {}
        self._scanning = False

    # ------------------------------------------------------------------
    # Roots
    # ------------------------------------------------------------------
    def _env_path(self, *names: str) -> str:
        lowered = {k.lower(): v for k, v in self._env.items()}
        for n in names:
            v = self._env.get(n) or lowered.get(n.lower())
            if v:
                return v
        return ""

    def _resolve_known(self, folder_name: str) -> Optional[str]:
        if self._known_resolver is not None:
            try:
                return self._known_resolver(folder_name)
            except Exception:  # noqa: BLE001
                logger.debug("Known-folder resolver failed", exc_info=True)
                return None
        try:
            from src.controllers.folder_opener import (
                KnownFolderResolver,
            )

            resolver = KnownFolderResolver(env=self._env)
            path = resolver.resolve(folder_name)
            return str(path) if path is not None else None
        except Exception:  # noqa: BLE001
            logger.debug("Known-folder resolution failed", exc_info=True)
            return None

    def _fixed_drives(self) -> list[str]:
        """Return fixed local drive roots (e.g. ['C:\\', 'D:\\'])."""
        drives = []
        if not hasattr(os, "listdrives") and os.name != "nt":
            # Non-Windows fallback: treat "/" as the only root.
            return [os.sep]
        try:
            import ctypes

            bitmask = ctypes.windll.kernel32.GetLogicalDrives()
            for i in range(26):
                if bitmask & (1 << i):
                    letter = chr(ord("A") + i)
                    root = f"{letter}:{os.sep}"
                    try:
                        if (
                            ctypes.windll.kernel32.GetDriveTypeW(root)
                            == _DRIVE_FIXED
                        ):
                            drives.append(root)
                    except Exception:  # noqa: BLE001
                        drives.append(root)
        except Exception:  # noqa: BLE001
            sys_drive = self._env_path("SystemDrive") or "C:"
            drives.append(sys_drive.rstrip("\\/") + os.sep)
        return drives

    def discover_roots(self) -> list[tuple[str, int]]:
        """Return ``(root_path, max_depth)`` pairs in priority order.

        Priority: user Known Folders > OneDrive > user profile > other
        fixed-drive roots (shallow). Duplicates and non-directories are
        removed.
        """
        roots: list[tuple[str, int]] = []
        seen: set[str] = set()

        def add(path: Optional[str], depth: int) -> None:
            if not path:
                return
            try:
                p = str(Path(path))
            except Exception:  # noqa: BLE001
                return
            key = os.path.normcase(os.path.normpath(p))
            if key in seen:
                return
            try:
                if not os.path.isdir(p):
                    return
            except OSError:
                return
            seen.add(key)
            roots.append((p, depth))

        # 1. Windows Known Folders (handles OneDrive redirection).
        for name in (
            "desktop", "documents", "downloads", "pictures",
            "videos", "music",
        ):
            add(self._resolve_known(name), self._max_depth)

        # 2. OneDrive roots (custom folders living beside the
        #    redirected known folders).
        for env_name in ("OneDrive", "OneDriveConsumer"):
            add(self._env_path(env_name), self._onedrive_depth)

        # 3. User profile (custom top-level folders like "Projects").
        profile = self._env_path("USERPROFILE", "HOME")
        add(profile, self._profile_depth)

        # 4. Other fixed local drives, shallow scan.
        system_drive = (self._env_path("SystemDrive") or "C:").upper()
        for drive in self._fixed_drives():
            if drive[0].upper() == system_drive[0].upper():
                continue
            add(drive, self._drive_depth)

        # 5. Test/extra roots always last.
        for r in self._extra_roots:
            add(r, self._max_depth)

        return roots

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------
    def _excluded_dir(self, name: str) -> bool:
        return name.lower() in self._excluded

    def scan(self, force: bool = False) -> dict:
        """Run one indexing pass. Returns scan statistics.

        Safe to call from a worker thread; serialized via a lock so a
        manual refresh cannot overlap a background one.
        """
        if self._store is None:
            return {"error": "no_store"}
        if self._scanning and not force:
            return {"skipped": "already_scanning"}
        with self._scan_lock:
            self._scanning = True
            try:
                return self._scan_locked()
            finally:
                self._scanning = False

    def _scan_locked(self) -> dict:
        store = self._store
        store.initialize()
        roots = self.discover_roots()
        stats = {
            "roots": len(roots), "seen": 0, "upserted": 0,
            "deleted": 0, "errors": 0, "truncated": False,
            "started": time.time(),
        }
        all_seen: set[str] = set()
        # Pre-load existing signatures to make writes incremental:
        # only rows whose (size, mtime) changed are rewritten.
        existing = {}
        try:
            for item in store.all_items():
                existing[os.path.normcase(item.path)] = item
        except Exception:  # noqa: BLE001
            logger.debug("Could not preload index", exc_info=True)

        changed: list[FileItem] = []
        for root, depth in roots:
            if stats["seen"] >= self._max_items:
                stats["truncated"] = True
                break
            try:
                for item in self._walk(root, depth, root, stats):
                    all_seen.add(os.path.normcase(item.path))
                    prev = existing.get(os.path.normcase(item.path))
                    if (
                        prev is None
                        or prev.mtime != item.mtime
                        or prev.size != item.size
                        or prev.kind != item.kind
                        or prev.name != item.name
                    ):
                        changed.append(item)
            except Exception:  # noqa: BLE001
                stats["errors"] += 1
                logger.debug("Root scan failed: %s", root, exc_info=True)

        try:
            stats["upserted"] = store.upsert(changed)
        except Exception:  # noqa: BLE001
            stats["errors"] += 1
            logger.exception("Index upsert failed")

        # Prune rows under scanned roots that no longer exist.
        stale = [
            p for p in existing
            if p not in all_seen
            and any(
                os.path.normcase(p).startswith(
                    os.path.normcase(r.rstrip("\\/") + os.sep)
                )
                for r, _ in roots
            )
        ]
        try:
            stats["deleted"] = store.delete_paths(stale)
        except Exception:  # noqa: BLE001
            stats["errors"] += 1
            logger.debug("Index prune failed", exc_info=True)

        stats["finished"] = time.time()
        stats["indexed"] = store.count()
        self._last_scan_at = time.time()
        self._last_scan_stats = stats
        logger.info(
            "File index scan: %d roots, %d seen, %d upserted, "
            "%d deleted, %d errors, %d total",
            stats["roots"], stats["seen"], stats["upserted"],
            stats["deleted"], stats["errors"], stats["indexed"],
        )
        return stats

    def _walk(self, dirpath: str, depth: int, root: str, stats: dict):
        """Yield FileItems under ``dirpath`` up to ``depth`` levels."""
        if depth < 0 or stats["seen"] >= self._max_items:
            return
        try:
            entries = list(os.scandir(dirpath))
        except OSError:
            stats["errors"] += 1
            return
        subdirs = []
        for entry in entries:
            if stats["seen"] >= self._max_items:
                stats["truncated"] = True
                return
            stats["seen"] += 1
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                stats["errors"] += 1
                continue
            name = entry.name
            try:
                is_dir = entry.is_dir(follow_symlinks=False)
            except OSError:
                is_dir = False
            path = entry.path
            if is_dir:
                item = FileItem(
                    name=name, path=path, kind=_KIND_DIR,
                    parent=dirpath, size=0, mtime=st.st_mtime,
                    root=root,
                )
                yield item
                # Never descend into reparse points (junctions,
                # symlinked dirs) or excluded/cache/system dirs.
                if _is_reparse(st):
                    continue
                if self._excluded_dir(name):
                    continue
                subdirs.append((path, depth - 1))
            else:
                try:
                    is_file = entry.is_file(follow_symlinks=False)
                except OSError:
                    is_file = False
                if not is_file:
                    continue
                yield FileItem(
                    name=name, path=path, kind=_KIND_FILE,
                    extension=_extension(name), parent=dirpath,
                    size=st.st_size, mtime=st.st_mtime, root=root,
                )
        for sub, d in subdirs:
            yield from self._walk(sub, d, root, stats)

    # ------------------------------------------------------------------
    # Background refresh
    # ------------------------------------------------------------------
    def start_background(self, interval: Optional[float] = None) -> Callable[[], None]:
        """Start a daemon thread that rescans every ``interval`` seconds.

        Returns a stop callable. Never blocks the GUI or audio threads.
        """
        if self._thread is not None and self._thread.is_alive():
            return self.stop_background
        interval = interval or self._refresh_interval
        self._stop.clear()

        def _loop():
            # Let the UI finish starting before the first scan.
            if self._stop.wait(2.0):
                return
            while not self._stop.is_set():
                try:
                    self.scan()
                except Exception:  # noqa: BLE001
                    logger.debug("Background index scan failed",
                                 exc_info=True)
                if self._stop.wait(interval):
                    break

        self._thread = threading.Thread(
            target=_loop, daemon=True, name="voiceos-fs-index"
        )
        self._thread.start()
        return self.stop_background

    def stop_background(self) -> None:
        self._stop.set()

    def ensure_fresh(self, max_age: float = 0.0) -> None:
        """Kick a scan when the index is empty or older than ``max_age``."""
        try:
            count = self._store.count() if self._store else 0
        except Exception:  # noqa: BLE001
            count = 0
        if count == 0 or (
            max_age and time.time() - self._last_scan_at > max_age
        ):
            self.scan()

    @property
    def last_scan_stats(self) -> dict:
        return dict(self._last_scan_stats)


# ----------------------------------------------------------------------
# Matcher
# ----------------------------------------------------------------------

class FileMatcher:
    """Rank FileItems against a spoken query.

    Matching is conservative: exact/normalized/token matches score
    high, fuzzy matches only when confidence is sufficient, and weak
    matches are reported as missing rather than silently opened.
    """

    EXACT = 100
    NAME_EXACT = 95
    TOKEN_SUBSET = 85
    EXT_MATCH = 10          # bonus when extension filter matches
    FUZZY_MIN = 0.82

    def __init__(self, items: Iterable[FileItem]) -> None:
        self._items = list(items)

    # -- query analysis -------------------------------------------------
    @staticmethod
    def _split_query(query: str) -> tuple[str, str, bool, bool]:
        """Return (clean_query, ext_filter, want_latest, want_dirs)."""
        q = (query or "").strip().strip(".!?…")
        latest = bool(
            re.search(r"\b(latest|newest|most recent|recent)\b", q, re.I)
        )
        # Modifier words never name the item itself.
        q = re.sub(
            r"\b(latest|newest|most\s+recent|recent(?:ly)?)\b",
            " ",
            q,
            flags=re.I,
        )
        want_dirs = bool(
            re.search(r"\bfolders?\b|\bdirector(y|ies)\b", q, re.I)
        )
        # Explicit ".pdf" style extension.
        m = re.search(r"\.([a-z0-9]{1,8})\b", q, re.I)
        ext = ""
        if m and m.group(1).lower() in _KNOWN_EXTENSIONS:
            ext = "." + m.group(1).lower()
            q = (q[: m.start()] + " " + q[m.end():]).strip()
        else:
            # Bare extension word: "pdf files", "word documents".
            m2 = re.search(
                r"\b(" + "|".join(sorted(_KNOWN_EXTENSIONS)) + r")\b",
                q, re.I,
            )
            if m2 and re.search(
                r"\b(files?|documents?|folders?|items?)\b", q, re.I
            ):
                ext = "." + m2.group(1).lower()
                q = (q[: m2.start()] + " " + q[m2.end():]).strip()
        # Filler tokens that never help identify an item.
        q = re.sub(
            r"\b(all|any|every|some|the|a|an|my|your|our)\b",
            " ",
            q,
            flags=re.I,
        )
        # A bare extension word ("pdf", "docx") is a type filter even
        # without an explicit "files" noun.
        m3 = re.search(
            r"^\s*(" + "|".join(sorted(_KNOWN_EXTENSIONS)) + r")\s*$",
            q,
            re.I,
        )
        if m3 and not ext:
            ext = "." + m3.group(1).lower()
            q = ""
        # Trailing scope words never name the item.
        q = re.sub(
            r"\b(files?|folders?|documents?|items?|directory|"
            r"directories|folder|on\s+my\s+(computer|pc)|"
            r"in\s+my\s+(files?|folders?|documents?))\b\.?\s*$",
            "",
            q,
            flags=re.I,
        ).strip(" .,")
        return q, ext, latest, want_dirs

    # -- scoring ---------------------------------------------------------
    def _score(self, query_norm: str, query_tokens: set, ext: str,
               item: FileItem) -> int:
        name_n = item.norm_name
        stem_n = item.norm_stem
        if not name_n:
            return 0
        if not query_norm:
            # Type-only query ("pdf files"): every matching-extension
            # file gets a modest equal score; ranking falls to mtime.
            return self.EXT_MATCH if (
                ext and item.extension == ext
            ) else 0
        if name_n == query_norm:
            return self.NAME_EXACT + (self.EXT_MATCH if ext and item.extension == ext else 0)
        # Query without spaces equals stem ("smartparkingsystem").
        compact_q = query_norm.replace(" ", "")
        if stem_n.replace(" ", "") == compact_q or name_n.replace(" ", "") == compact_q:
            return self.TOKEN_SUBSET
        name_tokens = set(name_n.split())
        stem_tokens = set(stem_n.split())
        # Parent-directory names carry real context for FILES
        # ("Assignment 1" disambiguates "Main.java" for a "java
        # assignment" query). Directories don't inherit parent tokens —
        # otherwise every subfolder of a match becomes a false candidate.
        parent_tokens = (
            set(normalize_name(Path(item.parent).name).split())
            if item.parent and not item.is_dir else set()
        )
        all_tokens = name_tokens | stem_tokens | parent_tokens
        if query_tokens and query_tokens <= all_tokens:
            score = self.TOKEN_SUBSET
        else:
            # Partial token coverage.
            hits = sum(1 for t in query_tokens if t in all_tokens)
            if not query_tokens or hits == 0:
                # Fuzzy whole-name fallback for STT spelling drift.
                import difflib

                ratio = max(
                    difflib.SequenceMatcher(None, query_norm, name_n).ratio(),
                    difflib.SequenceMatcher(None, query_norm, stem_n).ratio(),
                )
                if ratio < self.FUZZY_MIN:
                    return 0
                score = int(ratio * 70)
            else:
                coverage = hits / len(query_tokens)
                if coverage < 0.5:
                    return 0
                score = int(40 + 30 * coverage)
        if ext:
            if item.extension == ext:
                score += self.EXT_MATCH
            elif item.is_dir:
                # Folders have no extension; don't penalize hard, but
                # a file-type request should prefer real files.
                score -= 15
            else:
                score -= 25
        return score

    # -- public API ------------------------------------------------------
    def search(
        self,
        query: str,
        kinds: Optional[set] = None,
        limit: int = 10,
    ) -> list[tuple[FileItem, int]]:
        """Ranked ``(item, score)`` pairs, best first."""
        clean, ext, latest, want_dirs = self._split_query(query)
        q_norm = normalize_name(clean)
        q_tokens = set(q_norm.split()) if q_norm else set()
        allowed = kinds or {_KIND_FILE, _KIND_DIR}
        if not q_norm and not ext:
            # Generic browse request ("show me files and folders"):
            # return the most recently modified items, never everything.
            recent = sorted(
                (i for i in self._items if i.kind in allowed),
                key=lambda i: -i.mtime,
            )[: max(1, limit)]
            return [(i, 1) for i in recent]
        scored = []
        for item in self._items:
            if item.kind not in allowed:
                continue
            if want_dirs and not item.is_dir:
                continue
            s = self._score(q_norm, q_tokens, ext, item)
            if s <= 0:
                continue
            # Explicit extension in the query filters non-matching
            # files unless the name itself matched exactly.
            if ext and not item.is_dir and item.extension != ext \
                    and s < self.TOKEN_SUBSET:
                continue
            scored.append((item, s))
        scored.sort(key=lambda x: (-x[1], -x[0].mtime, x[0].name.lower()))
        if latest:
            scored.sort(key=lambda x: (-x[0].mtime, -x[1]))
        return scored[: max(1, limit)]

    def resolve(
        self,
        query: str,
        kinds: Optional[set] = None,
    ) -> ItemResolveResult:
        """Conservative single/ambiguous/missing resolution."""
        matches = self.search(query, kinds=kinds, limit=8)
        if not matches:
            return ItemResolveResult("missing")
        top_score = matches[0][1]
        # Never trust weak fuzzy hits for opening.
        if top_score < 60:
            return ItemResolveResult("missing", score=top_score)
        if len(matches) == 1:
            if top_score >= 80:
                return ItemResolveResult(
                    "found", item=matches[0][0], score=top_score
                )
            return ItemResolveResult(
                "ambiguous", options=[matches[0][0]], score=top_score
            )
        second = matches[1][1]
        if top_score >= 80 and top_score - second >= 15:
            return ItemResolveResult(
                "found", item=matches[0][0], score=top_score
            )
        plausible = [i for i, s in matches if s >= 55][:5]
        return ItemResolveResult(
            "ambiguous", options=plausible or [m[0] for m in matches[:5]],
            score=top_score,
        )

    @staticmethod
    def pick(reply: str, candidates: list[FileItem]) -> tuple[str, Optional[FileItem]]:
        """Map a spoken clarification reply to a candidate.

        Returns ``(status, item)`` where status is ``found``,
        ``ambiguous`` or ``missing``.
        """
        if not candidates:
            return "missing", None
        norm = normalize_name(reply)
        stripped = (reply or "").strip().strip(".!? ").lower()
        # Ordinal selection: "the second one", "2", "number three".
        tokens = re.split(r"\s+", stripped)
        for tok in tokens:
            tok = tok.strip(".,!?")
            if tok in _ORDINALS:
                idx = _ORDINALS[tok]
                if idx < len(candidates):
                    return "found", candidates[idx]
        # "latest" / "most recent".
        if re.search(r"\b(latest|newest|most recent)\b", stripped):
            best = max(candidates, key=lambda i: i.mtime)
            return "found", best
        # Exact normalized name match.
        for c in candidates:
            if c.norm_name == norm or c.norm_stem == norm:
                return "found", c
        # Unique substring match (reverse direction needs >=3 chars so
        # a one-letter name can't match inside an unrelated reply).
        hits = [
            c for c in candidates
            if norm and (
                norm in c.norm_name or norm in c.norm_stem
                or (len(c.norm_stem) >= 3 and c.norm_stem in norm)
            )
        ]
        if len(hits) == 1:
            return "found", hits[0]
        if len(hits) > 1:
            return "ambiguous", None
        return "missing", None


# ----------------------------------------------------------------------
# Follow-up context
# ----------------------------------------------------------------------

class FileContext:
    """Short-lived memory of the last search/open candidates.

    Lets follow-ups like "open it", "that folder", "the second one" and
    "open its parent folder" resolve against real prior results. The
    context expires after ``ttl`` seconds and is cleared on cancel.
    """

    def __init__(self, ttl: float = 300.0) -> None:
        self._ttl = ttl
        self._candidates: list[FileItem] = []
        self._query = ""
        self._kind = ""
        self._last_item: Optional[FileItem] = None
        self._at = 0.0

    def remember(
        self,
        candidates: Iterable[FileItem],
        query: str = "",
        kind: str = "",
        last_item: Optional[FileItem] = None,
    ) -> None:
        self._candidates = list(candidates)
        self._query = query
        self._kind = kind
        self._last_item = last_item or (
            self._candidates[0] if self._candidates else None
        )
        self._at = time.monotonic()

    def clear(self) -> None:
        self._candidates = []
        self._query = ""
        self._kind = ""
        self._last_item = None
        self._at = 0.0

    @property
    def expired(self) -> bool:
        return not self._candidates or (
            time.monotonic() - self._at > self._ttl
        )

    @property
    def candidates(self) -> list[FileItem]:
        return [] if self.expired else list(self._candidates)

    @property
    def last_item(self) -> Optional[FileItem]:
        return None if self.expired else self._last_item

    def parent_item(self) -> Optional[FileItem]:
        """The parent directory of the most recent item, if any."""
        item = self.last_item
        if item is None or not item.parent:
            return None
        return FileItem(
            name=Path(item.parent).name or item.parent,
            path=item.parent,
            kind=_KIND_DIR,
            parent=str(Path(item.parent).parent),
            mtime=0.0,
            root=item.root,
        )


# ----------------------------------------------------------------------
# Service facade
# ----------------------------------------------------------------------

class FileSystemService:
    """Facade used by command handlers for discovery/open/search.

    Args:
        indexer: :class:`FileSystemIndexer` (None disables indexing;
            handlers then rely on the live fallback only).
        store: optional :class:`FileIndexStore` used to load items.
        env: environment mapping.
        open_fn: ``(path) -> None`` opener (defaults to ``os.startfile``).
        known_folder_opener: optional ``FolderOpener``-compatible object
            with ``.open(name) -> (ok, detail)`` for the six aliases.
        live_fallback: when True, a bounded live walk runs for queries
            the index could not answer (covers freshly created items
            before the next background refresh).
        context_ttl: seconds a search result stays usable for
            follow-up commands.
    """

    def __init__(
        self,
        indexer: Optional[FileSystemIndexer] = None,
        store: Optional[FileIndexStore] = None,
        env: Optional[dict] = None,
        open_fn: Optional[Callable[[str], object]] = None,
        known_folder_opener=None,
        live_fallback: bool = True,
        context_ttl: float = 300.0,
        known_names: Optional[Iterable[str]] = None,
    ) -> None:
        self._indexer = indexer
        self._store = store if store is not None else (
            indexer._store if indexer is not None else None
        )
        self._env = env if env is not None else dict(os.environ)
        self._open = open_fn or getattr(os, "startfile", None)
        self._known_opener = known_folder_opener
        self._live_fallback = live_fallback
        self._context = FileContext(ttl=context_ttl)
        self._known_names = frozenset(
            n.lower() for n in (
                known_names
                or ("desktop", "documents", "downloads", "pictures",
                    "photos", "music", "videos", "movies")
            )
        )
        self._items_cache: Optional[list[FileItem]] = None
        self._items_loaded_at = 0.0
        self._items_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Index access
    # ------------------------------------------------------------------
    def _load_items(self, max_age: float = 30.0) -> list[FileItem]:
        """Return indexed items, refreshing the in-memory copy."""
        now = time.monotonic()
        with self._items_lock:
            if (
                self._items_cache is not None
                and now - self._items_loaded_at < max_age
            ):
                return self._items_cache
            items: list[FileItem] = []
            if self._store is not None:
                try:
                    self._store.initialize()
                    items = self._store.all_items()
                except Exception:  # noqa: BLE001
                    logger.debug("Index load failed", exc_info=True)
                    items = []
            self._items_cache = items
            self._items_loaded_at = now
            return items

    def invalidate_items_cache(self) -> None:
        with self._items_lock:
            self._items_cache = None

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        kinds: Optional[set] = None,
        limit: int = 10,
        use_live_fallback: bool = True,
    ) -> list[tuple[FileItem, int]]:
        """Ranked search over the index (plus bounded live fallback)."""
        items = self._load_items()
        results = FileMatcher(items).search(query, kinds=kinds, limit=limit)
        if not results and use_live_fallback and self._live_fallback:
            live = self._live_search(query, kinds=kinds, limit=limit)
            if live:
                return live
        return results

    # Generic scope nouns users append to names ("the Jarvis project
    # folder"). They never identify the item, so a missing primary
    # lookup retries once without them.
    _SCOPE_TOKENS = frozenset({
        "project", "projects", "folder", "folders", "file", "files",
        "document", "documents", "repo", "repos", "repository",
        "directory", "directories", "item", "items", "code", "space",
        "drive", "location", "place", "thing", "stuff",
    })

    @classmethod
    def _descope(cls, query: str) -> str:
        tokens = [
            t for t in normalize_name(query).split()
            if t not in cls._SCOPE_TOKENS
        ]
        return " ".join(tokens)

    def resolve(
        self,
        query: str,
        kinds: Optional[set] = None,
        use_live_fallback: bool = True,
    ) -> ItemResolveResult:
        """Resolve a spoken name to one item, many candidates or none."""
        res = self._resolve_from_index(query, kinds)
        effective = query
        if res.status == "missing":
            descoped = self._descope(query)
            if descoped and descoped != normalize_name(query):
                res = self._resolve_from_index(descoped, kinds)
                effective = descoped
        if res.status == "missing" and use_live_fallback and self._live_fallback:
            live_items = self._live_candidates(effective, kinds)
            if live_items:
                res = FileMatcher(live_items).resolve(effective, kinds=kinds)
        return res

    def _resolve_from_index(
        self, query: str, kinds: Optional[set]
    ) -> ItemResolveResult:
        items = self._load_items()
        return FileMatcher(items).resolve(query, kinds=kinds)

    # -- bounded live fallback ------------------------------------------
    def _live_roots(self) -> list[tuple[str, int]]:
        if self._indexer is not None:
            try:
                return self._indexer.discover_roots()
            except Exception:  # noqa: BLE001
                logger.debug("Root discovery failed", exc_info=True)
        return []

    def _live_candidates(
        self, query: str, kinds: Optional[set], max_results: int = 500
    ) -> list[FileItem]:
        """Walk roots once, collecting name-similar items (bounded)."""
        clean, ext, _latest, want_dirs = FileMatcher._split_query(query)
        q_norm = normalize_name(clean)
        q_tokens = set(q_norm.split()) if q_norm else set()
        if not q_norm and not ext:
            return []
        allowed = kinds or {_KIND_FILE, _KIND_DIR}
        seen: set[str] = set()
        matcher_items: list[FileItem] = []
        for root, depth in self._live_roots():
            for item in self._live_walk(root, depth):
                key = os.path.normcase(item.path)
                if key in seen:
                    continue
                seen.add(key)
                if item.kind not in allowed:
                    continue
                if want_dirs and not item.is_dir:
                    continue
                if ext and not item.is_dir and item.extension != ext:
                    continue
                # Cheap pre-filter on raw name tokens before scoring.
                name_tokens = set(item.norm_name.split()) | set(
                    item.norm_stem.split()
                )
                if q_tokens and not (q_tokens & name_tokens):
                    continue
                matcher_items.append(item)
                if len(matcher_items) >= max_results:
                    break
            if len(matcher_items) >= max_results:
                break
        return matcher_items

    def _live_walk(self, dirpath: str, depth: int):
        """Yield FileItems like the indexer but with a tighter budget."""
        budget = [4000]
        excluded = _EXCLUDED_DIR_NAMES

        def rec(path: str, d: int):
            if d < 0 or budget[0] <= 0:
                return
            try:
                entries = list(os.scandir(path))
            except OSError:
                return
            subdirs = []
            for entry in entries:
                if budget[0] <= 0:
                    return
                budget[0] -= 1
                try:
                    st = entry.stat(follow_symlinks=False)
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                if is_dir:
                    yield FileItem(
                        name=entry.name, path=entry.path,
                        kind=_KIND_DIR, parent=path, mtime=st.st_mtime,
                        root=dirpath,
                    )
                    if _is_reparse(st) or entry.name.lower() in excluded:
                        continue
                    subdirs.append((entry.path, d - 1))
                else:
                    try:
                        if not entry.is_file(follow_symlinks=False):
                            continue
                    except OSError:
                        continue
                    yield FileItem(
                        name=entry.name, path=entry.path,
                        kind=_KIND_FILE, extension=_extension(entry.name),
                        parent=path, size=st.st_size,
                        mtime=st.st_mtime, root=dirpath,
                    )
            for sub, nd in subdirs:
                yield from rec(sub, nd)

        yield from rec(dirpath, depth)

    def _live_search(
        self, query: str, kinds: Optional[set], limit: int
    ) -> list[tuple[FileItem, int]]:
        items = self._live_candidates(query, kinds)
        if not items:
            return []
        return FileMatcher(items).search(query, kinds=kinds, limit=limit)

    # ------------------------------------------------------------------
    # Context
    # ------------------------------------------------------------------
    @property
    def context(self) -> FileContext:
        return self._context

    def remember(
        self,
        items,
        query: str = "",
        kind: str = "",
        last_item: Optional[FileItem] = None,
    ) -> None:
        self._context.remember(items, query=query, kind=kind,
                               last_item=last_item)

    # ------------------------------------------------------------------
    # Known-folder aliases (preserve existing behaviour)
    # ------------------------------------------------------------------
    def is_known_folder(self, name: str) -> bool:
        from src.controllers.folder_opener import KnownFolderResolver

        return KnownFolderResolver.normalize(name) in self._known_names

    def open_known_folder(self, name: str):
        """Delegate to the reviewed FolderOpener allowlist path."""
        if self._known_opener is not None:
            return self._known_opener.open(name)
        try:
            from src.controllers.folder_opener import FolderOpener

            return FolderOpener().open(name)
        except Exception:  # noqa: BLE001
            logger.exception("Known-folder open failed")
            return False, "open_failed"

    # ------------------------------------------------------------------
    # Opening
    # ------------------------------------------------------------------
    @staticmethod
    def is_dangerous(item: FileItem) -> bool:
        return item.extension.lower() in _DANGEROUS_EXTENSIONS

    def open_item(
        self, item: FileItem, confirmed: bool = False
    ) -> tuple[bool, str]:
        """Open a validated item. Returns ``(ok, detail)``.

        ``detail`` may be ``needs_confirmation`` for dangerous file
        types when ``confirmed`` is False.
        """
        if self._open is None:
            return False, "unsupported_platform"
        # Re-validate against the live filesystem: the index may be
        # stale (item renamed or deleted since the scan).
        try:
            exists_dir = os.path.isdir(item.path)
            exists_file = os.path.isfile(item.path)
        except OSError:
            return False, "path_error"
        if item.is_dir:
            if not exists_dir:
                return False, "not_found"
        else:
            if not exists_file:
                return False, "not_found"
            if self.is_dangerous(item) and not confirmed:
                return False, "needs_confirmation"
        try:
            self._open(item.path)
        except FileNotFoundError:
            return False, "not_found"
        except PermissionError:
            return False, "permission_denied"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to open %s", item.path)
            return False, str(exc)
        logger.info("Opened %s %s", item.kind, item.path)
        return True, "ok"

    def open_folder_path(self, path: str):
        """Open an arbitrary verified directory in Explorer."""
        try:
            if not os.path.isdir(path):
                return False, "not_found"
        except OSError:
            return False, "path_error"
        if self._open is None:
            return False, "unsupported_platform"
        try:
            self._open(path)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to open folder %s", path)
            return False, str(exc)
        return True, "ok"


# ----------------------------------------------------------------------
# Convenience factory
# ----------------------------------------------------------------------

def _file_index_row_count(db_path: Path) -> Optional[int]:
    """Return the ``fs_items`` row count, or None when unreadable."""
    try:
        uri = db_path.resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            return int(
                conn.execute("SELECT COUNT(*) FROM fs_items").fetchone()[0]
            )
        finally:
            conn.close()
    except (OSError, sqlite3.Error):
        return None


def _migrate_legacy_file_index(db_path: Path, legacy_dir: Optional[Path]) -> None:
    """Copy a legacy project-local ``file_index.db`` to ``db_path``.

    Idempotent: does nothing when ``db_path`` already holds indexed
    rows or no legacy database is present. The copy uses the SQLite
    backup API so it is safe even while the source is open, and the
    destination is verified with ``PRAGMA integrity_check`` before the
    migration is reported as successful. The legacy file is never
    deleted. When both databases contain rows, the destination always
    wins and the divergence is logged — an older legacy copy is never
    allowed to overwrite newer indexed data.
    """
    if legacy_dir is None:
        return
    legacy_db = Path(legacy_dir) / "file_index.db"
    if not legacy_db.is_file():
        return
    dest_rows = _file_index_row_count(db_path)
    if db_path.exists() and dest_rows is None:
        logger.error(
            "File index destination %s exists but is not a readable "
            "SQLite database; skipping legacy migration from %s",
            db_path, legacy_db,
        )
        return
    if dest_rows:
        legacy_rows = _file_index_row_count(legacy_db)
        if legacy_rows:
            logger.warning(
                "File index databases diverged: keeping %s "
                "(%d rows); legacy %s also has %d rows and was "
                "left untouched",
                db_path, dest_rows, legacy_db, legacy_rows,
            )
        return
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        source = sqlite3.connect(str(legacy_db))
        try:
            dest = sqlite3.connect(str(db_path))
            try:
                source.backup(dest)
                ok = dest.execute("PRAGMA integrity_check").fetchone()[0]
            finally:
                dest.close()
        finally:
            source.close()
        if ok == "ok":
            logger.info(
                "Migrated file index %s -> %s", legacy_db, db_path
            )
        else:
            db_path.unlink(missing_ok=True)
            logger.error(
                "File index migration failed integrity check (%s); "
                "legacy database left in place at %s",
                ok, legacy_db,
            )
    except (OSError, sqlite3.Error) as exc:
        db_path.unlink(missing_ok=True)
        logger.error(
            "File index migration %s -> %s failed: %s",
            legacy_db, db_path, exc,
        )


def create_file_system_service(
    data_dir: Optional[Path | str] = None,
    index_db_path: Optional[Path | str] = None,
    migrate_from: Optional[Path | str] = None,
    env: Optional[dict] = None,
    extra_roots: Optional[Iterable[str]] = None,
    open_fn: Optional[Callable[[str], object]] = None,
    known_folder_opener=None,
    start_background: bool = False,
) -> tuple[FileSystemService, FileSystemIndexer]:
    """Build store + indexer + service for the real runtime.

    ``index_db_path`` selects the database file directly; otherwise the
    legacy ``<data_dir>/file_index.db`` layout is used. When
    ``migrate_from`` points at a directory containing a legacy
    ``file_index.db``, it is copied to the new location once (verified,
    never deleted).
    """
    if index_db_path is not None:
        db_path = Path(index_db_path)
    else:
        base = Path(data_dir) if data_dir else Path("data")
        db_path = base / "file_index.db"
    _migrate_legacy_file_index(db_path, Path(migrate_from) if migrate_from else None)
    store = FileIndexStore(db_path)
    indexer = FileSystemIndexer(
        store=store, env=env, extra_roots=extra_roots
    )
    service = FileSystemService(
        indexer=indexer,
        env=env,
        open_fn=open_fn,
        known_folder_opener=known_folder_opener,
    )
    if start_background:
        indexer.start_background()
    return service, indexer
