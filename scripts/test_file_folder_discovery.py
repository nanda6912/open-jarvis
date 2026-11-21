"""Automated tests for VoiceOS V10.1 file/folder discovery.

Builds a temporary directory tree (never touches real user documents),
runs the real FileSystemIndexer/FileMatcher/FileSystemService and the
real intent recognizer + command registry end-to-end, and writes
JSON/CSV reports to ``reports/``.

Usage:
    python scripts/test_file_folder_discovery.py            # full suite
    python scripts/test_file_folder_discovery.py --live     # + real opens
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.commands.app_handlers import _Svc, create_app_registry  # noqa: E402
from src.commands.intent import Intent, RuleBasedIntentRecognizer  # noqa: E402
from src.commands.registry import CommandRegistry  # noqa: E402
from src.controllers.file_system_discovery import (  # noqa: E402
    FileIndexStore,
    FileItem,
    FileMatcher,
    FileSystemIndexer,
    FileSystemService,
    create_file_system_service,
)

RESULTS: list[dict] = []


def record(section: str, name: str, ok: bool, detail: str = ""):
    RESULTS.append(
        {"section": section, "test": name, "status": "PASS" if ok else "FAIL",
         "detail": detail}
    )
    print(f"[{'PASS' if ok else 'FAIL'}] {section}/{name} {detail}")


# ----------------------------------------------------------------------
# Fakes for registry construction
# ----------------------------------------------------------------------
class _FakeDiscovery:
    def resolve(self, q):
        from src.controllers.application_discovery import ResolveResult
        return ResolveResult("missing")

    def discover(self, force=False):
        return []


class _FakeLauncher:
    def launch(self, app):
        return False, "fake"


class _FakeMonitor:
    def list_processes(self):
        return []


class _FakeWindows:
    def list_windows(self, include_hidden=False):
        return []

    def windows_for(self, t):
        return []


class _FakeFolders:
    def open(self, name):
        return True, "ok"


def build_service(root: Path, opened: list, env=None):
    """Real indexer+service over ``root`` only; opens are recorded."""
    store = FileIndexStore(root.parent / "idx_test.db")
    indexer = FileSystemIndexer(
        store=store,
        env=env or {},
        extra_roots=[str(root)],
        known_folder_resolver=lambda name: None,
        max_depth=12,
    )
    service = FileSystemService(
        indexer=indexer,
        env=env or {},
        open_fn=lambda p: opened.append(p),
        known_folder_opener=_FakeFolders(),
        live_fallback=True,
    )
    stats = indexer.scan()
    return service, indexer, stats


def make_tree(base: Path) -> dict:
    """Create the fixture tree; returns paths of interest."""
    paths = {}

    def w(rel, content="x"):
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return p

    def d(rel):
        p = base / rel
        p.mkdir(parents=True, exist_ok=True)
        return p

    d("Smart Parking System")
    d("Smart Parking System/src/main")
    d("TeamFund")
    d("Java Projects")
    d("Java Projects/Assignment 1")
    d("Documents/Resume Folder")
    d("Ünïcodé Földér/")
    d("Duplicate/Projects")
    d("Other/Projects")
    w("resume.pdf", "%PDF-1.4 fake")
    w("Documents/requirements.docx", "doc")
    w("Documents/report final.pdf", "pdf")
    w("Java Projects/Assignment 1/Main.java", "class Main {}")
    w("notes.txt", "hello")
    w("photo.png", "png")
    w("Duplicate/resume.pdf", "pdf2")
    w("Other/resume.pdf", "pdf3")
    w("Ünïcodé Földér/résumé cv.pdf", "pdf")
    w("run.bat", "echo hi")
    w("script.ps1", "echo hi")
    paths["smart_parking"] = base / "Smart Parking System"
    paths["teamfund"] = base / "TeamFund"
    paths["java_projects"] = base / "Java Projects"
    paths["resume_pdf"] = base / "resume.pdf"
    paths["nested_java"] = base / "Java Projects/Assignment 1/Main.java"
    paths["unicode_dir"] = base / "Ünïcodé Földér"
    paths["unicode_file"] = base / "Ünïcodé Földér/résumé cv.pdf"
    paths["dup1"] = base / "Duplicate/Projects"
    paths["dup2"] = base / "Other/Projects"
    return paths


def dispatch(registry, text):
    intent = RuleBasedIntentRecognizer().recognize(text)
    return registry.dispatch(intent), intent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="perform real os.startfile opens")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="voiceos_v101_"))
    tree = tmp / "storage"
    paths = make_tree(tree)

    opened: list[str] = []
    live = args.live
    open_fn = (lambda p: __import__("os").startfile(p)) if live \
        else (lambda p: opened.append(p))

    store = FileIndexStore(tmp / "idx.db")
    indexer = FileSystemIndexer(
        store=store, env={}, extra_roots=[str(tree)],
        known_folder_resolver=lambda name: None, max_depth=12,
    )
    folders = _FakeFolders()
    if live:
        from src.controllers.folder_opener import FolderOpener
        folders = FolderOpener()
    service = FileSystemService(
        indexer=indexer, env={}, open_fn=open_fn,
        known_folder_opener=folders, live_fallback=True,
    )

    t0 = time.time()
    stats = indexer.scan()
    scan_secs = time.time() - t0
    record("discovery", "index_scan", stats["seen"] > 10,
           f"{stats['seen']} entries in {scan_secs:.2f}s")

    # ---------------- discovery ----------------
    items = store.all_items()
    names = {i.name.lower() for i in items}
    for label, path in paths.items():
        record("discovery", f"indexed_{label}", path.name.lower() in names
               or any(i.path == str(path) for i in items),
               str(path))
    dirs = {i.path for i in items if i.kind == "dir"}
    record("discovery", "nested_folder_indexed",
           str(paths["nested_java"].parent) in dirs)
    record("discovery", "unicode_indexed",
           str(paths["unicode_dir"]) in dirs
           and str(paths["unicode_file"]) in {i.path for i in items})

    # ---------------- matching ----------------
    def res(query, kinds=None):
        return service.resolve(query, kinds=kinds)

    r = res("smart parking system", {"dir"})
    record("matching", "exact_folder", r.status == "found"
           and r.item.path == str(paths["smart_parking"]),
           f"{r.status} {getattr(r.item,'path','')}")
    r = res("smart parking", {"dir"})
    record("matching", "partial_folder", r.status == "found"
           and "Smart Parking" in r.item.name, r.status)
    r = res("resume.pdf", {"file"})
    record("matching", "exact_file_with_ext", r.status in ("found", "ambiguous"),
           r.status)
    r = res("teamfund", {"dir"})
    record("matching", "single_token_folder", r.status == "found"
           and r.item.name == "TeamFund", r.status)
    r = res("java assignment", {"file"})
    record("matching", "token_file", r.status == "found"
           and r.item.path == str(paths["nested_java"]),
           f"{r.status} {getattr(r.item,'path','')}")
    r = res("résumé cv", {"file"})
    record("matching", "unicode_file", r.status == "found"
           and r.item.path == str(paths["unicode_file"]),
           f"{r.status} {getattr(r.item,'path','')}")
    r = res("does not exist anywhere", {"dir", "file"})
    record("matching", "missing_item", r.status == "missing", r.status)
    r = res("projects", {"dir"})
    record("matching", "duplicate_folders_ambiguous",
           r.status == "ambiguous" and len(r.options) >= 2,
           f"{r.status} n={len(r.options)}")
    r = res("nonexistent folder qwerty", {"dir"})
    record("matching", "weak_match_not_opened", r.status == "missing",
           r.status)

    # pick / clarification
    status, item = FileMatcher.pick("the second one",
                                  [FileItem("A", "/a", "dir"),
                                   FileItem("B", "/b", "dir")])
    record("matching", "ordinal_pick", status == "found" and item.name == "B")
    status, _ = FileMatcher.pick("cancel it", [FileItem("A", "/a", "dir")])
    record("matching", "pick_no_match", status == "missing")

    # ---------------- opening (fake opener records paths) ----------------
    if not live:
        ok, detail = service.open_item(paths["smart_parking"] and FileItem(
            name="Smart Parking System", path=str(paths["smart_parking"]),
            kind="dir"))
        record("opening", "open_folder_recorded", ok and opened
               and opened[-1] == str(paths["smart_parking"]), detail)
        fi = FileItem(name="resume.pdf", path=str(paths["resume_pdf"]),
                      kind="file", extension=".pdf")
        ok, _ = service.open_item(fi)
        record("opening", "open_file_recorded",
               ok and opened[-1] == str(paths["resume_pdf"]))
        danger = FileItem(name="run.bat", path=str(tree / "run.bat"),
                          kind="file", extension=".bat")
        ok, detail = service.open_item(danger)
        record("opening", "dangerous_requires_confirm",
               (not ok) and detail == "needs_confirmation", detail)
        ok, _ = service.open_item(danger, confirmed=True)
        record("opening", "dangerous_confirmed_opens",
               ok and opened[-1].endswith("run.bat"))
        missing = FileItem(name="gone.txt", path=str(tree / "gone.txt"),
                           kind="file", extension=".txt")
        ok, detail = service.open_item(missing)
        record("opening", "deleted_item_not_found",
               (not ok) and detail == "not_found", detail)
        # no arbitrary process execution: opener only ever got real paths
        record("opening", "only_paths_opened",
               all(isinstance(p, str) and os.path.isabs(p) for p in opened))
    else:
        record("opening", "live_mode", True, "real os.startfile used")

    # ---------------- context ----------------
    service.remember(
        [FileItem("First", "/f1", "file"), FileItem("Second", "/f2", "file")],
        query="test", kind="file")
    st, it = FileMatcher.pick("second", service.context.candidates)
    record("context", "remembered_candidates", st == "found"
           and it.name == "Second")
    service.context._at -= 10000  # force expiry
    record("context", "expired_context_empty",
           service.context.candidates == [])

    # ---------------- registry / intents ----------------
    svc = _Svc(_FakeDiscovery(), _FakeLauncher(), _FakeMonitor(),
               _FakeWindows(), fs=service, folders=folders)
    registry = create_app_registry(svc)

    cases = [
        ("Open Smart Parking System folder", "OPEN_FOLDER"),
        ("Open projects folder", "OPEN_FOLDER"),
        ("Open TeamFund folder", "OPEN_FOLDER"),
        ("Open Java Projects folder", "OPEN_FOLDER"),
        ("Open resume.pdf", "OPEN_FILE"),
        ("Open Java assignment", "OPEN_FILE"),
        ("Open the file named requirements.docx", "OPEN_FILE"),
        ("Find my resume files", "SEARCH_FILES"),
        ("Search for PDF files", "SEARCH_FILES"),
        ("Find Smart Parking project", "SEARCH_FILES"),
        ("Search for Java files", "SEARCH_FILES"),
        ("Find Projects folders", "SEARCH_FOLDERS"),
        ("Show me files and folders", "SEARCH_LOCAL_ITEMS"),
        ("Open it", "OPEN_LOCAL_ITEM"),
        ("Open the second one", "OPEN_LOCAL_ITEM"),
        ("Open its parent folder", "OPEN_LOCAL_ITEM"),
    ]
    for text, expected in cases:
        result, intent = dispatch(registry, text)
        record("intents", f"route:{text}", intent.name == expected,
               f"{intent.name} entities={intent.entities}")

    # end-to-end folder open through the registry
    opened.clear()
    result, _ = dispatch(registry, "Open Smart Parking System folder")
    record("e2e", "open_custom_folder",
           result.success and result.executed
           and opened and opened[-1] == str(paths["smart_parking"]),
           result.message)

    opened.clear()
    result, _ = dispatch(registry, "Open TeamFund folder")
    record("e2e", "open_teamfund",
           result.success and opened and opened[-1] == str(paths["teamfund"]),
           result.message)

    # ambiguous folder -> clarify -> pick second
    opened.clear()
    result, _ = dispatch(registry, "Open Projects folder")
    record("e2e", "ambiguous_projects_clarify",
           result.success and result.data.get("clarify") is not None,
           result.message)
    if result.data.get("clarify"):
        clarify = result.data["clarify"]
        # candidates order = resolver's options (score order)
        pick_res = clarify["resolver"]("the second one")
        record("e2e", "clarify_second_found", pick_res.status == "found",
               pick_res.status)
        if pick_res.status == "found":
            intent = Intent("OPEN_FOLDER",
                            {"folder": "projects"}, raw_text="second")
            clarify["apply"](intent, pick_res.app)
            result2 = registry.dispatch(intent)
            record("e2e", "clarify_second_opens",
                   result2.executed and opened
                   and opened[-1] == pick_res.app.path,
                   result2.message)

    # file open end-to-end (duplicate names -> clarify -> pick first)
    opened.clear()
    result, _ = dispatch(registry, "Open resume.pdf")
    if result.data.get("clarify"):
        pick_res = result.data["clarify"]["resolver"]("the first one")
        intent = Intent("OPEN_FILE", {"query": "resume.pdf"})
        result.data["clarify"]["apply"](intent, pick_res.app)
        result = registry.dispatch(intent)
    record("e2e", "open_pdf", result.success and result.executed
           and any(p.endswith("resume.pdf") for p in opened),
           result.message)

    # dangerous file requires confirmation through registry
    opened.clear()
    result, _ = dispatch(registry, "Open run.bat")
    record("e2e", "bat_needs_confirm", result.confirmation_required
           and not opened, result.message)
    if result.confirmation_required:
        intent = Intent("OPEN_FILE", {"query": "run.bat", "_confirmed": True})
        result2 = registry.dispatch(intent)
        record("e2e", "bat_confirmed_opens",
               result2.executed and opened and opened[-1].endswith("run.bat"),
               result2.message)

    # search does not open
    opened.clear()
    result, _ = dispatch(registry, "Find my resume files")
    record("e2e", "search_does_not_open",
           result.success and not opened and result.data.get("count", 0) >= 1,
           result.message)

    # follow-up "open it" after a search
    result, _ = dispatch(registry, "Find resume")
    opened.clear()
    result2, _ = dispatch(registry, "Open it")
    record("e2e", "open_it_after_search",
           result2.success and result2.executed and opened,
           result2.message)

    # missing file honest failure
    result, _ = dispatch(registry, "Open zzznotafile.pdf")
    record("e2e", "missing_file_fails",
           (not result.success) and "couldn't find" in result.message.lower(),
           result.message)

    # known folder alias still works through discovery handler path:
    # "documents" is a known name; fake FolderOpener returns ok.
    result, _ = dispatch(registry, "Open Documents folder")
    record("e2e", "known_folder_alias", result.success and result.executed,
           result.message)

    # ---------------- report ----------------
    out = PROJECT_ROOT / "reports"
    out.mkdir(exist_ok=True)
    passed = sum(1 for r in RESULTS if r["status"] == "PASS")
    failed = [r for r in RESULTS if r["status"] == "FAIL"]
    payload = {
        "suite": "v101_file_folder_discovery",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "live": bool(live),
        "scan_seconds": round(scan_secs, 3),
        "indexed_entries": stats.get("indexed"),
        "passed": passed,
        "failed": len(failed),
        "results": RESULTS,
    }
    (out / "v101_file_folder_test_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    with (out / "v101_file_folder_test_results.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "test", "status", "detail"])
        w.writeheader()
        w.writerows(RESULTS)
    print(f"\n{passed} passed, {len(failed)} failed -> reports/v101_*")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
