"""Local file/folder command handlers (V10.1).

Executes discovery-backed OPEN_FOLDER / OPEN_FILE / SEARCH_* /
OPEN_LOCAL_ITEM intents through the injected ``svc.fs``
(:class:`FileSystemService`). Spoken text is never used as a command;
opening goes through Windows Shell (Explorer for folders, registered
file associations for files) on validated absolute paths only.

Ambiguous matches become clarifications (never silent guesses); weak
matches are reported as missing; dangerous file types require the
standard yes-confirmation lifecycle before opening.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.commands.handlers import _BaseHandler
from src.commands.registry import CommandResult, PERMISSION_SYSTEM
from src.controllers.file_system_discovery import (
    FileItem,
    FileMatcher,
    ItemResolveResult,
)

logger = logging.getLogger(__name__)

_KIND_DIR = "dir"
_KIND_FILE = "file"

_MAX_SPOKEN_RESULTS = 3


def _fs(svc):
    return getattr(svc, "fs", None)


def _unavailable(intent_name: str) -> CommandResult:
    return CommandResult(
        intent_name, False,
        message="I couldn't reach the file search service.",
        error="service_unavailable",
    )


def _item_pick_result(status: str, item=None, options=None):
    """Adapter matching the session controller's clarify contract.

    The clarify flow reads ``res.status``, ``res.app`` and
    ``res.options``; ``app`` carries the chosen :class:`FileItem`.
    """

    class _R:
        pass

    r = _R()
    r.status = status
    r.app = item
    r.options = list(options or ([] if item is None else [item]))
    r.score = 0
    return r


def _clarify_payload(candidates: list, intent_name: str,
                     message: str):
    """Build the ``data["clarify"]`` payload for candidate selection."""
    def resolver(text: str):
        status, item = FileMatcher.pick(text, candidates)
        if status == "found" and item is not None:
            return _item_pick_result("found", item=item)
        if status == "ambiguous":
            return _item_pick_result("ambiguous", options=candidates)
        return _item_pick_result("missing")

    def apply(intent, payload):
        intent.entities["_item"] = payload

    return {
        "clarify": {
            "resolver": resolver,
            "apply": apply,
            "message": message,
            "ambiguous_message": "Which one did you mean?",
            # V10.8 Task 4: lets the session controller classify this
            # as entity_ambiguous rather than a generic failure.
            "entity_issue": "ambiguous",
        }
    }


def _spoken_names(items: list, limit: int = _MAX_SPOKEN_RESULTS) -> str:
    names = [i.name for i in items[:limit]]
    if len(items) > limit:
        names.append(f"and {len(items) - limit} more")
    return ", ".join(names)


def _open_result(intent_name: str, item: FileItem, fs,
                 confirmed: bool = False) -> CommandResult:
    """Open ``item`` and shape the CommandResult (danger gate aware)."""
    ok, detail = fs.open_item(item, confirmed=confirmed)
    kind_word = "folder" if item.is_dir else "file"
    if detail == "needs_confirmation":
        return CommandResult(
            intent_name, True,
            message=(
                f"{item.name} looks like a program or script. "
                "Should I open it anyway? Say yes to confirm."
            ),
            confirmation_required=True,
            data={"item": item.path, "dangerous": True},
        )
    if not ok:
        if detail == "not_found":
            message = f"I couldn't find {item.name} anymore."
        elif detail == "permission_denied":
            message = f"I don't have permission to open {item.name}."
        else:
            message = f"I couldn't open {item.name}."
        return CommandResult(
            intent_name, False, message=message,
            error=detail or "open_failed",
            data={"item": item.path},
        )
    if item.is_dir:
        message = f"Opening your {item.name} folder."
    else:
        message = f"Opening {item.name}."
    return CommandResult(
        intent_name, True, message=message,
        data={
            "item": item.name, "path": item.path, "kind": item.kind,
            "parent": item.parent,
        },
        executed=True,
    )


class OpenDiscoveredFolderHandler(_BaseHandler):
    """OPEN_FOLDER — known aliases first, then arbitrary discovery."""

    intent_name = "OPEN_FOLDER"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc):
        self._svc = svc

    def validate(self, entities: dict) -> Optional[str]:
        folder = entities.get("folder")
        if not folder or not str(folder).strip():
            return "missing_folder"
        return None

    def execute(self, entities: dict) -> CommandResult:
        fs = _fs(self._svc)
        if fs is None:
            # Legacy wiring without the discovery service: keep the
            # reviewed known-folder allowlist behaviour.
            from src.commands.handlers import SUPPORTED_FOLDERS

            name = str(entities.get("folder") or "").strip().lower()
            opener = getattr(self._svc, "folders", None)
            if opener is None or name not in SUPPORTED_FOLDERS:
                return CommandResult(
                    self.intent_name, False,
                    message=f"I can't open {name or 'that folder'}.",
                    error="unsupported_folder",
                )
            ok, detail = opener.open(name)
            label = name.capitalize()
            if not ok:
                return CommandResult(
                    self.intent_name, False,
                    message=f"I couldn't open your {label} folder.",
                    error=detail or "open_failed",
                )
            return CommandResult(
                self.intent_name, True,
                message=f"Opening your {label} folder.",
                data={"folder": name}, executed=True,
            )

        preset = entities.get("_item")
        if preset is not None:
            if not isinstance(preset, FileItem):
                return self._fail("invalid_item")
            if not preset.is_dir:
                # A file was picked for a folder request: open it as a
                # file instead of failing confusingly.
                return _open_result(self.intent_name, preset, fs,
                                    confirmed=bool(entities.get("_confirmed")))
            return _open_result(self.intent_name, preset, fs)

        name = str(entities.get("folder") or "").strip()
        if not name:
            return self._fail("missing_folder")

        # Preserve the reviewed known-folder path (Desktop, Documents,
        # Downloads, Pictures, Music, Videos + aliases).
        if fs.is_known_folder(name):
            ok, detail = fs.open_known_folder(name)
            label = name.capitalize()
            if not ok:
                if detail == "folder_not_found":
                    message = f"I couldn't find your {label} folder."
                else:
                    message = f"I couldn't open your {label} folder."
                return CommandResult(
                    self.intent_name, False, message=message,
                    error=detail or "open_failed",
                )
            return CommandResult(
                self.intent_name, True,
                message=f"Opening your {label} folder.",
                data={"folder": name}, executed=True,
            )

        res = fs.resolve(name, kinds={_KIND_DIR})
        if res.status == "found" and res.item is not None:
            fs.remember([res.item], query=name, kind="folder",
                        last_item=res.item)
            return _open_result(self.intent_name, res.item, fs)
        if res.status == "ambiguous" and res.options:
            message = self._ambiguous_message(res.options)
            return CommandResult(
                self.intent_name, True, message=message,
                data=_clarify_payload(
                    res.options, self.intent_name, message
                ),
            )
        # Missing: try files too — "open projects folder" where only a
        # file matched is still worth an honest answer, not a guess.
        fres = fs.resolve(name, kinds={_KIND_FILE})
        if fres.status == "found" and fres.item is not None:
            return CommandResult(
                self.intent_name, False,
                message=(
                    f"I found {fres.item.name}, but that's a file, "
                    "not a folder. Say \"open file\" to open it."
                ),
                error="not_a_folder",
                data={"item": fres.item.name},
            )
        return CommandResult(
            self.intent_name, False,
            message=f"I couldn't find a folder named {name}.",
            error="folder_not_found",
        )

    @staticmethod
    def _ambiguous_message(options: list) -> str:
        names = [i.name for i in options[:5]]
        if len(names) == 1:
            return f"I found {names[0]}. Did you mean that folder?"
        return (
            "I found several folders: " + ", ".join(names)
            + ". Which one should I open?"
        )

    def _fail(self, error: str) -> CommandResult:
        return CommandResult(
            self.intent_name, False,
            message="I need a folder name.", error=error,
        )


class OpenFileHandler(_BaseHandler):
    """OPEN_FILE — resolve and open a file via Windows association."""

    intent_name = "OPEN_FILE"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc):
        self._svc = svc

    def validate(self, entities: dict) -> Optional[str]:
        if entities.get("_item") is not None:
            return None
        query = entities.get("query")
        if not query or not str(query).strip():
            return "missing_query"
        return None

    def execute(self, entities: dict) -> CommandResult:
        fs = _fs(self._svc)
        if fs is None:
            return _unavailable(self.intent_name)
        confirmed = bool(entities.get("_confirmed"))

        preset = entities.get("_item")
        if preset is not None:
            if not isinstance(preset, FileItem):
                return self._fail("invalid_item")
            return _open_result(self.intent_name, preset, fs,
                                confirmed=confirmed)

        query = str(entities.get("query") or "").strip()
        if entities.get("latest"):
            item = self._latest_match(fs, query)
            if item is None:
                return CommandResult(
                    self.intent_name, False,
                    message=f"I couldn't find a recent file matching {query}.",
                    error="file_not_found",
                )
            fs.remember([item], query=query, kind="file", last_item=item)
            return _open_result(self.intent_name, item, fs,
                                confirmed=confirmed)

        res = fs.resolve(query, kinds={_KIND_FILE})
        if res.status == "missing":
            # Folders can satisfy "open X" when no file matched — but
            # only report it; the user decides.
            dres = fs.resolve(query, kinds={_KIND_DIR})
            if dres.status == "found" and dres.item is not None:
                return CommandResult(
                    self.intent_name, False,
                    message=(
                        f"I found a folder named {dres.item.name}, "
                        "not a file. Say \"open folder\" to open it."
                    ),
                    error="not_a_file",
                    data={"item": dres.item.name},
                )
            return CommandResult(
                self.intent_name, False,
                message=f"I couldn't find a file matching {query}.",
                error="file_not_found",
            )
        if res.status == "ambiguous" and res.options:
            message = (
                "I found several files: "
                + _spoken_names(res.options)
                + ". Which one should I open?"
            )
            return CommandResult(
                self.intent_name, True, message=message,
                data=_clarify_payload(res.options, self.intent_name,
                                      message),
            )
        item = res.item
        fs.remember([item] + [o for o in res.options if o != item],
                    query=query, kind="file", last_item=item)
        return _open_result(self.intent_name, item, fs,
                            confirmed=confirmed)

    @staticmethod
    def _latest_match(fs, query: str) -> Optional[FileItem]:
        """Most recently modified strong match for 'latest' queries."""
        results = fs.search(query, kinds={_KIND_FILE}, limit=25)
        strong = [i for i, s in results if s >= 80]
        pool = strong or [i for i, _ in results]
        if not pool:
            return None
        return max(pool, key=lambda i: i.mtime)

    def _fail(self, error: str) -> CommandResult:
        return CommandResult(
            self.intent_name, False,
            message="I need a file name.", error=error,
        )


class _SearchHandler(_BaseHandler):
    """Base for SEARCH_FILES / SEARCH_FOLDERS / SEARCH_LOCAL_ITEMS.

    Search never opens anything: it speaks a concise summary and
    remembers the candidates so "open it" / "the second one" work.
    """

    intent_name = ""
    permission = PERMISSION_SYSTEM
    executable = True
    kinds = None
    label = "items"

    def __init__(self, svc):
        self._svc = svc

    def validate(self, entities: dict) -> Optional[str]:
        query = entities.get("query")
        if not query or not str(query).strip():
            return "missing_query"
        return None

    def execute(self, entities: dict) -> CommandResult:
        fs = _fs(self._svc)
        if fs is None:
            return _unavailable(self.intent_name)
        query = str(entities.get("query") or "").strip()

        # SEARCH_FILES prefers files but happily reports folders when
        # only a folder matched ("find Smart Parking project").
        kinds = self.kinds
        results = fs.search(query, kinds=kinds, limit=10)
        items = [i for i, _ in results]
        if not items and kinds == {_KIND_FILE}:
            alt = fs.search(query, kinds={_KIND_DIR}, limit=10)
            if alt:
                results = alt
                items = [i for i, _ in alt]
                kinds = {_KIND_DIR}

        if not items:
            fs.remember([], query=query, kind=self.label)
            return CommandResult(
                self.intent_name, True,
                message=f"I couldn't find any {self.label} matching {query}.",
                executed=True,
            )

        fs.remember(items, query=query, kind=self.label)
        if len(items) == 1:
            item = items[0]
            kind_word = "folder" if item.is_dir else "file"
            message = f"I found {item.name}, a {kind_word}."
        else:
            message = (
                f"I found {len(items)} {self.label}: "
                f"{_spoken_names(items)}. Say \"open the first one\" "
                "to open one."
            )
        return CommandResult(
            self.intent_name, True, message=message,
            data={
                "query": query, "count": len(items),
                "items": [
                    {"name": i.name, "path": i.path, "kind": i.kind,
                     "parent": i.parent}
                    for i in items[:10]
                ],
            },
            executed=True,
        )


class SearchFilesHandler(_SearchHandler):
    intent_name = "SEARCH_FILES"
    kinds = {_KIND_FILE}
    label = "files"


class SearchFoldersHandler(_SearchHandler):
    intent_name = "SEARCH_FOLDERS"
    kinds = {_KIND_DIR}
    label = "folders"


class SearchLocalItemsHandler(_SearchHandler):
    intent_name = "SEARCH_LOCAL_ITEMS"
    kinds = {_KIND_FILE, _KIND_DIR}
    label = "items"


class OpenLocalItemHandler(_BaseHandler):
    """OPEN_LOCAL_ITEM — contextual 'open it / second one / parent'."""

    intent_name = "OPEN_LOCAL_ITEM"
    permission = PERMISSION_SYSTEM
    executable = True

    def __init__(self, svc):
        self._svc = svc

    def validate(self, entities: dict) -> Optional[str]:
        return None

    def execute(self, entities: dict) -> CommandResult:
        fs = _fs(self._svc)
        if fs is None:
            return _unavailable(self.intent_name)
        confirmed = bool(entities.get("_confirmed"))
        selector = str(entities.get("selector") or "it").strip().lower()
        ctx = fs.context

        if selector == "parent":
            parent = ctx.parent_item()
            if parent is None:
                return CommandResult(
                    self.intent_name, False,
                    message="I don't have a previous result to open.",
                    error="no_context",
                )
            ok, detail = fs.open_folder_path(parent.path)
            if not ok:
                return CommandResult(
                    self.intent_name, False,
                    message=f"I couldn't open {parent.name}.",
                    error=detail or "open_failed",
                )
            return CommandResult(
                self.intent_name, True,
                message=f"Opening {parent.name}.",
                data={"item": parent.name, "path": parent.path,
                      "kind": _KIND_DIR},
                executed=True,
            )

        candidates = ctx.candidates
        if not candidates:
            return CommandResult(
                self.intent_name, False,
                message="I don't have a previous search to open from. "
                        "Try finding the file or folder first.",
                error="no_context",
            )

        if selector in ("it", "that", "this", "them", "the last one",
                        "the previous one", "last"):
            item = ctx.last_item or candidates[0]
        else:
            status, item = FileMatcher.pick(selector, candidates)
            if status != "found" or item is None:
                return CommandResult(
                    self.intent_name, False,
                    message="I couldn't tell which one you meant.",
                    error="ambiguous_selection",
                )
        return _open_result(self.intent_name, item, fs,
                            confirmed=confirmed)


def create_file_handlers(svc):
    """Return ``{intent_name: handler}`` for the V10.1 intents."""
    return {
        "OPEN_FOLDER": OpenDiscoveredFolderHandler(svc),
        "OPEN_FILE": OpenFileHandler(svc),
        "SEARCH_FILES": SearchFilesHandler(svc),
        "SEARCH_FOLDERS": SearchFoldersHandler(svc),
        "SEARCH_LOCAL_ITEMS": SearchLocalItemsHandler(svc),
        "OPEN_LOCAL_ITEM": OpenLocalItemHandler(svc),
    }
