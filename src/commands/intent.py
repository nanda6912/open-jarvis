"""Offline intent recognition for VoiceOS.

A rule-based parser maps recognized speech to a structured
:class:`Intent` (name + extracted entities). It is fully offline, has
no external dependencies and never executes anything — it only
understands.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Canonical folder names Jarvis knows about (Task 6+ may expand this).
KNOWN_FOLDERS = (
    "downloads",
    "documents",
    "desktop",
    "pictures",
    "photos",
    "music",
    "videos",
    "movies",
)


@dataclass
class Intent:
    """Structured result of intent recognition.

    Attributes:
        name: One of the supported intent names or ``"UNKNOWN"``.
        entities: Extracted arguments (app_name, folder, query, amount).
        raw_text: The original recognized text.
        confidence: Parser confidence (rule-based: 1.0 or 0.0).
    """

    name: str
    entities: dict = field(default_factory=dict)
    raw_text: str = ""
    confidence: float = 1.0


# ----------------------------------------------------------------------
# Entity extractors
# ----------------------------------------------------------------------
_FILLER_WORDS = (
    "the ", "a ", "an ", "my ", "this ", "that ", "up ",
)
_TRAILING_FILLER = re.compile(r"\b(please|for me|now|thanks|thank you)\b\.?\s*$")


def _strip_filler(text: str) -> str:
    """Remove leading articles/possessives and trailing politeness."""
    t = text.strip().strip(".!?,")
    t = _TRAILING_FILLER.sub("", t).strip()
    changed = True
    while changed:
        changed = False
        for f in _FILLER_WORDS:
            if t.lower().startswith(f):
                t = t[len(f):]
                changed = True
    # Drop leading command-verb residue like "to ".
    if t.lower().startswith("to "):
        t = t[3:]
    # Strip leftover standalone filler words ("the", "my", ...).
    t = re.sub(
        r"^(the|a|an|my|this|that|it)\b[\s.!?,]*", "", t, flags=re.I
    ).strip()
    t = re.sub(
        r"[\s.!?,]*\b(the|a|an|my|this|that|it|please)\b[\s.!?,]*$",
        "",
        t,
        flags=re.I,
    ).strip()
    return t.strip(".!?,")


def _extract_app_name(text: str) -> str:
    """Extract an application name from the text after the verb."""
    # V10.5: "this PC"/"my computer" are Windows shell destinations,
    # not app names. _strip_filler would reduce them to "pc"/"computer"
    # and fuzzy resolution picked "PC Manager" — keep the canonical
    # alias so discovery maps them to File Explorer instead.
    low = (text or "").lower()
    if re.search(r"\bthis\s+pc\b", low):
        return "this pc"
    if re.search(r"\bmy\s+computer\b", low):
        return "my computer"
    t = _strip_filler(text)
    # "open the chrome app" -> "chrome"
    t = re.sub(r"\b(app|application|program)\b\.?\s*$", "", t, flags=re.I)
    # "bring notepad to front" -> "notepad"
    t = re.sub(r"\s+to\s+(the\s+)?front\s*$", "", t, flags=re.I)
    return t.strip()


def _extract_folder(text: str) -> str:
    """Extract a folder name; prefers an explicit ``<name> folder``.

    An explicit "X folder"/"X directory" capture wins over the
    known-folder scan so custom names that merely *contain* a known
    word ("my Music Projects folder") resolve to the real name.
    """
    low = text.lower()
    m = re.search(
        r"(?:my |the |a |an )?([a-z0-9_\- ]+?)\s+"
        r"(?:folders?|directories|directory)\b",
        low,
    )
    if m:
        cand = m.group(1).strip()
        # Strip leading verb/politeness residue ("open my X folder",
        # "show me the X folder", "go to X folder").
        cand = re.sub(
            r"^(?:please\s+)?(?:can you\s+|could you\s+)?"
            r"(?:i\s+)?(?:want\s+to\s+|would\s+like\s+to\s+|wanna\s+)?"
            r"(?:open|show|go\s+to|launch|take\s+me\s+to|find|search|"
            r"look\s+for|list|display)\s+",
            "",
            cand,
            flags=re.I,
        ).strip()
        cand = re.sub(r"^me\s+", "", cand, flags=re.I).strip()
        # Strip leading articles/possessives, possibly repeated.
        prev = None
        while prev != cand:
            prev = cand
            cand = re.sub(
                r"^(?:the|a|an|my|this|that)\s+", "", cand,
                flags=re.I,
            ).strip()
        if cand:
            return cand
    for name in KNOWN_FOLDERS:
        if re.search(rf"\b{re.escape(name)}\b", low):
            return name
    return ""


def _extract_open_name(text: str, match=None) -> str:
    """Extract the item name from an open-file utterance.

    Keeps extension dots ("resume.pdf") and strips verbs, articles and
    "the file named/called" scaffolding.
    """
    if match is not None and match.lastindex and match.lastindex >= 1:
        t = match.group(match.lastindex)
    else:
        t = text
    t = (t or "").strip().strip(".!?,")
    t = re.sub(
        r"^(?:please\s+)?(?:can you\s+|could you\s+)?"
        r"(?:i\s+)?(?:want\s+to\s+|would\s+like\s+to\s+|wanna\s+)?"
        r"(?:open|launch|start|run|show|go\s+to|take\s+me\s+to|"
        r"bring\s+up|fire\s+up|load|display|view)\s+",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"^(?:the\s+|a\s+|an\s+|my\s+)?"
        r"(?:file|document|item|one)\s+(?:named|called)\s+",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"^(?:the|a|an|my|this|that)\s+", "", t, flags=re.I
    ).strip()
    t = re.sub(
        r"\s*\b(please|for me|now|thanks|thank you)\b\.?\s*$",
        "",
        t,
        flags=re.I,
    ).strip()
    return t.strip(" .,")


def _extract_open_file(text: str, match=None) -> dict:
    """Entities for OPEN_FILE: query name + 'latest' modifier."""
    name = _extract_open_name(text, match)
    latest = bool(
        re.search(r"\b(latest|newest|most recent)\b", text, re.I)
    )
    return {"query": name, "latest": latest}


def _extract_context_selector(text: str, match=None) -> dict:
    """Entities for OPEN_LOCAL_ITEM follow-up commands."""
    t = (text or "").lower().strip()
    selector = "it"
    if re.search(r"\bparent\b", t):
        selector = "parent"
    else:
        m = re.search(
            r"\b(first|second|third|fourth|fifth|1st|2nd|3rd|"
            r"latest|newest|most\s+recent|last)\b", t
        )
        if m:
            selector = re.sub(r"\s+", " ", m.group(1)).strip()
    return {"selector": selector}


_WEB_ENGINES = ("youtube", "google", "wikipedia")


def _extract_web_search(text: str, match=None) -> dict:
    """Entities for SEARCH_WEB: query + destination engine.

    General web requests must never degrade into local file searches
    ("Search YouTube" returned JavaScript filenames). The engine is
    inferred from explicit destination words; the query keeps the
    user's actual search text.
    """
    t = (text or "").strip().strip(".!?,")
    low = t.lower()
    engine = "web"
    for eng in _WEB_ENGINES:
        if re.search(rf"\b{eng}\b", low):
            engine = eng
            break
    # Drop leading request scaffolding and the search verb.
    t = re.sub(
        r"^(?:please\s+)?(?:can you\s+|could you\s+)?"
        r"(?:i\s+)?(?:want\s+to\s+|would\s+like\s+to\s+|wanna\s+)?"
        r"(?:search|look\s+up|google|find|browse|check|open)\s+",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(r"^for\s+", "", t, flags=re.I)
    # Drop web-scope phrases wherever they appear.
    t = re.sub(
        r"\b(?:on|in|via|using|through|from)\s+(?:the\s+|my\s+)?"
        r"(?:web|internet|online|youtube|google|wikipedia)\b",
        " ",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"\b(?:the\s+)?(?:web|internet|online)\s+for\s+",
        " ",
        t,
        flags=re.I,
    )
    # Engine name used as destination: "youtube cats" -> query "cats".
    for eng in _WEB_ENGINES:
        t = re.sub(rf"\b{eng}\s+for\s+", " ", t, flags=re.I)
        t = re.sub(rf"^\s*{eng}\s+", "", t, flags=re.I)
    # Trailing/standalone scope words are not part of the query
    # ("jarvis online" -> "jarvis").
    t = re.sub(
        r"\b(?:web|internet|online)\s*$", "", t, flags=re.I
    )
    t = re.sub(r"^\s*(?:web|internet|online)\s+", "", t, flags=re.I)
    t = re.sub(r"\s+", " ", t).strip(" .,!?")
    # A query that is just the engine name means "open the site".
    if t.lower() in _WEB_ENGINES or t.lower() in (
        "the web", "web", "internet", "online",
    ):
        t = ""
    return {"query": _strip_filler(t), "engine": engine}


def _extract_search_query(text: str) -> str:
    """Extract the query from a file-search utterance."""
    t = text.strip().strip(".!?,")
    t = re.sub(
        r"^(?:please\s+)?(?:can you\s+|could you\s+)?"
        r"(?:search|find|look|show|list|display)\b",
        "",
        t,
        flags=re.I,
    ).strip()
    t = re.sub(r"^me\s+", "", t, flags=re.I).strip()
    # "search for X" / "look up X" / "show me X"
    t = re.sub(r"^(?:for|up)\s+", "", t, flags=re.I)
    # Drop scope phrases: "my files", "the documents", "files for ..."
    t = re.sub(
        r"^(?:my|the|our)\s+(?:files?|folders?|documents?|computer|pc)\s+",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(r"^for\s+", "", t, flags=re.I)
    t = re.sub(
        r"^(?:files?|folders?|documents?)\s+",
        "",
        t,
        flags=re.I,
    )
    # "... in my files / on my computer" trailing scope
    t = re.sub(
        r"\s+(?:in|on|from)\s+(?:my|the|our)\s+"
        r"(?:files?|folders?|computer|pc|documents?)\b.*$",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(r"\s*\bfiles?\b\s*$", "", t, flags=re.I)
    return _strip_filler(t)


_WINDOW_TARGET_STOPWORDS = {
    "from", "to", "on", "at", "in", "of", "for", "the", "a", "an",
    "my", "this", "that", "it", "window", "please",
}
_WINDOW_PRONOUNS = {"", "window", "this window", "the window", "it",
                    "this", "that"}
_WINDOW_RESIDUE_FILLER = {
    "the", "a", "an", "my", "this", "that", "it", "window",
    "active", "current", "please",
}


def _window_target(match) -> dict:
    """Extract the window target after the leading verb.

    Returns ``target`` plus ``needs_target`` when the utterance contained
    a target-shaped fragment that cleaned down to nothing usable (e.g.
    "Restore from." -> the word "from" is never a valid app target).
    """
    t = re.sub(
        r"^(?:minimi[sz]e|maximi[sz]e|un-?maximi[sz]e|restore|"
        r"bring\s+back|focus)\s+",
        "",
        match.group(0),
        flags=re.I,
    )
    t = re.sub(r"[\s.!?,]+$", "", t)
    t = re.sub(r"\bwindow\b\s*$", "", t, flags=re.I)
    t = re.sub(r"[\s.!?,]+$", "", t)
    residue = t.strip()
    target = _extract_app_name(t)
    # Strip leading prepositions/articles the app extractor keeps.
    prev = None
    while prev != target:
        prev = target
        target = re.sub(
            r"^(from|to|on|at|in|of|for|into|over)\s+", "",
            target, flags=re.I,
        ).strip()
    target = target.strip(" .!?,")
    norm = target.lower()
    # Residues made purely of articles/pronouns ("the", "this") refer
    # to the active window — only meaningful leftovers ("from") ask
    # for a target.
    residue_words = set(re.findall(r"[a-z]+", residue.lower()))
    meaningful = residue_words - _WINDOW_RESIDUE_FILLER
    needs_target = bool(
        meaningful
        and (not target or norm in _WINDOW_TARGET_STOPWORDS)
    )
    if norm in _WINDOW_PRONOUNS or norm in _WINDOW_TARGET_STOPWORDS:
        target = ""
    return {"target": target, "needs_target": needs_target}


def _clean_move_target(text: str) -> str:
    """Normalize a move-command window target ('this window' -> '')."""
    target = _extract_app_name(text or "")
    target = target.strip(" .!?,").lower()
    if target in _WINDOW_PRONOUNS:
        return ""
    return target


def _monitor_spec(text: str) -> str:
    """Clean a spoken monitor phrase down to its descriptor tokens."""
    t = (text or "").strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t, flags=re.I)
    t = re.sub(
        r"\b(the|my|your|a|an|to|on|onto|in|into|please)\b", " ",
        t, flags=re.I,
    )
    t = re.sub(r"\b(monitors?|screens?|displays?)\b", " ", t, flags=re.I)
    # STT confuses "right" with "write"/"rite" in monitor phrases
    # ("move chrome to write monitor"). Safe here: this function only
    # ever sees the monitor-destination fragment of a command.
    t = re.sub(r"\b(write|rite|wright)\b", "right", t, flags=re.I)
    # STT renders "left" as "less" inside monitor phrases
    # ("to less the monitor", "less than monitor"). Scoped to the
    # monitor-destination fragment only.
    t = re.sub(r"\bless\b", "left", t, flags=re.I)
    t = re.sub(r"\bthan\b|\bthen\b", " ", t, flags=re.I)
    return " ".join(t.split())


# ----------------------------------------------------------------------
# Power-action recognition (unsupported Windows operations)
# ----------------------------------------------------------------------
# Windows power phrases must NEVER end the voice session and are never
# executed — they map to UNSUPPORTED_WINDOWS_ACTION so the user gets an
# honest "I can't do that" instead of a silent session exit or a power
# change. Keep this list conservative: only phrases that unambiguously
# refer to OS power/session state belong here.
_POWER_OBJECT = (
    r"(?:computer|pc|windows|laptop|machine|workstation|desktop|system)"
)
# Words that are unambiguously OS power actions even with no object
# ("shut down.", "power off.", "restart."). "turn off"/"switch off"
# are deliberately excluded — they collide with mute ("turn off the
# volume") and are only power actions with a computer object.
_POWER_WORDS = (
    r"shut\s?down|power\s+off|power\s+down|restart|hibernate|"
    r"sign\s*(?:out|off)"
)
_POWER_WORDS_WITH_OBJECT = _POWER_WORDS + r"|turn\s+off|switch\s+off"
# Narrow STT-mishearing variants observed in the 2026-11-20 production
# transcript ("Cancel sherdown.", "Shed down computer.", "Sheep my
# computer."). Each variant still requires the full command shape —
# an explicit computer object or end-anchored position — so ordinary
# uses of "sheep"/"shed"/"sherdown" can never become power commands.
_SHUTDOWN_SPOKEN = r"shut\s?down|sher\s?down|shed\s?down"
_SLEEP_SPOKEN = r"sleep|sheep"


def _extract_power_action(text: str, match=None) -> dict:
    """Return a coarse ``action`` label for the refusal message."""
    low = text.lower()
    if re.search(r"\bsleep\b|\bto\s+sleep\b", low):
        return {"action": "sleep"}
    if re.search(r"\bshut\s?down\b", low):
        return {"action": "shut_down"}
    if re.search(r"\bpower\s+(?:off|down)\b", low):
        return {"action": "power_off"}
    if re.search(r"\brestart\b", low):
        return {"action": "restart"}
    if re.search(r"\bhibernate\b", low):
        return {"action": "hibernate"}
    if re.search(r"\bsign\s*(?:out|off)\b", low):
        return {"action": "sign_out"}
    return {"action": "power"}


_NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80,
    "ninety": 90, "hundred": 100,
}
_NUMBER_WORD_RE = "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True))


def _spoken_number(text: str) -> Optional[int]:
    """Parse a spoken number ('forty five', '100') to 0-100 or None."""
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        try:
            value = int(m.group(1))
        except ValueError:
            return None
        return value if 0 <= value <= 100 else None
    words = re.findall(r"[a-z]+", text.lower())
    total = 0
    seen = False
    for w in words:
        if w not in _NUMBER_WORDS:
            if seen:
                break
            continue
        seen = True
        total += _NUMBER_WORDS[w]
    return total if seen and 0 <= total <= 100 else None


def _extract_set_volume(text: str, match=None) -> dict:
    """Extract the target level for 'set volume to N'."""
    m = re.search(
        r"\b(?:to|at)\s+(\d{1,3}|[a-z]+(?:\s+[a-z]+){0,2})\b",
        text,
        flags=re.I,
    )
    amount = None
    if m:
        candidate = m.group(1)
        dm = re.search(r"\d{1,3}", candidate)
        if dm:
            try:
                amount = int(dm.group(0))
            except ValueError:
                amount = None
        else:
            amount = _spoken_number(candidate)
    if amount is None:
        amount = _extract_volume_amount(text)
    return {"amount": amount}


def _extract_notification_message(text: str, match=None) -> dict:
    """Extract the user-supplied notification body."""
    t = (text or "").strip().strip(".!?,")
    t = re.sub(
        r"^(?:please\s+)?(?:can\s+you\s+|could\s+you\s+)?"
        r"(?:(?:send|show|display|post|pop\s+up|give\s+me|create|"
        r"make|set)\s+(?:me\s+)?(?:a\s+|an\s+|the\s+)?notifications?\b|"
        r"notify\s+me\b|remind\s+me\b)",
        "",
        t,
        flags=re.I,
    )
    t = re.sub(
        r"^\s*(?:that|to|about|saying|which\s+says|with\s+the\s+text)"
        r"[\s:,-]*",
        "",
        t,
        flags=re.I,
    )
    return {"message": _strip_filler(t)}


def _extract_volume_amount(text: str) -> Optional[int]:
    """Extract an optional numeric volume amount (0-100)."""
    m = re.search(r"\b(\d{1,3})\b", text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _extract_adjust_amount(text: str) -> Optional[int]:
    """Extract a relative volume step ('by 10', 'ten percent', '5%').

    Falls back to spoken number words anywhere in the utterance, but
    only when the number is the *last* numeric content — phrases like
    "increase volume in the morning" yield None.
    """
    # Explicit percent phrasing first: "10 percent", "10%".
    m = re.search(
        r"\b(\d{1,3})\s*(?:%|percent|per\s*cent)\b", text, flags=re.I
    )
    if m:
        try:
            value = int(m.group(1))
            return value if 0 <= value <= 100 else None
        except ValueError:
            pass
    # "by <number>" — digits or spoken words.
    m = re.search(
        r"\bby\s+(\d{1,3}|[a-z]+(?:\s+[a-z]+){0,2})\b", text, flags=re.I
    )
    if m:
        candidate = m.group(1)
        dm = re.search(r"\d{1,3}", candidate)
        if dm:
            try:
                value = int(dm.group(0))
                return value if 0 < value <= 100 else None
            except ValueError:
                pass
        else:
            value = _spoken_number(candidate)
            if value is not None and value > 0:
                return value
    # Bare spoken number at the end: "increase volume ten".
    tail = re.search(
        rf"((?:{_NUMBER_WORD_RE})(?:\s+(?:{_NUMBER_WORD_RE})){{0,2}})"
        r"\s*[.!?]*\s*$",
        text,
        flags=re.I,
    )
    if tail:
        value = _spoken_number(tail.group(1))
        if value is not None and value > 0:
            return value
    # Bare digits anywhere: "increase volume 10".
    return _extract_volume_amount(text)


# ----------------------------------------------------------------------
# Recognizer
# ----------------------------------------------------------------------
class RuleBasedIntentRecognizer:
    """Offline rule-based intent recognizer.

    Rules are evaluated in priority order; the first match wins.
    """

    name = "rule-based"

    # Each rule: (regex, intent_name, entity_extractor or None)
    _RULES: list[tuple] = []

    def __init__(self) -> None:
        self._rules = self._build_rules()

    # -- rule construction -------------------------------------------------
    @staticmethod
    def _build_rules() -> list[tuple]:
        R = RuleBasedIntentRecognizer
        return [
            # 0a. Cancel a scheduled (countdown) power action. Must
            #     precede the power rules so "cancel shutdown" is never
            #     read as a fresh shutdown request. Distinct from the
            #     yes/no confirmation flow: this only fires when a
            #     shutdown/restart countdown is actually armed.
            (
                re.compile(
                    r"\b(?:cancel|abort|stop)\b[^.!?]*\b(?:"
                    + _SHUTDOWN_SPOKEN +
                    r"|restart|reboot|power\s+(?:off|down)|power\s+action|"
                    r"scheduled\s+power)\b|"
                    r"\bcancel\s+(?:the\s+|any\s+)?(?:pending\s+|"
                    r"scheduled\s+)?(?:power\s+)?action\b",
                    re.I,
                ),
                "CANCEL_PENDING_POWER_ACTION",
                None,
            ),
            # 0b. Windows power actions (Milestone 1). Sleep, restart
            #     and shutdown are real intents behind the confirmation
            #     lifecycle; hibernate/sign-out stay unsupported. These
            #     rules must precede END_SESSION and CLOSE_APPLICATION
            #     so "shut down my computer" can never end the session
            #     or be misread as closing an app named "down".
            #
            #     Bare power words only match at end-of-utterance —
            #     "restart the download" is never a power command.
            #     Object forms require an explicit computer object so
            #     "turn off the volume" stays mute.
            (
                re.compile(
                    r"^\s*(?:please\s+|hey\s+|ok(?:ay)?\s+)?"
                    r"(?:jarvis\s+)?go\s+to\s+sleep"
                    r"(?:\s+(?:now|please))*\s*[.!?]*\s*$|"
                    rf"\b(?:{_SLEEP_SPOKEN})\s+(?:the\s+|this\s+|my\s+|"
                    rf"your\s+)?{_POWER_OBJECT}\b|"
                    rf"\b(?:put|send)\b[^.!?]*\b{_POWER_OBJECT}\b"
                    r"[^.!?]*\b(?:to|into)\s+sleep\b|"
                    rf"^\s*(?:please\s+|hey\s+)?(?:{_SLEEP_SPOKEN})"
                    r"\s*[.!?]*\s*$",
                    re.I,
                ),
                "SLEEP_COMPUTER",
                lambda t, m: {"action": "sleep"},
            ),
            (
                re.compile(
                    rf"\b(?:restart|reboot)\b[^.!?]*\b{_POWER_OBJECT}\b|"
                    rf"\b(?:restart|reboot)\s*[.!?]*\s*$",
                    re.I,
                ),
                "RESTART_COMPUTER",
                lambda t, m: {"action": "restart"},
            ),
            (
                re.compile(
                    rf"\b(?:{_SHUTDOWN_SPOKEN})\b[^.!?]*\b{_POWER_OBJECT}\b|"
                    rf"\b(?:{_SHUTDOWN_SPOKEN})\s*[.!?]*\s*$|"
                    rf"\bpower\s+(?:off|down)\b[^.!?]*\b{_POWER_OBJECT}\b|"
                    r"\bpower\s+(?:off|down)\s*[.!?]*\s*$|"
                    rf"\bturn\s+(?:off|out)\b[^.!?]*\b{_POWER_OBJECT}\b|"
                    rf"\bswitch\s+off\b[^.!?]*\b{_POWER_OBJECT}\b|"
                    rf"\bturn\b[^.!?]*\b{_POWER_OBJECT}\b[^.!?]*\b(?:off|down)\b",
                    re.I,
                ),
                "SHUTDOWN_COMPUTER",
                lambda t, m: {"action": "shutdown"},
            ),
            # 0c. Remaining Windows power phrases stay unsupported —
            #     recognized honestly, never executed.
            (
                re.compile(
                    rf"\b(?:hibernate|sign\s*(?:out|off))\b[^.!?]*"
                    rf"\b{_POWER_OBJECT}\b|"
                    rf"\b(?:hibernate|sign\s*(?:out|off))\s*[.!?]*\s*$",
                    re.I,
                ),
                "UNSUPPORTED_WINDOWS_ACTION",
                _extract_power_action,
            ),
            # 1. Session control — unambiguous phrases first.
            (
                re.compile(
                    r"\b(goodbye|bye|good\s*bye)(\s+jarvis)?\b|"
                    r"\bend\s+(the\s+|this\s+)?session\b|"
                    r"\bthat'?s\s+all\b|\bthat\s+is\s+all\b|"
                    r"\bgo\s+(back\s+)?to\s+standby\b|"
                    r"\bstop\s+listening\b|"
                    r"\bhang\s+up\b",
                    re.I,
                ),
                "END_SESSION",
                None,
            ),
            (
                # Bare "exit"/"quit" only when nothing follows.
                re.compile(r"^\s*(exit|quit|goodbye|bye)\s*[.!?]*\s*$", re.I),
                "END_SESSION",
                None,
            ),
            # 2. Informational intents.
            (
                re.compile(
                    r"\bwhat\s+time\s+(is\s+it|is\s+that)\b|"
                    r"\bwhat'?s\s+the\s+time\b|"
                    r"\bcurrent\s+time\b|"
                    r"\btell\s+me\s+the\s+time\b|"
                    r"\btell\s+me\s+the\s+current\s+time\b|"
                    r"\btell\s+me\s+what\s+time\s+it\s+is\b|"
                    r"\bwhat\s+is\s+the\s+current\s+time\b|"
                    r"\btime\s+is\s+it\b",
                    re.I,
                ),
                "GET_CURRENT_TIME",
                None,
            ),
            (
                re.compile(
                    r"\bwhat\s+(is\s+)?(the\s+)?(date|day)\b|"
                    r"\bwhat'?s\s+(the\s+)?(date|day)\b|"
                    r"\bwhat\s+day\s+is\s+(it|today)\b|"
                    r"\bcurrent\s+date\b|"
                    r"\btoday'?s\s+date\b|"
                    r"\btell\s+me\s+(the\s+)?(date|day)\b|"
                    r"\btell\s+me\s+today'?s\s+date\b|"
                    r"\bdate\s+is\s+it\b|"
                    r"\bwhat'?s\s+today\s+date\b|"
                    r"\bwhat'?s\s+the\s+date\s+today\b|"
                    r"\bwhat\s+is\s+the\s+date\s+today\b|"
                    r"\bwhat\s+day\s+is\s+today\b|"
                    r"\btell\s+me\s+what\s+(day|date)\s+it\s+is\b|"
                    r"\bwhat\s+is\s+today'?s\s+date\b",
                    re.I,
                ),
                "GET_CURRENT_DATE",
                None,
            ),
            # 2b. Windows system awareness (P2) — read-only status,
            #     screen capture and notifications.
            (
                re.compile(
                    r"\bbattery\b|\bpower\s+status\b|\bcharging\b|"
                    r"\bon\s+battery\b|\brunning\s+on\s+battery\b|"
                    r"\bhow\s+much\s+battery\b|"
                    r"\bbattery\s+(?:percentage|level|life|status)\b|"
                    r"\b(?:am|is)\s+(?:i|my\s+\w+|the\s+\w+|\w+)\s+"
                    r"charging\b",
                    re.I,
                ),
                "GET_BATTERY_STATUS",
                None,
            ),
            (
                re.compile(
                    r"\bconnected\s+to\s+the\s+internet\b|"
                    r"\binternet\s+(?:status|connection|connectivity)\b|"
                    r"\b(?:am\s+i|are\s+we)\s+"
                    r"(?:connected|online|on\s+the\s+internet)\b|"
                    r"\bdo\s+i\s+have\s+(?:an?\s+)?internet\b",
                    re.I,
                ),
                "GET_INTERNET_STATUS",
                None,
            ),
            (
                re.compile(
                    r"\bwi-?fi\b|\bwifi\b|\bwireless\b|\bwlan\b",
                    re.I,
                ),
                "GET_WIFI_STATUS",
                None,
            ),
            (
                re.compile(
                    r"\bethernet\b|\bwired\s+(?:connection|network|"
                    r"adapter|internet)\b|\bnetwork\s+cable\b",
                    re.I,
                ),
                "GET_ETHERNET_STATUS",
                None,
            ),
            (
                re.compile(
                    r"\bscreenshot\b|\bscreen\s*shot\b|"
                    r"\b(?:take|capture|snap|grab)\b[^.!?]*"
                    r"\b(?:screen|display|screenshot|screen\s*shot)\b",
                    re.I,
                ),
                "TAKE_SCREENSHOT",
                None,
            ),
            (
                re.compile(
                    r"\bnotify\s+me\b|\bremind\s+me\b|"
                    r"\b(?:send|show|display|post|pop\s+up|give\s+me|"
                    r"create|make|set)\b[^.!?]*\bnotifications?\b",
                    re.I,
                ),
                "SEND_NOTIFICATION",
                _extract_notification_message,
            ),
            # 3. Volume intents (set/get before mute/unmute/up/down).
            #    Two shapes: "set (the) volume ... to/at ..." and the
            #    STT-truncated "(that/the/my) volume to N" where the
            #    leading verb was lost but the level is explicit.
            (
                re.compile(
                    r"\bset\s+(?:the\s+|my\s+)?(?:volume|sound|audio)\b"
                    r"[^.!?]*\b(?:to|at)\b|"
                    # Adjacent "volume to N" only — "volume down to
                    # zero"/"up to fifty" stay directional intents.
                    r"\b(?:the\s+|my\s+|that\s+)?"
                    r"(?:volume|sound|audio)\s+"
                    rf"(?:to|at)\s+(?:\d{{1,3}}|{_NUMBER_WORD_RE})\b",
                    re.I,
                ),
                "SET_VOLUME",
                _extract_set_volume,
            ),
            (
                re.compile(
                    r"\bwhat'?s\s+the\s+volume\b|"
                    r"\bwhat\s+is\s+the\s+volume\b|"
                    r"\bcurrent\s+volume\b|"
                    r"\bget\s+(?:the\s+|my\s+)?volume\b|"
                    r"\bcheck\s+(?:the\s+|my\s+)?volume\b|"
                    r"\bvolume\s+(?:level|percentage|percent)\b",
                    re.I,
                ),
                "GET_VOLUME",
                None,
            ),
            (
                re.compile(
                    r"\bmute\b|\bsilence\b|"
                    r"\bturn\s+(the\s+|it\s+|sound\s+|volume\s+)?(off|down\s+to\s+zero)\b|"
                    r"\bvolume\s+off\b|"
                    r"\bmute\s+(the\s+|my\s+)?(volume|sound|audio)\b",
                    re.I,
                ),
                "MUTE_VOLUME",
                lambda t, m: {"amount": _extract_volume_amount(t)},
            ),
            (
                re.compile(
                    r"\bunmute\b|"
                    r"\bturn\s+(the\s+|it\s+|sound\s+|volume\s+)?(on|up\s+from\s+zero)\b|"
                    r"\bvolume\s+on\b",
                    re.I,
                ),
                "UNMUTE_VOLUME",
                lambda t, m: {"amount": _extract_volume_amount(t)},
            ),
            (
                re.compile(
                    r"\b(increase|raise|crank|bump)\b|"
                    r"\bturn\s+up\b|"
                    r"\b(turn|switch)\b[^.!?]*\b(volume|sound|it|audio)\b"
                    r"[^.!?]*\b(up|higher|louder)\b|"
                    r"\b(volume|sound)\s+up\b|"
                    r"\bmake\b[^.!?]*\b(louder|higher)\b|"
                    r"^\s*(louder|higher)\b.*$",
                    re.I,
                ),
                "INCREASE_VOLUME",
                lambda t, m: {"amount": _extract_adjust_amount(t)},
            ),
            (
                re.compile(
                    r"\b(decrease|lower|reduce|drop)\b|"
                    r"\bturn\s+down\b|"
                    r"\b(turn|switch)\b[^.!?]*\b(volume|sound|it|audio)\b"
                    r"[^.!?]*\b(down|lower|quieter)\b|"
                    r"\b(volume|sound)\s+down\b|"
                    r"\bmake\b[^.!?]*\b(quieter|lower)\b|"
                    r"^\s*(quieter|lower)\b.*$",
                    re.I,
                ),
                "DECREASE_VOLUME",
                lambda t, m: {"amount": _extract_adjust_amount(t)},
            ),
            # 3b. "Find <CapitalizedName>" is an app lookup, not a file
            #     search — unless the utterance carries file/folder
            #     vocabulary, in which case it stays a local search
            #     ("Find Smart Parking project" is not an app lookup).
            (
                re.compile(
                    r"(?i:\bfind)\s+(?![^.!?]*\b(?:files?|folders?|"
                    r"documents?|pdf|images?|pictures?|photos?|videos?|"
                    r"projects?|assignments?|resumes?|spreadsheets?|"
                    r"directories|directory)\b)(?=[A-Z])(.+)",
                ),
                "FIND_APPLICATION",
                lambda t, m: {"app_name": _extract_app_name(m.group(1))},
            ),
            # 3c. Web search — MUST precede SEARCH_FILES so "Search
            #     YouTube" / "look up X online" never becomes a local
            #     file search. Requires an explicit web destination or
            #     engine name; file/folder vocabulary still wins below
            #     only when no web scope is present.
            (
                re.compile(
                    r"\b(?:search|look\s+up|google|find|browse)\b"
                    r"[^.!?]*\b(?:web|internet|online|youtube|"
                    r"wikipedia)\b"
                    r"|\bgoogle\s+(?!chrome\b|chrome\s)(?!\s*$)"
                    r"[^.!?]*"
                    r"|\byoutube\s+(?:search\s+)?for\s+[^.!?]+",
                    re.I,
                ),
                "SEARCH_WEB",
                _extract_web_search,
            ),
            # 4. File/folder search. Folder- and item-scoped searches
            #    must precede the generic SEARCH_FILES rule so "find
            #    folders" never degrades into a file-name query.
            (
                re.compile(
                    r"\b(?:find|search|look|show|list|display)\b"
                    r"(?!\s+(?:the\s+)?(?:application|app|program)\b)"
                    r"[^.!?]*\b(?:files?\s+and\s+folders?|"
                    r"folders?\s+and\s+files?|everything|all\s+"
                    r"(?:my\s+|the\s+)?(?:files?|items?|stuff)|"
                    r"local\s+(?:files?|items?))\b",
                    re.I,
                ),
                "SEARCH_LOCAL_ITEMS",
                lambda t, m: {"query": _extract_search_query(t)},
            ),
            (
                re.compile(
                    r"\b(?:find|search|look)\b"
                    r"(?!\s+(?:the\s+)?(?:application|app|program)\b)"
                    r"[^.!?]*\b(?:folders?|directories|directory)\b",
                    re.I,
                ),
                "SEARCH_FOLDERS",
                lambda t, m: {"query": _extract_search_query(t)},
            ),
            (
                re.compile(
                    r"\b(search|find|look)\b(?!\s+(?:the\s+)?(?:application|app|program)\b).*\b(for\s+)?",
                    re.I,
                ),
                "SEARCH_FILES",
                lambda t, m: {"query": _extract_search_query(t)},
            ),
            (
                re.compile(
                    r"\brefresh\b[^.!?]*\b(applications?|apps?|programs?|"
                    r"installed)\b|"
                    r"\b(rescan|re-?scan|update)\b[^.!?]*"
                    r"\b(applications?|apps?|programs?|installed)\b|"
                    r"\brefresh\s+installed\s+applications\b",
                    re.I,
                ),
                "REFRESH_APPLICATIONS",
                None,
            ),
            (
                re.compile(
                    r"\b(recent|newly\s+installed|latest)\b[^.!?]*"
                    r"\b(applications?|apps?|programs?|installs?)\b|"
                    r"\b(applications?|apps?)\s+(installed|added)\s+"
                    r"(recently|lately|since)\b",
                    re.I,
                ),
                "RECENT_APPLICATIONS",
                None,
            ),
            (
                re.compile(
                    r"\b(list|show)\b[^.!?]*\binstalled\b[^.!?]*"
                    r"\b(applications?|apps?|programs?)\b|"
                    r"\binstalled\s+(applications?|apps?|programs?)\b|"
                    r"\ball\s+(installed\s+)?(applications?|apps?)\b",
                    re.I,
                ),
                "LIST_INSTALLED_APPLICATIONS",
                None,
            ),
            # 4b. Windows app/window intents (Task 6). Verb+object rules
            #     capture the object into group 2 -> entity "target".
            (
                re.compile(
                    r"\bwhich\s+applications?\s+(are|is)\s+"
                    r"(currently\s+)?(running|open)\b|"
                    r"\bwhat\s+(applications?|apps?|programs?)\s+"
                    r"(are|is)\s+(currently\s+)?(running|open)\b|"
                    r"\blist\s+(running|open)\s+(applications?|apps?)\b|"
                    r"\bwhat'?s\s+running\b|\bwhat\s+is\s+running\b|"
                    r"\bshow\s+(me\s+)?(running|open)\s+(applications?|apps?)\b",
                    re.I,
                ),
                "LIST_RUNNING_APPLICATIONS",
                None,
            ),
            (
                re.compile(
                    r"\bshow\s+(me\s+)?(the\s+)?desktop\b|"
                    r"\b(display|view)\s+(the\s+)?desktop\b|"
                    r"\bminimi[sz]e\s+all\s+(windows?|applications?)\b|"
                    r"\bget\s+(?:all\s+)?(?:the\s+)?windows\s+"
                    r"out\s+of\s+the\s+way\b|"
                    r"\bclear\s+(?:the\s+)?(?:screen|desktop)\s+"
                    r"of\s+windows\b",
                    re.I,
                ),
                "SHOW_DESKTOP",
                None,
            ),
            # 4b-ii. Active-window controls (Milestone 1). These must
            #     precede CLOSE_APPLICATION / SWITCH_TO_APPLICATION /
            #     the generic minimize/restore rules so pronoun and
            #     "active window" targets never resolve to an app name.
            (
                re.compile(
                    r"\bclose\s+(?:the\s+|this\s+|that\s+|my\s+|"
                    r"current\s+|active\s+)*window\b"
                    r"(?:\s+(?:please|now|for\s+me))*\s*[.!?]*\s*$",
                    re.I,
                ),
                "CLOSE_ACTIVE_WINDOW",
                None,
            ),
            (
                re.compile(
                    r"\bminimi[sz]e\s+(?:the\s+|this\s+|that\s+|my\s+|"
                    r"current\s+)?active\s+window\b|"
                    r"^\s*(?:please\s+)?minimi[sz]e\s+window\s*"
                    r"(?:please|now)*\s*[.!?]*\s*$",
                    re.I,
                ),
                "MINIMIZE_ACTIVE_WINDOW",
                None,
            ),
            (
                re.compile(
                    r"\b(?:restore|un-?maximi[sz]e|bring\s+back)\s+"
                    r"(?:the\s+|this\s+|that\s+|my\s+|current\s+)?"
                    r"active\s+window\b",
                    re.I,
                ),
                "RESTORE_ACTIVE_WINDOW",
                None,
            ),
            (
                re.compile(
                    r"\b(?:switch|change|cycle|go|move|jump)\s+to\s+"
                    r"(?:the\s+|my\s+)?next\s+"
                    r"(?:window|application|app)\b|"
                    r"^\s*(?:please\s+)?next\s+"
                    r"(?:window|application|app)\b"
                    r"(?:\s+(?:please|now))*\s*[.!?]*\s*$",
                    re.I,
                ),
                "SWITCH_NEXT_WINDOW",
                None,
            ),
            (
                re.compile(
                    r"\b(?:switch|change|cycle|go|move|jump)\s+to\s+"
                    r"(?:the\s+|my\s+)?(?:previous|prev)\s+"
                    r"(?:window|application|app)\b|"
                    r"^\s*(?:please\s+)?(?:previous|prev)\s+"
                    r"(?:window|application|app)\b"
                    r"(?:\s+(?:please|now))*\s*[.!?]*\s*$",
                    re.I,
                ),
                "SWITCH_PREVIOUS_WINDOW",
                None,
            ),
            (
                re.compile(
                    r"\b(?:switch|change|move|go|cycle)\s+"
                    r"(?:\w+\s+)?to\s+(?:the\s+|my\s+)?next\s+"
                    r"(?:virtual\s+)?desktop\b|"
                    r"^\s*(?:please\s+)?next\s+(?:virtual\s+)?desktop\b"
                    r"(?:\s+(?:please|now))*\s*[.!?]*\s*$",
                    re.I,
                ),
                "SWITCH_NEXT_VIRTUAL_DESKTOP",
                None,
            ),
            (
                re.compile(
                    r"\b(?:switch|change|move|go|cycle)\s+"
                    r"(?:\w+\s+)?to\s+(?:the\s+|my\s+)?"
                    r"(?:previous|prev)\s+(?:virtual\s+)?desktop\b|"
                    r"^\s*(?:please\s+)?(?:previous|prev)\s+"
                    r"(?:virtual\s+)?desktop\b"
                    r"(?:\s+(?:please|now))*\s*[.!?]*\s*$",
                    re.I,
                ),
                "SWITCH_PREVIOUS_VIRTUAL_DESKTOP",
                None,
            ),
            (
                re.compile(
                    r"\bsnap\s+(?:the\s+|this\s+|that\s+|my\s+|"
                    r"current\s+)?(?:window|it)?\s*"
                    r"(?:to\s+the\s+|to\s+)?left\b|"
                    r"^\s*(?:please\s+)?snap\s+left\b"
                    r"(?:\s+(?:please|now))*\s*[.!?]*\s*$",
                    re.I,
                ),
                "SNAP_ACTIVE_WINDOW_LEFT",
                None,
            ),
            (
                re.compile(
                    r"\bsnap\s+(?:the\s+|this\s+|that\s+|my\s+|"
                    r"current\s+)?(?:window|it)?\s*"
                    r"(?:to\s+the\s+|to\s+)?right\b|"
                    r"^\s*(?:please\s+)?snap\s+right\b"
                    r"(?:\s+(?:please|now))*\s*[.!?]*\s*$",
                    re.I,
                ),
                "SNAP_ACTIVE_WINDOW_RIGHT",
                None,
            ),
            # 4c. Monitor intents (Task 6+). Ordered before the generic
            #     window/app rules so "on monitor 2" is never swallowed
            #     as part of an app name.
            (
                re.compile(
                    r"\bmove\s+(.+?)\s+(?:to|into|onto)\s+"
                    r"(?:the\s+|my\s+|your\s+)?"
                    r"(primary|main)\s+(monitor|screen|display)s?\b",
                    re.I,
                ),
                "MOVE_WINDOW_TO_PRIMARY_MONITOR",
                lambda t, m: {
                    "target": _clean_move_target(m.group(1)),
                    "monitor": "primary",
                },
            ),
            (
                re.compile(
                    r"\bmove\s+(.+?)\s+(?:to|into|onto)\s+(?:the\s+|my\s+|your\s+)?"
                    r"other\s+(monitor|screen|display)s?\b",
                    re.I,
                ),
                "MOVE_WINDOW_TO_OTHER_MONITOR",
                lambda t, m: {
                    "target": _clean_move_target(m.group(1)),
                    "monitor": "other",
                },
            ),
            (
                re.compile(
                    r"\bmove\s+(.+?)\s+(?:to|into|onto)\s+"
                    r"(?=[^.!?]*\b(monitors?|screens?|displays?|"
                    r"left|right|write|top|bottom|upper|lower|primary|"
                    r"secondary|other|another|laptop|external|"
                    r"first|second|third|\d+)\b)"
                    r"(.+?)\s*[.!?]*\s*$",
                    re.I,
                ),
                "MOVE_WINDOW_TO_MONITOR",
                lambda t, m: {
                    "target": _clean_move_target(m.group(1)),
                    "monitor": _monitor_spec(m.group(3)),
                },
            ),
            (
                re.compile(
                    r"\b(open|launch|start|run|fire\s+up|bring\s+up|"
                    r"execute|load)\s+(.+?)\s+on\s+"
                    r"(?=[^.!?]*\b(monitors?|screens?|displays?|"
                    r"left|right|write|top|bottom|upper|lower|primary|"
                    r"secondary|other|another|laptop|external|"
                    r"first|second|third|\d+)\b)"
                    r"(.+?)\s*[.!?]*\s*$",
                    re.I,
                ),
                "OPEN_APPLICATION_ON_MONITOR",
                lambda t, m: {
                    "app_name": _extract_app_name(m.group(2)),
                    "monitor": _monitor_spec(m.group(4)),
                },
            ),
            (
                re.compile(
                    r"\bwhich\s+(monitor|screen|display)\s+is\s+"
                    r"(.+?)\s+(on|running\s+on|currently\s+on)\b|"
                    r"\bwhat\s+(monitor|screen|display)\s+is\s+"
                    r"(.+?)\s+(on|using)\b|"
                    r"\bwhich\s+(monitor|screen|display)\s+"
                    r"(.+?)\s+(using|on)\b",
                    re.I,
                ),
                "GET_WINDOW_MONITOR",
                lambda t, m: {
                    "target": _extract_app_name(
                        next(
                            (g for g in reversed(m.groups())
                             if g and g.strip()
                             and g.strip().lower()
                             not in {"on", "using", "running on",
                                     "currently on"}),
                            "",
                        )
                    )
                },
            ),
            (
                re.compile(
                    r"\b(list|show)\s+(me\s+)?(my\s+)?"
                    r"(monitors?|screens?|displays?)\b|"
                    r"\bhow\s+many\s+(monitors?|screens?|displays?)\b|"
                    r"\bwhat\s+monitors?\s+(do|are)\b|"
                    r"\b(display|show)\s+(me\s+)?(my\s+)?"
                    r"(monitors?|screens?|displays?)\b",
                    re.I,
                ),
                "LIST_MONITORS",
                None,
            ),
            (
                re.compile(
                    r"\bwhich\s+(monitor|screen|display)\s+is\s+"
                    r"(the\s+)?primary\b|"
                    r"\bis\s+(the\s+)?primary\s+(monitor|screen|display)\b|"
                    r"^\s*primary\s+(monitor|screen|display)\s*[?.!]*\s*$",
                    re.I,
                ),
                "GET_PRIMARY_MONITOR",
                None,
            ),
            (
                re.compile(
                    r"\b(find|locate)\s+(the\s+)?(application|app|program)\s+"
                    r"(.+)|"
                    r"\bfind\s+(?=[A-Z])(.+)|"
                    r"\bwhere\s+is\s+(.+)\s+installed\b|"
                    r"\bis\s+(.+)\s+installed\b|"
                    r"\bdo\s+you\s+have\s+(.+)\b",
                    re.I,
                ),
                "FIND_APPLICATION",
                lambda t, m: {
                    "app_name": _extract_app_name(
                        next(
                            (
                                g for g in reversed(m.groups())
                                if g and g.strip()
                            ),
                            "",
                        )
                    )
                },
            ),
            (
                re.compile(
                    r"\b(switch\s+to|switch\s+over\s+to|switched\s+to|"
                    r"twitch\s+to|bring|"
                    r"jump\s+to|go\s+back\s+to|return\s+to)\s+(.+)",
                    re.I,
                ),
                "SWITCH_TO_APPLICATION",
                lambda t, m: {"app_name": _extract_app_name(m.group(2))},
            ),
            (
                re.compile(
                    r"\bminimi[sz]e\b\s*(?:the\s+|my\s+|this\s+|that\s+)?"
                    r"(window|it|(.+?)\s+window|(.+))?\s*[.!?]*\s*$",
                    re.I,
                ),
                "MINIMIZE_WINDOW",
                lambda t, m: _window_target(m),
            ),
            (
                re.compile(
                    r"\bmaximi[sz]e\b\s*(?:the\s+|my\s+|this\s+|that\s+)?"
                    r"(window|it|(.+?)\s+window|(.+))?\s*[.!?]*\s*$",
                    re.I,
                ),
                "MAXIMIZE_WINDOW",
                lambda t, m: _window_target(m),
            ),
            (
                re.compile(
                    r"\b(?:restore|un-?maximi[sz]e|bring\s+back)\b\s*"
                    r"(?:the\s+|my\s+|this\s+|that\s+)?"
                    r"(window|it|(.+?)\s+window|(.+))?\s*[.!?]*\s*$",
                    re.I,
                ),
                "RESTORE_WINDOW",
                lambda t, m: _window_target(m),
            ),
            (
                re.compile(
                    r"\bfocus\s+(?:the\s+|my\s+|this\s+|that\s+)?(.+)",
                    re.I,
                ),
                "FOCUS_WINDOW",
                lambda t, m: _window_target(m),
            ),
            # 4d. Workstation lock — explicit object required so "lock"
            #     alone or "lock the window" never triggers it.
            (
                re.compile(
                    r"\block\s+(?:the\s+|my\s+|this\s+|your\s+)?"
                    r"(?:computer|pc|windows|workstation|machine|"
                    r"laptop|desktop)\b",
                    re.I,
                ),
                "LOCK_WORKSTATION",
                None,
            ),
            # 5a. Contextual follow-ups: "open it", "open that folder",
            #     "open the second one", "open its parent folder".
            #     Resolved against the previous search context — never
            #     against arbitrary text.
            (
                re.compile(
                    r"\b(?:open|show|launch|go\s+to|take\s+me\s+to)\s+"
                    r"(?:its\s+|his\s+|her\s+|their\s+)?parent\s+"
                    r"(?:folder|directory|file)\b"
                    r"|\b(?:open|show)\s+(?:the\s+|my\s+)?parent\s+"
                    r"(?:folder|directory)\s+of\s+(?:it|that|this)\b"
                    r"|\b(?:open|show|launch|go\s+to|take\s+me\s+to)\s+"
                    r"(?:it|that|this|them|the\s+last\s+one|"
                    r"the\s+previous\s+one)\b"
                    r"(?:\s+(?:folder|file|document|one|item|directory))?"
                    r"\s*[.!?]*\s*$"
                    r"|\b(?:open|show)\s+(?:the\s+|my\s+)?"
                    r"(?:first|second|third|fourth|fifth|1st|2nd|3rd|"
                    r"latest|newest|most\s+recent|last)\s+"
                    r"(?:one|file|files|folder|folders|document|"
                    r"documents|item|items)\b"
                    r"(?:\s+(?:folder|file|document))?\s*[.!?]*\s*$"
                    r"|\b(?:open|show)\s+(?:the\s+|my\s+)?"
                    r"(?:file|folder|document|one|item)\s*[.!?]*\s*$",
                    re.I,
                ),
                "OPEN_LOCAL_ITEM",
                lambda t, m: _extract_context_selector(t, m),
            ),
            # 5b-i. Explicit folder requests: an open-ish verb plus the
            #     word "folder"/"directory" anywhere in the utterance.
            #     Any name is accepted; resolution happens in the
            #     handler (known-folder aliases first, then discovery).
            (
                re.compile(
                    r"\b(open|show|go\s+to|launch|take\s+me\s+to)\b[^.!?]*"
                    r"\b(?:folders?|directories|directory)\b",
                    re.I,
                ),
                "OPEN_FOLDER",
                lambda t, m: {"folder": _extract_folder(t)},
            ),
            # 5b-ii. Bare known-folder object with NO folder keyword:
            #     "open desktop", "open the downloads". The known name
            #     must BE the whole object — "open docker desktop" is
            #     an application, not a folder request.
            (
                re.compile(
                    r"\b(?:open|show|go\s+to|launch|take\s+me\s+to)\s+"
                    r"(?:the\s+|my\s+|your\s+)?"
                    r"(?:downloads|documents|desktop|pictures|photos|"
                    r"music|videos|movies)"
                    r"(?:\s+(?:folders?|directories|directory))?"
                    r"(?:\s+(?:please|for\s+me|now|thanks|thank\s+you))*"
                    r"\s*[.!?]*\s*$",
                    re.I,
                ),
                "OPEN_FOLDER",
                lambda t, m: {"folder": _extract_folder(t)},
            ),
            # 5c. File opening — open-ish verb plus file vocabulary or a
            #     filename extension, and NOT a folder request. Routed
            #     before OPEN_APPLICATION so "open resume.pdf" or
            #     "open Java assignment" never becomes an app lookup.
            (
                re.compile(
                    r"\b(?:open|launch|start|run|show|go\s+to|"
                    r"take\s+me\s+to|bring\s+up|fire\s+up|load|"
                    r"display|view)\s+"
                    r"(?![^.!?]*\b(?:folders?|directories|directory)\b)"
                    r"(?=[^.!?]*"
                    r"(?:\bfiles?\b|\bdocuments?\b|\bpdfs?\b|"
                    r"\bimages?\b|\bpictures?\b|\bphotos?\b|"
                    r"\bvideos?\b|\bresumes?\b|\bcv\b|\bassignments?\b|"
                    r"\breports?\b|\bspreadsheets?\b|\bpresentations?\b|"
                    r"\bcode\s+files?\b|\btext\s+files?\b|"
                    r"\.[a-z0-9]{1,8}\b))"
                    r"(.+?)\s*[.!?]*\s*$",
                    re.I,
                ),
                "OPEN_FILE",
                _extract_open_file,
            ),
            # 6. Application open/close (verb + object required). Group 2
            #    holds the object text after the verb.
            (
                re.compile(
                    r"\b(open|launch|start|run|bring\s+up|fire\s+up|execute|load)\b"
                    r"\s+(.+)",
                    re.I,
                ),
                "OPEN_APPLICATION",
                lambda t, m: {"app_name": _extract_app_name(m.group(2))},
            ),
            (
                re.compile(
                    r"\b(close|quit|exit|kill|terminate|end|shut)\b"
                    r"\s+(.+)",
                    re.I,
                ),
                "CLOSE_APPLICATION",
                lambda t, m: {"app_name": _extract_app_name(m.group(2))},
            ),
        ]

    # -- public API ---------------------------------------------------------
    def recognize(self, text: str) -> Intent:
        """Map ``text`` to an :class:`Intent`."""
        if not text or not text.strip():
            return Intent("UNKNOWN", raw_text=text or "", confidence=0.0)
        for pattern, name, extractor in self._rules:
            m = pattern.search(text)
            if not m:
                continue
            entities: dict = {}
            if extractor is not None:
                try:
                    entities = extractor(text, m) or {}
                except Exception:  # noqa: BLE001
                    logger.debug("Entity extraction failed", exc_info=True)
                    entities = {}
            # For verb+object intents the regex match object may carry a
            # better entity source; extractors above handle cleanup.
            logger.debug("Intent %s matched %r", name, text)
            return Intent(name=name, entities=entities, raw_text=text)

        # Conservative STT rescue: common mishearings of window verbs
        # ("Max ..." -> "maximize ...", "Mini ..." -> "minimize ...").
        # Only fires when the result is still a window-management intent
        # — never guesses an application or executes anything new.
        rescued = self._rescue_window_verb(text)
        if rescued is not None:
            return rescued
        rescued = self._rescue_move_verb(text)
        if rescued is not None:
            return rescued
        return Intent("UNKNOWN", raw_text=text, confidence=0.0)

    # Common STT truncations/abbreviations of window verbs. Kept tiny
    # and explicit so rescue stays predictable.
    _VERB_RESCUE = {
        "max": "maximize",
        "maxi": "maximize",
        "mini": "minimize",
        "min": "minimize",
        "rest": "restore",
        "unmax": "unmaximize",
    }
    _WINDOW_INTENTS = frozenset({
        "MINIMIZE_WINDOW", "MAXIMIZE_WINDOW", "RESTORE_WINDOW",
        "FOCUS_WINDOW",
    })

    def _rescue_window_verb(self, text: str) -> Optional[Intent]:
        words = text.strip().split()
        if not words:
            return None
        first = re.sub(r"[^a-z]", "", words[0].lower())
        replacement = self._VERB_RESCUE.get(first)
        if replacement is None:
            return None
        fixed = " ".join([replacement] + words[1:])
        for pattern, name, extractor in self._rules:
            m = pattern.search(fixed)
            if not m or name not in self._WINDOW_INTENTS:
                continue
            entities: dict = {}
            if extractor is not None:
                try:
                    entities = extractor(fixed, m) or {}
                except Exception:  # noqa: BLE001
                    entities = {}
            logger.info("Window-verb rescue: %r -> %r", text, fixed)
            return Intent(name=name, entities=entities, raw_text=text)
        return None

    # STT frequently hears "move" as "more"/"moe"/"mow". Rescue ONLY
    # when the remainder still has the exact "<target> to <monitor-ish
    # destination>" shape — arbitrary sentences containing "more" are
    # never converted into movement commands.
    _MOVE_MISHEARINGS = frozenset({"more", "moe", "mow", "muv", "muve"})
    _MONITOR_DEST_RE = re.compile(
        r"\bto\b[^.!?]*\b(monitors?|screens?|displays?|left|right|write|"
        r"top|bottom|upper|lower|primary|main|secondary|other|"
        r"another|first|second|third|one|two|three|\d+)\b",
        re.I,
    )

    def _rescue_move_verb(self, text: str) -> Optional[Intent]:
        words = text.strip().split()
        if len(words) < 3:
            return None
        first = re.sub(r"[^a-z]", "", words[0].lower())
        if first not in self._MOVE_MISHEARINGS:
            return None
        fixed = "move " + " ".join(words[1:])
        if not self._MONITOR_DEST_RE.search(fixed):
            return None
        for pattern, name, extractor in self._rules:
            m = pattern.search(fixed)
            if not m or not name.startswith("MOVE_WINDOW"):
                continue
            entities: dict = {}
            if extractor is not None:
                try:
                    entities = extractor(fixed, m) or {}
                except Exception:  # noqa: BLE001
                    entities = {}
            logger.info("Move-verb rescue: %r -> %r", text, fixed)
            return Intent(name=name, entities=entities, raw_text=text)
        return None


# Backwards-compatible alias used by older imports.
IntentRecognizer = RuleBasedIntentRecognizer
