"""Contextual recovery for misheard voice commands (V10.4).

When the primary rule-based recognizer returns ``UNKNOWN`` — typically
because the STT model garbled a command ("More crumpled less
commoditous." for "Move Chrome to the left monitor.") — this module
attempts a *conservative* reinterpretation:

- The utterance must still look like a command template (a known
  verb-ish onset plus an application target, optionally a monitor
  destination).
- The application fragment is matched against the currently running /
  known application names with a similarity score. A match is only
  accepted when it is clearly best (score floor + margin over the
  runner-up). Ambiguous candidates produce a clarification request,
  never a silent guess.
- The monitor fragment is matched against the monitor vocabulary
  (left/right/primary/secondary/first/second/other/...); unresolved
  fragments are passed through so the monitor handler asks "which
  monitor did you mean?" instead of guessing.

Nothing here executes anything: it only produces an :class:`Intent`
(or a clarification request) that flows through the normal registry,
handlers and safety rules. The original transcript is always preserved
in ``entities["_recovery"]`` alongside the interpreted fields.
"""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.commands.intent import Intent

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Similarity helpers
# ----------------------------------------------------------------------
def _norm(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return " ".join(t.split())


def _seq_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _tokens(text: str) -> list[str]:
    return _norm(text).split()


# STT confusions observed in monitor phrases; applied only to the
# monitor-destination fragment, never to app names.
_MONITOR_TOKEN_FIXES = {
    "write": "right",
    "rite": "right",
    "wright": "right",
    "wrigh": "right",
    "lef": "left",
    "riht": "right",
    # V10.5: STT renders "left" as "less" inside monitor phrases
    # ("to less the monitor"). Scoped to the monitor fragment only.
    "less": "left",
    # Glue words STT inserts inside monitor phrases ("less than
    # monitor" -> "left monitor").
    "than": "",
    "then": "",
}

# Canonical monitor phrases (normalized). Order matters only for
# tie-breaking readability; scoring picks the best ratio.
_MONITOR_VOCAB = (
    "left monitor", "right monitor", "left screen", "right screen",
    "left display", "right display",
    "primary monitor", "main monitor", "secondary monitor",
    "monitor one", "monitor two", "monitor three",
    "monitor 1", "monitor 2", "monitor 3",
    "first monitor", "second monitor", "third monitor",
    "other monitor", "another monitor", "the other monitor",
    "top monitor", "bottom monitor", "upper monitor", "lower monitor",
    "left", "right", "primary", "main", "secondary",
    "first", "second", "third", "other", "another",
    "top", "bottom", "upper", "lower",
)

# Spatial/role words that identify a monitor destination by themselves.
_MONITOR_ROLES = {
    "primary", "main", "secondary", "other", "another",
    "first", "second", "third", "top", "bottom", "upper", "lower",
}

# Verb onsets for each recoverable command family. Matching uses a
# sequence ratio so STT-garbled verbs ("more" ~ "move") still count.
_VERB_GROUPS = {
    "MOVE_WINDOW_TO_MONITOR": (
        ("move", "moved", "movie", "mover"),
        0.72,
    ),
    "OPEN_APPLICATION": (
        ("open", "opened", "opener", "pn"),  # "pn" covers dropped vowel
        0.70,
    ),
    "SWITCH_TO_APPLICATION": (
        ("switch", "switched", "swish", "twitch"),
        0.70,
    ),
    "CLOSE_APPLICATION": (
        ("close", "closed", "clothes"),  # "clothes" is a known STT hit
        0.70,
    ),
}

# Score floors / margins (0..1).
_APP_SCORE_MIN = 0.55
_APP_MARGIN_MIN = 0.08
_MONITOR_SCORE_MIN = 0.50
_MOVE_VERB_MIN = 0.72


@dataclass
class RecoveryResult:
    """Outcome of a recovery attempt."""

    status: str  # "matched" | "ambiguous" | "none"
    intent: Optional[Intent] = None
    confidence: float = 0.0
    reason: str = ""
    interpretation: str = ""
    clarify: Optional[dict] = None
    meta: dict = field(default_factory=dict)


def _app_label(app) -> str:
    return str(getattr(app, "name", app) or app)


def _score_app(fragment: str, app) -> float:
    """Similarity between a spoken fragment and an app name.

    Combines whole-string ratio with token coverage so "dev" scores
    well against "Devin" while "crumpled" does not accidentally match
    an unrelated long name.
    """
    frag = _norm(fragment)
    name = _norm(_app_label(app))
    if not frag or not name:
        return 0.0
    seq = _seq_ratio(frag, name)
    ft = set(frag.split())
    nt = set(name.split())
    # exe-stem tokens also count (chrome.exe -> chrome).
    exe = _norm(str(getattr(app, "exe_name", "") or ""))
    nt |= set(exe.replace(".exe", "").split())
    coverage = len(ft & nt) / len(ft) if ft else 0.0
    # Substring bonus: the fragment appears inside the app name
    # ("whats app" -> "whatsapp", "dev" -> "devin").
    compact_frag = frag.replace(" ", "")
    compact_name = name.replace(" ", "")
    substring = 1.0 if compact_frag and compact_frag in compact_name else 0.0
    return max(seq, 0.55 * seq + 0.45 * coverage, substring * 0.9)


def _score_monitor(fragment: str) -> tuple[float, str]:
    """Best (score, canonical_spec) for a monitor fragment."""
    frag = _norm(fragment)
    if not frag:
        return 0.0, ""
    tokens = [
        _MONITOR_TOKEN_FIXES.get(t, t) for t in frag.split()
    ]
    tokens = [t for t in tokens if t]
    fixed = " ".join(tokens)
    best_score, best_spec = 0.0, ""
    for vocab in _MONITOR_VOCAB:
        score = _seq_ratio(fixed, vocab)
        if score > best_score:
            best_score, best_spec = score, vocab
    # A lone spatial/role token is a strong signal even when the full
    # phrase ratio is modest.
    if len(tokens) == 1 and tokens[0] in _MONITOR_ROLES | {
        "left", "right", "top", "bottom"
    }:
        best_score = max(best_score, 1.0)
        best_spec = tokens[0] if tokens[0] in {
            "left", "right", "top", "bottom", "primary", "main",
            "secondary", "other", "another", "first", "second",
            "third", "upper", "lower",
        } else best_spec
    return best_score, best_spec


def _monitor_canonical_spec(score: float, spec: str, raw: str) -> str:
    """Pick the monitor entity to hand to the handler.

    High-confidence vocab matches collapse to the canonical phrase;
    weaker fragments pass through raw so the handler's resolver asks
    the user instead of guessing.
    """
    if score >= _MONITOR_SCORE_MIN and spec:
        return spec
    return raw


class MisheardCommandRecovery:
    """Recover likely command intents from garbled transcripts.

    Args:
        app_names: Callable returning candidate applications (running
            apps preferred). Items may be AppInfo-like objects or plain
            strings; ``.name`` is used when present.
        app_resolver: Optional ``query -> ResolveResult`` used to
            validate the recovered app name through the normal
            resolution path (not required for matching itself).
        monitor_resolver: Optional ``query -> MonitorResolveResult``
            used to validate monitor fragments.
    """

    def __init__(
        self,
        app_names: Optional[Callable[[], list]] = None,
        app_resolver: Optional[Callable[[str], object]] = None,
        monitor_resolver: Optional[Callable[..., object]] = None,
        app_score_min: float = _APP_SCORE_MIN,
        app_margin_min: float = _APP_MARGIN_MIN,
        monitor_score_min: float = _MONITOR_SCORE_MIN,
        move_verb_min: float = _MOVE_VERB_MIN,
    ) -> None:
        self._app_names = app_names or (lambda: [])
        self._app_resolver = app_resolver
        self._monitor_resolver = monitor_resolver
        self._app_score_min = app_score_min
        self._app_margin_min = app_margin_min
        self._monitor_score_min = monitor_score_min
        self._move_verb_min = move_verb_min
        self._apps_cache: tuple[float, list] = (0.0, [])

    # ------------------------------------------------------------------
    def _candidate_apps(self) -> list:
        """Candidate apps, cached briefly (recovery is rare)."""
        import time

        now = time.monotonic()
        if now - self._apps_cache[0] < 30.0 and self._apps_cache[1]:
            return self._apps_cache[1]
        try:
            apps = list(self._app_names() or [])
        except Exception:  # noqa: BLE001
            logger.debug("Recovery app listing failed", exc_info=True)
            apps = []
        self._apps_cache = (now, apps)
        return apps

    # ------------------------------------------------------------------
    def _match_app(self, fragment: str):
        """Return (status, app_or_options, score, reason).

        status: "found" | "ambiguous" | "missing".
        """
        frag = _norm(fragment)
        if not frag:
            return "missing", [], 0.0, "empty_target"
        apps = self._candidate_apps()
        if not apps:
            # No candidate list: fall back to the resolver for an
            # exact-ish hit only (never fuzzy-guess blind).
            if self._app_resolver is not None:
                try:
                    res = self._app_resolver(frag)
                except Exception:  # noqa: BLE001
                    res = None
                if getattr(res, "status", "") == "found" and getattr(
                    res, "app", None
                ) is not None:
                    return "found", res.app, 1.0, "resolver_exact"
            return "missing", [], 0.0, "no_candidates"
        scored = sorted(
            ((_score_app(frag, a), a) for a in apps),
            key=lambda t: t[0], reverse=True,
        )
        best_score, best = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best_score < self._app_score_min:
            return "missing", [], best_score, "below_threshold"
        if second and best_score - second < self._app_margin_min:
            tied = [a for s, a in scored if best_score - s < self._app_margin_min]
            return "ambiguous", tied[:4], best_score, "close_scores"
        return "found", best, best_score, "unique_best"

    # ------------------------------------------------------------------
    def _detect_verb(self, tokens: list[str]) -> tuple[str, int]:
        """Return (intent_name, verb_token_count) for the onset."""
        if not tokens:
            return "", 0
        first = tokens[0]
        # Two-token verb onsets: "switch to X", "move to X".
        if len(tokens) >= 2:
            pair = f"{first} {tokens[1]}"
            for name, (verbs, thresh) in _VERB_GROUPS.items():
                for v in verbs:
                    if _seq_ratio(pair, v) >= max(thresh, 0.8):
                        return name, 2
        for name, (verbs, thresh) in _VERB_GROUPS.items():
            for v in verbs:
                if _seq_ratio(first, v) >= thresh:
                    return name, 1
        return "", 0

    # ------------------------------------------------------------------
    def recover(self, text: str) -> RecoveryResult:
        """Attempt recovery; never raises, never executes."""
        try:
            return self._recover(text)
        except Exception:  # noqa: BLE001
            logger.exception("Command recovery failed")
            return RecoveryResult("none", reason="error")

    def _recover(self, text: str) -> RecoveryResult:
        raw = (text or "").strip()
        tokens = _tokens(raw)
        # Too short to be a garbled command; leave to UNKNOWN path.
        if len(tokens) < 3:
            return RecoveryResult("none", reason="too_short")

        intent_name, verb_len = self._detect_verb(tokens)
        if not intent_name:
            return RecoveryResult("none", reason="no_verb")

        rest = tokens[verb_len:]
        if not rest:
            return RecoveryResult("none", reason="no_target")

        meta = {
            "original_transcript": raw,
            "verb_guess": intent_name,
        }

        if intent_name == "MOVE_WINDOW_TO_MONITOR":
            return self._recover_move(rest, raw, meta)
        return self._recover_app_op(intent_name, rest, raw, meta)

    # ------------------------------------------------------------------
    def _recover_move(self, rest: list[str], raw: str, meta: dict):
        """Parse '<app...> [to] <monitor...>' with fuzzy slots."""
        # Drop leading filler/prepositions ("to", "into", "onto").
        while rest and rest[0] in {"to", "into", "onto", "on", "in"}:
            rest = rest[1:]
        if not rest:
            return RecoveryResult("none", reason="no_target")

        best = None  # (combined, i, app_status, app, app_score, mon_score, mon_spec)
        n = len(rest)
        for i in range(1, n):
            target_tokens = rest[:i]
            mon_tokens = rest[i:]
            # Monitor fragment may itself start with "to"-like glue.
            while mon_tokens and mon_tokens[0] in {
                "to", "too", "two", "the", "a", "an", "my", "your",
                "into", "in", "onto", "than", "then",
            }:
                mon_tokens = mon_tokens[1:]
            if not mon_tokens:
                continue
            target = " ".join(target_tokens)
            mon_raw = " ".join(mon_tokens)
            mon_score, mon_spec = _score_monitor(mon_raw)
            # Also try the raw monitor resolver for canonical phrases.
            resolved_spec = mon_raw
            if self._monitor_resolver is not None:
                try:
                    mres = self._monitor_resolver(mon_raw)
                    if getattr(mres, "status", "") == "found":
                        mon_score = max(mon_score, 0.9)
                except Exception:  # noqa: BLE001
                    pass
            app_status, app, app_score, app_reason = self._match_app(target)
            if app_status == "missing":
                continue
            combined = app_score + mon_score
            if best is None or combined > best[0]:
                best = (combined, i, app_status, app, app_score,
                        mon_score, mon_spec, mon_raw, target, app_reason)

        if best is None:
            return RecoveryResult("none", reason="no_parse", meta=meta)

        (_combined, _i, app_status, app, app_score, mon_score,
         mon_spec, mon_raw, target, app_reason) = best

        interpretation = (
            f"move {_app_label(app) if app_status == 'found' else target} "
            f"to {mon_spec or mon_raw}"
        )
        meta.update({
            "interpretation": interpretation,
            "app_score": round(app_score, 3),
            "app_match_reason": app_reason,
            "monitor_score": round(mon_score, 3),
            "monitor_spec_raw": mon_raw,
            "monitor_spec_interpreted": mon_spec or mon_raw,
        })

        entities = {
            "target": (
                _app_label(app) if app_status == "found" else target
            ),
            "monitor": _monitor_canonical_spec(
                mon_score, mon_spec, mon_raw
            ),
            "_recovery": meta,
        }
        confidence = round(min(1.0, (app_score + mon_score) / 2.0), 3)

        if app_status == "ambiguous":
            return RecoveryResult(
                "ambiguous",
                confidence=confidence,
                reason="ambiguous_app",
                interpretation=interpretation,
                clarify=self._app_clarify(app, entities, meta),
                meta=meta,
            )

        intent = Intent(
            name="MOVE_WINDOW_TO_MONITOR",
            entities=entities,
            raw_text=raw,
            confidence=confidence,
        )
        return RecoveryResult(
            "matched", intent=intent, confidence=confidence,
            reason="move_recovered", interpretation=interpretation,
            meta=meta,
        )

    # ------------------------------------------------------------------
    def _recover_app_op(self, intent_name: str, rest: list[str],
                        raw: str, meta: dict):
        """Parse '<app...>' for open/switch/close."""
        # Strip filler words at the edges.
        while rest and rest[0] in {"the", "a", "an", "my", "to", "up"}:
            rest = rest[1:]
        while rest and rest[-1] in {"please", "now", "app", "application"}:
            rest = rest[:-1]
        fragment = " ".join(rest)
        if not fragment:
            return RecoveryResult("none", reason="no_target", meta=meta)

        app_status, app, app_score, app_reason = self._match_app(fragment)
        meta.update({
            "interpretation": (
                f"{intent_name.lower().replace('_', ' ')} "
                f"{_app_label(app) if app_status == 'found' else fragment}"
            ),
            "app_score": round(app_score, 3),
            "app_match_reason": app_reason,
        })
        if app_status == "missing":
            return RecoveryResult(
                "none", reason="app_below_threshold",
                confidence=round(app_score, 3), meta=meta,
            )

        entity_name = "app_name"
        entities = {
            entity_name: (
                _app_label(app) if app_status == "found" else fragment
            ),
            "_recovery": meta,
        }
        confidence = round(app_score, 3)
        if app_status == "ambiguous":
            return RecoveryResult(
                "ambiguous", confidence=confidence,
                reason="ambiguous_app",
                interpretation=meta["interpretation"],
                clarify=self._app_clarify(app, entities, meta,
                                         intent_name=intent_name),
                meta=meta,
            )
        intent = Intent(
            name=intent_name, entities=entities, raw_text=raw,
            confidence=confidence,
        )
        return RecoveryResult(
            "matched", intent=intent, confidence=confidence,
            reason="app_op_recovered",
            interpretation=meta["interpretation"], meta=meta,
        )

    # ------------------------------------------------------------------
    def _app_clarify(self, options, entities, meta, intent_name=None):
        """Clarification payload for an ambiguous app match.

        Compatible with the session controller's generic clarify flow:
        ``resolver(text)`` returns an object with ``status``/``app``/
        ``options``; ``apply(intent, app)`` fills ``_app_info``.
        """
        from src.controllers.application_discovery import ResolveResult

        opts = list(options)

        def resolver(text: str):
            t = _norm(text)
            if not t:
                return ResolveResult("missing")
            hits = [
                a for a in opts
                if t and (
                    t in _norm(_app_label(a))
                    or _norm(_app_label(a)) in t
                    or any(tok and tok in _norm(_app_label(a))
                           for tok in t.split()[:2])
                )
            ]
            if len(hits) == 1:
                return ResolveResult("found", app=hits[0])
            if hits:
                return ResolveResult("ambiguous", options=hits)
            return ResolveResult("missing")

        def apply(intent, app):
            intent.entities["_app_info"] = app
            intent.entities["app_name"] = _app_label(app)
            if intent.name == "MOVE_WINDOW_TO_MONITOR":
                intent.entities["target"] = _app_label(app)

        names = [_app_label(a) for a in opts[:4]]
        return {
            "resolver": resolver,
            "apply": apply,
            "message": (
                "I heard a command but I'm not sure which application "
                "you meant: " + ", ".join(names) + ". Which one?"
            ),
            "ambiguous_message": "Which application did you mean?",
            "_recovery": meta,
            # V10.8 Task 4: entity classification for diagnostics.
            "entity_issue": "ambiguous",
        }
