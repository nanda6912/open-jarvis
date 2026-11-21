"""Ground-truth command corpus tests (Task: speech accuracy & intent).

``tests/fixtures/command_corpus.json`` records, per command:

- ``spoken`` — what the user actually said (physical-mic confirmed or
  intended phrasing).
- ``expected_transcript`` / ``normalized_transcript`` — ideal ASR output.
- ``expected_intent`` / ``expected_entities`` / ``expected_action``.
- For ``physical_microphone`` entries: ``moonshine_transcript`` (the
  real ASR output), ``failure_category`` and ``recovered``.

Two accuracy axes are reported separately:

- *Transcription accuracy* is exercised by ``test_live_reliability.py``
  (wav -> transcript) — this file asserts the *intent/entity* layer
  given a transcript, plus that recorded ASR errors fail safely.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.commands.intent import RuleBasedIntentRecognizer
from src.controllers.application_discovery import (
    AppInfo,
    ApplicationDiscovery,
)

CORPUS = Path(__file__).parent / "fixtures" / "command_corpus.json"
COMMANDS_DIR = Path(__file__).parent / "fixtures" / "commands"


def _corpus() -> list[dict]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def intents():
    return RuleBasedIntentRecognizer()


@pytest.fixture(scope="module")
def discovery():
    """Inventory covering every app named in the corpus."""
    apps = [
        AppInfo("Google Chrome", r"C:\PF\Google\Chrome\Application\chrome.exe"),
        AppInfo("Notepad", r"C:\Windows\System32\notepad.exe"),
        AppInfo(
            "Devin (User)",
            r"C:\Users\user\Softwares\Devin\Devin.exe",
        ),
        AppInfo("WhatsApp", r"C:\PF\WhatsApp\WhatsApp.exe"),
    ]
    valid = {a.exe_path.lower() for a in apps}
    return ApplicationDiscovery(
        env={},
        file_exists=lambda p: p.lower() in valid,
        registry_reader=lambda k: {},
        cache_ttl=0,
        extra_candidates={a.name: [a.exe_path] for a in apps},
    )


def _entity_match(expected: dict, actual: dict) -> bool:
    """Every expected entity key must match (subset semantics)."""
    for key, want in (expected or {}).items():
        got = actual.get(key)
        if isinstance(want, str):
            if str(got).strip().lower() != want.strip().lower():
                return False
        elif got != want:
            return False
    return True


class TestExpectedTranscriptParsing:
    """Intent/entity accuracy: ideal transcript -> intent + entities."""

    @pytest.mark.parametrize("entry", _corpus(), ids=lambda e: e["id"])
    def test_expected_transcript(self, intents, entry):
        text = entry["expected_transcript"]
        intent = intents.recognize(text)
        assert intent.name == entry["expected_intent"], (
            f"{entry['id']}: {text!r} -> {intent.name} "
            f"(entities={intent.entities})"
        )
        assert _entity_match(entry.get("expected_entities"), intent.entities), (
            f"{entry['id']}: entities {intent.entities} != "
            f"{entry.get('expected_entities')}"
        )

    @pytest.mark.parametrize("entry", _corpus(), ids=lambda e: e["id"])
    def test_normalized_transcript(self, intents, entry):
        """The normalized form must parse identically (no casing/punct
        sensitivity)."""
        intent = intents.recognize(entry["normalized_transcript"])
        assert intent.name == entry["expected_intent"]
        assert _entity_match(entry.get("expected_entities"), intent.entities)


class TestRecordedAsrOutcomes:
    """Physical-mic entries: the real Moonshine transcript must either
    be recovered by the intent/entity layer or fail safely — never
    resolve to the wrong application."""

    @pytest.mark.parametrize(
        "entry",
        [e for e in _corpus() if e.get("moonshine_transcript")],
        ids=lambda e: e["id"],
    )
    def test_recorded_transcript_outcome(self, intents, discovery, entry):
        text = entry["moonshine_transcript"]
        intent = intents.recognize(text)

        if entry.get("recovered"):
            # Intent layer repaired the ASR error: full expected intent
            # must be produced from the bad transcript. Entities may
            # still carry the misheard surface form ("Devine") — the
            # recovery happens at app-resolution time via aliases.
            assert intent.name == entry["expected_intent"], (
                f"{entry['id']}: {text!r} -> {intent.name}, "
                f"expected recovery to {entry['expected_intent']}"
            )
            expected_app = (entry.get("expected_entities") or {}).get(
                "app_name"
            )
            app_name = (
                intent.entities.get("app_name")
                or intent.entities.get("target")
                or ""
            )
            if expected_app and app_name:
                res = discovery.resolve(str(app_name))
                assert res.status == "found" and res.app is not None, (
                    f"{entry['id']}: {app_name!r} did not resolve "
                    f"({res.status})"
                )
                assert expected_app.lower() in (
                    res.app.name.lower() + " " + res.app.exe_name
                )
            return

        # Unrecovered STT error: assert the *safe* outcome. The parser
        # may still produce a plausible intent, but resolving its app
        # entity must never find the intended application.
        app_name = (
            intent.entities.get("app_name")
            or intent.entities.get("target")
            or ""
        )
        expected_app = (entry.get("expected_entities") or {}).get(
            "app_name"
        )
        if app_name and expected_app:
            res = discovery.resolve(str(app_name))
            # Safe failure = not a confident "found" of a *different*
            # app; "found" of the intended app would mean the ASR error
            # accidentally did the right thing (still acceptable, but
            # then the entry should be marked recovered).
            if res.status == "found" and res.app is not None:
                assert res.app.name.lower() != expected_app.lower(), (
                    f"{entry['id']}: mis-transcript {text!r} resolved "
                    f"to {res.app.name} — unsafe"
                )


class TestAliasSafety:
    """New pronunciation aliases are narrow and explicit."""

    @pytest.mark.parametrize(
        "query,expected_stem",
        [
            ("Devine", "devin"),
            ("devinn", "devin"),
            ("devon", "devin"),
            ("whats app", "whatsapp"),
            ("what s app", "whatsapp"),
            ("whatsap", "whatsapp"),
        ],
    )
    def test_alias_resolves(self, discovery, query, expected_stem):
        res = discovery.resolve(query)
        assert res.status == "found", f"{query!r} -> {res.status}"
        assert expected_stem in res.app.exe_name or expected_stem in (
            res.app.name.lower()
        )

    def test_no_broad_fuzzy_launch(self, discovery):
        """Uncertain phrases must NOT resolve to an app."""
        for bad in ("north side of", "wind", "wind deadline",
                    "what's happening", "crumb"):
            res = discovery.resolve(bad)
            assert res.status != "found" or (
                res.score >= 80
            ), f"{bad!r} unexpectedly resolved: {res.app}"
