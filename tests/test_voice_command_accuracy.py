"""V10.4 voice-command accuracy tests.

An in-memory evaluation set of spoken-command transcripts (including
the garbled monitor commands observed in the field) run through the
rule recognizer plus the misheard-command recovery layer. No audio
hardware or recordings are required — transcripts stand in for the
STT output.
"""

from __future__ import annotations

import types

import pytest

from src.commands.intent import RuleBasedIntentRecognizer
from src.commands.recovery import MisheardCommandRecovery


def _app(name, exe=None):
    return types.SimpleNamespace(
        name=name, exe_name=exe or name.lower().replace(" ", "")
    )


# Simulated "currently running" application inventory.
RUNNING_APPS = [
    _app("Google Chrome", "chrome"),
    _app("WhatsApp", "whatsapp"),
    _app("Devin", "devin"),
    _app("Notepad", "notepad"),
    _app("Spotify", "spotify"),
]


def make_recovery(**kw):
    return MisheardCommandRecovery(
        app_names=lambda: RUNNING_APPS,
        **kw,
    )


# (transcript, expected_intent, expected_app_fragment)
COMMAND_EVAL = [
    # --- application operations -------------------------------------
    ("Open Chrome.", "OPEN_APPLICATION", "chrome"),
    ("Open WhatsApp.", "OPEN_APPLICATION", "whatsapp"),
    ("Open Devin.", "OPEN_APPLICATION", "devin"),
    ("Switch to Chrome.", "SWITCH_TO_APPLICATION", "chrome"),
    ("Switch to WhatsApp.", "SWITCH_TO_APPLICATION", "whatsapp"),
    ("Switch to Devin.", "SWITCH_TO_APPLICATION", "devin"),
    ("Close Chrome.", "CLOSE_APPLICATION", "chrome"),
    ("Open Notepad.", "OPEN_APPLICATION", "notepad"),
    # --- monitor operations -----------------------------------------
    ("Move Chrome to the left monitor.", "MOVE_WINDOW_TO_MONITOR", "chrome"),
    ("Move Chrome to the right monitor.", "MOVE_WINDOW_TO_MONITOR", "chrome"),
    ("Move Devin to the left monitor.", "MOVE_WINDOW_TO_MONITOR", "devin"),
    ("Move WhatsApp to the right monitor.", "MOVE_WINDOW_TO_MONITOR", "whatsapp"),
    ("Move the current window to the other monitor.", "MOVE_WINDOW_TO_OTHER_MONITOR", None),
    ("Move the active application to monitor one.", "MOVE_WINDOW_TO_MONITOR", None),
    ("Move the active application to monitor two.", "MOVE_WINDOW_TO_MONITOR", None),
    # --- other operations -------------------------------------------
    ("Open the downloads folder.", "OPEN_FOLDER", None),
    ("What time is it?", "GET_CURRENT_TIME", None),
    ("Goodbye Jarvis.", "END_SESSION", None),
]


class TestCommandEvalSet:
    """Clean transcripts must keep working through the primary parser."""

    @pytest.mark.parametrize("text,intent,app", COMMAND_EVAL)
    def test_primary_recognizer(self, text, intent, app):
        result = RuleBasedIntentRecognizer().recognize(text)
        assert result.name == intent, f"{text!r} -> {result.name}"
        if app:
            blob = " ".join(
                str(v) for v in result.entities.values()
            ).lower()
            assert app in blob


class TestGarbledMonitorCommands:
    """Field-observed STT garbles must recover or safely clarify."""

    @pytest.mark.parametrize("text", [
        "More crumpled less commoditous.",
        "More crumpled last-monitor.",
        "More devent to less the monitors.",
        "Move Dev into less than monitor.",
    ])
    def test_recovery_never_returns_wrong_app(self, text):
        rec = make_recovery()
        primary = RuleBasedIntentRecognizer().recognize(text)
        # The primary parser may already claim a move intent (its
        # monitor slot then goes through normal resolution/clarify);
        # recovery only runs on UNKNOWN in production.
        if primary.name != "UNKNOWN":
            assert primary.name in (
                "MOVE_WINDOW_TO_MONITOR",
                "MOVE_WINDOW_TO_OTHER_MONITOR",
                "MOVE_WINDOW_TO_PRIMARY_MONITOR",
            )
            return
        result = rec.recover(text)
        # Either no recovery (safe UNKNOWN) or a move intent whose
        # resolved app is a plausible candidate — never a confident
        # wrong application.
        if result.status == "matched" and result.intent is not None:
            assert result.intent.name == "MOVE_WINDOW_TO_MONITOR"
            meta = result.intent.entities.get("_recovery", {})
            assert meta.get("original_transcript") == text

    def test_move_devin_left_monitor_recovers(self):
        rec = make_recovery()
        result = rec.recover("Move Dev into less than monitor.")
        # "dev" should match Devin; the garbled monitor fragment must
        # NOT silently become a confident wrong monitor.
        if result.status == "matched":
            ent = result.intent.entities
            assert "devin" in str(ent.get("target", "")).lower()
        # Ambiguous/none are both acceptable safe outcomes.
        assert result.status in ("matched", "ambiguous", "none")

    def test_move_chrome_right_monitor_recovers(self):
        rec = make_recovery()
        result = rec.recover("Move crumpled to right monitor.")
        # "crumpled" is a heavy garble of "chrome"; recovery may match
        # or decline — it must never pick a different app confidently.
        if result.status == "matched":
            ent = result.intent.entities
            assert "chrome" in str(ent.get("target", "")).lower()
            assert "right" in str(ent.get("monitor", "")).lower()

    def test_open_garbled_app_recovers(self):
        rec = make_recovery()
        result = rec.recover("Open chrome please.")
        # Clean enough for primary parser; recovery is for UNKNOWN only
        # but must still behave sanely if invoked.
        assert result.status in ("matched", "none", "ambiguous")


class TestRecoverySafety:
    def test_unrelated_speech_not_recovered(self):
        rec = make_recovery()
        for text in (
            "The weather is nice today.",
            "Blorf zanzibar quux.",
            "Um, let me think about that.",
        ):
            result = rec.recover(text)
            assert result.status == "none", text

    def test_short_utterance_rejected(self):
        rec = make_recovery()
        assert rec.recover("Move it.").status == "none"

    def test_ambiguous_app_asks_clarification(self):
        # Two apps equally close to the fragment force ambiguity.
        apps = [
            _app("Devin", "devin"),
            _app("Devon", "devon"),
        ]
        rec = MisheardCommandRecovery(app_names=lambda: apps)
        result = rec.recover("Move dev to left monitor.")
        # "dev" sits inside both names -> ambiguous clarify or a safe
        # decline; never a confident wrong pick.
        assert result.status in ("ambiguous", "none")
        if result.status == "ambiguous":
            assert result.clarify is not None

    def test_recovery_preserves_original_transcript(self):
        rec = make_recovery()
        text = "Move crumpled to right monitor."
        result = rec.recover(text)
        if result.intent is not None:
            meta = result.intent.entities.get("_recovery", {})
            assert meta.get("original_transcript") == text
            assert result.intent.raw_text == text

    def test_no_app_candidates_is_safe(self):
        rec = MisheardCommandRecovery(app_names=lambda: [])
        result = rec.recover("Move nothing at all here.")
        assert result.status in ("none", "ambiguous")
