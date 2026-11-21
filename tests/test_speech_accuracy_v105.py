"""V10.5 speech-accuracy tests: intent/entity correctness on the
controlled command vocabulary, plus conservative routing rules.

These are recognition-level tests (transcript -> intent/entities).
Execution-level checks live in the monitor/app handler tests.
"""

from __future__ import annotations

import pytest

from src.commands.intent import RuleBasedIntentRecognizer


@pytest.fixture(scope="module")
def parser():
    return RuleBasedIntentRecognizer()


# (transcript, expected_intent, expected entity fragment or None)
APP_CASES = [
    ("Open Chrome", "OPEN_APPLICATION", "chrome"),
    ("Open WhatsApp", "OPEN_APPLICATION", "whatsapp"),
    ("Open Devin", "OPEN_APPLICATION", "devin"),
    ("Open Docker Desktop", "OPEN_APPLICATION", "docker"),
    ("Open Calculator", "OPEN_APPLICATION", "calculator"),
    ("Switch to Chrome", "SWITCH_TO_APPLICATION", "chrome"),
    ("Switch to WhatsApp", "SWITCH_TO_APPLICATION", "whatsapp"),
    ("Open this PC", "OPEN_APPLICATION", "this pc"),
    ("Open my computer", "OPEN_APPLICATION", "my computer"),
]


@pytest.mark.parametrize("text,expected,frag", APP_CASES)
def test_app_commands(parser, text, expected, frag):
    intent = parser.recognize(text)
    assert intent.name == expected, f"{text!r} -> {intent.name}"
    blob = " ".join(str(v) for v in intent.entities.values()).lower()
    assert frag in blob, f"{text!r} entities={intent.entities}"


FOLDER_FILE_CASES = [
    ("Open Downloads", "OPEN_FOLDER", "downloads"),
    ("Open Desktop folder", "OPEN_FOLDER", "desktop"),
    ("Open Jarvis folder", "OPEN_FOLDER", "jarvis"),
    ("Search resume files", "SEARCH_FILES", "resume"),
    ("Open Smart Parking System folder", "OPEN_FOLDER", None),
]


@pytest.mark.parametrize("text,expected,frag", FOLDER_FILE_CASES)
def test_folder_file_commands(parser, text, expected, frag):
    intent = parser.recognize(text)
    assert intent.name == expected, f"{text!r} -> {intent.name} {intent.entities}"
    if frag:
        blob = " ".join(str(v) for v in intent.entities.values()).lower()
        assert frag in blob


GENERAL_CASES = [
    ("What time is it?", "GET_CURRENT_TIME"),
    ("What's today's date?", "GET_CURRENT_DATE"),
    ("List running applications", "LIST_RUNNING_APPLICATIONS"),
    ("Goodbye Jarvis", "END_SESSION"),
]


@pytest.mark.parametrize("text,expected", GENERAL_CASES)
def test_general_commands(parser, text, expected):
    intent = parser.recognize(text)
    assert intent.name == expected, f"{text!r} -> {intent.name}"


def test_search_youtube_is_web_not_files(parser):
    """Regression: 'Search YouTube' must never become SEARCH_FILES."""
    intent = parser.recognize("Search YouTube")
    assert intent.name == "SEARCH_WEB"
    assert intent.entities.get("engine") == "youtube"


def test_web_search_variants(parser):
    for text in (
        "Search the web for python tutorials",
        "Look up jarvis online",
        "Search youtube for lofi beats",
        "Google maps directions",
    ):
        intent = parser.recognize(text)
        assert intent.name == "SEARCH_WEB", f"{text!r} -> {intent.name}"
        assert intent.entities.get("query") or text == "Search YouTube"


def test_file_search_still_wins_without_web_scope(parser):
    """Local file/folder vocabulary must not be stolen by SEARCH_WEB."""
    intent = parser.recognize("Search files for resume")
    assert intent.name == "SEARCH_FILES"
    intent = parser.recognize("Find my resume pdf")
    assert intent.name == "SEARCH_FILES"


def test_open_google_chrome_is_app_not_web(parser):
    """'google' inside an app name must not trigger SEARCH_WEB."""
    intent = parser.recognize("Open Google Chrome")
    assert intent.name == "OPEN_APPLICATION"
    assert "chrome" in str(intent.entities.get("app_name", "")).lower()


def test_this_pc_not_pc_manager(parser):
    """'this pc' must stay a shell destination, not fuzzy to PC Manager."""
    intent = parser.recognize("Open this PC")
    assert intent.entities.get("app_name", "").lower() == "this pc"


def test_original_transcript_preserved(parser):
    raw = "Move Chrome into less than monitor"
    intent = parser.recognize(raw)
    assert intent.raw_text == raw
