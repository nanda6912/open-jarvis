"""V10.3 intent-classification accuracy tests.

Covers the app/folder/file disambiguation that regressed in the
field: "Open Docker Desktop" must be an application request, never a
folder; bare known folders stay folders; phonetic STT variants
resolve through the discovery engine's skeleton matching.
"""

from __future__ import annotations

import pytest

from src.commands.intent import RuleBasedIntentRecognizer
from src.controllers.application_discovery import ApplicationDiscovery

rec = RuleBasedIntentRecognizer()


# (utterance, expected_intent, entity_checks)
INTENT_CASES = [
    # --- applications -------------------------------------------------
    ("Open Chrome", "OPEN_APPLICATION", {"app_name": "chrome"}),
    ("Open WhatsApp", "OPEN_APPLICATION", {"app_name": "whatsapp"}),
    ("Open Devin", "OPEN_APPLICATION", {"app_name": "devin"}),
    # Regression: "desktop" inside a longer app name must NOT route to
    # OPEN_FOLDER (was the reported V10.3 failure).
    ("Open Docker Desktop", "OPEN_APPLICATION",
     {"app_name": "docker desktop"}),
    ("Open Docker Desktop app", "OPEN_APPLICATION",
     {"app_name": "docker desktop"}),
    ("Switch to WhatsApp", "SWITCH_TO_APPLICATION",
     {"app_name": "whatsapp"}),
    # --- folders ------------------------------------------------------
    ("Open Desktop", "OPEN_FOLDER", {"folder": "desktop"}),
    ("Open the Desktop folder", "OPEN_FOLDER", {"folder": "desktop"}),
    ("Open Desktop folder", "OPEN_FOLDER", {"folder": "desktop"}),
    ("Open downloads folder", "OPEN_FOLDER", {"folder": "downloads"}),
    ("Open downloads", "OPEN_FOLDER", {"folder": "downloads"}),
    ("Open the downloads please", "OPEN_FOLDER",
     {"folder": "downloads"}),
    ("Open Smart Parking System folder", "OPEN_FOLDER",
     {"folder": "smart parking system"}),
    ("Open the Jarvis project folder", "OPEN_FOLDER",
     {"folder": "jarvis project"}),
    ("Show me the pictures folder", "OPEN_FOLDER",
     {"folder": "pictures"}),
    ("Go to documents folder", "OPEN_FOLDER",
     {"folder": "documents"}),
    # --- files --------------------------------------------------------
    ("Open resume.pdf", "OPEN_FILE", {}),
    ("Open requirements.docx", "OPEN_FILE", {}),
    ("Open the latest PDF report", "OPEN_FILE", {}),
    # --- search -------------------------------------------------------
    ("Find my resume files", "SEARCH_FILES", {}),
    ("Search for PDF files", "SEARCH_FILES", {}),
    ("Find Smart Parking project", "SEARCH_FILES", {}),
    # --- windows / monitors -------------------------------------------
    ("Move Chrome to left monitor", "MOVE_WINDOW_TO_MONITOR",
     {"target": "chrome"}),
    ("Move Chrome to the right monitor", "MOVE_WINDOW_TO_MONITOR",
     {"target": "chrome"}),
    ("Move this window to monitor two", "MOVE_WINDOW_TO_MONITOR", {}),
    ("List running applications", "LIST_RUNNING_APPLICATIONS", {}),
    ("What applications are running", "LIST_RUNNING_APPLICATIONS", {}),
    # --- session ------------------------------------------------------
    ("Goodbye Jarvis", "END_SESSION", {}),
    # --- contextual ---------------------------------------------------
    ("Open it", "OPEN_LOCAL_ITEM", {"selector": "it"}),
    ("Open the second one", "OPEN_LOCAL_ITEM", {"selector": "second"}),
    ("Open its parent folder", "OPEN_LOCAL_ITEM",
     {"selector": "parent"}),
]


@pytest.mark.parametrize(
    "text,intent_name,entities", INTENT_CASES,
    ids=[c[0] for c in INTENT_CASES],
)
def test_intent_classification(text, intent_name, entities):
    intent = rec.recognize(text)
    assert intent.name == intent_name, (
        f"{text!r} -> {intent.name} (entities={intent.entities})"
    )
    for key, want in entities.items():
        got = str(intent.entities.get(key, "")).lower()
        assert want in got or got in want, (
            f"{text!r}: entity {key}={got!r}, expected ~{want!r}"
        )


def _disc(*names: str) -> ApplicationDiscovery:
    return ApplicationDiscovery(
        env={},
        file_exists=lambda p: str(p).lower().endswith(".exe"),
        registry_reader=lambda k: {},
        cache_ttl=0,
        extra_candidates={
            n: [rf"C:\Apps\{n}.exe"] for n in names
        },
    )


class TestAppNameResolution:
    """Phonetic STT variants must resolve without hardcoded aliases."""

    APPS = ("Devin", "Docker Desktop", "Google Chrome", "WhatsApp",
            "Notepad")

    @pytest.fixture
    def disc(self):
        return _disc(*self.APPS)

    @pytest.mark.parametrize("query", [
        "devin", "devine", "divin", "devon", "deven",
    ])
    def test_devin_variants_resolve(self, disc, query):
        result = disc.resolve(query)
        assert result.status == "found", (
            f"{query!r} -> {result.status}"
        )
        assert result.app is not None
        assert result.app.name == "Devin"

    @pytest.mark.parametrize("query,app", [
        ("docker desktop", "Docker Desktop"),
        ("google chrome", "Google Chrome"),
        ("whatsapp", "WhatsApp"),
        ("whats app", "WhatsApp"),
        ("note pad", "Notepad"),
    ])
    def test_exact_and_alias_resolution(self, disc, query, app):
        result = disc.resolve(query)
        assert result.status == "found", (
            f"{query!r} -> {result.status} {result.options}"
        )
        assert result.app.name == app

    def test_weak_partial_match_is_ambiguous_not_launched(self, disc):
        # "chrome" only substring-matches "Google Chrome" (score 60);
        # safe behavior is a clarification, never a silent launch.
        result = disc.resolve("chrome")
        assert result.status == "ambiguous"

    def test_two_skeleton_equal_apps_are_ambiguous(self):
        # "Devin" and "Divan" share the skeleton "dvn"; resolution
        # must clarify, never silently pick.
        disc = _disc("Devin", "Divan")
        result = disc.resolve("divin")
        assert result.status == "ambiguous"
        assert len(result.options) >= 1

    def test_unknown_app_is_missing_not_guessed(self, disc):
        result = disc.resolve("grown")
        assert result.status == "missing"


class TestFolderVsAppBoundary:
    """Names containing folder vocabulary stay folders; app names
    containing folder words stay applications."""

    @pytest.mark.parametrize("text", [
        "Open Docker Desktop",
        "Open Docker Desktop app",
        "Launch Docker Desktop",
        "Open the Docker Desktop program",
    ])
    def test_desktop_inside_app_name_is_application(self, text):
        intent = rec.recognize(text)
        assert intent.name == "OPEN_APPLICATION", (
            f"{text!r} -> {intent.name}"
        )
        assert "docker" in intent.entities.get("app_name", "").lower()

    @pytest.mark.parametrize("text,folder", [
        ("Open Desktop", "desktop"),
        ("Open Desktop folder", "desktop"),
        ("Open the Desktop directory", "desktop"),
        ("Open my Downloads folder", "downloads"),
        ("Open Music folder", "music"),
    ])
    def test_explicit_folder_requests_stay_folders(self, text, folder):
        intent = rec.recognize(text)
        assert intent.name == "OPEN_FOLDER", (
            f"{text!r} -> {intent.name}"
        )
        assert folder in intent.entities.get("folder", "")
