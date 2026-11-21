"""Live-voice reliability tests on recorded audio fixtures.

Fixtures in tests/fixtures/commands/*.wav were synthesized with the
production TTS voice (Piper/VITS) — they exercise the real STT model,
intent parser and app resolver end-to-end. Skips cleanly when the STT
model is not installed.
"""

from __future__ import annotations

import json
import os
import time
import wave
from pathlib import Path

import numpy as np
import pytest

from src.commands.intent import RuleBasedIntentRecognizer
from src.controllers.application_discovery import (
    AppInfo,
    ApplicationDiscovery,
)
from src.voice.stt_engine import create_recognizer

FIXTURES = Path(__file__).parent / "fixtures" / "commands"
MODELS = Path.home() / "AppData" / "Local" / "VoiceOS" / "models" / "stt"


def _load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        return (
            np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
            .astype(np.float32) / 32768.0
        )


@pytest.fixture(scope="module")
def recognizer():
    if not MODELS.exists():
        pytest.skip("STT model not installed")
    rec = create_recognizer(MODELS)
    if getattr(rec, "name", "") == "null":
        pytest.skip("STT recognizer unavailable")
    yield rec
    rec.close()


@pytest.fixture(scope="module")
def pipeline():
    """STT + intent + a realistic app inventory."""
    rec = create_recognizer(MODELS) if MODELS.exists() else None
    apps = [
        AppInfo("Google Chrome", r"C:\PF\Google\Chrome\Application\chrome.exe"),
        AppInfo("Visual Studio Code", r"C:\Users\x\AppData\Local\Programs\Microsoft VS Code\Code.exe"),
        AppInfo("Notepad", r"C:\Windows\System32\notepad.exe"),
        AppInfo("Spotify", r"C:\Users\x\AppData\Roaming\Spotify\Spotify.exe"),
        AppInfo("Microsoft Edge", r"C:\PF86\Microsoft\Edge\Application\msedge.exe"),
    ]
    valid = {a.exe_path.lower() for a in apps}
    disc = ApplicationDiscovery(
        env={}, file_exists=lambda p: p.lower() in valid,
        registry_reader=lambda k: {}, cache_ttl=0,
        extra_candidates={a.name: [a.exe_path] for a in apps},
    )
    return rec, RuleBasedIntentRecognizer(), disc


def _transcribe_file(rec, wav: Path):
    audio = _load_wav(wav)
    t = time.time()
    text = rec.transcribe(audio)
    return text.strip(), time.time() - t, len(audio) / 16000.0


class TestRecordedCommandPipeline:
    """End-to-end: wav -> transcript -> intent -> app resolution."""

    EXPECTED = {
        "open_chrome.wav": ("OPEN_APPLICATION", "chrome"),
        "open_vscode.wav": ("OPEN_APPLICATION", "code"),
        "open_notepad.wav": ("OPEN_APPLICATION", "notepad"),
        # NOTE: the TTS frontend lacks "Spotify" in its lexicon (OOV),
        # so this fixture's audio literally says "Open." — the honest
        # expectation is a fragment/UNKNOWN, which the controller turns
        # into a short "need more detail" reply. Spotify resolution is
        # covered by unit tests in test_dynamic_discovery.py.
        "open_spotify.wav": ("UNKNOWN", None),
        "open_nonexistent.wav": ("UNKNOWN", None),
        "switch_chrome.wav": ("SWITCH_TO_APPLICATION", "chrome"),
        "close_chrome.wav": ("CLOSE_APPLICATION", "chrome"),
        "what_time.wav": ("GET_CURRENT_TIME", None),
        "set_volume_50.wav": ("SET_VOLUME", None),
        "increase_volume_10.wav": ("INCREASE_VOLUME", None),
        "move_chrome_monitor_two.wav": ("MOVE_WINDOW_TO_MONITOR", None),
        "switch_notepad.wav": ("SWITCH_TO_APPLICATION", "notepad"),
        "hello_jarvis.wav": (None, None),   # wake phrase, not a command
        "hey_jarvis.wav": (None, None),
        "silence.wav": ("", None),
        "background_speech.wav": ("UNKNOWN", None),
    }

    @pytest.mark.parametrize("wav_name", sorted(EXPECTED))
    def test_fixture(self, pipeline, wav_name):
        rec, intents, disc = pipeline
        if rec is None or getattr(rec, "name", "") == "null":
            pytest.skip("STT unavailable")
        wav = FIXTURES / wav_name
        if not wav.exists():
            pytest.skip("fixture missing")
        text, latency, dur = _transcribe_file(rec, wav)
        exp_intent, exp_app = self.EXPECTED[wav_name]

        if exp_intent is None:
            # Wake-phrase fixtures: transcript should mention Jarvis.
            assert "jarvis" in text.lower() or not text
            return
        if exp_intent == "":
            assert text == ""
            return

        intent = intents.recognize(text)
        assert intent.name == exp_intent, (
            f"{wav_name}: transcript={text!r} intent={intent.name} "
            f"expected={exp_intent}"
        )
        if exp_app:
            res = disc.resolve(intent.entities.get("app_name", ""))
            assert res.status == "found", (
                f"{wav_name}: app resolution failed for "
                f"{intent.entities} (transcript={text!r})"
            )
            assert exp_app in res.app.exe_name or exp_app in res.app.name.lower()

    def test_metrics_summary(self, pipeline, capsys):
        """Print accuracy/latency metrics for the report."""
        rec, intents, disc = pipeline
        if rec is None or getattr(rec, "name", "") == "null":
            pytest.skip("STT unavailable")
        rows = []
        for wav_name, (exp_intent, exp_app) in sorted(self.EXPECTED.items()):
            wav = FIXTURES / wav_name
            if not wav.exists():
                continue
            text, latency, dur = _transcribe_file(rec, wav)
            intent = intents.recognize(text) if text else None
            ok_intent = (
                exp_intent is None
                or (exp_intent == "" and text == "")
                or (intent is not None and intent.name == exp_intent)
            )
            rows.append((wav_name, text, intent.name if intent else "-",
                         round(latency * 1000), ok_intent))
        print("\n=== Recorded-audio pipeline metrics ===")
        for r in rows:
            print(f"{r[0]:24s} {r[3]:>5}ms  {r[1]!r:50s} -> {r[2]} {'OK' if r[4] else 'MISS'}")
        n = len(rows)
        hits = sum(1 for r in rows if r[4])
        print(f"intent accuracy: {hits}/{n} = {hits/n:.0%}")
        print(f"mean latency: {sum(r[3] for r in rows)/n:.0f} ms")


class TestControllerRobustness:
    """Unit-level checks for dedupe/wake/noise handling (no audio)."""

    def _controller(self):
        from src.core.session_manager import SessionManager
        from src.core.state_manager import AppState, StateManager
        from src.voice.voice_session_controller import VoiceSessionController

        class STT:
            def __init__(self): self._cb = {}
            def set_callbacks(self, **kw): self._cb.update(kw)
            def start(self): pass
            def stop(self): pass
            @property
            def is_running(self): return True
            def emit(self, t): self._cb["on_transcript"](t)

        class TTS:
            def __init__(self): self.spoken = []; self._cb = {}
            def set_callbacks(self, **kw): self._cb.update(kw)
            def start(self): pass
            def stop(self): pass
            @property
            def is_running(self): return True
            def speak(self, t):
                self.spoken.append(t); self._cb["on_finished"](t)

        state = StateManager(); state.transition_to(AppState.STANDBY)
        ctl = VoiceSessionController(
            state_manager=state,
            session_manager=SessionManager(timeout_seconds=60),
            speech_recognition=STT(), speech_output=TTS(),
        )
        return ctl

    def test_duplicate_transcript_ignored(self):
        ctl = self._controller()
        ctl.handle_wake_word()
        ctl._tts.spoken.clear()
        ctl._stt.emit("Open Chrome.")
        n1 = len(ctl._tts.spoken)
        ctl._stt.emit("Open Chrome.")  # immediate duplicate
        assert len(ctl._tts.spoken) == n1
        ctl.shutdown()

    def test_wake_variants_acknowledged(self):
        ctl = self._controller()
        ctl.handle_wake_word()
        ctl._tts.spoken.clear()
        for variant in ("Hey Jarvis", "Jarvis", "Below Jarvis"):
            ctl._stt.emit(variant)
        assert ctl._tts.spoken.count("Yes?") == 3
        ctl.shutdown()

    def test_noise_fragment_silent(self):
        ctl = self._controller()
        ctl.handle_wake_word()
        ctl._tts.spoken.clear()
        ctl._stt.emit("uh")
        ctl._stt.emit("...")
        assert ctl._tts.spoken == []
        ctl.shutdown()

    def test_fragment_verb_gets_short_response(self):
        ctl = self._controller()
        ctl.handle_wake_word()
        ctl._tts.spoken.clear()
        ctl._stt.emit("Open.")
        assert any("more detail" in s for s in ctl._tts.spoken)
        assert not any("don't know how to handle" in s for s in ctl._tts.spoken)
        ctl.shutdown()
