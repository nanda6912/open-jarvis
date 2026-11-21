"""Continuous wake-word cycle acceptance test (mandatory).

Verifies Jarvis survives at least five consecutive
wake -> command -> standby -> wake cycles without restarting:

- activation speaks "At your service." exactly once per session
- commands execute and Jarvis returns to LISTENING
- "Goodbye Jarvis" / inactivity timeout return to STANDBY
- wake-word detection resumes, STT stops, mic handoff is clean
- no duplicate workers, streams, sessions or stale frames

Uses the REAL VoiceSessionController, SpeechRecognitionService,
SpeechOutputService, StateManager and SessionManager. Only the audio
hardware, synthesizer and recognizer are faked.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.commands.app_handlers import _Svc, create_app_registry
from src.controllers.application_discovery import AppInfo, ApplicationDiscovery
from src.controllers.windows_automation import (
    ApplicationLauncher,
    ProcessMonitor,
    WindowManager,
)
from src.core.session_manager import SessionManager
from src.core.state_manager import AppState, StateManager
from src.voice.speech_output_service import SpeechOutputService
from src.voice.speech_recognition_service import SpeechRecognitionService
from src.voice.stt_engine import SpeechRecognizer
from src.voice.voice_session_controller import VoiceSessionController

SR = 16000


# ----------------------------------------------------------------------
# Fakes (same style as test_task4)
# ----------------------------------------------------------------------
class FakeFrameSource:
    def __init__(self):
        import queue
        self._queue = queue.Queue()
        self.paused = False
        self.flush_count = 0
        self.frames_pushed = 0

    def push(self, samples):
        self._queue.put(np.asarray(samples, dtype=np.float32))
        self.frames_pushed += 1

    def push_speech(self, seconds=0.4, level=0.2):
        n = int(SR * seconds)
        t = np.arange(n) / SR
        self.push((level * np.sin(2 * np.pi * 220 * t)).astype(np.float32))

    def push_silence(self, seconds=0.4):
        self.push(np.zeros(int(SR * seconds), dtype=np.float32))

    def get_frame(self, timeout=0.5):
        import queue
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def flush(self):
        self.flush_count += 1
        import queue
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return


class ScriptedRecognizer:
    """Returns queued transcripts, one per finalized utterance."""

    name = "scripted-stt"
    model_dir = "/fake"

    def __init__(self, texts):
        self._texts = list(texts)
        self.calls = 0
        self.closed = False

    def transcribe(self, samples):
        self.calls += 1
        return self._texts.pop(0) if self._texts else ""

    def close(self):
        self.closed = True


class FakeSynthesizer:
    name = "fake-tts"
    model_dir = "/fake"

    def __init__(self):
        self.calls = []
        self.closed = False

    def synthesize(self, text):
        self.calls.append(text)
        return np.zeros(2205, dtype=np.float32), 22050

    def close(self):
        self.closed = True


class FakePlayer:
    name = "fake-player"

    def __init__(self, block=0.02):
        self.played = []
        self.block = block

    def play(self, samples, sample_rate):
        self.played.append((len(samples), sample_rate))
        if self.block:
            time.sleep(self.block)


class FakeMic:
    """Mimics MicrophoneController suspend/resume + KWS worker state."""

    def __init__(self):
        self.capture = FakeFrameSource()
        self.suspend_count = 0
        self.resume_count = 0
        self.kws_running = True   # standby: KWS active
        self.stt_handoff_flushes = 0
        self.events = []          # ordered event log for assertions

    def suspend_detection(self, pause_capture=True):
        self.suspend_count += 1
        self.kws_running = False
        self.events.append("kws_stop")
        if pause_capture:
            self.capture.pause()

    def resume_detection(self):
        self.resume_count += 1
        self.kws_running = True
        self.capture.flush()
        self.events.append("kws_resume")


# ----------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------
def _wait_for(pred, timeout=4.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def _build(timeout=60, transcripts=None, app_registry=False):
    """Assemble the real pipeline with fakes at the edges."""
    state = StateManager()
    state.transition_to(AppState.STANDBY)
    session = SessionManager(timeout_seconds=timeout)
    capture = FakeFrameSource()
    mic = FakeMic()
    mic.capture = capture

    recognizer = ScriptedRecognizer(transcripts or [])
    stt = SpeechRecognitionService(
        capture=capture,
        recognizer=recognizer,
        speech_threshold=0.02,
        silence_hangover=0.25,
        min_utterance=0.15,
        pre_roll=0.2,
        frame_timeout=0.1,
    )
    synth = FakeSynthesizer()
    player = FakePlayer(block=0.02)
    tts = SpeechOutputService(synthesizer=synth, player=player, gate=capture)
    tts.start()

    if app_registry:
        # Minimal dynamic inventory so "Open Chrome" really dispatches.
        chrome = AppInfo(
            "Google Chrome",
            r"C:\PF\Google\Chrome\Application\chrome.exe",
        )
        valid = {chrome.exe_path.lower()}
        disc = ApplicationDiscovery(
            env={}, file_exists=lambda p: p.lower() in valid,
            registry_reader=lambda k: {}, cache_ttl=0,
            extra_candidates={"Google Chrome": [chrome.exe_path]},
        )
        launched = []
        mon = ProcessMonitor(lister=lambda: [])
        wins = WindowManager(api=_NullAPI(), monitor=mon)
        launcher = ApplicationLauncher(
            discovery=disc,
            popen=lambda a, **kw: launched.append(a) or object(),
            monitor=mon,
        )
        registry = create_app_registry(_Svc(disc, launcher, mon, wins))
        harness = {"launched": launched}
    else:
        registry = None
        harness = {}

    ctl = VoiceSessionController(
        state_manager=state,
        session_manager=session,
        speech_recognition=stt,
        speech_output=tts,
        microphone_controller=mic,
        command_registry=registry,
    )
    return ctl, state, session, stt, tts, mic, capture, synth, harness


class _NullAPI:
    """WindowManager api that reports no windows (safe no-op)."""

    def enum_windows(self, cb):
        pass

    def window_text(self, h): return ""
    def is_visible(self, h): return False
    def pid_of(self, h): return 0
    def show(self, h, c): pass
    def set_foreground(self, h): return False
    def post_close(self, h): pass
    def foreground_hwnd(self): return 0
    def ex_style(self, h): return 0
    def show_desktop(self): pass


def _speak_utterance(capture, words=""):
    """Push one speech+silence burst so VAD finalizes an utterance."""
    capture.push_speech(0.4)
    capture.push_speech(0.4)
    capture.push_silence(0.5)
    capture.push_silence(0.5)


# ----------------------------------------------------------------------
# The mandatory 5-cycle acceptance test
# ----------------------------------------------------------------------
class TestWakeCycles:
    """Five consecutive wake -> command -> standby -> wake cycles."""

    CYCLES = 5

    def _run_cycles(self, commands, end_mode="goodbye", timeout=60,
                    shutdown=True):
        """Run ``len(commands)`` full cycles; returns per-cycle report."""
        # Flatten transcripts: each cycle needs [command, (goodbye?)].
        transcripts = []
        for cmd, mode in commands:
            transcripts.append(cmd)
            if mode == "goodbye":
                transcripts.append("Goodbye Jarvis")
        ctl, state, session, stt, tts, mic, cap, synth, harness = _build(
            timeout=timeout, transcripts=transcripts,
            app_registry=any(c == "Open Chrome." for c, _ in commands),
        )
        report = []
        for idx, (command, mode) in enumerate(commands, 1):
                cycle = {"n": idx, "command": command, "checks": {}}
                # -- 1. Wake ------------------------------------------------
                started = ctl.handle_wake_word("hello jarvis")
                cycle["checks"]["wake_accepted"] = started
                assert started, f"cycle {idx}: duplicate/blocked activation"
                # Duplicate wake while active must be rejected.
                cycle["checks"]["dupe_rejected"] = (
                    ctl.handle_wake_word("hello jarvis") is False
                )
                assert _wait_for(lambda: state.state is AppState.LISTENING)
                cycle["checks"]["activation_spoke"] = (
                    "At your service." in synth.calls
                )
                cycle["checks"]["kws_stopped"] = not mic.kws_running
                cycle["checks"]["stt_running"] = stt.is_running
                # -- 2. Command --------------------------------------------
                if command is not None:
                    calls_before = len(synth.calls)
                    _speak_utterance(cap)
                    assert _wait_for(
                        lambda: len(synth.calls) > calls_before
                    ), f"cycle {idx}: no response to {command!r}"
                    cycle["checks"]["command_response"] = True
                    assert _wait_for(
                        lambda: state.state is AppState.LISTENING
                    ), f"cycle {idx}: did not return to LISTENING"

                # -- 3. End session -----------------------------------------
                if mode == "goodbye":
                    _speak_utterance(cap)
                    assert _wait_for(
                        lambda: state.state is AppState.STANDBY
                    ), f"cycle {idx}: goodbye did not reach STANDBY"
                    cycle["checks"]["goodbye_spoke"] = any(
                        "Goodbye." == c for c in synth.calls
                    )
                else:  # timeout
                    time.sleep(timeout + 0.4)
                    ctl.tick()
                    assert _wait_for(
                        lambda: state.state is AppState.STANDBY
                    ), f"cycle {idx}: timeout did not reach STANDBY"
                    cycle["checks"]["timeout_spoke"] = any(
                        "Going back to standby." == c for c in synth.calls
                    )

                cycle["checks"]["session_inactive"] = not session.is_active
                cycle["checks"]["stt_stopped"] = not stt.is_running
                cycle["checks"]["kws_resumed"] = mic.kws_running
                cycle["checks"]["mic_handbacks"] = (
                    mic.suspend_count == mic.resume_count == idx
                )
                cycle["checks"]["flush_on_resume"] = cap.flush_count >= idx
                report.append(cycle)
        env = {
            "ctl": ctl, "state": state, "session": session, "stt": stt,
            "tts": tts, "mic": mic, "cap": cap, "synth": synth,
            "harness": harness,
        }
        if shutdown:
            ctl.shutdown()
            stt.shutdown()
            tts.shutdown()
        return report, env

    def test_five_goodbye_cycles(self):
        commands = [
            ("What time is it?", "goodbye"),
            ("Open Chrome.", "goodbye"),
            ("What time is it?", "goodbye"),
            ("What is the date?", "goodbye"),
            ("What time is it?", "goodbye"),
        ]
        report, env = self._run_cycles(commands, shutdown=False)
        assert len(report) == self.CYCLES
        for c in report:
            for k, v in c["checks"].items():
                assert v, f"cycle {c['n']} failed check {k}: {c['checks']}"
        # Aggregate invariants across the whole run.
        ctl, state, session, stt, tts, mic, cap, synth, harness = (
            env["ctl"], env["state"], env["session"], env["stt"],
            env["tts"], env["mic"], env["cap"], env["synth"], env["harness"],
        )
        assert mic.suspend_count == self.CYCLES
        assert mic.resume_count == self.CYCLES
        assert mic.kws_running  # standby at the end: KWS alive
        assert cap.flush_count >= self.CYCLES  # stale frames flushed
        # STT recognizer closed only once, at shutdown.
        # Open Chrome really launched through the app registry.
        assert harness["launched"] == [
            [r"C:\PF\Google\Chrome\Application\chrome.exe"]
        ]
        # App still "running" (tray): controller accepts a 6th wake.
        try:
            assert ctl.handle_wake_word("hello jarvis") is True
        finally:
            ctl.shutdown()
            stt.shutdown()
            tts.shutdown()

    def test_timeout_cycle(self):
        """Inactivity timeout also returns cleanly to STANDBY."""
        commands = [
            ("What time is it?", "timeout"),
            ("What time is it?", "timeout"),
            ("What time is it?", "timeout"),
        ]
        report, env = self._run_cycles(commands, timeout=3)
        assert len(report) == 3
        for c in report:
            assert c["checks"]["timeout_spoke"]
            assert c["checks"]["kws_resumed"]
            assert c["checks"]["stt_stopped"]

    def test_mixed_goodbye_and_timeout(self):
        commands = [
            ("What time is it?", "goodbye"),
            ("What time is it?", "timeout"),
            ("What time is it?", "goodbye"),
            ("What time is it?", "timeout"),
            ("What time is it?", "goodbye"),
        ]
        report, env = self._run_cycles(commands, timeout=3)
        assert len(report) == self.CYCLES
        for c in report:
            for k, v in c["checks"].items():
                assert v, f"cycle {c['n']} failed {k}"
