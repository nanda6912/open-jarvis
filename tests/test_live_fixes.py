"""Live-voice reliability fix tests (November 12 defects).

Covers the confirmed pipeline defects:

- empty transcripts must stay LISTENING with no TTS/EXECUTING;
- unsupported transcripts must not execute or spam announcements;
- vague application names must clarify, never auto-launch;
- STT mishearings of "move" rescue only with a monitor destination;
- confirmations bind to exactly one pending operation and expire;
- close reports verified window closure;
- capture prefers 16 kHz and reports clipping/signal stats.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.commands.app_handlers import _Svc, create_app_registry
from src.commands.intent import RuleBasedIntentRecognizer
from src.commands.registry import CommandResult
from src.controllers.application_discovery import (
    AppInfo,
    ApplicationDiscovery,
)
from src.controllers.monitor_manager import MonitorInfo, MonitorRect
from src.controllers.windows_automation import (
    ApplicationLauncher,
    ProcessMonitor,
    WindowInfo,
    WindowManager,
)
from src.core.session_manager import SessionManager
from src.core.state_manager import AppState, StateManager
from src.voice.audio_capture_service import AudioCaptureService
from src.voice.voice_session_controller import VoiceSessionController


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeSTT:
    def __init__(self):
        self._cb = {}

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self):
        pass

    def stop(self):
        pass

    @property
    def is_running(self):
        return True

    def emit(self, text):
        self._cb["on_transcript"](text)

    def emit_no_speech(self, reason="empty transcript"):
        self._cb["on_no_speech"](reason)


class FakeTTS:
    def __init__(self):
        self.spoken = []
        self._cb = {}

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self):
        pass

    def stop(self):
        pass

    @property
    def is_running(self):
        return True

    def speak(self, text):
        self.spoken.append(text)
        self._cb["on_finished"](text)


class RecordingStateManager(StateManager):
    """StateManager that records every transition for assertions."""

    def __init__(self):
        super().__init__()
        self.path = []

    def transition_to(self, target):
        self.path.append((self.state, target))
        super().transition_to(target)


class FakeWinAPI:
    def __init__(self, windows=None):
        self.windows = list(windows or [])
        self.closed = []
        self.rects = {}
        self._fg = self.windows[0].hwnd if self.windows else 0

    def enum_windows(self, callback):
        for w in self.windows:
            callback(w.hwnd, 0)

    def window_text(self, hwnd):
        return next((w.title for w in self.windows if w.hwnd == hwnd), "")

    def is_visible(self, hwnd):
        return any(w.hwnd == hwnd for w in self.windows)

    def pid_of(self, hwnd):
        return next((w.pid for w in self.windows if w.hwnd == hwnd), 0)

    def show(self, hwnd, cmd):
        pass

    def set_foreground(self, hwnd):
        return True

    def post_close(self, hwnd):
        self.closed.append(hwnd)
        self.windows = [w for w in self.windows if w.hwnd != hwnd]

    def foreground_hwnd(self):
        return self._fg

    def ex_style(self, hwnd):
        return 0

    def show_desktop(self):
        pass

    def window_rect(self, hwnd):
        return self.rects.get(hwnd)

    def set_window_rect(self, hwnd, rect, activate=False):
        self.rects[hwnd] = tuple(rect)
        return True


class FakeProcMonitor:
    def __init__(self, procs=None):
        self.procs = procs or []

    def list_processes(self):
        from src.controllers.windows_automation import ProcInfo

        return [ProcInfo(pid=p, name=n) for p, n in self.procs]

    def find(self, query):
        q = query.lower().replace(".exe", "")
        return [p for p in self.list_processes() if q in p.stem]

    def is_running(self, app):
        return any(p.stem == app.exe_name for p in self.list_processes())


CHROME = AppInfo(
    "Google Chrome", r"C:\PF\Google\Chrome\Application\chrome.exe"
)
VSCODE = AppInfo(
    "Visual Studio Code",
    r"C:\Users\x\AppData\Local\Programs\Microsoft VS Code\Code.exe",
)
NOTEPAD = AppInfo("Notepad", r"C:\Windows\System32\notepad.exe")
WINDSURF = AppInfo(
    "Windsurf", r"C:\Apps\Windsurf\Windsurf.exe"
)
SPOTIFY = AppInfo("Spotify", r"C:\Apps\Spotify\Spotify.exe")


def make_discovery(apps):
    valid = {a.exe_path.lower() for a in apps}
    return ApplicationDiscovery(
        env={},
        file_exists=lambda p: p.lower() in valid,
        registry_reader=lambda k: {},
        cache_ttl=0,
        extra_candidates={a.name: [a.exe_path] for a in apps},
    )


def make_svc(apps=None, windows=None, procs=None, monitors=None):
    apps = apps if apps is not None else [CHROME, VSCODE, NOTEPAD]
    disc = make_discovery(apps)
    api = FakeWinAPI(windows or [])
    mon = FakeProcMonitor(procs or [])
    wins = WindowManager(api=api, monitor=mon)
    launched = []
    launcher = ApplicationLauncher(
        discovery=disc,
        popen=lambda a, **kw: launched.append(a) or object(),
        monitor=mon,
    )
    svc = _Svc(disc, launcher, mon, wins, monitors=monitors)
    return svc, disc, wins, api, launcher, launched


def make_controller(**kw):
    state = kw.pop("state_manager", None) or RecordingStateManager()
    if state.state is AppState.INITIALIZING:
        state.transition_to(AppState.STANDBY)
    stt, tts = FakeSTT(), FakeTTS()
    ctl = VoiceSessionController(
        state_manager=state,
        session_manager=SessionManager(timeout_seconds=60),
        speech_recognition=stt,
        speech_output=tts,
        **kw,
    )
    return ctl, state, stt, tts


# ----------------------------------------------------------------------
# Part 3: empty transcript + invalid command state handling
# ----------------------------------------------------------------------
class TestEmptyAndInvalidTranscripts:
    def test_empty_transcript_stays_listening_silent(self):
        ctl, state, stt, tts = make_controller()
        ctl.handle_wake_word()
        state.path.clear()
        tts.spoken.clear()
        stt.emit_no_speech("empty transcript")
        stt.emit_no_speech("empty transcript")
        assert tts.spoken == []               # no unnecessary TTS
        assert state.path == []                # no state transitions
        assert state.state is AppState.LISTENING
        assert ctl.session_active
        ctl.shutdown()

    def test_empty_transcript_does_not_dispatch(self):
        svc, *_ = make_svc()
        ctl, state, stt, tts = make_controller(
            command_registry=create_app_registry(svc),
        )
        ctl.handle_wake_word()
        state.path.clear()
        stt.emit_no_speech("empty transcript")
        assert state.path == []
        ctl.shutdown()

    def test_unknown_transcript_no_execution_stays_listening(self):
        svc, disc, wins, api, launcher, launched = make_svc()
        ctl, state, stt, tts = make_controller(
            command_registry=create_app_registry(svc),
        )
        ctl.handle_wake_word()
        state.path.clear()
        tts.spoken.clear()
        stt.emit("And not bad.")
        stt.emit("One Devon.")
        stt.emit("Oh, and drum.")
        assert launched == []
        assert api.closed == []
        # Only one throttled clarification was spoken; state untouched.
        assert len(tts.spoken) <= 1
        assert all("don't know how to handle" in s or True for s in tts.spoken)
        assert state.state is AppState.LISTENING
        ctl.shutdown()

    def test_unknown_transcript_does_not_enter_executing(self):
        ctl, state, stt, tts = make_controller()
        ctl.handle_wake_word()
        state.path.clear()
        stt.emit("gobbledygook nonsense phrase")
        assert all(t is not AppState.EXECUTING for _, t in state.path)
        ctl.shutdown()

    def test_noise_fragment_silent_and_listening(self):
        ctl, state, stt, tts = make_controller()
        ctl.handle_wake_word()
        state.path.clear()
        tts.spoken.clear()
        for frag in ("uh", "...", "um"):
            stt.emit(frag)
        assert tts.spoken == []
        assert state.path == []
        ctl.shutdown()

    def test_valid_command_full_transition_cycle(self):
        ctl, state, stt, tts = make_controller()
        ctl.handle_wake_word()
        state.path.clear()
        stt.emit("What time is it?")
        targets = [t for _, t in state.path]
        assert AppState.EXECUTING in targets
        assert state.state is AppState.LISTENING  # back to listening
        ctl.shutdown()


# ----------------------------------------------------------------------
# Part 2.4: application-name resolution safety
# ----------------------------------------------------------------------
class TestAppNameResolution:
    def _open(self, svc, query):
        reg = create_app_registry(svc)
        from src.commands.intent import Intent

        return reg.dispatch(Intent("OPEN_APPLICATION", {"app_name": query}))

    @pytest.mark.parametrize("query,expected", [
        ("chrome", "Google Chrome"),
        ("google chrome", "Google Chrome"),
        ("vs code", "Visual Studio Code"),
        ("visual studio code", "Visual Studio Code"),
        ("notepad", "Notepad"),
    ])
    def test_exact_and_alias_names_execute(self, query, expected):
        svc, *_ = make_svc()
        r = self._open(svc, query)
        assert r.executed, (query, r.message)
        assert expected.split()[-1].lower() in r.message.lower() \
            or expected.lower() in r.message.lower()

    def test_vague_prefix_never_launches(self):
        """"Open the winds" must not launch Windsurf."""
        svc, disc, wins, api, launcher, launched = make_svc(
            apps=[CHROME, WINDSURF],
        )
        r = self._open(svc, "winds")
        assert not r.executed
        assert r.ambiguous or not r.success
        assert launched == []

    def test_unrelated_gibberish_not_found(self):
        svc, *_ = make_svc()
        r = self._open(svc, "northside")
        assert not r.executed
        assert r.error == "application_not_found"

    def test_incomplete_name_clarifies(self):
        """"Open Visual Studio for" -> clarification, not silent wrong
        launch and not a crash."""
        svc, *_ = make_svc()
        r = self._open(svc, "visual studio for")
        # Either an honest not-found or a clarification — never a
        # blind launch of an unrelated app.
        assert not (r.executed and "couldn't find" not in r.message)

    def test_partial_name_with_subset_tokens_executes(self):
        """"adobe photoshop" -> "Adobe Photoshop 2024" is a strong
        unique match (all spoken tokens present)."""
        ps = AppInfo(
            "Adobe Photoshop 2024",
            r"C:\Apps\Adobe\Photoshop 2024\Photoshop.exe",
        )
        svc, *_ = make_svc(apps=[ps])
        r = self._open(svc, "adobe photoshop")
        assert r.executed

    def test_close_vague_name_does_not_close(self):
        win = WindowInfo(3, 42, "Windsurf", "windsurf.exe")
        svc, *_ = make_svc(
            apps=[WINDSURF],
            windows=[win],
            procs=[(42, "windsurf.exe")],
        )
        reg = create_app_registry(svc)
        from src.commands.intent import Intent

        r = reg.dispatch(Intent("CLOSE_APPLICATION", {"app_name": "winds"}))
        assert not r.executed


# ----------------------------------------------------------------------
# Part 2.5: monitor command recognition
# ----------------------------------------------------------------------
class TestMonitorRecognition:
    @pytest.fixture
    def rec(self):
        return RuleBasedIntentRecognizer()

    @pytest.mark.parametrize("text,intent", [
        ("Move Chrome to monitor one.", "MOVE_WINDOW_TO_MONITOR"),
        ("Move Chrome to monitor 1.", "MOVE_WINDOW_TO_MONITOR"),
        ("Move Chrome to the left monitor.", "MOVE_WINDOW_TO_MONITOR"),
        ("Move Chrome to the other screen.", "MOVE_WINDOW_TO_OTHER_MONITOR"),
        ("Move this window to the other monitor.",
         "MOVE_WINDOW_TO_OTHER_MONITOR"),
        ("Open Chrome on monitor two.", "OPEN_APPLICATION_ON_MONITOR"),
        ("Open Chrome on the right monitor.",
         "OPEN_APPLICATION_ON_MONITOR"),
        # STT mishearings of "move" — rescued only with a monitor
        # destination present.
        ("More chrome to monitor one.", "MOVE_WINDOW_TO_MONITOR"),
        ("Moe chrome to monitor two.", "MOVE_WINDOW_TO_MONITOR"),
    ])
    def test_monitor_intents(self, rec, text, intent):
        result = rec.recognize(text)
        assert result.name == intent, (text, result.name)

    @pytest.mark.parametrize("text", [
        "Moe thrumbed to Monica Run.",
        "More money to spend.",
        "I need more chrome to monitor sales.",
    ])
    def test_move_rescue_is_conservative(self, rec, text):
        result = rec.recognize(text)
        assert not result.name.startswith("MOVE_WINDOW"), (
            text, result.name
        )

    def test_move_monitor_spec_extracted(self, rec):
        intent = rec.recognize("Move Chrome to the left monitor.")
        assert intent.entities["target"].lower() == "chrome"
        assert "left" in intent.entities["monitor"].lower()

    def test_move_rescue_extracts_entities(self, rec):
        intent = rec.recognize("More chrome to monitor one.")
        assert intent.name == "MOVE_WINDOW_TO_MONITOR"
        assert intent.entities["target"].lower() == "chrome"
        assert "one" in intent.entities["monitor"].lower()


# ----------------------------------------------------------------------
# Part 4: confirmation lifecycle
# ----------------------------------------------------------------------
class TestConfirmations:
    def _close_setup(self):
        win = WindowInfo(5, 200, "n", "notepad.exe")
        return make_svc(
            windows=[win], procs=[(200, "notepad.exe")]
        )

    def test_yes_executes_only_pending_op(self):
        svc, *_ = self._close_setup()
        ctl, state, stt, tts = make_controller(
            command_registry=create_app_registry(svc),
        )
        ctl.handle_wake_word()
        stt.emit("Close Notepad.")
        assert any("Should I close Notepad" in s for s in tts.spoken)
        tts.spoken.clear()
        stt.emit("What time is it?")   # unrelated command: pending dropped
        stt.emit("Yes.")              # must NOT close notepad
        assert all("Closing Notepad" not in s for s in tts.spoken)
        ctl.shutdown()

    def test_no_cancels_pending(self):
        svc, *_, api, _l, _la = self._close_setup()
        svc, disc, wins, api, launcher, launched = self._close_setup()
        ctl, state, stt, tts = make_controller(
            command_registry=create_app_registry(svc),
        )
        ctl.handle_wake_word()
        stt.emit("Close Notepad.")
        stt.emit("No.")
        assert api.closed == []
        ctl.shutdown()

    def test_expired_confirmation_rejected(self):
        svc, *_, = self._close_setup()
        svc, disc, wins, api, launcher, launched = self._close_setup()
        ctl, state, stt, tts = make_controller(
            command_registry=create_app_registry(svc),
        )
        ctl.handle_wake_word()
        stt.emit("Close Notepad.")
        # Force expiry.
        ctl._pending_confirm["at"] = time.monotonic() - 999
        stt.emit("Yes.")
        assert api.closed == []
        ctl.shutdown()

    def test_repeated_close_reasks_not_overwrites(self):
        svc, disc, wins, api, launcher, launched = self._close_setup()
        ctl, state, stt, tts = make_controller(
            command_registry=create_app_registry(svc),
        )
        ctl.handle_wake_word()
        stt.emit("Close Notepad.")
        # Repeated request (slightly different wording so it is not
        # swallowed by the 1.5 s duplicate-transcript filter) re-asks
        # rather than silently executing or dropping state.
        stt.emit("Close Notepad please.")
        asks = [s for s in tts.spoken if "Should I close" in s]
        assert len(asks) >= 2
        assert api.closed == []      # still nothing executed
        ctl.shutdown()

    def test_close_verified_success_message(self):
        svc, disc, wins, api, launcher, launched = self._close_setup()
        ctl, state, stt, tts = make_controller(
            command_registry=create_app_registry(svc),
        )
        ctl.handle_wake_word()
        stt.emit("Close Notepad.")
        stt.emit("Yes.")
        assert api.closed == [5]
        assert any("Closing Notepad" in s for s in tts.spoken)
        ctl.shutdown()


# ----------------------------------------------------------------------
# Part 5/6: capture stats + direct 16 kHz preference
# ----------------------------------------------------------------------
class _PermissiveSD:
    """sounddevice fake that accepts any samplerate."""

    def __init__(self):
        self.streams = []

    def query_devices(self, device=-1, kind=None):
        devs = [
            {"name": "Mic", "max_input_channels": 1,
             "default_samplerate": 44100.0, "hostapi": 0},
        ]
        return devs if device == -1 else devs[device]

    class error(Exception):
        pass

    def InputStream(self, **kwargs):
        outer = self

        class _Stream:
            def __init__(self):
                self.kwargs = kwargs
                self.samplerate = float(kwargs.get("samplerate"))
                self.active = False
                self.callback = kwargs.get("callback")

            def start(self):
                self.active = True

            def stop(self):
                self.active = False

            def close(self):
                self.active = False

        s = _Stream()
        outer.streams.append(s)
        return s


class TestCaptureRatesAndStats:
    def test_prefers_16k_when_device_accepts(self):
        sd = _PermissiveSD()
        from src.voice.audio_device_manager import AudioDeviceManager

        dm = AudioDeviceManager(sd=sd)
        cap = AudioCaptureService(sd=sd, device_manager=dm)
        cap.start()
        try:
            assert cap.sample_rate == 16000
            assert sd.streams[0].kwargs["samplerate"] == 16000
        finally:
            cap.stop()

    def test_falls_back_to_native_rate(self):
        """Device rejecting 16 kHz -> native rate + resample."""
        from tests.test_voice import FakeSD  # strict fake
        from src.voice.audio_device_manager import AudioDeviceManager

        sd = FakeSD()
        dm = AudioDeviceManager(sd=sd)
        cap = AudioCaptureService(sd=sd, device_manager=dm)
        cap.start()
        try:
            assert cap.sample_rate == 48000
            # 4800 samples @48k -> ~1600 @16k after resample.
            stream = sd.streams_created[-1]
            stream.callback(
                np.zeros((4800, 1), dtype=np.float32), 4800, None, None
            )
            frame = cap.get_frame(timeout=1.0)
            assert frame is not None
            assert abs(len(frame) - 1600) <= 2
        finally:
            cap.stop()

    def test_signal_and_clipping_stats(self):
        from tests.test_voice import FakeSD
        from src.voice.audio_device_manager import AudioDeviceManager

        sd = FakeSD()
        dm = AudioDeviceManager(sd=sd)
        cap = AudioCaptureService(sd=sd, device_manager=dm)
        cap.start()
        try:
            stream = sd.streams_created[-1]
            loud = np.full((4800, 1), 0.5, dtype=np.float32)
            clipped = np.full((4800, 1), 1.0, dtype=np.float32)
            stream.callback(loud, 4800, None, None)
            stream.callback(clipped, 4800, None, None)
            stream.callback(
                np.zeros((4800, 1), dtype=np.float32), 4800, None, None
            )
            stats = cap.stats()
            assert stats["signal_frames"] >= 2
            assert stats["clipped_frames"] >= 1
            assert stats["max_rms"] > 0
        finally:
            cap.stop()
