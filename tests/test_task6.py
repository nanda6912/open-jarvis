"""Tests for Task 6: app discovery, launching, window management,
command handlers, confirmation flow — all Windows ops mocked."""

from __future__ import annotations

import pytest

from src.commands.app_handlers import _Svc, create_app_registry
from src.commands.intent import RuleBasedIntentRecognizer
from src.commands.registry import CommandRegistry, CommandResult
from src.controllers.application_discovery import (
    AppInfo,
    ApplicationDiscovery,
)
from src.controllers.windows_automation import (
    ApplicationLauncher,
    ProcessMonitor,
    WindowInfo,
    WindowManager,
)
from src.core.session_manager import SessionManager
from src.core.state_manager import AppState, StateManager
from src.voice.voice_session_controller import VoiceSessionController


# ----------------------------------------------------------------------
# Fakes
# ----------------------------------------------------------------------
class FakeAPI:
    """Fake user32 for WindowManager."""

    def __init__(self, windows=None):
        self.windows = windows or []  # list[WindowInfo]
        self.shown = []
        self.foreground_calls = []
        self.closed = []
        self.desktop_shown = 0
        self._fg = self.windows[0].hwnd if self.windows else 0

    def enum_windows(self, callback):
        import ctypes
        from ctypes import wintypes

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def cb(hwnd, lp):
            return callback(hwnd, lp)

        for w in self.windows:
            callback(w.hwnd, 0)

    def window_text(self, hwnd):
        for w in self.windows:
            if w.hwnd == hwnd:
                return w.title
        return ""

    def is_visible(self, hwnd):
        return any(w.hwnd == hwnd for w in self.windows)

    def pid_of(self, hwnd):
        for w in self.windows:
            if w.hwnd == hwnd:
                return w.pid
        return 0

    def show(self, hwnd, cmd):
        self.shown.append((hwnd, cmd))

    def set_foreground(self, hwnd):
        self.foreground_calls.append(hwnd)
        return True

    def post_close(self, hwnd):
        self.closed.append(hwnd)
        # WM_CLOSE actually destroys the window; mirror that so close
        # verification sees the window disappear.
        self.windows = [w for w in self.windows if w.hwnd != hwnd]

    def foreground_hwnd(self):
        return self._fg

    def ex_style(self, hwnd):
        return 0

    def show_desktop(self):
        self.desktop_shown += 1


class FakeMonitor:
    def __init__(self, procs=None):
        self.procs = procs or []  # list[(pid, name)]

    def list_processes(self):
        from src.controllers.windows_automation import ProcInfo
        return [ProcInfo(pid=p, name=n) for p, n in self.procs]

    def find(self, query):
        q = query.lower().replace(".exe", "")
        return [p for p in self.list_processes() if q in p.stem]

    def is_running(self, app):
        return any(p.stem == app.exe_name for p in self.list_processes())


CHROME = AppInfo("Google Chrome", r"C:\Program Files\Google\Chrome\Application\chrome.exe")
NOTEPAD = AppInfo("Notepad", r"C:\Windows\System32\notepad.exe")
EXPLORER = AppInfo("File Explorer", r"C:\Windows\explorer.exe")


def make_discovery(apps=None):
    apps = apps if apps is not None else [CHROME, NOTEPAD, EXPLORER]
    valid = {a.exe_path.lower() for a in apps}
    return ApplicationDiscovery(
        env={},
        file_exists=lambda p: p.lower() in valid,
        registry_reader=lambda k: {},
        start_apps_lister=lambda: [],
        extra_candidates={
            a.name: [a.exe_path] for a in apps
        },
        cache_ttl=0,
    )


def make_windows(windows=None, procs=None):
    api = FakeAPI(windows or [])
    mon = FakeMonitor(procs or [])
    return WindowManager(api=api, monitor=mon), api, mon


def make_launcher(discovery, procs=None, popen=None):
    launched = []
    popen = popen or (lambda args, **kw: launched.append(args) or object())
    mon = FakeMonitor(procs or [])
    return ApplicationLauncher(discovery, popen=popen, monitor=mon), launched


def make_svc(**kw):
    disc = kw.get("discovery") or make_discovery()
    wins, api, mon = make_windows(
        kw.get("windows"), kw.get("procs")
    )
    launcher, launched = make_launcher(
        disc, kw.get("procs"), kw.get("popen")
    )
    monitor = kw.get("monitor") or mon
    svc = _Svc(disc, launcher, monitor, wins)
    return svc, disc, wins, api, launcher, launched


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------
class TestDiscovery:
    def test_seed_discovery_finds_apps(self):
        d = make_discovery()
        apps = d.discover()
        names = {a.name for a in apps}
        assert {"Google Chrome", "Notepad", "File Explorer"} <= names

    def test_find_by_name(self):
        d = make_discovery()
        assert d.find("chrome").name == "Google Chrome"
        assert d.find("note pad") is not None or d.find("notepad")

    def test_find_missing_returns_none(self):
        d = make_discovery()
        assert d.find("sublime text ultra") is None

    def test_invalid_exe_rejected(self):
        d = ApplicationDiscovery(
            env={}, file_exists=lambda p: False,
            registry_reader=lambda k: {},
        start_apps_lister=lambda: [],
        )
        assert d.discover() == []
        assert not d.is_app_path_allowed("C:\\fake\\app.exe")

    def test_non_exe_rejected(self):
        d = make_discovery()
        assert not d.is_app_path_allowed("C:\\readme.txt")

    def test_cache_and_refresh(self):
        calls = {"n": 0}
        d = ApplicationDiscovery(
            env={}, file_exists=lambda p: True,
            registry_reader=lambda k: {},
        start_apps_lister=lambda: [],
            extra_candidates={"TestApp": ["C:\\TestApp\\app.exe"]},
            cache_ttl=60,
        )
        d.discover()
        d.discover()  # cached
        assert d.find("testapp") is not None

    def test_app_paths_registry(self):
        def reader(key):
            if "App Paths" in key:
                return {"myapp.exe": {"": r"C:\Apps\myapp.exe"}}
            return {}
        d = ApplicationDiscovery(
            env={}, file_exists=lambda p: p.endswith("myapp.exe"),
            registry_reader=reader,
            start_apps_lister=lambda: [],
        )
        apps = d.discover()
        assert any(a.exe_path.endswith("myapp.exe") for a in apps)


# ----------------------------------------------------------------------
# Launcher / monitor / window manager units
# ----------------------------------------------------------------------
class TestLauncher:
    def test_launch_uses_list_form_no_shell(self):
        svc, disc, wins, api, launcher, launched = make_svc()
        app = launcher.resolve("chrome")
        ok, detail = launcher.launch(app)
        assert ok and launched == [[CHROME.exe_path]]

    def test_launch_missing_app(self):
        svc, disc, wins, api, launcher, launched = make_svc()
        assert launcher.resolve("nonexistent app") is None

    def test_launch_invalid_path_fails(self):
        disc = make_discovery(apps=[])
        bad = AppInfo("Bad", r"C:\nope\bad.exe")
        launcher, _ = make_launcher(disc)
        ok, detail = launcher.launch(bad)
        assert not ok and detail == "invalid_path"

    def test_monitor_find_and_running(self):
        mon = FakeMonitor([(100, "chrome.exe"), (200, "notepad.exe")])
        assert mon.is_running(CHROME)
        assert not mon.is_running(EXPLORER)
        assert mon.find("chrome")[0].pid == 100


class TestWindowManager:
    def _windows(self):
        return [
            WindowInfo(1, 100, "GitHub - Chrome", "chrome.exe"),
            WindowInfo(2, 100, "New Tab - Chrome", "chrome.exe"),
            WindowInfo(3, 200, "Untitled - Notepad", "notepad.exe"),
        ]

    def test_list_windows(self):
        wins, api, mon = make_windows(self._windows(),
                                      [(100, "chrome.exe"),
                                       (200, "notepad.exe")])
        out = wins.list_windows()
        assert len(out) == 3

    def test_multiple_windows_same_app(self):
        wins, api, mon = make_windows(self._windows(),
                                      [(100, "chrome.exe")])
        found = wins.windows_for("chrome")
        assert len(found) == 2  # both chrome windows

    def test_minimize_maximize_restore(self):
        wins, api, mon = make_windows(self._windows())
        wins.minimize(1); wins.maximize(2); wins.restore(3)
        assert api.shown == [(1, 6), (2, 3), (3, 9)]

    def test_focus_app(self):
        wins, api, mon = make_windows(self._windows(),
                                      [(100, "chrome.exe")])
        ok, title = wins.focus_app("chrome")
        assert ok and api.foreground_calls == [1]

    def test_focus_missing(self):
        wins, api, mon = make_windows([], [])
        ok, detail = wins.focus_app("chrome")
        assert not ok and detail == "not_running"

    def test_close_windows_posts_wm_close(self):
        wins, api, mon = make_windows(self._windows(),
                                      [(100, "chrome.exe")])
        n = wins.close_windows("chrome")
        assert n == 2 and api.closed == [1, 2]

    def test_show_desktop(self):
        wins, api, mon = make_windows()
        wins.show_desktop()
        assert api.desktop_shown == 1


# ----------------------------------------------------------------------
# Intents (Task 6 additions)
# ----------------------------------------------------------------------
class TestNewIntents:
    @pytest.fixture
    def rec(self):
        return RuleBasedIntentRecognizer()

    @pytest.mark.parametrize("text", [
        "Switch to Chrome.", "Bring notepad to front",
        "Go back to Chrome",
    ])
    def test_switch(self, rec, text):
        assert rec.recognize(text).name == "SWITCH_TO_APPLICATION"

    @pytest.mark.parametrize("text", [
        "Which applications are running?",
        "What apps are open?",
        "List running applications",
        "What's running?",
    ])
    def test_list_running(self, rec, text):
        assert rec.recognize(text).name == "LIST_RUNNING_APPLICATIONS"

    @pytest.mark.parametrize("text", [
        "Minimize Chrome.", "Minimize the window", "Minimize it",
    ])
    def test_minimize(self, rec, text):
        assert rec.recognize(text).name == "MINIMIZE_WINDOW"

    @pytest.mark.parametrize("text", [
        "Maximize Chrome.", "Maximize the window",
    ])
    def test_maximize(self, rec, text):
        assert rec.recognize(text).name == "MAXIMIZE_WINDOW"

    @pytest.mark.parametrize("text", [
        "Restore the window", "Restore notepad window",
        "Unmaximize",
    ])
    def test_restore(self, rec, text):
        assert rec.recognize(text).name == "RESTORE_WINDOW"

    def test_focus_window(self, rec):
        assert rec.recognize("Focus window").name == "FOCUS_WINDOW"
        assert rec.recognize("Focus Chrome").name == "FOCUS_WINDOW"

    @pytest.mark.parametrize("text", [
        "Show desktop", "Display the desktop", "Minimize all windows",
    ])
    def test_show_desktop(self, rec, text):
        assert rec.recognize(text).name == "SHOW_DESKTOP"

    @pytest.mark.parametrize("text", [
        "Is VS Code installed?", "Do you have Chrome?",
        "Find the application Spotify",
        "Where is notepad installed?",
    ])
    def test_find_app(self, rec, text):
        assert rec.recognize(text).name == "FIND_APPLICATION"


# ----------------------------------------------------------------------
# Handlers via registry
# ----------------------------------------------------------------------
class TestAppHandlers:
    def test_open_app_executes(self):
        svc, *_ = make_svc()
        reg = create_app_registry(svc)
        r = reg.dispatch(
            __import__("src.commands.intent", fromlist=["Intent"]).Intent(
                "OPEN_APPLICATION", {"app_name": "chrome"}
            )
        )
        assert r.executed and "Opening Google Chrome" in r.message

    def test_open_missing_app(self):
        svc, *_ = make_svc()
        reg = create_app_registry(svc)
        from src.commands.intent import Intent
        r = reg.dispatch(Intent("OPEN_APPLICATION", {"app_name": "spotify"}))
        assert not r.success and r.error == "application_not_found"

    def test_close_requires_confirmation(self):
        svc, *_ = make_svc()
        reg = create_app_registry(svc)
        from src.commands.intent import Intent
        r = reg.dispatch(Intent("CLOSE_APPLICATION", {"app_name": "notepad"}))
        assert r.confirmation_required and not r.executed
        assert "close notepad" in r.message.lower()

    def test_close_after_confirm(self):
        svc, disc, wins, api, launcher, launched = make_svc(
            windows=[WindowInfo(5, 200, "x", "notepad.exe")],
            procs=[(200, "notepad.exe")],
        )
        reg = create_app_registry(svc)
        from src.commands.intent import Intent
        r = reg.dispatch(Intent(
            "CLOSE_APPLICATION",
            {"app_name": "notepad", "_confirmed": True},
        ))
        assert r.executed and "Closing Notepad" in r.message
        assert api.closed == [5]

    def test_close_not_running(self):
        svc, *_ = make_svc(procs=[])
        reg = create_app_registry(svc)
        from src.commands.intent import Intent
        r = reg.dispatch(Intent(
            "CLOSE_APPLICATION",
            {"app_name": "notepad", "_confirmed": True},
        ))
        assert r.executed and "doesn't appear to be running" in r.message

    def test_switch_to_app(self):
        svc, *_ = make_svc(
            windows=[WindowInfo(5, 100, "Chrome", "chrome.exe")],
            procs=[(100, "chrome.exe")],
        )
        reg = create_app_registry(svc)
        from src.commands.intent import Intent
        r = reg.dispatch(Intent("SWITCH_TO_APPLICATION", {"app_name": "chrome"}))
        assert r.executed and "Switching to Google Chrome" in r.message

    def test_switch_not_running(self):
        svc, *_ = make_svc()
        reg = create_app_registry(svc)
        from src.commands.intent import Intent
        r = reg.dispatch(Intent("SWITCH_TO_APPLICATION", {"app_name": "chrome"}))
        assert not r.success and r.error == "not_running"

    def test_find_app(self):
        svc, *_ = make_svc()
        reg = create_app_registry(svc)
        from src.commands.intent import Intent
        r = reg.dispatch(Intent("FIND_APPLICATION", {"app_name": "chrome"}))
        assert r.executed and "found Google Chrome" in r.message
        r2 = reg.dispatch(Intent("FIND_APPLICATION", {"app_name": "spotify"}))
        assert "couldn't find" in r2.message

    def test_list_running(self):
        svc, *_ = make_svc(
            windows=[
                WindowInfo(1, 100, "a", "chrome.exe"),
                WindowInfo(2, 100, "b", "chrome.exe"),
                WindowInfo(3, 200, "c", "notepad.exe"),
            ],
            procs=[(100, "chrome.exe"), (200, "notepad.exe")],
        )
        reg = create_app_registry(svc)
        from src.commands.intent import Intent
        r = reg.dispatch(Intent("LIST_RUNNING_APPLICATIONS", {}))
        assert r.executed
        assert "chrome" in r.message.lower() and "notepad" in r.message.lower()

    def test_minimize_window_handler(self):
        svc, disc, wins, api, launcher, launched = make_svc(
            windows=[WindowInfo(1, 100, "Chrome", "chrome.exe")],
            procs=[(100, "chrome.exe")],
        )
        reg = create_app_registry(svc)
        from src.commands.intent import Intent
        r = reg.dispatch(Intent("MINIMIZE_WINDOW", {"target": "chrome"}))
        assert r.executed and "Minimizing Google Chrome" in r.message
        assert api.shown == [(1, 6)]

    def test_window_op_not_running(self):
        svc, *_ = make_svc()
        reg = create_app_registry(svc)
        from src.commands.intent import Intent
        r = reg.dispatch(Intent("MAXIMIZE_WINDOW", {"target": "chrome"}))
        assert not r.success and r.error == "not_running"

    def test_window_op_no_target_uses_foreground(self):
        svc, disc, wins, api, launcher, launched = make_svc(
            windows=[WindowInfo(9, 300, "fg", "app.exe")],
        )
        api._fg = 9
        reg = create_app_registry(svc)
        from src.commands.intent import Intent
        r = reg.dispatch(Intent("FOCUS_WINDOW", {"target": ""}))
        assert r.executed

    def test_show_desktop_handler(self):
        svc, disc, wins, api, launcher, launched = make_svc()
        reg = create_app_registry(svc)
        from src.commands.intent import Intent
        r = reg.dispatch(Intent("SHOW_DESKTOP", {}))
        assert r.executed and api.desktop_shown == 1

    def test_time_still_executes(self):
        svc, *_ = make_svc()
        reg = create_app_registry(svc)
        from src.commands.intent import Intent
        r = reg.dispatch(Intent("GET_CURRENT_TIME", {}))
        assert r.executed and "current time" in r.message.lower()


# ----------------------------------------------------------------------
# Session integration: confirmation flow
# ----------------------------------------------------------------------
class _StubSTT:
    def __init__(self):
        self._cb = {}

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self): pass
    def stop(self): pass

    @property
    def is_running(self): return True

    def emit(self, text): self._cb["on_transcript"](text)


class _StubTTS:
    def __init__(self):
        self.spoken = []
        self._cb = {}

    def set_callbacks(self, **kw):
        self._cb.update(kw)

    def start(self): pass
    def stop(self): pass

    @property
    def is_running(self): return True

    def speak(self, text):
        self.spoken.append(text)
        self._cb["on_finished"](text)


def _controller(svc):
    state = StateManager()
    state.transition_to(AppState.STANDBY)
    ctl = VoiceSessionController(
        state_manager=state,
        session_manager=SessionManager(timeout_seconds=60),
        speech_recognition=_StubSTT(),
        speech_output=_StubTTS(),
        command_registry=create_app_registry(svc),
        intent_recognizer=RuleBasedIntentRecognizer(),
    )
    return ctl


class TestSessionIntegration:
    def test_open_chrome_speaks_opening(self):
        svc, *_ = make_svc()
        ctl = _controller(svc)
        ctl._stt.emit if False else None
        stt = ctl._stt
        ctl.handle_wake_word()
        ctl._tts.spoken.clear()
        stt.emit("Open Chrome.")
        assert any("Opening Google Chrome" in s for s in ctl._tts.spoken)
        ctl.shutdown()

    def test_close_asks_confirmation_then_yes_closes(self):
        svc, disc, wins, api, launcher, launched = make_svc(
            windows=[WindowInfo(5, 200, "n", "notepad.exe")],
            procs=[(200, "notepad.exe")],
        )
        ctl = _controller(svc)
        stt, tts = ctl._stt, ctl._tts
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("Close Notepad.")
        assert any("Should I close Notepad" in s for s in tts.spoken)
        assert api.closed == []
        tts.spoken.clear()
        stt.emit("Yes.")
        assert any("Closing Notepad" in s for s in tts.spoken)
        assert api.closed == [5]
        ctl.shutdown()

    def test_close_confirmation_no_aborts(self):
        svc, disc, wins, api, launcher, launched = make_svc(
            windows=[WindowInfo(5, 200, "n", "notepad.exe")],
            procs=[(200, "notepad.exe")],
        )
        ctl = _controller(svc)
        stt, tts = ctl._stt, ctl._tts
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("Close Notepad.")
        tts.spoken.clear()
        stt.emit("No.")
        assert any("won't close" in s for s in tts.spoken)
        assert api.closed == []
        ctl.shutdown()

    def test_list_running_speaks_apps(self):
        svc, *_ = make_svc(
            windows=[
                WindowInfo(1, 100, "a", "chrome.exe"),
                WindowInfo(3, 200, "c", "notepad.exe"),
            ],
        )
        ctl = _controller(svc)
        stt, tts = ctl._stt, ctl._tts
        ctl.handle_wake_word()
        tts.spoken.clear()
        stt.emit("Which applications are running?")
        assert any("currently have" in s for s in tts.spoken)
        ctl.shutdown()
