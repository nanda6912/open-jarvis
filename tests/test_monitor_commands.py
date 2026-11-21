"""Tests for monitor voice commands: move window, open-on-monitor,
monitor queries — all Win32 APIs mocked."""

from __future__ import annotations

import pytest

from src.commands.app_handlers import _Svc, create_app_registry
from src.commands.intent import RuleBasedIntentRecognizer
from src.commands.monitor_handlers import create_monitor_handlers
from src.commands.registry import CommandRegistry
from src.controllers.application_discovery import (
    AppInfo,
    ApplicationDiscovery,
)
from src.controllers.monitor_manager import (
    MonitorInfo,
    MonitorManager,
    MonitorRect,
)
from src.controllers.windows_automation import (
    ApplicationLauncher,
    ProcessMonitor,
    WindowInfo,
    WindowManager,
)

CHROME = AppInfo(
    "Google Chrome",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
)
NOTEPAD = AppInfo("Notepad", r"C:\Windows\System32\notepad.exe")
VSCODE = AppInfo(
    "Visual Studio Code",
    r"C:\Users\x\AppData\Local\Programs\Microsoft VS Code\Code.exe",
)


def _mon(index, device, bounds, work=None, primary=False, handle=None):
    return MonitorInfo(
        index=index, device=device,
        bounds=MonitorRect(*bounds),
        work_area=MonitorRect(*(work or bounds)),
        primary=primary, handle=handle or index,
    )


# Two physical displays: external LEFT (not primary), laptop RIGHT
# (primary) — matching the target machine's arrangement. Logical
# numbering (MonitorManager): Monitor 1 = primary/right DISPLAY1,
# Monitor 2 = left DISPLAY2. Fake handles stay positional: handle 1 =
# DISPLAY2 (left), handle 2 = DISPLAY1 (right/primary).
MONS = [
    _mon(1, "\\\\.\\DISPLAY2", (-1920, 0, 0, 1080), primary=False),
    _mon(2, "\\\\.\\DISPLAY1", (0, 0, 1920, 1080), primary=True),
]


class FakeWinApi:
    """Fake user32 surface used by WindowManager/monitor ops."""

    def __init__(self, windows=None):
        self.windows = list(windows or [])  # WindowInfo
        self.rects = {}      # hwnd -> (l,t,r,b)
        self.show_cmd = {}   # hwnd -> showCmd int
        self.moved = []      # (hwnd, rect)
        self.procs = []      # (pid, name)
        self._fg = self.windows[0].hwnd if self.windows else 0

    # -- window enumeration (WindowManager) -------------------------------
    def enum_windows(self, callback):
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
        self.show_cmd[hwnd] = cmd

    def set_foreground(self, hwnd):
        return True

    def post_close(self, hwnd):
        pass

    def foreground_hwnd(self):
        return self._fg

    def ex_style(self, hwnd):
        return 0

    def show_desktop(self):
        pass

    # -- geometry ----------------------------------------------------------
    def window_rect(self, hwnd):
        return self.rects.get(hwnd, (100, 100, 500, 400))

    def set_window_rect(self, hwnd, rect, activate=False):
        self.rects[hwnd] = tuple(rect)
        self.moved.append((hwnd, tuple(rect)))
        return True

    def window_show_cmd(self, hwnd):
        return self.show_cmd.get(hwnd, 1)

    def normal_rect(self, hwnd):
        return self.rects.get(hwnd, (100, 100, 500, 400))


class FakeMonitorApi:
    def __init__(self, mons, rects=None):
        self._mons = mons
        self.hwnd_map = {}
        self._rects = rects if rects is not None else {}

    def enum_monitors(self):
        return [m.handle for m in self._mons]

    def monitor_info(self, hmon):
        for m in self._mons:
            if m.handle == hmon:
                return m.bounds, m.work_area, m.primary, m.device
        raise KeyError(hmon)

    def monitor_for_hwnd(self, hwnd):
        # Prefer the window's current rect center (post-move position).
        rect = self._rects.get(hwnd)
        if rect is not None:
            return self.monitor_for_point(
                (rect[0] + rect[2]) // 2, (rect[1] + rect[3]) // 2
            )
        return self.hwnd_map.get(hwnd, 0)

    def monitor_for_point(self, x, y):
        for m in self._mons:
            b = m.bounds
            if b.left <= x < b.right and b.top <= y < b.bottom:
                return m.handle
        return 0


class FakeProc:
    def __init__(self, pid):
        self.pid = pid


def make_svc(windows=None, mons=None, apps=None, popen=None, sleep=None):
    mons = mons if mons is not None else MONS
    apps = apps if apps is not None else [CHROME, NOTEPAD, VSCODE]
    api = FakeWinApi(windows)
    mon_api = FakeMonitorApi(mons, rects=api.rects)
    proc_mon = ProcessMonitor(lister=lambda: api.procs)
    win_api = api
    wm = WindowManager(api=win_api, monitor=proc_mon)
    mm = MonitorManager(api=mon_api)
    valid = {a.exe_path.lower() for a in apps}
    disc = ApplicationDiscovery(
        env={}, file_exists=lambda p: p.lower() in valid,
        registry_reader=lambda k: {}, cache_ttl=0,
        extra_candidates={a.name: [a.exe_path] for a in apps},
    )
    launcher = ApplicationLauncher(
        discovery=disc,
        popen=popen or (lambda cmd: FakeProc(4242)),
        monitor=proc_mon,
    )
    svc = _Svc(disc, launcher, proc_mon, wm, monitors=mm)
    svc._test_api = api
    svc._test_monitors = mons
    return svc


def _win(hwnd, pid, title, process):
    return WindowInfo(hwnd=hwnd, pid=pid, title=title, process=process)


def _dispatch(svc, text):
    intent = RuleBasedIntentRecognizer().recognize(text)
    reg = CommandRegistry()
    for name, h in create_monitor_handlers(svc).items():
        reg.register(name, h)
    # Also register app/window handlers for combined flows.
    from src.commands.app_handlers import create_app_registry
    full = create_app_registry(svc)
    for n in full.registered_intents:
        reg.register(n, full._handlers[n])
    return intent, reg.dispatch(intent)


class TestIntentParsing:
    @pytest.mark.parametrize("text,name", [
        ("List my monitors.", "LIST_MONITORS"),
        ("How many monitors are connected?", "LIST_MONITORS"),
        ("Which monitor is primary?", "GET_PRIMARY_MONITOR"),
        ("Move Chrome to monitor 2.", "MOVE_WINDOW_TO_MONITOR"),
        ("Move VS Code to my second screen.", "MOVE_WINDOW_TO_MONITOR"),
        ("Move this window to the other monitor.",
         "MOVE_WINDOW_TO_OTHER_MONITOR"),
        ("Move Notepad to the left monitor.", "MOVE_WINDOW_TO_MONITOR"),
        ("Move Chrome to the primary monitor.",
         "MOVE_WINDOW_TO_PRIMARY_MONITOR"),
        ("Move this window to my laptop screen.", "MOVE_WINDOW_TO_MONITOR"),
        ("Move Spotify to the external monitor.", "MOVE_WINDOW_TO_MONITOR"),
        ("Which monitor is Chrome on?", "GET_WINDOW_MONITOR"),
        ("What monitor is VS Code using?", "GET_WINDOW_MONITOR"),
        ("Open Chrome on monitor 2.", "OPEN_APPLICATION_ON_MONITOR"),
        ("Open VS Code on my second screen.", "OPEN_APPLICATION_ON_MONITOR"),
        ("Launch Spotify on the left monitor.",
         "OPEN_APPLICATION_ON_MONITOR"),
        ("Open Notepad on my laptop display.",
         "OPEN_APPLICATION_ON_MONITOR"),
        ("Open Chrome on the primary monitor.",
         "OPEN_APPLICATION_ON_MONITOR"),
    ])
    def test_intent_names(self, text, name):
        intent = RuleBasedIntentRecognizer().recognize(text)
        assert intent.name == name, f"{text!r} -> {intent.name}"

    def test_open_on_monitor_entities(self):
        intent = RuleBasedIntentRecognizer().recognize(
            "Open Chrome on monitor 2."
        )
        assert "chrome" in intent.entities["app_name"].lower()
        assert intent.entities["monitor"]

    def test_move_entities(self):
        intent = RuleBasedIntentRecognizer().recognize(
            "Move Chrome to the left monitor."
        )
        assert intent.entities["target"].lower() == "chrome"
        assert "left" in intent.entities["monitor"]

    def test_plain_open_still_works(self):
        intent = RuleBasedIntentRecognizer().recognize("Open Chrome.")
        assert intent.name == "OPEN_APPLICATION"


class TestMonitorQueries:
    def test_list_monitors(self):
        svc = make_svc()
        _i, res = _dispatch(svc, "List my monitors.")
        assert res.success and res.executed
        assert "2 monitors" in res.message
        assert "left" in res.message and "right" in res.message

    def test_primary_monitor(self):
        svc = make_svc()
        _i, res = _dispatch(svc, "Which monitor is primary?")
        assert res.success
        assert "primary" in res.message.lower()
        assert "monitor 1" in res.message  # primary = logical 1

    def test_window_monitor(self):
        svc = make_svc(windows=[
            _win(11, 100, "Chrome", "chrome.exe"),
        ])
        svc._test_monitors[0]  # noqa
        svc._test_api = svc._test_api
        # hwnd 11 lives on the left monitor (handle 1 = DISPLAY2,
        # logical monitor 2 under the primary-first numbering).
        svc._svc_monitors = None
        # Wire hwnd->monitor through the fake monitor api.
        svc.monitors._api.hwnd_map[11] = 1
        _i, res = _dispatch(svc, "Which monitor is Chrome on?")
        assert res.success
        assert "monitor 2" in res.message


class TestMoveWindow:
    def _svc_with_chrome(self, **kw):
        api_windows = kw.pop(
            "windows",
            [_win(11, 100, "Chrome", "chrome.exe")],
        )
        svc = make_svc(windows=api_windows, **kw)
        # Default: window starts on the LEFT monitor (negative X).
        svc._test_api.rects[11] = (-1700, 100, -1300, 400)
        return svc

    def test_move_to_monitor_1_is_right_primary(self):
        """'monitor one' = logical Monitor 1 = right/primary display."""
        svc = self._svc_with_chrome()
        svc.monitors._api.hwnd_map[11] = 1  # starts on left monitor
        _i, res = _dispatch(svc, "Move Chrome to monitor 1.")
        assert res.success and res.executed, res.message
        assert "monitor 1" in res.message or "right" in res.message
        # Window rect must now be inside the right monitor's work area.
        rect = svc._test_api.rects[11]
        assert rect[0] >= 0 and rect[2] <= 1920

    def test_move_to_monitor_2_is_left(self):
        """'monitor two' = logical Monitor 2 = left display."""
        svc = self._svc_with_chrome()
        svc._test_api.rects[11] = (100, 100, 500, 400)  # start: right
        svc.monitors._api.hwnd_map[11] = 2  # starts on right
        _i, res = _dispatch(svc, "Move Chrome to monitor 2.")
        assert res.success and res.executed, res.message
        rect = svc._test_api.rects[11]
        assert rect[2] <= 0  # entirely on the left display

    def test_move_to_left_monitor(self):
        svc = self._svc_with_chrome()
        svc._test_api.rects[11] = (100, 100, 500, 400)  # start: right
        svc.monitors._api.hwnd_map[11] = 2  # starts on right
        _i, res = _dispatch(svc, "Move Chrome to the left monitor.")
        assert res.success and res.executed, res.message
        rect = svc._test_api.rects[11]
        assert rect[2] <= 0  # entirely on the left display

    def test_move_this_window_other_monitor(self):
        fg = _win(22, 200, "Notepad", "notepad.exe")
        svc = make_svc(windows=[fg])
        svc.monitors._api.hwnd_map[22] = 2
        _i, res = _dispatch(svc, "Move this window to the other monitor.")
        assert res.success and res.executed, res.message
        rect = svc._test_api.rects[22]
        assert rect[2] <= 0  # moved to left monitor (negative X)

    def test_move_maximized_window(self):
        svc = self._svc_with_chrome()
        svc._test_api.show_cmd[11] = 3  # maximized
        svc.monitors._api.hwnd_map[11] = 1
        _i, res = _dispatch(svc, "Move Chrome to monitor 1.")
        assert res.success and res.executed, res.message
        # Maximized state reapplied.
        assert svc._test_api.show_cmd.get(11) == 3

    def test_move_minimized_window_restored(self):
        svc = self._svc_with_chrome()
        svc._test_api.show_cmd[11] = 2  # minimized
        svc.monitors._api.hwnd_map[11] = 1
        _i, res = _dispatch(svc, "Move Chrome to monitor 1.")
        assert res.success and res.executed, res.message
        assert svc._test_api.show_cmd.get(11) == 9  # SW_RESTORE seen

    def test_move_missing_app(self):
        svc = make_svc(windows=[])
        _i, res = _dispatch(svc, "Move Chrome to monitor 2.")
        assert not res.success
        assert res.error == "not_running"

    def test_move_invalid_monitor(self):
        svc = self._svc_with_chrome()
        _i, res = _dispatch(svc, "Move Chrome to monitor 9.")
        assert not res.executed

    def test_already_on_monitor(self):
        svc = self._svc_with_chrome()
        svc._test_api.rects[11] = (-1700, 100, -1300, 400)  # already left
        svc.monitors._api.hwnd_map[11] = 1
        _i, res = _dispatch(svc, "Move Chrome to monitor 2.")
        assert res.success
        assert "already" in res.message


class TestOpenOnMonitor:
    def test_open_chrome_on_monitor_2(self):
        launched = []

        def popen(cmd):
            launched.append(cmd)
            return FakeProc(777)

        svc = make_svc(popen=popen)
        # Window appears for pid 777 after "launch".
        orig_windows_for_pid = svc.windows.windows_for_pid

        def wfp(pid):
            if pid == 777:
                return [_win(55, 777, "Chrome", "chrome.exe")]
            return orig_windows_for_pid(pid)

        svc.windows.windows_for_pid = wfp
        svc.monitors._api.hwnd_map[55] = 1
        _i, res = _dispatch(svc, "Open Chrome on monitor 2.")
        assert res.success and res.executed, res.message
        # Monitor 2 = left display under the primary-first numbering.
        assert "left" in res.message or "monitor 2" in res.message
        assert launched  # exe launched via popen, no shell

    def test_open_on_left_monitor(self):
        svc = make_svc()
        svc.windows.windows_for_pid = lambda pid: [
            _win(56, 777, "Notepad", "notepad.exe")
        ]
        svc.monitors._api.hwnd_map[56] = 2
        _i, res = _dispatch(svc, "Open Notepad on the left monitor.")
        assert res.success and res.executed, res.message
        rect = svc._test_api.rects[56]
        assert rect[2] <= 0

    def test_open_app_not_found(self):
        svc = make_svc()
        _i, res = _dispatch(svc, "Open Blorp on monitor 2.")
        assert not res.success
        assert res.error == "application_not_found"

    def test_open_missing_monitor_spec(self):
        svc = make_svc()
        intent = RuleBasedIntentRecognizer().recognize(
            "Open Chrome on monitor 2."
        )
        intent.entities["monitor"] = ""
        reg = create_app_registry(svc)
        res = reg.dispatch(intent)
        assert not res.success
        assert res.error == "missing_monitor"

    def test_open_window_timeout_still_reports(self):
        svc = make_svc(
            sleep=lambda s: None,
        )
        svc.windows.windows_for_pid = lambda pid: []
        svc.windows.windows_for = lambda t: []
        # Shrink timeout via handler re-creation.
        from src.commands.monitor_handlers import OpenOnMonitorHandler
        h = OpenOnMonitorHandler(svc, window_timeout=0.0,
                                 sleep=lambda s: None,
                                 monotonic=__import__("time").monotonic)
        reg = CommandRegistry()
        reg.register("OPEN_APPLICATION_ON_MONITOR", h)
        intent = RuleBasedIntentRecognizer().recognize(
            "Open Chrome on monitor 2."
        )
        res = reg.dispatch(intent)
        assert res.success and res.executed
        assert "couldn't find its window" in res.message

    def test_existing_instance_new_window(self):
        """App reuses a process: a NEW window for the same exe appears."""
        svc = make_svc()
        old = _win(60, 900, "Chrome", "chrome.exe")
        new = _win(61, 900, "Chrome", "chrome.exe")
        state = {"spawned": False}

        def list_windows(include_hidden=False):
            return [old] + ([new] if state["spawned"] else [])

        svc.windows.list_windows = list_windows
        svc.windows.windows_for_pid = lambda pid: []

        real_launch = svc.launcher.launch_process

        def launch(app):
            state["spawned"] = True
            return real_launch(app)

        svc.launcher.launch_process = launch
        svc.monitors._api.hwnd_map[61] = 2
        _i, res = _dispatch(svc, "Open Chrome on monitor 2.")
        assert res.success and res.executed, res.message
        # Only the NEW window was moved.
        assert svc._test_api.rects.get(61) is not None
        assert svc._test_api.rects.get(60) is None


class TestDefaultMonitorLaunch:
    """Plain OPEN_APPLICATION must place the window on Monitor 1."""

    def _svc(self, windows=(), popen=None, mons=None):
        svc = make_svc(
            windows=list(windows), mons=mons, popen=popen,
        )
        return svc

    def test_open_defaults_to_monitor_1(self):
        """Plain OPEN_APPLICATION targets logical Monitor 1, which is
        the Windows PRIMARY/right display under the primary-first
        numbering."""
        svc = self._svc()
        svc.windows.windows_for_pid = lambda pid: [
            _win(70, 777, "Chrome", "chrome.exe")
        ]
        _i, res = _dispatch(svc, "Open Chrome.")
        assert res.success and res.executed, res.message
        # Primary monitor in the fixture is index 2 (right, x >= 0).
        rect = svc._test_api.rects[70]
        assert rect[0] >= 0 and rect[2] <= 1920, rect
        assert (
            "primary" in res.message.lower()
            or "right" in res.message.lower()
            or "monitor 2" in res.message
        )

    def test_open_explicit_monitor_2_overrides(self):
        """Explicit 'monitor two' = left display, overriding default."""
        svc = self._svc()
        svc.windows.windows_for_pid = lambda pid: [
            _win(71, 777, "Chrome", "chrome.exe")
        ]
        _i, res = _dispatch(svc, "Open Chrome on monitor 2.")
        assert res.success and res.executed, res.message
        rect = svc._test_api.rects[71]
        assert rect[2] <= 0  # left display

    def test_open_left_alias_is_monitor_2(self):
        svc = self._svc()
        svc.windows.windows_for_pid = lambda pid: [
            _win(72, 777, "Chrome", "chrome.exe")
        ]
        _i, res = _dispatch(svc, "Open Chrome on the left monitor.")
        assert res.success and res.executed, res.message
        assert svc._test_api.rects[72][2] <= 0

    def test_open_secondary_alias_is_monitor_2(self):
        svc = self._svc()
        svc.windows.windows_for_pid = lambda pid: [
            _win(73, 777, "Chrome", "chrome.exe")
        ]
        _i, res = _dispatch(svc, "Open Chrome on the secondary monitor.")
        assert res.success and res.executed, res.message
        assert svc._test_api.rects[73][2] <= 0

    def test_open_primary_alias_is_monitor_1(self):
        svc = self._svc()
        svc.windows.windows_for_pid = lambda pid: [
            _win(74, 777, "Chrome", "chrome.exe")
        ]
        _i, res = _dispatch(svc, "Open Chrome on the primary monitor.")
        assert res.success and res.executed, res.message
        assert svc._test_api.rects[74][0] >= 0

    def test_single_monitor_uses_only_display(self):
        single = [_mon(1, "\\\\.\\DISPLAY1", (0, 0, 1920, 1080),
                       primary=True)]
        svc = self._svc(mons=single)
        svc.windows.windows_for_pid = lambda pid: [
            _win(75, 777, "Notepad", "notepad.exe")
        ]
        _i, res = _dispatch(svc, "Open Notepad.")
        assert res.success and res.executed, res.message
        rect = svc._test_api.rects[75]
        assert 0 <= rect[0] and rect[2] <= 1920

    def test_monitor_failure_falls_back_to_plain_launch(self):
        """Discovery failure must not claim Monitor-1 placement."""
        svc = self._svc()

        class BadMonitors:
            def monitors(self, refresh=True):
                raise RuntimeError("display api gone")

            def resolve(self, *a, **k):
                raise RuntimeError("display api gone")

            def primary(self):
                raise RuntimeError("display api gone")

        svc.monitors = BadMonitors()
        launched = []
        svc.launcher._popen = lambda cmd: launched.append(cmd) or FakeProc(1)
        _i, res = _dispatch(svc, "Open Chrome.")
        assert res.success and res.executed
        assert launched  # app still opened via secure launcher
        assert "monitor" not in res.message.lower()

    def test_unrelated_window_never_moved(self):
        other = _win(80, 999, "Spotify", "spotify.exe")
        svc = self._svc(windows=[other])
        svc._test_api.rects[80] = (100, 100, 500, 400)
        svc.windows.windows_for_pid = lambda pid: [
            _win(81, 777, "Chrome", "chrome.exe")
        ]
        _i, res = _dispatch(svc, "Open Chrome.")
        assert res.success and res.executed, res.message
        assert svc._test_api.rects.get(80) == (100, 100, 500, 400)
        assert svc._test_api.rects.get(81) is not None

    def test_failed_placement_not_reported_as_success(self):
        svc = self._svc()
        svc.windows.windows_for_pid = lambda pid: [
            _win(82, 777, "Chrome", "chrome.exe")
        ]
        # Force move failure.
        svc.windows.move_to_work_area = lambda h, w, margin=0: (False, False)
        _i, res = _dispatch(svc, "Open Chrome.")
        assert res.executed
        assert "couldn't verify" in res.message or "couldn't" in res.message

    def test_existing_instance_window_reused_safely(self):
        """Existing app instance: only the NEW window is moved."""
        old = _win(90, 900, "Chrome", "chrome.exe")
        svc = self._svc(windows=[old])
        svc._test_api.rects[90] = (100, 100, 500, 400)
        state = {"spawned": False}
        new = _win(91, 900, "Chrome", "chrome.exe")

        def list_windows(include_hidden=False):
            return [old] + ([new] if state["spawned"] else [])

        svc.windows.list_windows = list_windows
        svc.windows.windows_for_pid = lambda pid: []
        real = svc.launcher.launch_process

        def launch(app):
            state["spawned"] = True
            return real(app)

        svc.launcher.launch_process = launch
        _i, res = _dispatch(svc, "Open Chrome.")
        assert res.success and res.executed, res.message
        assert svc._test_api.rects.get(91) is not None
        assert svc._test_api.rects.get(90) == (100, 100, 500, 400)

    def test_default_launch_follows_primary_not_index(self):
        """Default launch targets PRIMARY even when it is monitor 1."""
        flipped = [
            _mon(1, "\\\\.\\DISPLAY2", (-1920, 0, 0, 1080), primary=True),
            _mon(2, "\\\\.\\DISPLAY1", (0, 0, 1920, 1080), primary=False),
        ]
        svc = self._svc(mons=flipped)
        svc.windows.windows_for_pid = lambda pid: [
            _win(95, 777, "Chrome", "chrome.exe")
        ]
        _i, res = _dispatch(svc, "Open Chrome.")
        assert res.success and res.executed, res.message
        # Primary is the LEFT display in this layout, so the default
        # launch still lands on negative X — via primary, not index 1.
        assert svc._test_api.rects[95][2] <= 0


class TestMonitorClarifyFlow:
    def test_restore_from_asks_target(self):
        """'Restore from.' must not treat 'from' as an app."""
        svc = make_svc()
        intent = RuleBasedIntentRecognizer().recognize("Restore from.")
        assert intent.name == "RESTORE_WINDOW"
        assert intent.entities.get("target", "") in ("", None)
        reg = create_app_registry(svc)
        res = reg.dispatch(intent)
        assert res.success and not res.executed
        assert "restore" in res.message.lower()
        assert res.data.get("clarify") is not None

    def test_window_target_never_from(self):
        for text in ("Restore from.", "Restore the from window.",
                     "Minimize from.", "Maximize from it."):
            intent = RuleBasedIntentRecognizer().recognize(text)
            tgt = (intent.entities.get("target") or "").lower()
            assert tgt not in ("from", "it"), (text, tgt)

    def test_missing_target_clarify_reply(self):
        """Controller-level: clarification reply re-dispatches intent."""
        from src.core.session_manager import SessionManager
        from src.core.state_manager import AppState, StateManager
        from src.voice.voice_session_controller import (
            VoiceSessionController,
        )

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

        svc = make_svc(windows=[
            _win(11, 100, "Chrome", "chrome.exe"),
        ])
        svc._test_api.rects[11] = (100, 100, 500, 400)
        state = StateManager(); state.transition_to(AppState.STANDBY)
        ctl = VoiceSessionController(
            state_manager=state,
            session_manager=SessionManager(timeout_seconds=60),
            speech_recognition=STT(), speech_output=TTS(),
            command_registry=create_app_registry(svc),
        )
        ctl.handle_wake_word()
        ctl._tts.spoken.clear()
        ctl._stt.emit("Restore from.")
        assert any("which application" in s.lower()
                   for s in ctl._tts.spoken)
        ctl._stt.emit("Chrome")
        assert any("restor" in s.lower() for s in ctl._tts.spoken)
        ctl.shutdown()
