"""Regression tests for normalized app identity and window management.

Covers the physical-log failures:
- switching to running apps (incl. packaged apps hosted by
  ApplicationFrameHost) resolving by identity, not PFN fragments
- honest focus failure vs. "not running"
- default launch onto the PRIMARY monitor
- verified window moves with actual-destination reporting
- multi-window clarification: ordinal/title selection and cancel
"""

from __future__ import annotations

import pytest

from src.commands.app_handlers import _Svc, create_app_registry
from src.commands.intent import Intent, RuleBasedIntentRecognizer
from src.commands.monitor_handlers import create_monitor_handlers
from src.commands.registry import CommandRegistry
from src.controllers.app_identity import (
    identity_for_app,
    window_matches_app,
)
from src.controllers.application_discovery import (
    AppInfo,
    ApplicationDiscovery,
)
from src.controllers.monitor_manager import MonitorManager
from src.controllers.windows_automation import (
    ApplicationLauncher,
    ProcessMonitor,
    WindowInfo,
    WindowManager,
)
from tests.test_monitor_commands import (
    CHROME,
    MONS,
    NOTEPAD,
    FakeMonitorApi,
    FakeProc,
    FakeWinApi,
    _mon,
    _win,
)

WHATSAPP_AUMID = "5319275A.WhatsAppDesktop_cv1g1gvanyjgm!App"
DEVIN = AppInfo(
    "Devin", r"C:\Users\user\AppData\Local\Programs\Devin\Devin.exe"
)


def make_identity_svc(
    windows=None,
    procs=None,
    apps=None,
    start_apps=None,
    mons=None,
    popen=None,
):
    """_Svc wired like test_monitor_commands.make_svc but with
    start_apps (packaged-app) discovery support."""
    mons = mons if mons is not None else MONS
    apps = apps if apps is not None else [CHROME, NOTEPAD, DEVIN]
    api = FakeWinApi(windows or [])
    mon_api = FakeMonitorApi(mons, rects=api.rects)
    api.procs = list(procs or [])
    proc_mon = ProcessMonitor(
        lister=lambda: api.procs,
        pid_details=lambda pid: ("", ""),
    )
    wm = WindowManager(api=api, monitor=proc_mon)
    mm = MonitorManager(api=mon_api)
    valid = {a.exe_path.lower() for a in apps if a.exe_path}
    disc = ApplicationDiscovery(
        env={}, file_exists=lambda p: p.lower() in valid,
        registry_reader=lambda k: {}, cache_ttl=0,
        extra_candidates={
            a.name: [a.exe_path] for a in apps if a.exe_path
        },
        start_apps_lister=(
            (lambda: list(start_apps)) if start_apps is not None
            else (lambda: [])
        ),
    )
    launcher = ApplicationLauncher(
        discovery=disc,
        popen=popen or (lambda cmd: FakeProc(4242)),
        monitor=proc_mon,
    )
    svc = _Svc(disc, launcher, proc_mon, wm, monitors=mm)
    svc._test_api = api
    return svc


def _dispatch(svc, text):
    intent = RuleBasedIntentRecognizer().recognize(text)
    reg = CommandRegistry()
    for name, h in create_monitor_handlers(svc).items():
        reg.register(name, h)
    full = create_app_registry(svc)
    for n in full.registered_intents:
        reg.register(n, full._handlers[n])
    return intent, reg.dispatch(intent)


class TestPackagedAppIdentity:
    """WhatsApp-style MSIX apps: PFN!App launch ids, windows hosted
    by ApplicationFrameHost."""

    def _svc(self, windows):
        return make_identity_svc(
            windows=windows,
            procs=[(300, "ApplicationFrameHost.exe")],
            start_apps=[("WhatsApp", WHATSAPP_AUMID)],
        )

    def test_identity_never_uses_pfn_fragment(self):
        app = AppInfo(
            "WhatsApp", "", source="packaged",
            launch_kind="aumid", aumid=WHATSAPP_AUMID,
        )
        ident = identity_for_app(app)
        assert ident.name == "WhatsApp"
        # The publisher-hash PFN fragment is never a process candidate
        # or display name.
        assert all("5319275a" not in p for p in ident.process_names)
        assert ident.package_family == (
            "5319275a.whatsappdesktop_cv1g1gvanyjgm"
        )

    def test_afh_window_matches_by_title(self):
        app = AppInfo(
            "WhatsApp", "", source="packaged",
            launch_kind="aumid", aumid=WHATSAPP_AUMID,
        )
        win = _win(50, 300, "WhatsApp", "ApplicationFrameHost.exe")
        assert window_matches_app(win, app)

    def test_pfn_window_matches(self):
        app = AppInfo(
            "WhatsApp", "", source="packaged",
            launch_kind="aumid", aumid=WHATSAPP_AUMID,
        )
        win = WindowInfo(
            hwnd=51, pid=400, title="WhatsApp",
            process="WhatsApp.exe",
            package_family="5319275A.WhatsAppDesktop_cv1g1gvanyjgm",
        )
        assert window_matches_app(win, app)

    def test_unrelated_window_does_not_match(self):
        app = AppInfo(
            "WhatsApp", "", source="packaged",
            launch_kind="aumid", aumid=WHATSAPP_AUMID,
        )
        win = _win(52, 100, "GitHub - Chrome", "chrome.exe")
        assert not window_matches_app(win, app)

    def test_switch_to_packaged_app(self):
        svc = self._svc([_win(50, 300, "WhatsApp",
                             "ApplicationFrameHost.exe")])
        _i, res = _dispatch(svc, "Switch to WhatsApp.")
        assert res.success and res.executed, res.message
        assert "whatsapp" in res.message.lower()
        assert "5319275a" not in res.message.lower()

    def test_list_running_shows_display_name(self):
        svc = self._svc([_win(50, 300, "WhatsApp",
                             "ApplicationFrameHost.exe")])
        _i, res = _dispatch(svc, "What applications are running?")
        assert res.executed
        assert "whatsapp" in res.message.lower()
        assert "applicationframehost" not in res.message.lower()
        assert "5319275a" not in res.message.lower()

    def test_is_running_via_package_family(self):
        app = AppInfo(
            "WhatsApp", "", source="packaged",
            launch_kind="aumid", aumid=WHATSAPP_AUMID,
        )
        mon = ProcessMonitor(
            lister=lambda: [(300, "ApplicationFrameHost.exe")],
            pid_details=lambda pid: (
                "", "5319275A.WhatsAppDesktop_cv1g1gvanyjgm"
            ),
        )
        assert mon.is_running(app)

    def test_is_running_false_for_other_pfn(self):
        app = AppInfo(
            "WhatsApp", "", source="packaged",
            launch_kind="aumid", aumid=WHATSAPP_AUMID,
        )
        mon = ProcessMonitor(
            lister=lambda: [(300, "ApplicationFrameHost.exe")],
            pid_details=lambda pid: ("", "OTHER.Publisher_x_y"),
        )
        assert not mon.is_running(app)

    def test_dotted_process_stem_matches(self):
        """WhatsApp.Root.exe -> stem segments include 'whatsapp'."""
        app = AppInfo(
            "WhatsApp", "", source="packaged",
            launch_kind="aumid", aumid=WHATSAPP_AUMID,
        )
        win = WindowInfo(
            hwnd=53, pid=13760, title="WhatsApp",
            process="WhatsApp.Root.exe",
            package_family="5319275A.WhatsAppDesktop_cv1g1gvanyjgm",
        )
        assert window_matches_app(win, app)

    def test_hidden_window_restored_on_switch(self):
        """Tray-hidden packaged window: switch restores then focuses."""
        svc = self._svc([_win(50, 300, "WhatsApp",
                             "ApplicationFrameHost.exe")])
        api = svc._test_api
        state = {"vis": False}
        api.is_visible = lambda h: (h != 50) or state["vis"]

        def show(h, cmd):
            api.show_cmd[h] = cmd
            if h == 50 and cmd == 9:  # SW_RESTORE
                state["vis"] = True

        api.show = show
        api._fg = 50
        _i, res = _dispatch(svc, "Switch to WhatsApp.")
        assert res.success and res.executed, res.message
        assert "whatsapp" in res.message.lower()

    def test_tray_hidden_app_still_listed_as_running(self):
        """A packaged app with only a hidden top-level window (tray)
        is still reported under its canonical name."""
        win = WindowInfo(
            hwnd=50, pid=13760, title="WhatsApp",
            process="WhatsApp.Root.exe",
            package_family="5319275A.WhatsAppDesktop_cv1g1gvanyjgm",
            visible=False,
        )
        svc = make_identity_svc(
            windows=[win], procs=[(13760, "WhatsApp.Root.exe")],
            start_apps=[("WhatsApp", WHATSAPP_AUMID)],
        )
        svc._test_api.is_visible = lambda h: h != 50
        _i, res = _dispatch(svc, "What applications are running?")
        assert res.executed
        assert "whatsapp" in res.message.lower()
        assert "5319275a" not in res.message.lower()
        assert "whatsapproot" not in res.message.lower()


class TestDevinSwitchRegression:
    """Physical-log scenario: launch Devin, then switch to it."""

    def test_launch_then_switch_devin(self):
        svc = make_identity_svc(
            windows=[], procs=[], apps=[DEVIN],
        )
        # Devin's window appears after launch (async window creation).
        state = {"launched": False}
        devin_win = _win(60, 4242, "Devin", "Devin.exe")

        def list_windows(include_hidden=False):
            return [devin_win] if state["launched"] else []

        svc.windows.list_windows = list_windows
        svc.windows.windows_for_pid = lambda pid: (
            [devin_win] if state["launched"] and pid == 4242 else []
        )
        real = svc.launcher.launch_process

        def launch(app):
            state["launched"] = True
            return real(app)

        svc.launcher.launch_process = launch
        # Launch via OPEN_APPLICATION_ON_MONITOR to primary.
        _i, res = _dispatch(svc, "Open Devin on the primary monitor.")
        assert res.success and res.executed, res.message
        # Simulate the window having received foreground.
        svc._test_api._fg = 60
        # Now switch — must find the window by identity.
        _i, res2 = _dispatch(svc, "Switch to Devin.")
        assert res2.success and res2.executed, res2.message
        assert "devin" in res2.message.lower()


class TestFocusHonesty:
    def test_focus_failure_is_not_not_running(self):
        win = _win(70, 100, "Chrome", "chrome.exe")
        svc = make_identity_svc(windows=[win], procs=[(100, "chrome.exe")])
        svc._test_api.set_foreground = lambda h: False
        svc._test_api._fg = 999  # something else stays foreground
        _i, res = _dispatch(svc, "Switch to Chrome.")
        assert not res.success
        assert res.error == "focus_failed"
        assert "doesn't appear to be running" not in res.message
        assert "couldn't bring it to the front" in res.message


class TestPrimaryDefaultLaunch:
    @pytest.mark.parametrize("app_name", ["Chrome", "Notepad"])
    def test_plain_open_lands_on_primary(self, app_name):
        svc = make_identity_svc()
        stem = "chrome" if app_name == "Chrome" else "notepad"
        svc.windows.windows_for_pid = lambda pid: [
            _win(80, 4242, app_name, f"{stem}.exe")
        ]
        _i, res = _dispatch(svc, f"Open {app_name}.")
        assert res.success and res.executed, res.message
        rect = svc._test_api.rects[80]
        # Monitor 1 = primary/right display (x in [0, 1920]) under
        # the primary-first numbering.
        assert rect[0] >= 0 and rect[2] <= 1920, rect

    def test_explicit_monitor_overrides_primary(self):
        svc = make_identity_svc()
        svc.windows.windows_for_pid = lambda pid: [
            _win(81, 4242, "Notepad", "notepad.exe")
        ]
        _i, res = _dispatch(svc, "Open Notepad on monitor two.")
        assert res.success and res.executed, res.message
        rect = svc._test_api.rects[81]
        # Monitor 2 is the LEFT display (negative X) — explicit wins
        # over the Monitor-1 default.
        assert rect[2] <= 0, rect


class _FakeSTT:
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

    def emit(self, t):
        self._cb["on_transcript"](t)


class _FakeTTS:
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

    def speak(self, t):
        self.spoken.append(t)
        self._cb["on_finished"](t)


def _controller(svc):
    from src.core.session_manager import SessionManager
    from src.core.state_manager import AppState, StateManager
    from src.voice.voice_session_controller import (
        VoiceSessionController,
    )

    state = StateManager()
    state.transition_to(AppState.STANDBY)
    ctl = VoiceSessionController(
        state_manager=state,
        session_manager=SessionManager(timeout_seconds=60),
        speech_recognition=_FakeSTT(), speech_output=_FakeTTS(),
        command_registry=create_app_registry(svc),
    )
    ctl.handle_wake_word()
    ctl._tts.spoken.clear()
    return ctl


class TestWindowClarification:
    """Multi-window move: ordinal/title selection, cancel, expiry."""

    def _two_chrome(self):
        w1 = _win(11, 100, "GitHub - Chrome", "chrome.exe")
        w2 = _win(12, 100, "Docs - Chrome", "chrome.exe")
        svc = make_identity_svc(
            windows=[w1, w2], procs=[(100, "chrome.exe")]
        )
        svc._test_api.rects[11] = (-1700, 100, -1300, 400)
        svc._test_api.rects[12] = (-1600, 100, -1200, 400)
        svc.monitors._api.hwnd_map[11] = 1
        svc.monitors._api.hwnd_map[12] = 1
        return svc, w1, w2

    def test_ordinal_selection_moves_only_that_window(self):
        svc, w1, w2 = self._two_chrome()
        ctl = _controller(svc)
        ctl._stt.emit("Move Chrome to monitor 1.")
        assert any("which one" in s.lower()
                   for s in ctl._tts.spoken)
        ctl._stt.emit("the second one")
        assert any("moved" in s.lower() for s in ctl._tts.spoken)
        # Only window 12 moved onto monitor 1 (x >= 0).
        assert svc._test_api.rects[12][0] >= 0
        assert svc._test_api.rects[11][2] <= 0
        ctl.shutdown()

    def test_title_selection(self):
        svc, w1, w2 = self._two_chrome()
        ctl = _controller(svc)
        ctl._stt.emit("Move Chrome to monitor 1.")
        ctl._stt.emit("GitHub")
        assert any("moved" in s.lower() for s in ctl._tts.spoken)
        assert svc._test_api.rects[11][0] >= 0
        assert svc._test_api.rects[12][2] <= 0
        ctl.shutdown()

    def test_cancel_never_executes(self):
        svc, w1, w2 = self._two_chrome()
        ctl = _controller(svc)
        ctl._stt.emit("Move Chrome to monitor 1.")
        ctl._stt.emit("cancel")
        assert any("never mind" in s.lower()
                   for s in ctl._tts.spoken)
        # Nothing moved.
        assert svc._test_api.rects[11][2] <= 0
        assert svc._test_api.rects[12][2] <= 0
        ctl.shutdown()

    def test_unrelated_command_does_not_use_stale_selection(self):
        svc, w1, w2 = self._two_chrome()
        ctl = _controller(svc)
        ctl._stt.emit("Move Chrome to monitor 1.")
        # An unrelated command must route normally — not consume the
        # pending window selection.
        ctl._stt.emit("What time is it?")
        assert svc._test_api.rects[11][2] <= 0
        assert svc._test_api.rects[12][2] <= 0
        ctl.shutdown()

    def test_expired_clarification_does_not_execute(self):
        import time as _time

        svc, w1, w2 = self._two_chrome()
        ctl = _controller(svc)
        ctl._stt.emit("Move Chrome to monitor 1.")
        # Age the pending clarification past its window.
        pending = ctl._pending_confirm
        assert pending is not None and pending["kind"] == "clarify"
        pending["at"] = _time.monotonic() - 120
        ctl._stt.emit("the first one")
        # Expired: falls through to normal routing; no window moved.
        assert svc._test_api.rects[11][2] <= 0
        assert svc._test_api.rects[12][2] <= 0
        ctl.shutdown()
