"""VoiceOS V9.2 regression tests.

Covers the mandatory monitor-numbering convention (Monitor 1 =
primary/right, Monitor 2 = left), default Monitor-1 launches for every
app kind, packaged-app asynchronous window readiness, hidden-window
movement, and STT alias safety.
"""

from __future__ import annotations

import pytest

from src.commands.app_handlers import _Svc
from src.commands.intent import RuleBasedIntentRecognizer
from src.commands.registry import CommandRegistry
from src.controllers.application_discovery import AppInfo
from src.controllers.monitor_manager import MonitorManager, MonitorRect
from src.controllers.windows_automation import WindowInfo
from tests.test_app_identity import (
    DEVIN,
    WHATSAPP_AUMID,
    _dispatch,
    make_identity_svc,
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
    make_svc,
)

WHATSAPP = AppInfo(
    "WhatsApp", "", source="packaged",
    launch_kind="aumid", aumid=WHATSAPP_AUMID,
)


def _wa_svc(windows=(), procs=None, popen=None, **kw):
    """_Svc with WhatsApp discoverable as a packaged (AUMID) app."""
    return make_identity_svc(
        windows=list(windows),
        procs=procs if procs is not None else [
            (13760, "WhatsApp.Root.exe"),
            (300, "ApplicationFrameHost.exe"),
        ],
        apps=[WHATSAPP, CHROME, NOTEPAD, DEVIN],
        start_apps=[("WhatsApp", WHATSAPP_AUMID)],
        popen=popen,
        **kw,
    )


class TestLogicalNumbering:
    """Monitor 1 = primary; Monitor 2 = the other display."""

    def test_enumeration_order_does_not_change_numbering(self):
        api = FakeMonitorApi(list(reversed(MONS)))
        mm = MonitorManager(api=api)
        mons = mm.monitors()
        assert mons[0].device == "\\\\.\\DISPLAY1"  # primary first
        assert mons[0].primary is True
        assert mons[1].device == "\\\\.\\DISPLAY2"

    def test_monitor_one_is_right_and_two_is_left(self):
        svc = make_svc()
        one = svc.monitors.resolve("monitor one").monitor
        two = svc.monitors.resolve("monitor two").monitor
        assert one.primary is True
        assert one.bounds.left == 0       # right display
        assert two.bounds.left == -1920   # left display

    def test_numbering_survives_topology_change(self):
        api = FakeMonitorApi(list(MONS))  # copy: we mutate _mons
        mm = MonitorManager(api=api)
        assert mm.get(1).device == "\\\\.\\DISPLAY1"
        # Disconnect the left display: primary remains Monitor 1.
        api._mons.pop(0)
        mons = mm.monitors()
        assert len(mons) == 1
        assert mons[0].index == 1 and mons[0].primary is True
        # Reconnect on the LEFT again: still Monitor 2.
        api._mons.insert(0, MONS[0])
        assert mm.get(2).device == "\\\\.\\DISPLAY2"

    def test_list_monitors_reports_user_mapping(self):
        svc = make_svc()
        _i, res = _dispatch(svc, "List my monitors.")
        assert res.success
        # Monitor 1 must be described as right/primary; Monitor 2 left.
        m1 = svc.monitors.get(1)
        m2 = svc.monitors.get(2)
        assert m1.primary and m1.bounds.left == 0
        assert not m2.primary and m2.bounds.left < 0
        assert "monitor 1" in res.message
        assert "monitor 2" in res.message


class TestDefaultLaunchAllApps:
    """Every app kind defaults to Monitor 1 (right/primary)."""

    @pytest.mark.parametrize("app_name,stem", [
        ("Chrome", "chrome"), ("Notepad", "notepad"), ("Devin", "devin"),
    ])
    def test_plain_open_lands_on_monitor_1(self, app_name, stem):
        svc = make_identity_svc()
        svc.windows.windows_for_pid = lambda pid: [
            _win(90, 4242, app_name, f"{stem}.exe")
        ]
        _i, res = _dispatch(svc, f"Open {app_name}.")
        assert res.success and res.executed, res.message
        rect = svc._test_api.rects[90]
        assert rect[0] >= 0 and rect[2] <= 1920, rect

    def test_whatsapp_defaults_to_monitor_1(self):
        svc = _wa_svc()
        win = WindowInfo(
            hwnd=91, pid=555, title="WhatsApp",
            process="WhatsApp.Root.exe",
            package_family="5319275A.WhatsAppDesktop_cv1g1gvanyjgm",
        )
        state = {"spawned": False}
        svc.windows.list_windows = lambda include_hidden=False: (
            [win] if state["spawned"] else []
        )
        launches = []
        svc.launcher._aumid_opener = lambda t: (
            launches.append(t), state.update(spawned=True)
        )
        _i, res = _dispatch(svc, "Open WhatsApp.")
        assert res.success and res.executed, res.message
        assert len(launches) == 1
        rect = svc._test_api.rects[91]
        assert rect[0] >= 0 and rect[2] <= 1920, rect

    def test_explicit_monitor_2_overrides_for_whatsapp(self):
        svc = _wa_svc()
        win = WindowInfo(
            hwnd=92, pid=555, title="WhatsApp",
            process="WhatsApp.Root.exe",
            package_family="5319275A.WhatsAppDesktop_cv1g1gvanyjgm",
        )
        svc.windows.list_windows = lambda include_hidden=False: [win]
        svc.launcher._aumid_opener = lambda t: None
        _i, res = _dispatch(svc, "Open WhatsApp on monitor two.")
        assert res.success and res.executed, res.message
        rect = svc._test_api.rects[92]
        assert rect[2] <= 0, rect  # left display

    def test_explicit_monitor_1_for_devin(self):
        svc = make_identity_svc(apps=[DEVIN])
        svc.windows.windows_for_pid = lambda pid: [
            _win(93, 4242, "Devin", "Devin.exe")
        ]
        _i, res = _dispatch(svc, "Open Devin on monitor one.")
        assert res.success and res.executed, res.message
        rect = svc._test_api.rects[93]
        assert rect[0] >= 0 and rect[2] <= 1920


class TestPackagedWindowReadiness:
    """WhatsApp-style async window creation after AUMID launch."""

    def _delayed_wa(self, delay_calls=3):
        svc = _wa_svc()
        win = WindowInfo(
            hwnd=95, pid=777, title="WhatsApp",
            process="WhatsApp.Root.exe",
            package_family="5319275A.WhatsAppDesktop_cv1g1gvanyjgm",
        )
        state = {"calls": 0, "spawned": False}

        def list_windows(include_hidden=False):
            state["calls"] += 1
            if state["spawned"] and state["calls"] > delay_calls:
                return [win]
            return []

        svc.windows.list_windows = list_windows
        launches = []

        def opener(target):
            launches.append(target)
            state["spawned"] = True

        svc.launcher._aumid_opener = opener
        return svc, win, launches

    def test_delayed_window_is_found_and_moved(self):
        svc, win, launches = self._delayed_wa(delay_calls=3)
        _i, res = _dispatch(svc, "Open WhatsApp on monitor two.")
        assert res.success and res.executed, res.message
        assert len(launches) == 1  # no duplicate launch
        rect = svc._test_api.rects[95]
        assert rect[2] <= 0  # moved to left monitor

    def test_reused_running_instance_not_relaunched(self):
        """'Open WhatsApp' while running reuses the instance: a single
        existing window is moved without spawning a second launch."""
        svc = _wa_svc()
        win = WindowInfo(
            hwnd=96, pid=555, title="WhatsApp",
            process="WhatsApp.Root.exe",
            package_family="5319275A.WhatsAppDesktop_cv1g1gvanyjgm",
        )
        svc.windows.list_windows = lambda include_hidden=False: [win]
        launches = []
        svc.launcher._aumid_opener = lambda t: launches.append(t)
        _i, res = _dispatch(svc, "Open WhatsApp on monitor two.")
        assert res.success and res.executed, res.message
        # AUMID "launch" is activate-only (one call, no new process);
        # the existing window was moved, not duplicated.
        assert len(launches) == 1
        assert svc._test_api.rects[96][2] <= 0

    def test_hidden_whatsapp_window_moved_and_restored(self):
        """Tray-hidden WhatsApp: 'Move WhatsApp to monitor two' finds
        the hidden top-level window, restores it, moves and verifies."""
        win = WindowInfo(
            hwnd=97, pid=555, title="WhatsApp",
            process="WhatsApp.Root.exe",
            package_family="5319275A.WhatsAppDesktop_cv1g1gvanyjgm",
            visible=False,
        )
        svc = _wa_svc(
            windows=[win],
            procs=[(555, "WhatsApp.Root.exe"),
                   (300, "ApplicationFrameHost.exe")],
        )
        api = svc._test_api
        state = {"vis": False}
        api.is_visible = lambda h: (h != 97) or state["vis"]

        def show(h, cmd):
            api.show_cmd[h] = cmd
            if h == 97 and cmd == 9:
                state["vis"] = True

        api.show = show
        api.rects[97] = (-900, 200, -500, 700)  # somewhere on left
        svc.monitors._api.hwnd_map[97] = 1
        _i, res = _dispatch(svc, "Move WhatsApp to monitor one.")
        assert res.success and res.executed, res.message
        rect = api.rects[97]
        assert rect[0] >= 0 and rect[2] <= 1920, rect
        assert api.show_cmd.get(97) == 9  # SW_RESTORE seen


class TestSttAliasSafety:
    def test_write_monitor_resolves_right(self):
        intent = RuleBasedIntentRecognizer().recognize(
            "Move Chrome to write monitor."
        )
        assert intent.name == "MOVE_WINDOW_TO_MONITOR"
        svc = make_svc(
            windows=[_win(11, 100, "Chrome", "chrome.exe")]
        )
        svc._test_api.rects[11] = (-1700, 100, -1300, 400)
        svc.monitors._api.hwnd_map[11] = 1
        intent, res = _dispatch(svc, "Move Chrome to write monitor.")
        assert intent.name == "MOVE_WINDOW_TO_MONITOR"
        assert res.success and res.executed, res.message
        assert svc._test_api.rects[11][0] >= 0  # moved right

    def test_monitor_too_resolves_second(self):
        svc = make_svc()
        res = svc.monitors.resolve("monitor too")
        assert res.status == "found"
        assert res.monitor.index == 2
