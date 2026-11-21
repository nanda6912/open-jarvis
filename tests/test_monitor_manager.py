"""Tests for MonitorManager — all Win32 monitor APIs mocked."""

from __future__ import annotations

import pytest

from src.controllers.monitor_manager import (
    MonitorError,
    MonitorInfo,
    MonitorManager,
    MonitorRect,
)


def _mon(index, device, bounds, work=None, primary=False, handle=None):
    b = MonitorRect(*bounds)
    w = MonitorRect(*(work or bounds))
    return MonitorInfo(
        index=index, device=device, bounds=b, work_area=w,
        primary=primary, handle=handle if handle is not None else index,
    )


class FakeMonitorApi:
    """Fake _MonitorApi: list of (bounds, work, primary, device)."""

    def __init__(self, monitors):
        # monitors: list of dicts with bounds/work/primary/device
        self._mons = list(monitors)  # copy: tests mutate _mons
        self.enum_calls = 0

    def enum_monitors(self):
        self.enum_calls += 1
        return [i + 1 for i in range(len(self._mons))]

    def monitor_info(self, hmon):
        m = self._mons[hmon - 1]
        return (
            MonitorRect(*m["bounds"]),
            MonitorRect(*m.get("work", m["bounds"])),
            m.get("primary", False),
            m["device"],
        )

    def monitor_for_hwnd(self, hwnd):
        return getattr(self, "_hwnd_map", {}).get(hwnd, 0)

    def monitor_for_point(self, x, y):
        for i, m in enumerate(self._mons):
            l, t, r, b = m["bounds"]
            if l <= x < r and t <= y < b:
                return i + 1
        return 0


def _manager(monitors):
    return MonitorManager(api=FakeMonitorApi(monitors))


TWO_SIDE_BY_SIDE = [
    # External monitor physically LEFT (negative X), not primary.
    {"bounds": (-1920, 0, 0, 1080), "primary": False,
     "device": "\\\\.\\DISPLAY2"},
    # Laptop display physically RIGHT, primary.
    {"bounds": (0, 0, 1920, 1080), "primary": True,
     "device": "\\\\.\\DISPLAY1"},
]

ONE_MONITOR = [
    {"bounds": (0, 0, 1920, 1080), "primary": True,
     "device": "\\\\.\\DISPLAY1"},
]

THREE_STACKED = [
    {"bounds": (-2560, -200, -640, 880), "primary": False,
     "device": "\\\\.\\DISPLAY3"},
    {"bounds": (0, 0, 1920, 1080), "primary": True,
     "device": "\\\\.\\DISPLAY1"},
    {"bounds": (1920, -1080, 3840, 0), "primary": False,
     "device": "\\\\.\\DISPLAY2"},
]


class TestDiscovery:
    def test_single_monitor(self):
        mm = _manager(ONE_MONITOR)
        mons = mm.monitors()
        assert len(mons) == 1
        assert mons[0].index == 1
        assert mons[0].primary is True
        assert mons[0].width == 1920 and mons[0].height == 1080

    def test_two_monitors_primary_is_monitor_1(self):
        mm = _manager(TWO_SIDE_BY_SIDE)
        mons = mm.monitors()
        assert len(mons) == 2
        # Monitor 1 is ALWAYS the primary display (right side here),
        # never the leftmost.
        assert mons[0].device == "\\\\.\\DISPLAY1"
        assert mons[0].primary is True
        assert mons[0].bounds.left == 0
        assert mons[1].device == "\\\\.\\DISPLAY2"
        assert mons[1].bounds.left == -1920

    def test_three_monitors_primary_first_then_position(self):
        mm = _manager(THREE_STACKED)
        mons = mm.monitors()
        assert len(mons) == 3
        # Monitor 1 = primary; the rest sorted left->right:
        # DISPLAY3 (-2560) then DISPLAY2 (1920).
        assert [m.device for m in mons] == [
            "\\\\.\\DISPLAY1", "\\\\.\\DISPLAY3", "\\\\.\\DISPLAY2",
        ]
        assert [m.index for m in mons] == [1, 2, 3]
        assert mons[1].bounds.left == -2560  # negative coords handled

    def test_reversed_arrangement(self):
        # Laptop LEFT, external RIGHT (opposite of this machine).
        mons = [
            {"bounds": (0, 0, 1920, 1080), "primary": True,
             "device": "\\\\.\\DISPLAY1"},
            {"bounds": (1920, 0, 3840, 1080), "primary": False,
             "device": "\\\\.\\DISPLAY2"},
        ]
        mm = _manager(mons)
        res = mm.resolve("left monitor")
        assert res.status == "found"
        assert res.monitor.device == "\\\\.\\DISPLAY1"
        res = mm.resolve("right monitor")
        assert res.monitor.device == "\\\\.\\DISPLAY2"

    def test_above_below_arrangement(self):
        mons = [
            {"bounds": (0, -1080, 1920, 0), "primary": False,
             "device": "\\\\.\\DISPLAY2"},
            {"bounds": (0, 0, 1920, 1080), "primary": True,
             "device": "\\\\.\\DISPLAY1"},
        ]
        mm = _manager(mons)
        assert mm.resolve("top monitor").monitor.device == "\\\\.\\DISPLAY2"
        assert mm.resolve("bottom monitor").monitor.device == "\\\\.\\DISPLAY1"

    def test_work_area_differs_from_bounds(self):
        mons = [
            {"bounds": (0, 0, 1920, 1080),
             "work": (0, 0, 1920, 1040), "primary": True,
             "device": "\\\\.\\DISPLAY1"},
        ]
        mm = _manager(mons)
        m = mm.monitors()[0]
        assert m.bounds.height == 1080
        assert m.work_area.height == 1040  # taskbar accounted for

    def test_refresh_reenumerates(self):
        api = FakeMonitorApi(ONE_MONITOR)
        mm = MonitorManager(api=api)
        mm.monitors()
        api._mons.append(
            {"bounds": (1920, 0, 3840, 1080), "primary": False,
             "device": "\\\\.\\DISPLAY2"}
        )
        assert mm.count() == 2  # no stale cache
        assert api.enum_calls >= 2

    def test_disconnect_returns_fewer(self):
        api = FakeMonitorApi(TWO_SIDE_BY_SIDE)
        mm = MonitorManager(api=api)
        assert mm.count() == 2
        api._mons.pop(0)
        assert mm.count() == 1

    def test_enum_failure_raises(self):
        class Bad:
            def enum_monitors(self):
                raise RuntimeError("boom")
        with pytest.raises(MonitorError):
            MonitorManager(api=Bad()).monitors()

    def test_invalid_number(self):
        mm = _manager(TWO_SIDE_BY_SIDE)
        assert mm.get(99) is None
        res = mm.resolve("monitor 9")
        assert res.status == "missing"


class TestAliases:
    def setup_method(self):
        self.mm = _manager(TWO_SIDE_BY_SIDE)

    @pytest.mark.parametrize("q,device", [
        ("primary monitor", "\\\\.\\DISPLAY1"),
        ("the primary screen", "\\\\.\\DISPLAY1"),
        ("monitor 1", "\\\\.\\DISPLAY1"),   # primary = index 1
        ("monitor 2", "\\\\.\\DISPLAY2"),   # left secondary = index 2
        ("2", "\\\\.\\DISPLAY2"),
        ("first monitor", "\\\\.\\DISPLAY1"),
        ("second monitor", "\\\\.\\DISPLAY2"),
        ("left monitor", "\\\\.\\DISPLAY2"),
        ("the left screen", "\\\\.\\DISPLAY2"),
        ("right monitor", "\\\\.\\DISPLAY1"),
        ("right display", "\\\\.\\DISPLAY1"),
        ("secondary monitor", "\\\\.\\DISPLAY2"),
        # Windows cannot identify the laptop panel reliably -> the
        # resolver refuses to guess and reports missing (caller asks).
        ("laptop screen", None),
        ("external monitor", "\\\\.\\DISPLAY2"),
        ("display 2", "\\\\.\\DISPLAY2"),
    ])
    def test_alias_resolution(self, q, device):
        res = self.mm.resolve(q)
        if device is None:
            assert res.status == "missing", q
            return
        assert res.status == "found", q
        assert res.monitor.device == device, q

    def test_other_monitor_relative_to_left(self):
        left = self.mm.resolve("left monitor").monitor
        res = self.mm.resolve("other monitor", relative_to=left)
        assert res.status == "found"
        assert res.monitor.device == "\\\\.\\DISPLAY1"

    def test_other_monitor_defaults_to_primary(self):
        res = self.mm.resolve("other monitor")
        assert res.status == "found"
        assert res.monitor.device == "\\\\.\\DISPLAY2"

    def test_missing_monitor(self):
        res = self.mm.resolve("monitor 7")
        assert res.status == "missing"
        res = self.mm.resolve("quantum flux capacitor")
        assert res.status == "missing"

    def test_bare_monitor_ambiguous(self):
        res = self.mm.resolve("monitor")
        assert res.status == "ambiguous"
        assert len(res.options) == 2

    def test_single_monitor_bare_found(self):
        mm = _manager(ONE_MONITOR)
        res = mm.resolve("monitor")
        assert res.status == "found"

    def test_spoken_label(self):
        # Spatial labels follow geometry, not logical numbering:
        # monitor 1 (primary) sits on the right, monitor 2 on the left.
        assert self.mm.spoken_label(self.mm.get(1)) == "the right monitor"
        assert self.mm.spoken_label(self.mm.get(2)) == "the left monitor"

    def test_describe_includes_geometry(self):
        d = self.mm.describe(self.mm.get(1))
        assert "monitor 1" in d and "1920x1080" in d

    def test_summary_two_monitors(self):
        s = self.mm.summary()
        assert "2 monitors" in s
        assert "left" in s and "right" in s


class TestWindowMapping:
    def test_monitor_for_hwnd_and_point(self):
        api = FakeMonitorApi(TWO_SIDE_BY_SIDE)
        api._hwnd_map = {1234: 1}
        mm = MonitorManager(api=api)
        assert mm.monitor_for_hwnd(1234).device == "\\\\.\\DISPLAY2"
        assert mm.monitor_for_hwnd(9999) is None
        # Point on the right monitor.
        assert mm.monitor_for_point(100, 100).device == "\\\\.\\DISPLAY1"
        # Point on the left (negative X) monitor.
        assert mm.monitor_for_point(-100, 100).device == "\\\\.\\DISPLAY2"
