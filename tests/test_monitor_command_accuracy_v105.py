"""V10.5 monitor-command accuracy tests.

Covers the observed STT garbles ("less than monitor", "into the left
monitor", "monitor two") and the physical mapping invariant:
Monitor 1 = RIGHT = primary, Monitor 2 = LEFT = secondary — the
resolver must keep that geometry, never hard-code indices.
"""

from __future__ import annotations

import pytest

from src.commands.intent import RuleBasedIntentRecognizer
from src.commands.recovery import MisheardCommandRecovery
from src.controllers.monitor_manager import (
    MonitorManager,
    MonitorRect,
)


class _FakeMonitorApi:
    """Fake _MonitorApi: list of monitor dicts (bounds/work/primary)."""

    def __init__(self, monitors):
        self._mons = list(monitors)

    def enum_monitors(self):
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
        return 0

    def monitor_for_point(self, x, y):
        return 0


@pytest.fixture(scope="module")
def parser():
    return RuleBasedIntentRecognizer()


MOVE_CASES = [
    ("Move Chrome to the left monitor", "left"),
    ("Move Chrome to the right monitor", "right"),
    ("Move Devin to the left monitor", "left"),
    ("Move WhatsApp to the right monitor", "right"),
    ("Move Chrome into less than monitor", "left"),
    ("Move Dev into the left monitor", "left"),
    ("Move WhatsApp to monitor two", "two"),
    ("Move Chrome to monitor one", "one"),
    ("Move Chrome to the first monitor", "first"),
    ("Move Chrome to the second monitor", "second"),
]


@pytest.mark.parametrize("text,expected_side", MOVE_CASES)
def test_move_to_named_monitor(parser, text, expected_side):
    intent = parser.recognize(text)
    assert intent.name == "MOVE_WINDOW_TO_MONITOR", (
        f"{text!r} -> {intent.name} {intent.entities}"
    )
    assert intent.entities.get("monitor") == expected_side


def test_move_to_other_monitor(parser):
    for text in (
        "Move the current window to the other monitor",
        "Move current window to another monitor",
    ):
        intent = parser.recognize(text)
        assert intent.name in (
            "MOVE_WINDOW_TO_OTHER_MONITOR",
            "MOVE_WINDOW_TO_MONITOR",
        ), f"{text!r} -> {intent.name}"


def test_move_to_primary_secondary(parser):
    intent = parser.recognize("Move Chrome to the primary monitor")
    assert intent.name == "MOVE_WINDOW_TO_PRIMARY_MONITOR"
    intent = parser.recognize("Move Chrome to the secondary monitor")
    assert intent.name in (
        "MOVE_WINDOW_TO_MONITOR", "MOVE_WINDOW_TO_PRIMARY_MONITOR",
    )


# Two-monitor geometry: primary display physically on the RIGHT,
# secondary on the LEFT — the production mapping
# (Monitor 1 = RIGHT = primary, Monitor 2 = LEFT = secondary).
_TWO_MONITORS = [
    {"bounds": (0, 0, 1920, 1080), "primary": True,
     "device": "\\\\.\\DISPLAY1"},
    {"bounds": (-1920, 0, 0, 1080), "primary": False,
     "device": "\\\\.\\DISPLAY2"},
]


@pytest.fixture
def monitor_manager():
    return MonitorManager(api=_FakeMonitorApi(_TWO_MONITORS))


def test_physical_mapping_right_is_primary(monitor_manager):
    """Spatial alias resolution follows geometry, not list order."""
    res = monitor_manager.resolve("left monitor")
    assert res.status == "found"
    assert res.monitor is not None and res.monitor.primary is False
    assert res.monitor.bounds.left < 0  # physically left of primary
    res = monitor_manager.resolve("right monitor")
    assert res.status == "found"
    assert res.monitor is not None and res.monitor.primary is True


def test_numbered_monitor_aliases(monitor_manager):
    """'monitor one'/'monitor two' resolve by index."""
    for spec in ("monitor one", "monitor 1", "first monitor"):
        res = monitor_manager.resolve(spec)
        assert res.status == "found", f"{spec} -> {res.status}"
    for spec in ("monitor two", "monitor 2", "second monitor"):
        res = monitor_manager.resolve(spec)
        assert res.status == "found", f"{spec} -> {res.status}"


class _FakeApp:
    def __init__(self, name):
        self.name = name
        self.exe_name = ""
        self.exe_path = ""


class _FakeDiscovery:
    def __init__(self, names):
        self._apps = [_FakeApp(n) for n in names]

    def discover(self):
        return self._apps

    def resolve(self, name):
        from src.controllers.application_discovery import ResolveResult
        for a in self._apps:
            if a.name.lower() == name.lower():
                return ResolveResult("found", app=a, score=100)
        return ResolveResult("missing", score=0)


def test_recovery_fixless_than_monitor():
    """Garbled 'Move Dev into less than monitor' recovers to a real
    monitor destination — the exact observed runtime failure."""
    rec = MisheardCommandRecovery(
        app_names=lambda: [_FakeApp("Devin"), _FakeApp("Chrome")],
        app_resolver=_FakeDiscovery(["Devin", "Chrome"]).resolve,
        monitor_resolver=lambda spec: type(
            "R", (), {"status": "found", "monitor": object()}
        )(),
    )
    result = rec.recover("Move Dev into less than monitor")
    assert result.status == "matched", result.reason
    assert result.intent is not None
    assert result.intent.name == "MOVE_WINDOW_TO_MONITOR"
