"""V10.8 HUD topmost/Z-order regression tests.

Runs offscreen (QT_QPA_PLATFORM=offscreen). A fake ``user32`` records
every Get/SetWindowLongPtrW and SetWindowPos call so the native
window-style behaviour can be asserted without a real display.

These tests prove the *calls and styles* the HUD issues; they do NOT
prove real Chrome Z-order behaviour — that requires physical
validation on the Windows desktop (see
``docs/V10_8_HUD_TOPMOST_FIX_REPORT.md``).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import inspect

import pytest

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QShowEvent
from PySide6.QtWidgets import QApplication

import src.ui.hologram.hud_compact as hud_compact
from src.ui.hologram.hud_compact import (
    _HWND_TOPMOST,
    _SWP_NOACTIVATE,
    _SWP_NOMOVE,
    _SWP_NOSIZE,
    _SWP_NOZORDER,
    _WS_EX_NOACTIVATE,
    _WS_EX_TOPMOST,
    _WS_EX_TOOLWINDOW,
    _WS_EX_TRANSPARENT,
    CompactHologramHUD,
)

HUD_TEST_HWND = 0x000000000004D0C8

# Realistic baseline exstyle: layered | toolwindow | noactivate.
_BASE_EXSTYLE = 0x00080000 | 0x00000080 | 0x08000000


class FakeUser32:
    """Records Win32 calls and simulates GWL_EXSTYLE storage."""

    def __init__(self, exstyle: int = _BASE_EXSTYLE):
        # Styles are tracked per-HWND: a recreated native window starts
        # with the baseline exstyle, not the old window's bits.
        self._default_exstyle = exstyle & 0xFFFFFFFF
        self.styles = {}
        self.get_calls = []
        self.set_calls = []
        self.swp_calls = []
        self.swp_result = True

    def GetWindowLongPtrW(self, hwnd, index):
        self.get_calls.append((hwnd, index))
        return self.styles.get(hwnd, self._default_exstyle)

    def SetWindowLongPtrW(self, hwnd, index, value):
        self.set_calls.append((hwnd, index, int(value)))
        self.styles[hwnd] = int(value) & 0xFFFFFFFF
        return 1  # previous value (nonzero => unambiguous success)

    def SetWindowPos(self, hwnd, insert_after, x, y, cx, cy, flags):
        self.swp_calls.append(
            {"hwnd": hwnd, "insert_after": insert_after,
             "x": x, "y": y, "cx": cx, "cy": cy, "flags": flags})
        return self.swp_result

    # -- helpers ------------------------------------------------------
    def topmost_calls(self):
        """SetWindowPos calls that target the topmost band."""
        return [c for c in self.swp_calls
                if c["insert_after"] == _HWND_TOPMOST]

    def has_flag(self, bit: int, hwnd: int = HUD_TEST_HWND) -> bool:
        return bool(self.styles.get(hwnd, self._default_exstyle) & bit)


class _HwndHolder:
    def __init__(self, value: int = HUD_TEST_HWND):
        self.value = value


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def hud_env(qapp, monkeypatch):
    """CompactHologramHUD + fake user32 + controllable HWND."""
    fake = FakeUser32()
    holder = _HwndHolder()
    monkeypatch.setattr(hud_compact, "_user32", lambda: fake)
    monkeypatch.setattr(
        CompactHologramHUD, "_hwnd", lambda self: holder.value)
    hud = CompactHologramHUD(target_fps=30)
    yield hud, fake, holder
    try:
        hud.close()
    except Exception:  # noqa: BLE001
        pass


def _topmost_flags_ok(call) -> bool:
    f = call["flags"]
    return (
        f & _SWP_NOMOVE
        and f & _SWP_NOSIZE
        and f & _SWP_NOACTIVATE
        and not (f & _SWP_NOZORDER)
    )


# ---------------------------------------------------------------------------
# 1-2. Native style + initial topmost establishment
# ---------------------------------------------------------------------------
def test_overlay_chrome_sets_topmost_exstyle(hud_env):
    hud, fake, _ = hud_env
    hud._apply_native_chrome(reason="test")
    assert fake.has_flag(_WS_EX_TOPMOST)
    assert fake.has_flag(_WS_EX_NOACTIVATE)
    assert fake.has_flag(_WS_EX_TOOLWINDOW)
    # Locked by default: click-through must be present too.
    assert fake.has_flag(_WS_EX_TRANSPARENT)


def test_show_establishes_hwnd_topmost_band(hud_env):
    hud, fake, _ = hud_env
    hud.showEvent(QShowEvent())
    calls = fake.topmost_calls()
    assert calls, "no SetWindowPos(HWND_TOPMOST) issued on show"
    assert all(_topmost_flags_ok(c) for c in calls)
    # Position/size must not be touched by the topmost call.
    assert all(c["x"] == 0 and c["y"] == 0
               and c["cx"] == 0 and c["cy"] == 0 for c in calls)


def test_ensure_window_topmost_uses_hwnd_topmost(hud_env):
    _, fake, holder = hud_env
    assert hud_compact.ensure_window_topmost(holder.value, "unit") is True
    calls = fake.topmost_calls()
    assert len(calls) == 1
    assert calls[0]["hwnd"] == holder.value
    assert _topmost_flags_ok(calls[0])


# ---------------------------------------------------------------------------
# 3-5. Lock / unlock / relock preserve topmost
# ---------------------------------------------------------------------------
def test_locked_mode_preserves_topmost_and_clickthrough(hud_env):
    hud, fake, _ = hud_env
    hud._apply_native_chrome(reason="test")
    # Toggle unlocked->locked so the lock path actually runs
    # (set_position_locked early-returns when state is unchanged).
    hud.set_position_locked(False)
    fake.swp_calls.clear()
    hud.set_position_locked(True)
    assert fake.has_flag(_WS_EX_TRANSPARENT)
    assert fake.has_flag(_WS_EX_TOPMOST)
    assert fake.has_flag(_WS_EX_NOACTIVATE)
    assert fake.has_flag(_WS_EX_TOOLWINDOW)
    assert fake.topmost_calls(), "lock dropped topmost band membership"


def test_unlocked_mode_preserves_topmost_without_transparent(hud_env):
    hud, fake, _ = hud_env
    hud._apply_native_chrome(reason="test")
    hud.set_position_locked(False)
    assert not fake.has_flag(_WS_EX_TRANSPARENT)
    assert fake.has_flag(_WS_EX_TOPMOST)
    assert fake.has_flag(_WS_EX_NOACTIVATE)
    assert fake.has_flag(_WS_EX_TOOLWINDOW)
    assert fake.topmost_calls()


def test_relock_restores_clickthrough_keeps_topmost(hud_env):
    hud, fake, _ = hud_env
    hud._apply_native_chrome(reason="test")
    hud.set_position_locked(False)
    fake.swp_calls.clear()
    hud.set_position_locked(True)
    assert fake.has_flag(_WS_EX_TRANSPARENT)
    assert fake.has_flag(_WS_EX_TOPMOST)
    assert fake.topmost_calls(), "relock lost topmost"


def test_topmost_survives_exstyle_only_change(hud_env):
    """A pure TRANSPARENT toggle uses SWP_NOZORDER — it must never be
    the call that establishes or drops the topmost band."""
    _, fake, holder = hud_env
    hud_compact.set_click_through(holder.value, True)
    frame_calls = [
        c for c in fake.swp_calls if c["flags"] & 0x0020  # FRAMECHANGED
    ]
    assert frame_calls
    assert all(c["insert_after"] != _HWND_TOPMOST for c in frame_calls)
    # And NO call may use HWND_NOTOPMOST (-2).
    assert all(c["insert_after"] != -2 for c in fake.swp_calls)


# ---------------------------------------------------------------------------
# 6-7. Move / resize / screen change do not lose topmost
# ---------------------------------------------------------------------------
def test_drag_release_reasserts_topmost(hud_env, qapp):
    hud, fake, _ = hud_env
    hud._apply_native_chrome(reason="test")
    hud.set_position_locked(False)
    fake.swp_calls.clear()
    hud._drag_offset = QPoint(10, 10)
    ev = QMouseEvent(
        QEvent.Type.MouseButtonRelease, QPointF(50, 50),
        Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier)
    hud.mouseReleaseEvent(ev)
    assert fake.topmost_calls(), "drag end did not re-assert topmost"


def test_resize_reasserts_topmost(hud_env, qapp):
    hud, fake, _ = hud_env
    hud.show()
    qapp.processEvents()
    hud.resize(320, 320)
    hud._apply_native_chrome(reason="test")
    fake.swp_calls.clear()
    # Non-square resize -> resizeEvent re-asserts topmost.
    hud.resize(310, 300)
    qapp.processEvents()
    assert fake.topmost_calls(), "resize did not re-assert topmost"
    assert fake.has_flag(_WS_EX_TOPMOST)


def test_screen_change_reasserts_topmost(hud_env):
    hud, fake, _ = hud_env
    hud._apply_native_chrome(reason="test")
    fake.swp_calls.clear()
    hud._on_screen_changed(None)
    assert fake.topmost_calls(), "screen change did not re-assert topmost"


# ---------------------------------------------------------------------------
# 8. Recreated HWND receives full native treatment
# ---------------------------------------------------------------------------
def test_hwnd_recreation_reapplies_styles(hud_env):
    hud, fake, holder = hud_env
    hud._apply_native_chrome(reason="init")
    holder.value = 0x000000000004E000  # simulate WinIdChange
    fake.set_calls.clear()
    fake.swp_calls.clear()
    hud._apply_native_chrome(reason="hwnd-recreate")
    assert fake.set_calls, "exstyle not rewritten for new HWND"
    assert all(c[0] == holder.value for c in fake.set_calls)
    tops = fake.topmost_calls()
    assert tops and all(c["hwnd"] == holder.value for c in tops)


def test_winidchange_event_schedules_chrome(hud_env, qapp, monkeypatch):
    hud, fake, _ = hud_env
    calls = []
    orig = hud._apply_native_chrome
    hud._apply_native_chrome = lambda reason="x": (
        calls.append(reason), orig(reason))
    # QEvent.Type.WinIdChange may not be constructible offscreen; send
    # through the real event() path with the actual enum value.
    ev = QEvent(QEvent.Type.WinIdChange)
    hud.event(ev)
    qapp.processEvents()
    assert "hwnd-recreate" in calls


# ---------------------------------------------------------------------------
# 9. Never activates / steals focus
# ---------------------------------------------------------------------------
def test_hud_never_activates(hud_env):
    hud, _, _ = hud_env
    flags = hud.windowFlags()
    assert flags & Qt.WindowType.WindowDoesNotAcceptFocus
    assert flags & Qt.WindowType.WindowStaysOnTopHint
    assert hud.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    # The compact HUD module must never call activateWindow().
    src = inspect.getsource(hud_compact)
    assert "activateWindow(" not in src
    # And SetWindowPos topmost calls always carry NOACTIVATE.
    _, fake, _ = hud_env
    hud._apply_native_chrome(reason="test")
    for c in fake.topmost_calls():
        assert c["flags"] & _SWP_NOACTIVATE


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def test_describe_window_state_flags(hud_env):
    _, fake, holder = hud_env
    fake.styles[holder.value] = (_WS_EX_TOPMOST | _WS_EX_TRANSPARENT
                                 | _WS_EX_NOACTIVATE | _WS_EX_TOOLWINDOW)
    text = hud_compact.describe_window_state(holder.value)
    assert "topmost=True" in text
    assert "transparent=True" in text
    assert "noactivate=True" in text
    assert "toolwindow=True" in text


def test_setwindowpos_failure_is_logged_not_raised(hud_env, caplog):
    _, fake, _ = hud_env
    fake.swp_result = False
    import logging
    with caplog.at_level(logging.WARNING):
        assert hud_compact.ensure_window_topmost(0x1234, "fail-test") \
            is False
    assert any("TOPMOST failed" in r.message for r in caplog.records)
