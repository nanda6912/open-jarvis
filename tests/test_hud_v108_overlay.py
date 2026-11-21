"""V10.8 compact floating-HUD overlay tests.

Runs offscreen (QT_QPA_PLATFORM=offscreen). Uses fake QScreen /
MonitorManager / HUD objects — no real display, GPU or microphone.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PySide6.QtCore import QObject, QRect
from PySide6.QtWidgets import QApplication

from src.controllers.monitor_manager import MonitorInfo, MonitorRect
from src.ui.hologram.hud_compact import CompactHologramHUD
from src.ui.hologram.hud_controller import (
    HUD_SIZE_DEFAULT,
    HUD_SIZE_MAX,
    HUD_SIZE_MIN,
    HudController,
)
from src.ui.hologram.orb_renderer import OrbGLWidget


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeScreen:
    """QScreen-like object in *logical* pixels."""

    def __init__(self, name="", model="", manufacturer="", serial="",
                 geom=None, avail=None, dpr=1.0):
        self._name = name
        self._model = model
        self._manufacturer = manufacturer
        self._serial = serial
        self._geom = geom or QRect(0, 0, 1920, 1080)
        self._avail = avail or QRect(
            self._geom.x(), self._geom.y(),
            self._geom.width(), self._geom.height() - 48)
        self._dpr = dpr

    def geometry(self):
        return self._geom

    def availableGeometry(self):
        return self._avail

    def name(self):
        return self._name

    def model(self):
        return self._model

    def manufacturer(self):
        return self._manufacturer

    def serialNumber(self):
        return self._serial

    def devicePixelRatio(self):
        return self._dpr


def _laptop_screen():
    """1920x1080 @125% -> 1536x864 logical, primary, right-hand."""
    return FakeScreen(
        name="\\\\.\\DISPLAY1", model="", manufacturer="AU Optronics",
        serial="", geom=QRect(0, 0, 1536, 864),
        avail=QRect(0, 0, 1536, 816), dpr=1.25)


def _dell_screen():
    """Dell 1908FP, 1280x1024 @100%, left-hand at negative X."""
    return FakeScreen(
        name="DELL 1908FP", model="DELL 1908FP",
        manufacturer="Dell Inc.", serial="G474H8A6AA71",
        geom=QRect(-1280, 110, 1280, 1024),
        avail=QRect(-1280, 110, 1280, 976), dpr=1.0)


def _mon(index, device, bounds, primary=False):
    """MonitorInfo with *physical-pixel* bounds (PMv2-aware user32)."""
    l, t, r, b = bounds
    return MonitorInfo(
        index=index, device=device,
        bounds=MonitorRect(l, t, r, b),
        work_area=MonitorRect(l, t, r, b - 40),
        primary=primary)


def _two_monitor_setup():
    """(screens, monitors, primary) mirroring the real desktop."""
    laptop = _laptop_screen()
    dell = _dell_screen()
    screens = [laptop, dell]
    mons = [
        # Physical pixels: 1920x1080 and 1280x1024.
        _mon(1, "\\\\.\\DISPLAY1", (0, 0, 1920, 1080), primary=True),
        _mon(2, "\\\\.\\DISPLAY2", (-1280, 110, 0, 1134)),
    ]
    return screens, mons, laptop


class FakeMonitorManager:
    def __init__(self, mons):
        self._mons = list(mons)

    def set(self, mons):
        self._mons = list(mons)

    def monitors(self, refresh=True):
        return list(self._mons)


class GeomFakeHud(QObject):
    """QWidget-like fake with geometry + lock semantics."""

    def __init__(self, screen=None):
        super().__init__()
        self._geom = QRect(0, 0, 10, 10)
        self._visible = False
        self.set_calls = []
        self.lock_calls = []
        self._screen = screen
        self.hideRequested = _Sig()
        self.topmostChanged = _Sig()
        self.fullscreenChanged = _Sig()
        self.positionChanged = _Sig()

    def geometry(self):
        return self._geom

    def setGeometry(self, rect):
        r = QRect(rect)
        self.set_calls.append(r)
        self._geom = r

    def show(self):
        self._visible = True

    def hide(self):
        self._visible = False

    def isVisible(self):
        return self._visible

    def raise_(self):
        pass

    def activateWindow(self):
        pass

    def screen(self):
        return self._screen

    def set_topmost(self, on, block=True):
        pass

    def set_fullscreen(self, on, block=True):
        pass

    def set_position_locked(self, locked):
        self.lock_calls.append(bool(locked))

    def close(self):
        self.hide()

    def deleteLater(self):
        pass


class _Sig:
    def __init__(self):
        self._subs = []

    def connect(self, fn):
        self._subs.append(fn)

    def emit(self, *a):
        for fn in self._subs:
            fn(*a)


def _store(initial=None):
    store = dict(initial or {})
    return store, store.get, store.__setitem__


def _ctl(store=None, screens=None, mons=None, hud=None, config=None):
    store = store if store is not None else {}
    get = lambda k: store.get(k)
    set_ = lambda k, v: store.__setitem__(k, v)
    mm = FakeMonitorManager(mons or [])
    cfg = config or {}
    ctl = HudController(
        settings_get=get,
        settings_set=set_,
        monitor_manager=mm,
        config_get=lambda k: cfg.get(k),
        hud_factory=lambda **kw: hud,
        bridge_factory=lambda h, **kw: None,
    )
    ctl._screens = lambda: list(screens or [])
    # primary screen = first screen flagged by caller convention
    primary = next((s for s in (screens or [])
                    if getattr(s, "_is_primary", False)), None)
    if primary is None and screens:
        primary = screens[0]
    ctl._primary_screen = lambda: primary
    return ctl, mm, store


# ---------------------------------------------------------------------------
# 1. Exactly one HUD instance
# ---------------------------------------------------------------------------
def test_exactly_one_hud_and_one_orb(qapp):
    hud = CompactHologramHUD(target_fps=30)
    assert len(hud.findChildren(OrbGLWidget)) == 1
    assert hud.size().width() == hud.size().height() == HUD_SIZE_DEFAULT
    # Window flags: tool window (no taskbar), frameless, topmost,
    # non-activating.
    flags = hud.windowFlags()
    from PySide6.QtCore import Qt
    assert flags & Qt.WindowType.Tool
    assert flags & Qt.WindowType.FramelessWindowHint
    assert flags & Qt.WindowType.WindowStaysOnTopHint
    assert flags & Qt.WindowType.WindowDoesNotAcceptFocus
    hud.close()


# ---------------------------------------------------------------------------
# 2-6. Monitor selection / placement / DPI
# ---------------------------------------------------------------------------
def test_dell_identity_selection(qapp):
    screens, mons, primary = _two_monitor_setup()
    laptop, dell = screens
    ctl, mm, store = _ctl(
        screens=screens, mons=mons,
        config={"hud.monitor": "dell"})
    assert ctl._target_screen() is dell
    screen, reason = ctl._resolve_target_screen()
    assert screen is dell and "dell" in reason


def test_dell_match_without_optional_metadata(qapp):
    """Model/name alone must match even when manufacturer and serial
    are empty — Qt does not always expose EDID fields."""
    laptop = _laptop_screen()
    dell = FakeScreen(
        name="DELL 1908FP", model="DELL 1908FP",
        manufacturer="", serial="",
        geom=QRect(-1280, 110, 1280, 1024),
        avail=QRect(-1280, 110, 1280, 976), dpr=1.0)
    screens = [laptop, dell]
    mons = [
        _mon(1, "\\\\.\\DISPLAY1", (0, 0, 1920, 1080), primary=True),
        _mon(2, "\\\\.\\DISPLAY2", (-1280, 110, 0, 1134)),
    ]
    ctl, mm, store = _ctl(screens=screens, mons=mons,
                          config={"hud.monitor": "dell"})
    assert ctl._target_screen() is dell


def test_stale_numeric_user_pref_cannot_override_dell(qapp):
    """Regression: persisted ``hud.monitor='1'`` (auto-saved by the
    pre-V10.8 HUD) must not pin the HUD to the laptop when the config
    default is the Dell identity."""
    screens, mons, primary = _two_monitor_setup()
    laptop, dell = screens
    store = {"hud.monitor": "1"}   # stale legacy index -> laptop
    ctl, mm, _ = _ctl(store=store, screens=screens, mons=mons,
                      config={"hud.monitor": "dell"})
    assert ctl._target_screen() is dell
    # The legacy value is still a fallback when it is the only signal.
    ctl2, mm2, _ = _ctl(store={"hud.monitor": "1"}, screens=screens,
                        mons=mons, config={"hud.monitor": "auto"})
    assert ctl2._target_screen() is laptop


def test_deliberate_user_monitor_pref_respected(qapp):
    """An explicit non-numeric user identity wins over the config
    default; a deliberate ``displayN`` choice is honoured too."""
    screens, mons, primary = _two_monitor_setup()
    laptop, dell = screens
    # User explicitly chose the laptop by device identity.
    ctl, mm, _ = _ctl(store={"hud.monitor": "display1"},
                      screens=screens, mons=mons,
                      config={"hud.monitor": "dell"})
    assert ctl._target_screen() is laptop
    # User explicitly chose the Dell by EDID/model name.
    ctl2, mm2, _ = _ctl(store={"hud.monitor": "DELL 1908FP"},
                        screens=screens, mons=mons,
                        config={"hud.monitor": "auto"})
    assert ctl2._target_screen() is dell


def test_auto_prefers_dell_then_display2(qapp):
    screens, mons, primary = _two_monitor_setup()
    ctl, mm, store = _ctl(screens=screens, mons=mons,
                          config={"hud.monitor": "auto"})
    assert ctl._target_screen() is screens[1]


def test_display2_device_fallback(qapp):
    """When the Dell identity is absent but DISPLAY2 exists, the HUD
    still lands on the second Windows display device."""
    laptop = _laptop_screen()
    generic = FakeScreen(
        name="Generic PnP", model="", manufacturer="",
        serial="", geom=QRect(-1280, 110, 1280, 1024),
        avail=QRect(-1280, 110, 1280, 976), dpr=1.0)
    screens = [laptop, generic]
    mons = [
        _mon(1, "\\\\.\\DISPLAY1", (0, 0, 1920, 1080), primary=True),
        _mon(2, "\\\\.\\DISPLAY2", (-1280, 110, 0, 1134)),
    ]
    ctl, mm, store = _ctl(screens=screens, mons=mons,
                          config={"hud.monitor": "dell"})
    assert ctl._target_screen() is generic


def test_primary_fallback_single_monitor(qapp):
    laptop = _laptop_screen()
    mons = [_mon(1, "\\\\.\\DISPLAY1", (0, 0, 1920, 1080), primary=True)]
    ctl, mm, store = _ctl(screens=[laptop], mons=mons,
                          config={"hud.monitor": "dell"})
    assert ctl._target_screen() is laptop


def test_bottom_left_placement(qapp):
    screens, mons, primary = _two_monitor_setup()
    dell = screens[1]
    ctl, mm, store = _ctl(screens=screens, mons=mons,
                          config={"hud.monitor": "dell"})
    g = ctl._desired_geometry(dell)
    ag = dell.availableGeometry()
    side = HUD_SIZE_DEFAULT
    assert g.width() == g.height() == side
    # Bottom-left: 28 logical px from left and bottom edges.
    assert g.x() == ag.left() + 28
    assert g.y() == ag.top() + ag.height() - side - 28
    assert g.right() < ag.right() and g.bottom() <= ag.bottom()


def test_negative_screen_coordinates(qapp):
    screens, mons, primary = _two_monitor_setup()
    dell = screens[1]
    ctl, mm, store = _ctl(screens=screens, mons=mons,
                          config={"hud.monitor": "dell"})
    g = ctl._desired_geometry(dell)
    assert g.x() < 0  # left monitor lives at negative X
    ag = dell.availableGeometry()
    assert ag.left() <= g.x() and g.right() <= ag.right() + 1


def test_dpi_aware_monitor_bounds_matching(qapp):
    """MonitorManager bounds are physical px; QScreen is logical.
    The 125%-scaled laptop must still match via devicePixelRatio."""
    screens, mons, primary = _two_monitor_setup()
    laptop, dell = screens
    ctl, mm, store = _ctl(screens=screens, mons=mons)
    # Physical laptop bounds -> logical laptop screen.
    assert ctl._screen_for_bounds(mons[0].bounds) is laptop
    # Physical dell bounds -> dell screen.
    assert ctl._screen_for_bounds(mons[1].bounds) is dell
    # And back: display index resolution.
    assert ctl._display_index_for_screen(dell) == 2
    assert ctl._display_index_for_screen(laptop) == 1
    assert ctl._screen_for_display_index(2) is dell


def test_geometry_is_square(qapp):
    screens, mons, primary = _two_monitor_setup()
    ctl, mm, store = _ctl(screens=screens, mons=mons,
                          config={"hud.monitor": "dell"})
    for side in (HUD_SIZE_MIN, 300, HUD_SIZE_MAX):
        store["hud.size"] = str(side)
        g = ctl._desired_geometry(screens[1])
        assert g.width() == g.height() == side


def test_saved_position_and_size_restored(qapp):
    screens, mons, primary = _two_monitor_setup()
    dell = screens[1]
    store = {
        "hud.size": "300",
        "hud.position": "-1000,500",
        "hud.position_screen": "DELL 1908FP",
    }
    ctl, mm, _ = _ctl(store=store, screens=screens, mons=mons,
                      config={"hud.monitor": "dell"})
    g = ctl._desired_geometry(dell)
    assert (g.x(), g.y(), g.width()) == (-1000, 500, 300)
    # A position saved for a *different* screen is not applied.
    laptop = screens[0]
    g2 = ctl._desired_geometry(laptop)
    assert not (g2.x() == -1000 and g2.y() == 500)


# ---------------------------------------------------------------------------
# 9. Monitor removal / reconnection
# ---------------------------------------------------------------------------
def test_dell_disconnect_moves_to_primary_and_back(qapp):
    screens, mons, primary = _two_monitor_setup()
    laptop, dell = screens
    mutable_screens = list(screens)
    mm = FakeMonitorManager(mons)
    store, get, set_ = _store()
    hud = GeomFakeHud(screen=dell)
    ctl = HudController(
        settings_get=get, settings_set=set_, monitor_manager=mm,
        config_get=lambda k: {"hud.monitor": "dell"}.get(k),
        hud_factory=lambda **kw: hud,
        bridge_factory=lambda h, **kw: None,
    )
    ctl._screens = lambda: list(mutable_screens)
    ctl._primary_screen = lambda: laptop

    assert ctl._target_screen() is dell
    # Disconnect the Dell: screens + monitors shrink.
    mutable_screens[:] = [laptop]
    mm.set([mons[0]])
    assert ctl._target_screen() is laptop
    # Reconnect: identity resolution returns to the Dell.
    mutable_screens[:] = [laptop, dell]
    mm.set(mons)
    assert ctl._target_screen() is dell


def test_stale_pref_dell_reconnect_returns_to_dell(qapp):
    """With a stale numeric user pref + Dell config, disconnecting the
    Dell falls back to primary and reconnecting returns to the Dell."""
    screens, mons, primary = _two_monitor_setup()
    laptop, dell = screens
    mutable_screens = list(screens)
    mm = FakeMonitorManager(mons)
    store, get, set_ = _store({"hud.monitor": "1"})
    ctl = HudController(
        settings_get=get, settings_set=set_, monitor_manager=mm,
        config_get=lambda k: {"hud.monitor": "dell"}.get(k),
        hud_factory=lambda **kw: GeomFakeHud(),
        bridge_factory=lambda h, **kw: None,
    )
    ctl._screens = lambda: list(mutable_screens)
    ctl._primary_screen = lambda: laptop

    assert ctl._target_screen() is dell
    mutable_screens[:] = [laptop]
    mm.set([mons[0]])
    assert ctl._target_screen() is laptop
    mutable_screens[:] = [laptop, dell]
    mm.set(mons)
    assert ctl._target_screen() is dell


# ---------------------------------------------------------------------------
# 10. Lock / unlock interaction modes
# ---------------------------------------------------------------------------
def test_lock_unlock_persists_and_reaches_hud(qapp):
    store, get, set_ = _store()
    hud = GeomFakeHud()
    ctl = HudController(
        settings_get=get, settings_set=set_,
        hud_factory=lambda **kw: hud,
        bridge_factory=lambda h, **kw: None,
    )
    ctl.hud = hud
    assert ctl.is_position_locked() is True
    ctl.set_position_locked(False)
    assert store["hud.locked"] == "false"
    assert hud.lock_calls == [False]
    assert ctl.is_position_locked() is False
    ctl.set_position_locked(True)
    assert store["hud.locked"] == "true"
    assert hud.lock_calls == [False, True]


def test_unlocked_recheck_does_not_fight_user(qapp):
    """While unlocked with a stable display, the 5 s recheck must not
    snap the HUD back to the saved position."""
    screens, mons, primary = _two_monitor_setup()
    store = {"hud.locked": "false"}
    hud = GeomFakeHud(screen=screens[1])
    ctl, mm, _ = _ctl(store=store, screens=screens, mons=mons,
                      hud=hud, config={"hud.monitor": "dell"})
    ctl._target_screen = lambda: screens[1]
    ctl._shown = True
    ctl.hud = hud
    # Establish baseline signature.
    ctl._recheck_placement()
    n = len(hud.set_calls)
    # Simulate a user drag to a new position.
    hud.setGeometry(QRect(-900, 600, 320, 320))
    for _ in range(3):
        ctl._recheck_placement()
    assert len(hud.set_calls) == n + 1  # only the drag itself
    assert hud.geometry() == QRect(-900, 600, 320, 320)


# ---------------------------------------------------------------------------
# Compact window behaviour
# ---------------------------------------------------------------------------
def test_compact_window_square_enforced(qapp):
    hud = CompactHologramHUD(target_fps=30)
    hud.show()
    qapp.processEvents()
    hud.resize(300, 200)
    qapp.processEvents()
    assert hud.width() == hud.height()
    hud.close()


def test_compact_window_click_through_state(qapp):
    hud = CompactHologramHUD(target_fps=30)
    hud.show()
    qapp.processEvents()
    hud.set_position_locked(True)
    assert hud.position_locked is True
    hud.set_position_locked(False)
    assert hud.position_locked is False
    # Fallback path (offscreen winId may be 0): the widget attribute
    # mirrors the lock state when native exstyle is unavailable.
    hud.close()


def test_compact_window_status_and_orb_api(qapp):
    hud = CompactHologramHUD(target_fps=30)
    hud.set_orb_mode("listening")
    assert hud.orb.mode == "listening"
    hud.set_status("speaking", "jarvis output")
    assert hud.status_label.text() == "RESPONDING"
    hud.set_orb_level(0.8)
    assert hud.orb._level == pytest.approx(0.8)
    assert len(hud.waveform._samples) == 1
    hud.set_orb_speaking(True)
    hud.pulse_orb()
    hud.close()


# ---------------------------------------------------------------------------
# 11/12. Voice-state + audio-pipeline preservation
# ---------------------------------------------------------------------------
def test_bridge_drives_compact_hud(qapp):
    """The existing HudBridge must drive the compact window unchanged."""
    from tests.test_hud_v107 import FakeState, FakeMic, FakeSTT, FakeSession, _bridge

    hud = CompactHologramHUD(target_fps=30)
    bridge = _bridge(
        hud,
        state_manager=FakeState("STANDBY"),
        microphone_controller=FakeMic({"rms": 0.05}),
        stt_service=FakeSTT(0.0),
        session_controller=FakeSession(),
    )
    bridge._session_active = True
    bridge._last_activity = "listening"
    bridge._level_tick()
    assert hud.orb.mode == "listening"
    bridge._last_activity = "speaking"
    bridge._speaking_state = True
    bridge._level_tick()
    assert hud.orb.mode == "speaking"
    bridge._last_activity = "standby"
    bridge._speaking_state = False
    bridge._level_tick()
    assert hud.orb.mode == "standby"
    bridge.shutdown()
    hud.close()


def test_no_extra_microphone_stream_in_compact_module():
    import inspect

    import src.ui.hologram.hud_compact as mod

    src = inspect.getsource(mod)
    assert "sounddevice" not in src
    assert "import portaudio" not in src


def test_hud_controller_still_forwards_bridge_kwargs(qapp):
    """start() must keep forwarding every voice dependency to the bridge."""
    captured = {}

    class FakeHud:
        hideRequested = None
        topmostChanged = None
        fullscreenChanged = None
        positionChanged = None

    get, set_ = (lambda k: None), (lambda k, v: None)

    def bridge_factory(hud, **kw):
        captured.update(kw)
        return type("B", (), {"shutdown": lambda s: None})()

    ctl = HudController(
        get, set_,
        hud_factory=lambda **kw: FakeHud(),
        bridge_factory=bridge_factory,
    )
    mic = object()
    ok = ctl.start(
        voice_signals=object(), state_manager=object(),
        session_controller=object(), microphone_controller=mic,
        stt_service=object(), tts_service=object(),
        desktop_probe=object(),
    )
    assert ok is True
    assert captured["microphone_controller"] is mic
    ctl.shutdown()
