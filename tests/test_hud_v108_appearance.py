"""V10.8 balanced-appearance tests.

Offscreen; uses the real CompactHologramHUD/OrbGLWidget (GL object
creation is lazy — appearance state is plain Python until paintGL).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from PySide6.QtWidgets import QApplication

from src.ui.hologram.hud_appearance import (
    BALANCED,
    PARAMS,
    PRESETS,
    HudAppearance,
)
from src.ui.hologram.hud_compact import CompactHologramHUD
from src.ui.hologram.hud_controller import HudController


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def _store(initial=None):
    store = dict(initial or {})
    return store, store.get, store.__setitem__


def _cfg_get(cfg):
    """Dot-notation resolver over a nested dict (like config.get)."""
    def get(key, default=None):
        node = cfg
        for part in str(key).split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node
    return get


def _appearance(store=None, cfg=None, on_change=None):
    store = store if store is not None else {}
    get = lambda k: store.get(k)
    set_ = lambda k, v: store.__setitem__(k, v)
    return HudAppearance(get, set_,
                         config_get=_cfg_get(cfg or {}),
                         on_change=on_change), store


# ---------------------------------------------------------------------------
# 1-2. Balanced default on fresh install
# ---------------------------------------------------------------------------
def test_balanced_is_default_on_fresh_install(qapp):
    app_, store = _appearance()
    assert app_.preset == "balanced"
    for param in PARAMS:
        assert app_.value(param) == pytest.approx(BALANCED[param])


def test_balanced_values_match_spec(qapp):
    assert BALANCED["brightness"] == 0.65
    assert BALANCED["glow"] == 0.55
    assert BALANCED["opacity"] == 0.95
    assert BALANCED["ring"] == 0.75
    assert BALANCED["particle"] == 0.60
    assert BALANCED["waveform"] == 0.70
    assert BALANCED["speed"] == 1.00
    # Spec: presets exist and stay inside 0..1.
    for name, vals in PRESETS.items():
        for p in PARAMS:
            assert 0.0 <= vals[p] <= 1.0


def test_json_config_section_supplies_balanced(qapp):
    """config/default_settings.json hud.appearance -> balanced values."""
    cfg = {"hud": {"appearance": {
        "preset": "balanced", "brightness": 65, "glow": 55,
        "opacity": 95, "ring": 75, "particle": 60,
        "waveform": 70, "speed": 100}}}
    app_, _ = _appearance(cfg=cfg)
    assert app_.preset == "balanced"
    assert app_.value("brightness") == pytest.approx(0.65)


# ---------------------------------------------------------------------------
# 3-6. Presets, sliders, custom, reset
# ---------------------------------------------------------------------------
def test_preset_switch_updates_values_and_renderer(qapp):
    hud = CompactHologramHUD(target_fps=30)
    app_, store = _appearance()
    app_.apply_to(hud)
    assert hud.orb._appearance["brightness"] == pytest.approx(0.65)

    app_.set_preset("intense")
    assert app_.preset == "intense"
    assert app_.value("glow") == pytest.approx(PRESETS["intense"]["glow"])
    # on_change -> apply path used by the controller.
    app_.apply_to(hud)
    assert hud.orb._appearance["glow"] == pytest.approx(0.80)
    assert hud.orb._appearance["ring"] == pytest.approx(0.95)
    assert hud.waveform._intensity == pytest.approx(1.0)
    hud.close()


def test_slider_values_update_renderer(qapp):
    hud = CompactHologramHUD(target_fps=30)
    app_, store = _appearance()
    app_.set_value_percent("brightness", 40)
    app_.set_value_percent("particle", 30)
    app_.apply_to(hud)
    assert hud.orb._appearance["brightness"] == pytest.approx(0.40)
    assert hud.orb._appearance["particle"] == pytest.approx(0.30)
    hud.close()


def test_manual_slider_marks_custom(qapp):
    app_, store = _appearance()
    app_.set_preset("balanced")
    app_.set_value_percent("glow", 33)
    assert app_.preset == "custom"
    assert store["hud.appearance.preset"] == "custom"
    assert store["hud.appearance.glow"] == "33"
    # Preset re-apply restores a named preset.
    app_.set_preset("subtle")
    assert app_.preset == "subtle"
    assert app_.value("glow") == pytest.approx(
        PRESETS["subtle"]["glow"])


def test_reset_restores_balanced(qapp):
    app_, store = _appearance()
    app_.set_preset("intense")
    app_.set_value_percent("brightness", 10)
    app_.reset()
    assert app_.preset == "balanced"
    for p in PARAMS:
        assert app_.value(p) == pytest.approx(BALANCED[p])


# ---------------------------------------------------------------------------
# 7-8. Persistence + validation
# ---------------------------------------------------------------------------
def test_appearance_persists_across_restart(qapp):
    store = {}
    get = lambda k: store.get(k)
    set_ = lambda k, v: store.__setitem__(k, v)
    a1 = HudAppearance(get, set_, config_get=_cfg_get({}))
    a1.set_preset("subtle")
    a1.set_value_percent("waveform", 55)

    # "Restart": a brand-new manager over the same store.
    a2 = HudAppearance(get, set_, config_get=_cfg_get({}))
    assert a2.preset == "custom"
    assert a2.value("brightness") == pytest.approx(
        PRESETS["subtle"]["brightness"])
    assert a2.value("waveform") == pytest.approx(0.55)


def test_invalid_values_clamped_and_defaulted(qapp):
    store = {
        "hud.appearance.brightness": "250",   # -> clamp to 1.0
        "hud.appearance.glow": "-20",         # -> clamp to 0.0
        "hud.appearance.opacity": "garbage",  # -> balanced default
        "hud.appearance.ring": "",            # -> falls through to cfg
        "hud.appearance.preset": "bogus",     # -> balanced
    }
    cfg = {"hud": {"appearance": {"ring": 40}}}
    app_, _ = _appearance(store=store, cfg=cfg)
    assert app_.preset == "balanced"
    assert app_.value("brightness") == pytest.approx(1.0)
    assert app_.value("glow") == pytest.approx(0.0)
    assert app_.value("opacity") == pytest.approx(BALANCED["opacity"])
    assert app_.value("ring") == pytest.approx(0.40)


# ---------------------------------------------------------------------------
# 9-11. Voice-state modulation, GL stability, transparency
# ---------------------------------------------------------------------------
def test_voice_state_gain_does_not_overwrite_appearance(qapp):
    hud = CompactHologramHUD(target_fps=30)
    hud.orb.set_appearance(brightness=0.65, glow=0.55)
    hud.orb.set_mode("listening")
    # Run several animation ticks so the gain eases toward target.
    for _ in range(40):
        hud.orb._tick()
    # Mode gain moved away from 1.0 ...
    assert hud.orb._gain_b != pytest.approx(1.0)
    # ... but the saved appearance and the uniform base are untouched.
    assert hud.orb._appearance["brightness"] == pytest.approx(0.65)
    assert hud.orb._appearance["glow"] == pytest.approx(0.55)
    # Returning to standby eases back toward neutral.
    hud.orb.set_mode("standby")
    for _ in range(80):
        hud.orb._tick()
    assert hud.orb._gain_b == pytest.approx(1.0, abs=0.01)
    hud.close()


def test_animation_speed_scales_clock(qapp):
    hud = CompactHologramHUD(target_fps=30)
    orb = hud.orb
    orb.set_appearance(speed=0.5)
    t0 = orb._time
    for _ in range(10):
        orb._tick()
    slow_dt = orb._time - t0
    orb.set_appearance(speed=1.0)
    t0 = orb._time
    for _ in range(10):
        orb._tick()
    fast_dt = orb._time - t0
    assert fast_dt == pytest.approx(slow_dt * 2.0, rel=0.05)
    hud.close()


def test_appearance_never_rebuilds_gl(qapp):
    hud = CompactHologramHUD(target_fps=30)
    orb = hud.orb
    progs = (orb._prog_sphere, orb._prog_ring, orb._prog_point)
    buffers_before = list(getattr(orb, "_buffers", []))
    sphere_count = orb._sphere_vert_count
    orb.set_appearance(brightness=0.1, glow=1.0, opacity=0.2,
                       ring=0.3, particle=0.9, speed=0.5)
    # Same program objects, same buffer list, same vertex counts.
    assert (orb._prog_sphere, orb._prog_ring,
            orb._prog_point) == progs
    assert orb._buffers == buffers_before
    assert orb._sphere_vert_count == sphere_count
    hud.close()


def test_opacity_is_shader_level_not_window(qapp):
    from PySide6.QtCore import Qt
    hud = CompactHologramHUD(target_fps=30)
    hud.set_appearance(opacity=0.4, brightness=0.65)
    orb = hud.orb
    assert orb._appearance["opacity"] == pytest.approx(0.4)
    # Window translucency attribute and click-through lock unchanged.
    assert hud.testAttribute(
        Qt.WidgetAttribute.WA_TranslucentBackground)
    assert hud.windowOpacity() == pytest.approx(1.0)
    assert hud.position_locked is True
    hud.set_position_locked(False)
    assert hud.position_locked is False
    hud.close()


def test_controller_applies_appearance_on_start(qapp):
    store, get, set_ = _store()
    hud = CompactHologramHUD(target_fps=30)
    ctl = HudController(
        settings_get=get, settings_set=set_,
        hud_factory=lambda **kw: hud,
        bridge_factory=lambda h, **kw: None,
        config_get=_cfg_get({}),
    )
    ctl.hud = hud
    ctl._apply_appearance()
    assert hud.orb._appearance["brightness"] == pytest.approx(0.65)
    assert hud.waveform._intensity == pytest.approx(0.70)
    ctl.appearance.set_preset("intense")
    assert hud.orb._appearance["glow"] == pytest.approx(0.80)
    hud.close()
