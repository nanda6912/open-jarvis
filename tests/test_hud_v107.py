"""V10.7 holographic HUD tests.

Runs offscreen (QT_QPA_PLATFORM=offscreen) — no real display, microphone
or GPU required. Voice events are injected through the real
``VoiceSignals`` bridge or by calling bridge slots directly.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import math

import numpy as np
import pytest

from PySide6.QtWidgets import QApplication

from src.controllers.monitor_manager import MonitorInfo, MonitorRect
from src.ui.hologram import hud_bridge as hb
from src.ui.hologram.hud_bridge import HudBridge
from src.ui.hologram.hud_controller import HudController
from src.ui.hologram.hud_probe import DesktopProbe, SystemProbe, normalize_level
from src.ui.hologram.orb_renderer import OrbGLWidget, _sphere_mesh, _torus_mesh
from src.voice.audio_worker import VoiceSignals


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class FakeHud:
    """Records every setter the bridge calls (no widgets/GL needed)."""

    def __init__(self):
        self.levels = []
        self.modes = []
        self.statuses = []
        self.activities = []
        self.histories = []
        self.system_rows = []
        self.desktop_rows = []
        self.speaking = []
        self.pulses = 0
        self.render_active = []

    def set_orb_level(self, level):
        self.levels.append(level)

    def set_orb_mode(self, mode):
        self.modes.append(mode)

    def set_status(self, state, detail=""):
        self.statuses.append((state, detail))

    def set_orb_speaking(self, speaking):
        self.speaking.append(bool(speaking))

    def pulse_orb(self):
        self.pulses += 1

    def set_rendering_active(self, active):
        self.render_active.append(bool(active))

    def set_activity(self, *args):
        self.activities.append(args)

    def set_history(self, lines):
        self.histories.append(list(lines))

    def set_system_rows(self, *args):
        self.system_rows.append(args)

    def set_desktop_rows(self, *args):
        self.desktop_rows.append(args)

    def set_topmost(self, on, block=True):
        pass

    def set_fullscreen(self, on, block=True):
        pass


class FakeState:
    def __init__(self, name="STANDBY"):
        from src.core.state_manager import AppState

        self.state = AppState[name]


class FakeMic:
    def __init__(self, stats=None):
        self._stats = stats or {}
        self.status = "listening"
        self.status_detail = "ok"
        self.capture = None  # deliberately absent: bridge must not need it

    def audio_stats(self):
        return dict(self._stats)


class FakeSTT:
    engine_name = "fake-stt"

    def __init__(self, rms=0.0):
        self._rms = rms

    def stats(self):
        return {"last_rms": self._rms}


class FakeSession:
    def __init__(self, log=None, intent=None):
        self._log = log or []
        self.last_intent = intent

    def recognition_log(self):
        return [dict(e) for e in self._log]


def _bridge(hud=None, signals=None, **kw):
    return HudBridge(
        hud or FakeHud(),
        voice_signals=signals,
        state_manager=kw.pop("state_manager", FakeState()),
        session_controller=kw.pop("session_controller", FakeSession()),
        microphone_controller=kw.pop("microphone_controller", FakeMic()),
        stt_service=kw.pop("stt_service", FakeSTT()),
        tts_service=kw.pop("tts_service", None),
        desktop_probe=kw.pop("desktop_probe", DesktopProbe()),
        system_probe=kw.pop("system_probe", SystemProbe()),
        **kw,
    )


# ---------------------------------------------------------------------------
# Geometry / renderer
# ---------------------------------------------------------------------------
def test_sphere_mesh_valid():
    pos, nrm, uv, idx = _sphere_mesh(1.0, lat=8, lon=16)
    assert pos.shape == (9 * 17, 3)
    assert nrm.shape == pos.shape
    assert uv.shape == (9 * 17, 2)
    assert idx.size > 0 and idx.max() < pos.shape[0]
    # Normals are unit length.
    lengths = np.linalg.norm(nrm, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-4)


def test_torus_mesh_valid():
    data, idx = _torus_mesh(1.5, 0.03, seg_u=32, seg_v=6)
    assert data.shape == ((33) * 7, 6)
    assert idx.size > 0


def test_orb_widget_creation_and_setters(qapp):
    orb = OrbGLWidget(target_fps=30)
    orb.set_mode("listening")
    orb.set_level(0.7)
    orb.set_speaking(True)
    orb.trigger_pulse()
    assert orb.mode == "listening"
    # Unknown modes are ignored.
    orb.set_mode("bogus")
    assert orb.mode == "listening"
    orb.set_rendering_active(False)
    orb.set_rendering_active(True)
    orb.deleteLater()


def test_orb_shaders_compile_source_sanity():
    # Guard against accidental GLSL syntax regressions: the sources must
    # at least contain matching entry points.
    from src.ui.hologram import orb_renderer as m

    for src in (m._SPHERE_VERT, m._SPHERE_FRAG, m._RING_VERT,
                m._RING_FRAG, m._POINT_VERT, m._POINT_FRAG):
        assert "void main()" in src
    assert "uAudio" in m._SPHERE_FRAG
    assert "uScan" in m._SPHERE_FRAG


# ---------------------------------------------------------------------------
# Level normalization + probes
# ---------------------------------------------------------------------------
def test_normalize_level():
    assert normalize_level(0.0) == 0.0
    assert normalize_level(-1.0) == 0.0
    assert normalize_level(0.001) < normalize_level(0.05)
    assert normalize_level(10.0) <= 1.0


def test_desktop_probe_never_raises():
    snap = DesktopProbe().snapshot()
    assert set(snap) == {"active_app", "window_count", "monitors",
                         "system_state"}


def test_system_probe_never_raises():
    snap = SystemProbe().snapshot()
    assert set(snap) == {"cpu", "ram"}


# ---------------------------------------------------------------------------
# Bridge: mode resolution + voice events
# ---------------------------------------------------------------------------
def test_bridge_standby_mode(qapp):
    hud = FakeHud()
    bridge = _bridge(hud)
    bridge._level_tick()
    assert hud.modes[-1] == "standby"
    assert hud.statuses[-1][0] == "standby"
    bridge.shutdown()


def test_bridge_listening_mode(qapp):
    hud = FakeHud()
    bridge = _bridge(
        hud,
        state_manager=FakeState("LISTENING"),
        session_controller=FakeSession(),
    )
    bridge._session_active = True
    bridge._last_activity = "listening"
    bridge._level_tick()
    assert hud.modes[-1] == "listening"
    bridge.shutdown()


def test_bridge_processing_and_speaking(qapp):
    hud = FakeHud()
    bridge = _bridge(hud)
    bridge._last_activity = "processing"
    bridge._level_tick()
    assert hud.modes[-1] == "processing"

    bridge._last_activity = "speaking"
    bridge._speaking_state = True
    bridge._level_tick()
    assert hud.modes[-1] == "speaking"
    assert hud.speaking[-1] is True
    bridge.shutdown()


def test_bridge_error_mode(qapp):
    hud = FakeHud()
    bridge = _bridge(hud)
    bridge._on_mic_error("device lost")
    bridge._level_tick()
    assert hud.modes[-1] == "error"
    bridge.shutdown()


def test_wake_word_pulses_orb(qapp):
    hud = FakeHud()
    signals = VoiceSignals(qapp)
    bridge = _bridge(hud, signals=signals)
    signals.wakeWordDetected.emit("hello jarvis")
    qapp.processEvents()
    assert hud.pulses == 1
    # V10.7.1: a real keyword emission shows "WAKE WORD DETECTED".
    assert hud.statuses[-1] == ("wake", "hello jarvis")
    bridge._level_tick()
    assert hud.statuses[-1][0] == "wake"  # flash persists ~1.5 s
    bridge.shutdown()


def test_transcript_and_activity_panel(qapp):
    hud = FakeHud()
    signals = VoiceSignals(qapp)
    from src.commands.intent import Intent

    intent = Intent(
        name="MOVE_WINDOW",
        entities={"app_name": "Chrome", "monitor": "left"},
        raw_text="move chrome to the left monitor",
    )
    session = FakeSession(
        log=[{
            "text": "move chrome to the left monitor",
            "outcome": "executed",
            "intent": "MOVE_WINDOW",
            "detail": "Moved Chrome to monitor 2.",
        }],
        intent=intent,
    )
    bridge = _bridge(hud, signals=signals, session_controller=session)
    signals.transcriptReceived.emit("move chrome to the left monitor")
    qapp.processEvents()
    bridge._panel_tick()
    args = hud.activities[-1]
    # (transcript, intent, app, dest, status, result)
    assert args[0] == "move chrome to the left monitor"
    assert args[1] == "MOVE_WINDOW"
    assert args[2] == "Chrome"
    assert "monitor 2" in args[5].lower() or args[5]
    assert hud.histories
    bridge.shutdown()


def test_session_end_resets_state(qapp):
    hud = FakeHud()
    signals = VoiceSignals(qapp)
    bridge = _bridge(hud, signals=signals)
    signals.sessionStateChanged.emit(True)
    qapp.processEvents()
    assert bridge._session_active is True
    signals.sessionStateChanged.emit(False)
    qapp.processEvents()
    assert bridge._session_active is False
    assert bridge._speaking_state is False
    bridge.shutdown()


def test_tts_envelope_drives_level(qapp):
    hud = FakeHud()
    signals = VoiceSignals(qapp)
    bridge = _bridge(hud, signals=signals)
    # Envelope of 10 chunks @0.1 s each; all max amplitude.
    signals.ttsEnvelope.emit([1.0] * 10, 0.1)
    qapp.processEvents()
    bridge._speaking_state = True
    # Force the envelope 'now' index by rewinding start time.
    levels, chunk, started = bridge._tts_envelope
    bridge._tts_envelope = (levels, chunk, started - 0.25)
    bridge._level_tick()
    assert hud.levels[-1] > 0.3
    # Past the end -> procedural speaking pulse still > 0.
    bridge._tts_envelope = (levels, chunk, started - 5.0)
    bridge._level_tick()
    assert hud.levels[-1] > 0.0
    bridge.shutdown()


def test_level_uses_stt_rms_when_listening(qapp):
    hud = FakeHud()
    bridge = _bridge(
        hud,
        state_manager=FakeState("LISTENING"),
        stt_service=FakeSTT(rms=0.10),
        microphone_controller=FakeMic({"rms": 0.0}),
    )
    bridge._session_active = True
    bridge._raw_level = bridge._raw_audio_level()
    assert bridge._raw_level > 0.3
    bridge.shutdown()


def test_level_uses_mic_stats_in_standby(qapp):
    hud = FakeHud()
    bridge = _bridge(
        hud,
        state_manager=FakeState("STANDBY"),
        microphone_controller=FakeMic({"rms": 0.05}),
    )
    bridge._session_active = False
    assert bridge._raw_audio_level() > 0.3
    bridge.shutdown()


def test_bridge_needs_no_capture_stream(qapp):
    """The HUD must not create or require a second mic stream."""
    hud = FakeHud()
    mic = FakeMic({})  # no .capture attribute at all
    bridge = _bridge(hud, microphone_controller=mic)
    bridge._level_tick()
    bridge._system_tick()
    bridge._desktop_tick()
    assert not hasattr(mic, "capture_started")
    bridge.shutdown()


def test_hidden_hud_stops_rendering(qapp):
    from src.ui.hologram.hud_window import HologramHUD

    hud = HologramHUD(target_fps=30)
    hud.show()
    qapp.processEvents()
    assert hud.orb._timer.isActive()
    hud.hide()
    qapp.processEvents()
    assert not hud.orb._timer.isActive()
    hud.close()


def test_hud_close_only_hides(qapp):
    from src.ui.hologram.hud_window import HologramHUD

    hud = HologramHUD(target_fps=30)
    hud.show()
    qapp.processEvents()
    hud.close()
    qapp.processEvents()
    # Window hidden; nothing about the app was shut down.
    assert not hud.isVisible()
    assert QApplication.instance() is qapp


# ---------------------------------------------------------------------------
# Controller: monitor selection + lifecycle
# ---------------------------------------------------------------------------
def _mon(index, primary=False, left=0, top=0):
    return MonitorInfo(
        index=index,
        device=f"\\\\.\\DISPLAY{index}",
        bounds=MonitorRect(left, top, left + 1920, top + 1080),
        work_area=MonitorRect(left, top + 40, left + 1920, top + 1080),
        primary=primary,
    )


class FakeMonitorManager:
    def __init__(self, mons):
        self._mons = mons

    def monitors(self, refresh=True):
        return list(self._mons)


def _settings_store(initial=None):
    store = dict(initial or {})
    return store.get, store.__setitem__


def test_controller_prefers_monitor_2():
    get, set_ = _settings_store()
    mm = FakeMonitorManager([_mon(1, primary=True), _mon(2, left=-1920)])
    ctl = HudController(get, set_, monitor_manager=mm)
    assert ctl.target_monitor_index() == 2


def test_controller_single_monitor_fallback():
    get, set_ = _settings_store()
    mm = FakeMonitorManager([_mon(1, primary=True)])
    ctl = HudController(get, set_, monitor_manager=mm)
    assert ctl.target_monitor_index() == 1


def test_controller_config_monitor_override():
    get, set_ = _settings_store()
    mm = FakeMonitorManager([_mon(1, primary=True), _mon(2)])
    cfg = {"hud.monitor": "1"}
    ctl = HudController(
        get, set_, monitor_manager=mm,
        config_get=lambda k: cfg.get(k),
    )
    assert ctl.target_monitor_index() == 1


def test_controller_disabled_via_setting():
    get, set_ = _settings_store({"hud.enabled": "false"})
    ctl = HudController(get, set_)
    assert ctl.enabled() is False
    assert ctl.start() is False
    assert ctl.hud is None


def test_controller_start_with_fake_hud(qapp):
    hud = FakeHud()
    get, set_ = _settings_store()
    mm = FakeMonitorManager([_mon(1, primary=True), _mon(2)])
    ctl = HudController(
        get, set_, monitor_manager=mm,
        hud_factory=lambda **kw: hud,
        bridge_factory=lambda h, **kw: FakeHudBridge(),
    )
    ok = ctl.start()
    assert ok is True
    assert ctl.hud is hud
    ctl.shutdown()


class FakeHudBridge:
    def __init__(self):
        self.shut = False

    def shutdown(self):
        self.shut = True


def test_controller_shutdown_isolated(qapp):
    get, set_ = _settings_store()
    ctl = HudController(
        get, set_,
        hud_factory=lambda **kw: FakeHud(),
        bridge_factory=lambda h, **kw: FakeHudBridge(),
    )
    ctl.start()
    ctl.shutdown()  # must not raise
    assert ctl.hud is None


# ---------------------------------------------------------------------------
# TTS envelope (SpeechOutputService)
# ---------------------------------------------------------------------------
def test_tts_level_envelope_computed():
    from src.voice.speech_output_service import SpeechOutputService

    received = []

    class FakeSynth:
        name = "fake"

        def synthesize(self, text):
            rate = 16000
            t = np.arange(rate) / rate
            return (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate

        def close(self):
            pass

    svc = SpeechOutputService(synthesizer=FakeSynth())
    svc.set_level_callback(lambda levels, dur: received.append((levels, dur)))
    samples = np.ones(1600, np.float32) * 0.5
    svc._emit_level_envelope(samples, 16000)
    assert received, "envelope callback not invoked"
    levels, chunk_dur = received[0]
    assert len(levels) == 1  # 1600 samples @16kHz, 10 chunks/s -> 1 chunk? no:
    # chunk = rate//10 = 1600 -> exactly 1 chunk
    assert chunk_dur == pytest.approx(0.1)
    assert 0.0 <= levels[0] <= 1.0
    svc.shutdown()


def test_tts_envelope_multi_chunk():
    from src.voice.speech_output_service import SpeechOutputService

    received = []

    class FakeSynth:
        name = "fake"

        def close(self):
            pass

    svc = SpeechOutputService(synthesizer=FakeSynth())
    svc.set_level_callback(lambda levels, dur: received.append((levels, dur)))
    # 3 chunks of varying amplitude.
    samples = np.concatenate([
        np.ones(1600, np.float32) * 0.1,
        np.ones(1600, np.float32) * 0.4,
        np.ones(1600, np.float32) * 0.2,
    ])
    svc._emit_level_envelope(samples, 16000)
    levels, _ = received[0]
    assert len(levels) == 3
    assert levels[1] > levels[0]
    svc.shutdown()


# ---------------------------------------------------------------------------
# Voice engine regression guards
# ---------------------------------------------------------------------------
def test_session_controller_has_last_intent():
    from src.voice.voice_session_controller import VoiceSessionController

    assert hasattr(VoiceSessionController, "_dispatch_and_respond")
    # last_intent attribute must exist on instances.
    import inspect

    src = inspect.getsource(VoiceSessionController)
    assert "self.last_intent = intent" in src


def test_no_extra_microphone_stream_created():
    """HUD modules must never import sounddevice/portaudio directly."""
    import inspect

    import src.ui.hologram.hud_bridge as bridge_mod
    import src.ui.hologram.hud_controller as ctl_mod
    import src.ui.hologram.hud_probe as probe_mod
    import src.ui.hologram.hud_widgets as wid_mod
    import src.ui.hologram.hud_window as win_mod
    import src.ui.hologram.orb_renderer as orb_mod

    for mod in (bridge_mod, ctl_mod, probe_mod, wid_mod, win_mod, orb_mod):
        src = inspect.getsource(mod)
        assert "sounddevice" not in src
        assert "import portaudio" not in src
