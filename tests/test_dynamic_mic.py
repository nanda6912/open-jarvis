"""Tests for dynamic microphone selection and hot-plug switching.

Covers Auto (Windows Default) mode, default-input change detection,
earphone connect/disconnect, stable device identity across index
reshuffles, safe fallback, supported-rate probing/resampling, stream
restart safety, and wake-word continuity after a switch.
"""

from __future__ import annotations

import threading
import time
import types

import numpy as np
import pytest

from src.controllers.microphone_controller import (
    MODE_AUTO,
    MicrophoneController,
    STATUS_ERROR,
    STATUS_IDLE,
    STATUS_LISTENING,
)
from src.voice.audio_capture_service import AudioCaptureService
from src.voice.audio_device_manager import (
    AUTO_DEVICE_ID,
    AudioDeviceManager,
    DeviceUnavailableError,
)


# ----------------------------------------------------------------------
# Scriptable sounddevice fake
# ----------------------------------------------------------------------
def mic(name, hapi=2, ch=2, rate=48000.0, supported=None):
    """Build a fake input-device dict.

    ``supported``: iterable of rates the driver accepts, or None to accept
    anything (WASAPI shared mode typically accepts all rates).
    """
    d = {
        "name": name,
        "max_input_channels": ch,
        "default_samplerate": rate,
        "hostapi": hapi,
    }
    if supported is not None:
        d["supported_rates"] = {float(r) for r in supported}
    return d


class FakeStream:
    def __init__(self, sd, kwargs):
        self._sd = sd
        self.kwargs = kwargs
        self.active = False
        self.callback = kwargs.get("callback")
        self.closed = False
        rate = kwargs.get("samplerate")
        self.samplerate = float(rate) if rate else None

    def start(self):
        dev = int(self.kwargs.get("device", 0))
        rate = float(self.kwargs.get("samplerate", 0.0))
        fail = self._sd.fail_open.get(dev)
        if fail == "all" or (fail and rate in fail):
            raise RuntimeError(f"device {dev} rate {rate} rejected")
        rates = self._sd._devices[dev].get("supported_rates")
        if rates is not None and rate not in rates:
            raise ValueError(f"Invalid sample rate {rate}")
        self.active = True
        self._sd.streams.append(self)

    def stop(self):
        self.active = False

    def close(self):
        self.active = False
        self.closed = True


class FakeSD:
    """Mutable sounddevice fake supporting hot-plug and rate probing."""

    class error(Exception):
        pass

    def __init__(self, devices, default_index=0, fail_open=None):
        self._devices = list(devices)
        self.default = types.SimpleNamespace(device=(default_index, -1))
        # fail_open: {device_index: "all" | set(rates)}
        self.fail_open = fail_open or {}
        self.streams = []

    # -- device list mutation helpers (hot-plug simulation) --
    def set_devices(self, devices, default_index=0):
        self._devices = list(devices)
        self.default.device = (default_index, -1)

    def plug(self, device, make_default=True):
        self._devices.append(device)
        if make_default:
            self.default.device = (len(self._devices) - 1, -1)

    def unplug(self, index):
        name = self._devices[index]["name"]
        del self._devices[index]
        # Remap default index if it pointed at/after the removed slot.
        d = self.default.device[0]
        if d == index:
            self.default.device = (0, -1)
        elif d > index:
            self.default.device = (d - 1, -1)
        return name

    # -- sounddevice API --
    def query_devices(self, device=-1, kind=None):
        if device == -1:
            return self._devices
        return self._devices[device]

    def check_input_settings(self, device=None, channels=None,
                              samplerate=None, **kwargs):
        rates = self._devices[int(device)].get("supported_rates")
        if rates is not None and float(samplerate) not in rates:
            raise ValueError(f"Invalid sample rate {samplerate}")
        return None

    def InputStream(self, **kwargs):  # noqa: N802
        return FakeStream(self, kwargs)


class FakeEngine:
    name = "fake"

    def __init__(self, score=0.0, keyword="hello jarvis"):
        self.score = score
        self._kw = keyword
        self.processed = 0
        self.reset_count = 0
        self.closed = False

    def process(self, samples):
        self.processed += 1
        if self.score >= 1.0:
            self._pending = self._kw
            return 1.0
        return self.score

    def keyword_detected(self):
        kw = getattr(self, "_pending", None)
        self._pending = None
        return kw

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


@pytest.fixture
def settings():
    return {}


def make_controller(sd, settings=None, engine=None, interval=0.05):
    settings = settings if settings is not None else {}
    mgr = AudioDeviceManager(sd=sd)
    return MicrophoneController(
        device_manager=mgr,
        capture=AudioCaptureService(sd=sd, device_manager=mgr),
        engine=engine or FakeEngine(score=0.0),
        settings_get=lambda k: settings.get(k),
        settings_set=lambda k, v: settings.__setitem__(k, v),
        monitor_interval=interval,
    )


def wait_for(pred, timeout=3.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


def active_streams(sd):
    return [s for s in sd.streams if s.active]


# ----------------------------------------------------------------------
# Auto mode: default input changes
# ----------------------------------------------------------------------
class TestAutoMode:
    def test_defaults_to_auto(self, settings):
        ctl = make_controller(FakeSD([mic("Mic A")]), settings)
        assert ctl.selection_mode == MODE_AUTO
        assert ctl.selection_key == AUTO_DEVICE_ID
        ctl.shutdown()

    def test_auto_uses_windows_default(self, settings):
        sd = FakeSD([
            mic("Speakers out", ch=0),          # index 0: not an input
            mic("Realtek Headset Microphone"),  # index 1
            mic("Intel Microphone Array"),      # index 2
        ], default_index=1)
        ctl = make_controller(sd, settings)
        assert ctl.enable() is True
        assert ctl.active_device.name == "Realtek Headset Microphone"
        assert ctl.selection_mode == MODE_AUTO
        ctl.shutdown()

    def test_default_change_switches_capture(self, settings):
        """Earphones disconnected -> Windows default becomes Intel."""
        sd = FakeSD([
            mic("Realtek Headset Microphone"),
            mic("Intel Microphone Array"),
        ], default_index=0)
        ctl = make_controller(sd, settings)
        assert ctl.enable() is True
        assert ctl.active_device.name == "Realtek Headset Microphone"
        first_stream = sd.streams[-1]

        # Simulate unplug: Realtek gone, Intel becomes default (index 0).
        sd.set_devices([mic("Intel Microphone Array")], default_index=0)

        assert wait_for(
            lambda: ctl.active_device is not None
            and ctl.active_device.name == "Intel Microphone Array"
        )
        # Old stream was closed; exactly one stream is active.
        assert first_stream.closed or not first_stream.active
        assert len(active_streams(sd)) == 1
        assert ctl.is_listening
        assert ctl.diagnostics()["connected"] is True
        ctl.shutdown()

    def test_reconnect_switches_back(self, settings):
        sd = FakeSD([
            mic("Intel Microphone Array"),
            mic("Realtek Headset Microphone"),
        ], default_index=0)
        ctl = make_controller(sd, settings)
        assert ctl.enable() is True
        assert ctl.active_device.name == "Intel Microphone Array"

        # Earphones plugged in: Realtek appended and made default.
        sd.plug(mic("Realtek Headset Microphone"), make_default=True)
        assert wait_for(
            lambda: ctl.active_device is not None
            and ctl.active_device.name == "Realtek Headset Microphone"
        )
        assert len(active_streams(sd)) == 1
        ctl.shutdown()

    def test_no_duplicate_streams_while_switching(self, settings):
        sd = FakeSD([
            mic("Realtek Headset Microphone"),
            mic("Intel Microphone Array"),
        ], default_index=0)
        ctl = make_controller(sd, settings)
        assert ctl.enable() is True
        for _ in range(3):
            sd.set_devices(
                [mic("Intel Microphone Array"),
                 mic("Realtek Headset Microphone")],
                default_index=1,
            )
            time.sleep(0.15)
        assert len(active_streams(sd)) <= 1
        ctl.shutdown()

    def test_dead_stream_is_restarted(self, settings):
        """A stream that dies without close() is detected and reopened."""
        sd = FakeSD([mic("Mic A")], default_index=0)
        ctl = make_controller(sd, settings)
        assert ctl.enable() is True
        stream = sd.streams[-1]
        stream.active = False  # driver abort, object still referenced
        assert wait_for(lambda: ctl.capture.is_running)
        assert sd.streams[-1] is not stream
        ctl.shutdown()


# ----------------------------------------------------------------------
# Explicit device mode
# ----------------------------------------------------------------------
class TestExplicitMode:
    def test_select_realtek_pins_device(self, settings):
        sd = FakeSD([
            mic("Intel Microphone Array"),
            mic("Realtek Headset Microphone"),
        ], default_index=0)
        ctl = make_controller(sd, settings)
        assert ctl.enable() is True
        realtek = [d for d in ctl.devices
                   if "Realtek" in d.name][0]
        assert ctl.select_device(realtek.device_id) is True
        assert ctl.selection_mode == "device"
        assert ctl.active_device.name == "Realtek Headset Microphone"
        assert settings["voice.microphone_selection"] == "device"
        assert settings["voice.microphone_device"] == realtek.device_id
        ctl.shutdown()

    def test_select_auto_returns_to_default(self, settings):
        sd = FakeSD([
            mic("Realtek Headset Microphone"),
            mic("Intel Microphone Array"),
        ], default_index=1)
        ctl = make_controller(sd, settings)
        assert ctl.enable() is True
        assert ctl.active_device.name == "Intel Microphone Array"
        realtek = [d for d in ctl.devices
                   if "Realtek" in d.name][0]
        assert ctl.select_device(realtek.device_id) is True
        assert ctl.active_device.name == "Realtek Headset Microphone"
        assert ctl.select_device(AUTO_DEVICE_ID) is True
        assert ctl.selection_mode == MODE_AUTO
        assert ctl.active_device.name == "Intel Microphone Array"
        ctl.shutdown()

    def test_pinned_device_unplugged_falls_back(self, settings):
        """Selected mic disappears -> safe fallback, honest reporting."""
        sd = FakeSD([
            mic("Realtek Headset Microphone"),
            mic("Intel Microphone Array"),
        ], default_index=0)
        ctl = make_controller(sd, settings)
        assert ctl.enable() is True
        realtek = [d for d in ctl.devices
                   if "Realtek" in d.name][0]
        assert ctl.select_device(realtek.device_id) is True
        assert ctl.active_device.name == "Realtek Headset Microphone"

        # Unplug Realtek; Intel is already default (index 0 -> 0).
        sd.unplug(0)
        assert wait_for(
            lambda: ctl.active_device is not None
            and "Intel" in ctl.active_device.name
        )
        diag = ctl.diagnostics()
        assert diag["connected"] is True
        assert diag["fallback"] is not None
        assert "Realtek" in diag["fallback"] or "unavailable" in (
            diag["fallback"] or ""
        )
        # The user's pinned choice is NOT overwritten by the fallback.
        assert settings["voice.microphone_device"] == realtek.device_id
        ctl.shutdown()

    def test_missing_device_rejected(self, settings):
        sd = FakeSD([mic("Mic A")], default_index=0)
        ctl = make_controller(sd, settings)
        assert ctl.select_device("9:9:Missing Mic") is False
        ctl.shutdown()


# ----------------------------------------------------------------------
# Stable identity across sessions / index reshuffles
# ----------------------------------------------------------------------
class TestStableIdentity:
    def test_index_change_resolves_by_name(self):
        "tsounddevice indexes change between sessions; name must win."""
        devices_v1 = [
            mic("Realtek Headset Microphone"),
            mic("Intel Microphone Array"),
        ]
        sd1 = FakeSD(devices_v1, default_index=0)
        mgr1 = AudioDeviceManager(sd=sd1)
        realtek_v1 = [
            d for d in mgr1.list_input_devices()
            if "Realtek" in d.name
        ][0]
        persisted = realtek_v1.device_id  # e.g. "2:0:Realtek..."

        # Next session: device order swapped -> Realtek now index 1.
        devices_v2 = [devices_v1[1], devices_v1[0]]
        sd2 = FakeSD(devices_v2, default_index=1)
        mgr2 = AudioDeviceManager(sd=sd2)
        resolved = mgr2.resolve_device(persisted)
        assert "Realtek" in resolved.name
        assert resolved.index == 1  # followed the name, not the index

    def test_full_name_not_truncated_for_matching(self):
        long_name = (
            "Realtek(R) Audio USB Headset Microphone "
            "([Some Very Long Descriptor])"
        )
        sd = FakeSD([mic(long_name), mic("Intel Microphone Array")],
                    default_index=0)
        mgr = AudioDeviceManager(sd=sd)
        dev = mgr.list_input_devices()[0]
        assert dev.name == long_name  # untruncated
        found = mgr.find_device(dev.device_id)
        assert found is not None and found.name == long_name

    def test_missing_device_fallback_is_logged_and_honest(self, caplog):
        sd = FakeSD([mic("Intel Microphone Array")], default_index=0)
        mgr = AudioDeviceManager(sd=sd)
        with caplog.at_level("WARNING"):
            device = mgr.resolve_device("2:7:Realtek Headset Microphone")
        assert "Intel" in device.name
        assert "not found" in caplog.text


# ----------------------------------------------------------------------
# Sample-rate probing and resampling
# ----------------------------------------------------------------------
class TestSampleRates:
    def test_supported_rates_probed(self):
        sd = FakeSD([
            mic("WASAPI Mic", supported={48000.0, 44100.0}),
        ], default_index=0)
        mgr = AudioDeviceManager(sd=sd)
        dev = mgr.list_input_devices()[0]
        rates = mgr.supported_input_rates(
            dev, preferred=(16000.0, 48000.0)
        )
        assert rates[0] == 48000.0      # 16 kHz unsupported -> skipped
        assert 44100.0 in rates
        assert 16000.0 not in rates

    def test_stream_uses_supported_rate_and_resamples(self):
        """16 kHz rejected -> open at 48 kHz -> frames resampled to 16k."""
        sd = FakeSD([
            mic("WASAPI Mic", rate=48000.0,
                supported={48000.0, 44100.0}),
        ], default_index=0)
        mgr = AudioDeviceManager(sd=sd)
        cap = AudioCaptureService(sd=sd, device_manager=mgr)
        device = cap.start()
        assert cap.sample_rate == 48000
        stream = sd.streams[-1]
        assert stream.kwargs["samplerate"] == 48000.0
        # Feed 4800 samples @48k -> expect ~1600 @16k.
        stream.callback(
            np.zeros((4800, 1), dtype=np.float32), 4800, None, None
        )
        frame = cap.get_frame(timeout=1.0)
        assert frame is not None
        assert abs(len(frame) - 1600) <= 2
        cap.stop()

    def test_preferred_16k_when_supported(self):
        sd = FakeSD([
            mic("Native 16k Mic", rate=16000.0,
                supported={16000.0, 48000.0}),
        ], default_index=0)
        mgr = AudioDeviceManager(sd=sd)
        cap = AudioCaptureService(sd=sd, device_manager=mgr)
        cap.start()
        assert cap.sample_rate == 16000
        assert sd.streams[-1].kwargs["samplerate"] == 16000.0
        cap.stop()

    def test_all_rates_failing_raises_audio_error(self):
        sd = FakeSD(
            [mic("Dead Mic", rate=48000.0)],
            default_index=0,
            fail_open={0: "all"},
        )
        mgr = AudioDeviceManager(sd=sd)
        cap = AudioCaptureService(sd=sd, device_manager=mgr)
        with pytest.raises(Exception):
            cap.start()


# ----------------------------------------------------------------------
# Safe fallback
# ----------------------------------------------------------------------
class TestSafeFallback:
    def test_explicit_device_failure_falls_back_to_default(self, settings):
        sd = FakeSD([
            mic("Realtek Headset Microphone"),
            mic("Intel Microphone Array"),
        ], default_index=1)
        ctl = make_controller(sd, settings)
        realtek = [d for d in ctl.devices
                   if "Realtek" in d.name][0]
        assert ctl.enable() is True
        # Pin Realtek (index 0), then make every open on it fail.
        assert ctl.select_device(realtek.device_id) is True
        sd.fail_open[0] = "all"
        assert ctl.select_device(realtek.device_id) is True
        # Fallback engaged: Intel (default, index 1) is active and the
        # status detail says so — the requested device is NOT claimed.
        assert "Intel" in ctl.active_device.name
        assert ctl.diagnostics()["fallback"] is not None
        assert settings["voice.microphone_device"] == realtek.device_id
        ctl.shutdown()

    def test_no_devices_reports_error_not_crash(self, settings):
        sd = FakeSD([], default_index=-1)
        sd.default.device = (-1, -1)
        ctl = make_controller(sd, settings)
        assert ctl.enable() is False
        assert ctl.status == STATUS_ERROR
        ctl.shutdown()


# ----------------------------------------------------------------------
# Wake-word continuity across a switch
# ----------------------------------------------------------------------
class TestWakeWordAfterSwitch:
    def test_detection_survives_device_switch(self, settings):
        sd = FakeSD([
            mic("Realtek Headset Microphone"),
            mic("Intel Microphone Array"),
        ], default_index=0)
        engine = FakeEngine(score=1.0)
        ctl = make_controller(sd, settings, engine=engine)
        wakes = []
        ctl.set_callbacks(on_wake_word=wakes.append)
        assert ctl.enable() is True

        def feed():
            stream = sd.streams[-1]
            stream.callback(
                np.zeros((1600, 1), dtype=np.float32), 1600, None, None
            )

        # Wake word on Realtek.
        assert wait_for(lambda: wakes, timeout=3.0) or _feed_until(
            feed, wakes
        )
        assert wakes == ["hello jarvis"]
        # Session suspended detection; simulate session end.
        ctl.resume_detection()

        # Switch to Intel and verify wake word still fires.
        sd.set_devices([mic("Intel Microphone Array")], default_index=0)
        assert wait_for(
            lambda: ctl.active_device is not None
            and "Intel" in ctl.active_device.name
        )
        assert _feed_until(feed, wakes, want=2, timeout=4.0)
        assert wakes == ["hello jarvis", "hello jarvis"]
        assert len(active_streams(sd)) == 1
        ctl.shutdown()


def _feed_until(feed, hits, want=1, timeout=4.0):
    deadline = time.time() + timeout
    while time.time() < deadline and len(hits) < want:
        try:
            feed()
        except Exception:  # noqa: BLE001 - stream may be mid-restart
            pass
        time.sleep(0.05)
    return len(hits) >= want


# ----------------------------------------------------------------------
# Task 3: stable identity, ambiguity, honest fallback reporting
# ----------------------------------------------------------------------
class TestStableResolution:
    def test_index_reshuffle_is_not_a_fallback(self, settings):
        """Name-resolved device at a new index must not report fallback.

        Regression for the contradictory log pair: resolve_device
        reported 'resolved by name' while the status detail still said
        'fallback: <id> unavailable' because the strict check compared
        the ephemeral hostapi:index pair.
        """
        sd = FakeSD([
            mic("Realtek Headset Microphone"),
            mic("Intel Microphone Array"),
        ], default_index=0)
        ctl = make_controller(sd, settings)
        assert ctl.enable() is True
        intel = [d for d in ctl.devices if "Intel" in d.name][0]
        assert ctl.select_device(intel.device_id) is True

        # Same devices, enumeration order swapped -> new index.
        sd.set_devices([
            mic("Intel Microphone Array"),
            mic("Realtek Headset Microphone"),
        ], default_index=0)
        assert wait_for(
            lambda: ctl.active_device is not None
            and "Intel" in ctl.active_device.name
        )
        # The configured mic is still the configured mic: no fallback
        # notice.
        assert ctl.diagnostics()["fallback"] is None
        # A capture restart re-persists the id with the fresh index
        # (self-healing) while remaining name-resolvable.
        ctl.disable()
        assert ctl.enable() is True
        new_id = settings["voice.microphone_device"]
        assert "Intel" in new_id
        mgr = AudioDeviceManager(sd=sd)
        assert "Intel" in mgr.resolve_device(new_id).name
        ctl.shutdown()

    def test_duplicate_names_never_bind_by_index_guess(self):
        sd = FakeSD([
            mic("Mic Array", hapi=2),
            mic("Mic Array", hapi=2),
            mic("Other Mic", hapi=2),
        ], default_index=2)
        mgr = AudioDeviceManager(sd=sd)
        # Ambiguous name on one host API: must not silently pick index.
        # (Use a stale index so the id cannot exact-match a live device.)
        device = mgr.resolve_device("2:9:Mic Array")
        assert "Other" in device.name  # explicit default, not a guess
        assert mgr.find_device("2:9:Mic Array") is None

    def test_duplicate_names_different_hostapi_still_resolve(self):
        sd = FakeSD([
            mic("Mic Array", hapi=0),
            mic("Mic Array", hapi=2),
        ], default_index=0)
        mgr = AudioDeviceManager(sd=sd)
        got = mgr.resolve_device("2:1:Mic Array")
        assert got.hostapi == 2

    def test_truncated_and_trademark_names_match(self):
        sd = FakeSD([
            mic("Microphone Array (Intel® Smart "),
            mic("Realtek Headset Microphone"),
        ], default_index=0)
        mgr = AudioDeviceManager(sd=sd)
        dev = mgr.list_input_devices()[0]
        # Persisted MME-style truncated name resolves to itself.
        assert mgr.resolve_device(dev.device_id).index == dev.index
        # A fuller variant of the same endpoint also name-matches.
        assert mgr.find_device(
            f"{dev.hostapi}:{dev.index}:"
            "Microphone Array (Intel Smart Sound Technology)"
        ) is not None

    def test_stale_name_uses_default_not_index(self, caplog):
        sd = FakeSD([mic("Mic A"), mic("Mic B")], default_index=0)
        mgr = AudioDeviceManager(sd=sd)
        with caplog.at_level("WARNING"):
            got = mgr.resolve_device("2:1:Old Driver Name")
        # Index 1 is 'Mic B' but the stored name matches nothing —
        # binding it would risk hot-unplug index reuse, so the explicit
        # default (index 0) is used with a clear warning instead.
        assert got.name == "Mic A"
        assert "not found" in caplog.text
        # Nameless legacy id still resolves via hostapi:index.
        assert mgr.resolve_device("2:1").name == "Mic B"

    def test_matches_stable_id_ignores_index_reshuffle(self):
        sd = FakeSD([
            mic("Realtek Headset Microphone"),
            mic("Intel Microphone Array"),
        ], default_index=0)
        mgr = AudioDeviceManager(sd=sd)
        intel_v1 = [
            d for d in mgr.list_input_devices() if "Intel" in d.name
        ][0]
        persisted = intel_v1.device_id
        # Reordered inventory: Intel now at index 0.
        sd2 = FakeSD([
            mic("Intel Microphone Array"),
            mic("Realtek Headset Microphone"),
        ], default_index=0)
        mgr2 = AudioDeviceManager(sd=sd2)
        intel_v2 = mgr2.list_input_devices()[0]
        assert mgr2.matches_stable_id(intel_v2, persisted)
        assert not mgr2.matches_stable_id(
            mgr2.list_input_devices()[1], persisted
        )

    def test_index_reuse_by_different_mic_never_binds_it(self, settings):
        """Pinned mic gone + its index taken by another mic.

        The stored named id must not bind the usurper via hostapi:index;
        the controller falls back to the explicit default and keeps the
        pinned preference for when the real device returns.
        """
        sd = FakeSD([
            mic("Intel Microphone Array"),
            mic("Realtek Headset Microphone"),
        ], default_index=1)
        ctl = make_controller(sd, settings)
        assert ctl.enable() is True
        intel = [d for d in ctl.devices if "Intel" in d.name][0]
        assert ctl.select_device(intel.device_id) is True
        pinned = settings["voice.microphone_device"]

        # Intel disappears; a different mic takes index 0.
        sd.set_devices([
            mic("USB Gaming Mic"),
            mic("Realtek Headset Microphone"),
        ], default_index=1)
        assert wait_for(
            lambda: ctl.active_device is not None
            and "Realtek" in ctl.active_device.name
        )
        assert ctl.diagnostics()["fallback"] is not None
        # Pinned preference preserved — never overwritten by the
        # unverified fallback device.
        assert settings["voice.microphone_device"] == pinned
        ctl.shutdown()

    def test_missing_preferred_uses_default_and_reports(self, settings):
        sd = FakeSD([
            mic("Realtek Headset Microphone"),
            mic("Intel Microphone Array"),
        ], default_index=0)
        ctl = make_controller(sd, settings)
        assert ctl.enable() is True
        realtek = [d for d in ctl.devices if "Realtek" in d.name][0]
        assert ctl.select_device(realtek.device_id) is True
        sd.unplug(0)  # remove Realtek entirely
        assert wait_for(
            lambda: ctl.active_device is not None
            and "Intel" in ctl.active_device.name
        )
        diag = ctl.diagnostics()
        assert diag["fallback"] is not None
        # Pinned preference preserved for when the device returns.
        assert settings["voice.microphone_device"] == realtek.device_id
        ctl.shutdown()
