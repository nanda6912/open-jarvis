"""Tests for P2 Windows system awareness: battery, network, screenshot,
toast notifications. All Win32/COM interaction is faked except the
PNG-encoder validity check, which uses the real GDI path on Windows."""

from __future__ import annotations

import os
import struct
import time
import types

import pytest

from src.commands.app_handlers import _Svc, create_app_registry
from src.commands.intent import Intent, RuleBasedIntentRecognizer
from src.commands.status_handlers import (
    BatteryStatusHandler,
    EthernetStatusHandler,
    InternetStatusHandler,
    ScreenshotHandler,
    SendNotificationHandler,
    WifiStatusHandler,
)
from src.controllers.network_status_controller import (
    NetworkStatus,
    NetworkStatusController,
    WifiStatus,
)
from src.controllers.notification_controller import (
    NotificationController,
    sanitize_text,
)
from src.controllers.power_status_controller import (
    PowerStatusController,
    _SYSTEM_POWER_STATUS,
)
from src.controllers.screenshot_controller import (
    ScreenshotController,
    ScreenshotEngine,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_svc(**kw):
    return _Svc(
        None, None, None, None,
        power=kw.get("power"),
        network=kw.get("network"),
        screenshot=kw.get("screenshot"),
        notifier=kw.get("notifier"),
    )


def sps(ac=1, flag=0, pct=50, life=0xFFFFFFFF, full=0xFFFFFFFF):
    s = _SYSTEM_POWER_STATUS()
    s.ACLineStatus = ac
    s.BatteryFlag = flag
    s.BatteryLifePercent = pct
    s.SystemStatusFlag = 0
    s.BatteryLifeTime = life
    s.BatteryFullLifeTime = full
    return s


# ----------------------------------------------------------------------
# P2.1 — battery / power status
# ----------------------------------------------------------------------
class TestPowerStatusController:
    def test_charging(self):
        c = PowerStatusController(status_fn=lambda: sps(
            ac=1, flag=8, pct=65, life=0xFFFFFFFF))
        st = c.status()
        assert st.ac_line is True and st.charging is True
        assert st.percent == 65 and st.has_battery is True

    def test_discharging(self):
        c = PowerStatusController(status_fn=lambda: sps(
            ac=0, flag=1, pct=40, life=5400))
        st = c.status()
        assert st.ac_line is False and st.discharging is True
        assert st.life_seconds == 5400

    def test_no_battery_desktop(self):
        c = PowerStatusController(status_fn=lambda: sps(
            ac=1, flag=128, pct=255))
        st = c.status()
        assert st.has_battery is False and st.percent is None

    def test_unknown_values(self):
        c = PowerStatusController(status_fn=lambda: sps(
            ac=255, flag=255, pct=255))
        st = c.status()
        assert st.ac_line is None and st.has_battery is None
        assert st.percent is None and st.charging is None

    def test_unavailable_estimate(self):
        c = PowerStatusController(status_fn=lambda: sps(
            ac=0, flag=1, pct=30, life=0xFFFFFFFF))
        assert c.status().life_seconds is None
        c2 = PowerStatusController(status_fn=lambda: sps(
            ac=0, flag=1, pct=30, life=0))
        assert c2.status().life_seconds is None

    def test_api_failure_returns_none(self):
        c = PowerStatusController(status_fn=lambda: None)
        assert c.status() is None
        c2 = PowerStatusController(
            status_fn=lambda: (_ for _ in ()).throw(OSError("x")))
        assert c2.status() is None


class FakePower:
    def __init__(self, status):
        self._status = status

    def status(self):
        return self._status


class TestBatteryHandler:
    def _run(self, st):
        h = BatteryStatusHandler(make_svc(
            power=FakePower(st) if st != "FAIL" else None))
        return h.execute({})

    def _st(self, **kw):
        defaults = dict(
            ac_line=True, has_battery=True, percent=65,
            charging=True, discharging=False,
            life_seconds=None, full_life_seconds=None,
        )
        defaults.update(kw)
        return PowerStatusController._normalize(sps(
            ac=1 if kw.get("ac_line") else 0,
            flag=8 if kw.get("charging") else 1,
            pct=kw.get("percent") if kw.get("percent") is not None else 255,
            life=kw.get("life_seconds") or 0xFFFFFFFF,
        )) if False else types.SimpleNamespace(**defaults)

    def test_charging_message(self):
        r = self._run(self._st(percent=65, charging=True))
        assert r.executed and "65 percent" in r.message
        assert "charging" in r.message

    def test_discharging_with_estimate(self):
        r = self._run(self._st(
            ac_line=False, charging=False, discharging=True,
            percent=40, life_seconds=5400))
        assert "battery" in r.message.lower()
        assert "1 hour" in r.message

    def test_discharging_no_estimate(self):
        r = self._run(self._st(
            ac_line=False, charging=False, discharging=True,
            percent=40, life_seconds=None))
        assert "Estimated" not in r.message

    def test_no_battery(self):
        r = self._run(self._st(has_battery=False, percent=None))
        assert "doesn't have a battery" in r.message

    def test_unknown_percent(self):
        r = self._run(self._st(percent=None))
        assert "can't read" in r.message.lower()

    def test_service_missing(self):
        r = BatteryStatusHandler(make_svc()).execute({})
        assert not r.success and r.error == "service_unavailable"


# ----------------------------------------------------------------------
# P2.2 — network status
# ----------------------------------------------------------------------
class FakeWlan:
    def __init__(self, status):
        self._status = status

    def wifi_status(self):
        return self._status


class FakeIfTable:
    def __init__(self, up=True, fail=False):
        self._up = up
        self._fail = fail

    def ethernet_up(self):
        if self._fail:
            raise OSError("no")
        return self._up


def net_with(wifi_status=None, ethernet=None, internet=None):
    class _W:
        def __init__(self, s):
            self._s = s

        def wifi_status(self):
            if self._s == "RAISE":
                raise OSError("dead")
            return self._s

    class _I:
        def __init__(self, v):
            self._v = v

        def ethernet_up(self):
            if self._v == "RAISE":
                raise OSError("dead")
            return self._v

    return NetworkStatusController(
        wlan=_W(wifi_status if wifi_status is not None else WifiStatus(
            available=True, connected=True, ssid="HomeNet", signal=80)),
        iftable=_I(ethernet if ethernet is not None else False),
        internet_fn=(lambda: internet) if internet is not None else (
            lambda: (_ for _ in ()).throw(OSError("x"))),
    )


class TestNetworkController:
    def test_aggregates(self):
        net = net_with(
            wifi_status=WifiStatus(True, True, "Cafe", 55),
            ethernet=True, internet=True)
        st = net.status()
        assert st.internet is True
        assert st.wifi.connected and st.wifi.ssid == "Cafe"
        assert st.wifi.signal == 55
        assert st.ethernet is True

    def test_probe_failures_become_none(self):
        net = net_with(
            wifi_status="RAISE", ethernet="RAISE", internet=None)
        st = net.status()
        assert st.internet is None
        assert st.wifi.connected is None
        assert st.ethernet is None

    def test_no_wifi_adapter(self):
        net = net_with(
            wifi_status=WifiStatus(False, False, None, None))
        assert net.wifi().available is False


class TestNetworkHandlers:
    def test_internet_connected(self):
        reg = create_app_registry(make_svc(network=net_with(internet=True)))
        r = reg.dispatch(Intent("GET_INTERNET_STATUS", {}))
        assert r.executed and "connected to the internet" in r.message

    def test_internet_disconnected(self):
        reg = create_app_registry(make_svc(network=net_with(internet=False)))
        r = reg.dispatch(Intent("GET_INTERNET_STATUS", {}))
        assert "don't detect" in r.message

    def test_wifi_connected_ssid_signal(self):
        reg = create_app_registry(make_svc(network=net_with(
            wifi_status=WifiStatus(True, True, "MyNet", 72),
            internet=True)))
        r = reg.dispatch(Intent("GET_WIFI_STATUS", {}))
        assert r.executed
        assert "MyNet" in r.message and "72 percent" in r.message

    def test_wifi_disconnected(self):
        reg = create_app_registry(make_svc(network=net_with(
            wifi_status=WifiStatus(True, False, None, None))))
        r = reg.dispatch(Intent("GET_WIFI_STATUS", {}))
        assert "not connected" in r.message

    def test_wifi_no_adapter(self):
        reg = create_app_registry(make_svc(network=net_with(
            wifi_status=WifiStatus(False, False, None, None))))
        r = reg.dispatch(Intent("GET_WIFI_STATUS", {}))
        assert "Wi-Fi adapter" in r.message

    def test_wifi_unknown_ssid_signal(self):
        reg = create_app_registry(make_svc(network=net_with(
            wifi_status=WifiStatus(True, True, None, None))))
        r = reg.dispatch(Intent("GET_WIFI_STATUS", {}))
        assert "unknown network" in r.message
        assert "Signal" not in r.message

    def test_ethernet_connected(self):
        reg = create_app_registry(make_svc(network=net_with(ethernet=True)))
        r = reg.dispatch(Intent("GET_ETHERNET_STATUS", {}))
        assert "Ethernet" in r.message and "connected" in r.message.lower()

    def test_ethernet_disconnected(self):
        reg = create_app_registry(make_svc(network=net_with(ethernet=False)))
        r = reg.dispatch(Intent("GET_ETHERNET_STATUS", {}))
        assert "not connected" in r.message

    def test_missing_service(self):
        reg = create_app_registry(make_svc())
        for name in ("GET_INTERNET_STATUS", "GET_WIFI_STATUS",
                     "GET_ETHERNET_STATUS"):
            r = reg.dispatch(Intent(name, {}))
            assert not r.success and r.error == "service_unavailable"


# ----------------------------------------------------------------------
# P2.3 — screenshot
# ----------------------------------------------------------------------
class FakeShot:
    def __init__(self, ok=True, detail="C:\\fake\\shot.png", cap=(123, 1, 1)):
        self.ok = ok
        self.detail = detail
        self.cap = cap if ok else None
        self.saved = []
        self.released = []

    def take_screenshot(self):
        return (self.ok, self.detail)


def fake_engine(cap=(111, 2, 2), save_ok=True, write_bytes=True):
    saved = []
    released = []

    def save(c, path):
        saved.append((c, path))
        if save_ok and write_bytes:
            # Minimal valid PNG signature + IHDR-ish bytes.
            with open(path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        return save_ok

    def release(c):
        released.append(c)

    return ScreenshotEngine(
        capture_fn=lambda: cap,
        save_fn=save,
        release_fn=release,
    ), saved, released


class TestScreenshotController:
    def test_success_writes_png(self, tmp_path):
        eng, saved, released = fake_engine()
        ctl = ScreenshotController(out_dir=str(tmp_path), engine=eng)
        ok, detail = ctl.take_screenshot()
        assert ok
        assert detail.endswith(".png")
        assert os.path.dirname(detail) == str(tmp_path)
        assert os.path.getsize(detail) > 0
        with open(detail, "rb") as f:
            assert f.read(8) == b"\x89PNG\r\n\x1a\n"
        assert released  # GDI object released

    def test_unique_filenames(self, tmp_path):
        eng, saved, _ = fake_engine()
        ctl = ScreenshotController(out_dir=str(tmp_path), engine=eng)
        ok1, p1 = ctl.take_screenshot()
        ok2, p2 = ctl.take_screenshot()
        assert ok1 and ok2 and p1 != p2

    def test_capture_failure(self, tmp_path):
        eng = ScreenshotEngine(
            capture_fn=lambda: None,
            save_fn=lambda c, p: True,
            release_fn=lambda c: None,
        )
        ctl = ScreenshotController(out_dir=str(tmp_path), engine=eng)
        ok, detail = ctl.take_screenshot()
        assert not ok and detail == "capture_failed"

    def test_capture_raises(self, tmp_path):
        def boom():
            raise OSError("gdi dead")

        eng = ScreenshotEngine(
            capture_fn=boom, save_fn=lambda c, p: True,
            release_fn=lambda c: None)
        ctl = ScreenshotController(out_dir=str(tmp_path), engine=eng)
        ok, _ = ctl.take_screenshot()
        assert not ok

    def test_save_failure(self, tmp_path):
        eng, _, _ = fake_engine(save_ok=False)
        ctl = ScreenshotController(out_dir=str(tmp_path), engine=eng)
        ok, detail = ctl.take_screenshot()
        assert not ok and detail == "save_failed"

    def test_save_writes_nothing_fails(self, tmp_path):
        eng, _, _ = fake_engine(save_ok=True, write_bytes=False)
        ctl = ScreenshotController(out_dir=str(tmp_path), engine=eng)
        ok, detail = ctl.take_screenshot()
        assert not ok and detail == "save_failed"

    def test_creates_missing_directory(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "shots"
        eng, _, _ = fake_engine()
        ctl = ScreenshotController(out_dir=str(target), engine=eng)
        ok, detail = ctl.take_screenshot()
        assert ok and os.path.isdir(target)

    def test_real_gdi_capture_and_png(self, tmp_path):
        """Real Win32 capture -> real GDI+ PNG on this Windows host."""
        from src.controllers.screenshot_controller import (
            _Gdiplus,
            _Win32Capture,
        )

        cap = _Win32Capture()
        gdip = _Gdiplus()
        eng = ScreenshotEngine(
            capture_fn=cap.capture,
            save_fn=lambda c, p: gdip.save_png_from_hbitmap(c[0], p),
            release_fn=cap.release,
        )
        ctl = ScreenshotController(out_dir=str(tmp_path), engine=eng)
        ok, detail = ctl.take_screenshot()
        assert ok, detail
        with open(detail, "rb") as f:
            head = f.read(24)
        assert head[:8] == b"\x89PNG\r\n\x1a\n"
        # PNG IHDR width/height should be > 0.
        w, h = struct.unpack(">II", head[16:24])
        assert w > 0 and h > 0


class TestScreenshotHandler:
    def test_success_message_has_path(self, tmp_path):
        eng, _, _ = fake_engine()
        shot = ScreenshotController(out_dir=str(tmp_path), engine=eng)
        reg = create_app_registry(make_svc(screenshot=shot))
        r = reg.dispatch(Intent("TAKE_SCREENSHOT", {}))
        assert r.executed and "Screenshot saved to" in r.message
        assert r.data["path"].endswith(".png")

    def test_failure_not_reported_as_success(self):
        eng = ScreenshotEngine(
            capture_fn=lambda: None, save_fn=lambda c, p: True,
            release_fn=lambda c: None)
        shot = ScreenshotController(out_dir="C:\\nope", engine=eng)
        reg = create_app_registry(make_svc(screenshot=shot))
        r = reg.dispatch(Intent("TAKE_SCREENSHOT", {}))
        assert not r.success and "couldn't take" in r.message.lower()

    def test_missing_service(self):
        reg = create_app_registry(make_svc())
        r = reg.dispatch(Intent("TAKE_SCREENSHOT", {}))
        assert not r.success and r.error == "service_unavailable"


# ----------------------------------------------------------------------
# P2.4 — notifications
# ----------------------------------------------------------------------
class FakeNotifyApi:
    def __init__(self, ok=True, raise_exc=False):
        self.ok = ok
        self.raise_exc = raise_exc
        self.calls = []

    def show(self, title, message):
        self.calls.append((title, message))
        if self.raise_exc:
            raise OSError("toast dead")
        return self.ok


class TestNotificationController:
    def test_success(self):
        api = FakeNotifyApi()
        ctl = NotificationController(api=api, min_interval=0)
        assert ctl.notify("Jarvis", "Lunch is ready.") is True
        assert api.calls == [("Jarvis", "Lunch is ready.")]

    def test_invalid_input_rejected(self):
        api = FakeNotifyApi()
        ctl = NotificationController(api=api)
        assert ctl.notify("", "msg") is False
        assert ctl.notify("title", "   ") is False
        assert api.calls == []

    def test_text_sanitized_and_limited(self):
        api = FakeNotifyApi()
        ctl = NotificationController(api=api, min_interval=0)
        ctl.notify("T" * 200, "M" * 400 + "\x07bad")
        title, msg = api.calls[0]
        assert len(title) <= 63
        assert len(msg) <= 255
        assert "\x07" not in msg

    def test_delivery_failure(self):
        api = FakeNotifyApi(ok=False)
        ctl = NotificationController(api=api)
        assert ctl.notify("Jarvis", "hi") is False

    def test_delivery_exception(self):
        api = FakeNotifyApi(raise_exc=True)
        ctl = NotificationController(api=api)
        assert ctl.notify("Jarvis", "hi") is False

    def test_disabled(self):
        api = FakeNotifyApi()
        ctl = NotificationController(api=api, enabled=False)
        assert ctl.notify("Jarvis", "hi") is False
        assert api.calls == []

    def test_rate_limited(self):
        api = FakeNotifyApi()
        now = [1000.0]
        ctl = NotificationController(
            api=api, min_interval=5.0, clock=lambda: now[0])
        assert ctl.notify("Jarvis", "one") is True
        assert ctl.notify("Jarvis", "two") is False  # too soon
        now[0] += 10
        assert ctl.notify("Jarvis", "three") is True

    def test_max_per_minute(self):
        api = FakeNotifyApi()
        now = [0.0]
        ctl = NotificationController(
            api=api, min_interval=0.0, max_per_minute=3,
            clock=lambda: now[0])
        for i in range(3):
            now[0] += 1
            assert ctl.notify("Jarvis", f"m{i}") is True
        now[0] += 1
        assert ctl.notify("Jarvis", "overflow") is False

    def test_sanitize_text(self):
        assert sanitize_text("a\x00b\x07c", 100) == "a b c"
        assert len(sanitize_text("x" * 500, 63)) == 63


class TestSendNotificationHandler:
    def test_success(self):
        api = FakeNotifyApi()
        notifier = NotificationController(api=api, min_interval=0)
        reg = create_app_registry(make_svc(notifier=notifier))
        r = reg.dispatch(Intent(
            "SEND_NOTIFICATION", {"message": "build finished"}))
        assert r.executed and r.success
        assert api.calls == [("Jarvis", "build finished")]

    def test_missing_message_validation(self):
        reg = create_app_registry(make_svc(notifier=NotificationController(
            api=FakeNotifyApi())))
        r = reg.dispatch(Intent("SEND_NOTIFICATION", {"message": "  "}))
        assert not r.success and r.error == "missing_message"
        r2 = reg.dispatch(Intent("SEND_NOTIFICATION", {}))
        assert not r2.success and r2.error == "missing_message"

    def test_delivery_failure_not_success(self):
        notifier = NotificationController(api=FakeNotifyApi(ok=False))
        reg = create_app_registry(make_svc(notifier=notifier))
        r = reg.dispatch(Intent("SEND_NOTIFICATION", {"message": "hi"}))
        assert not r.success

    def test_missing_service(self):
        reg = create_app_registry(make_svc())
        r = reg.dispatch(Intent("SEND_NOTIFICATION", {"message": "hi"}))
        assert not r.success and r.error == "service_unavailable"


# ----------------------------------------------------------------------
# Intent recognition coverage
# ----------------------------------------------------------------------
class TestP2Intents:
    @pytest.fixture
    def rec(self):
        return RuleBasedIntentRecognizer()

    @pytest.mark.parametrize("text,name", [
        ("How much battery do I have?", "GET_BATTERY_STATUS"),
        ("What's my battery percentage?", "GET_BATTERY_STATUS"),
        ("Is my laptop charging?", "GET_BATTERY_STATUS"),
        ("Am I running on battery?", "GET_BATTERY_STATUS"),
        ("What's my power status?", "GET_BATTERY_STATUS"),
        ("Am I connected to the internet?", "GET_INTERNET_STATUS"),
        ("Is Wi-Fi connected?", "GET_WIFI_STATUS"),
        ("Which Wi-Fi network am I using?", "GET_WIFI_STATUS"),
        ("What's my Wi-Fi signal strength?", "GET_WIFI_STATUS"),
        ("Am I using Ethernet?", "GET_ETHERNET_STATUS"),
        ("Take a screenshot.", "TAKE_SCREENSHOT"),
        ("Capture my screen.", "TAKE_SCREENSHOT"),
        ("Take a screenshot of both monitors.", "TAKE_SCREENSHOT"),
        ("Notify me that the build finished.", "SEND_NOTIFICATION"),
        ("Send a notification saying lunch is ready.",
         "SEND_NOTIFICATION"),
        ("Remind me to call mom.", "SEND_NOTIFICATION"),
    ])
    def test_intent_mapping(self, rec, text, name):
        assert rec.recognize(text).name == name, text

    def test_existing_intents_unaffected(self, rec):
        assert rec.recognize("Turn the volume up.").name == \
            "INCREASE_VOLUME"
        assert rec.recognize("Open Downloads.").name == "OPEN_FOLDER"
        assert rec.recognize("Goodbye Jarvis").name == "END_SESSION"
        # V10.5: explicit web scope routes to SEARCH_WEB, never a local
        # file search (was SEARCH_FILES before the routing fix).
        assert rec.recognize("Search the internet for cats.").name == \
            "SEARCH_WEB"
