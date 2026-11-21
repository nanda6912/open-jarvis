"""P2 system-awareness command handlers.

Read-only status reporting (battery, internet, Wi-Fi, Ethernet),
screenshot capture and toast notifications. All real work goes through
the injected services on ``_Svc`` — never a shell, never invented
data: every handler reports "unknown/unavailable" honestly when the
underlying probe fails.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.commands.handlers import _BaseHandler
from src.commands.registry import CommandResult, PERMISSION_SYSTEM

logger = logging.getLogger(__name__)


def _format_duration(seconds: int) -> str:
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes = rem // 60
    if hours and minutes:
        return f"{hours} hour{'s' if hours != 1 else ''} and {minutes} minute{'s' if minutes != 1 else ''}"
    if hours:
        return f"{hours} hour{'s' if hours != 1 else ''}"
    if minutes:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    return "less than a minute"


class _StatusHandler(_BaseHandler):
    permission = PERMISSION_SYSTEM
    executable = True
    intent_name = ""
    service_attr = ""

    def __init__(self, svc):
        self._svc = svc

    def validate(self, entities: dict) -> Optional[str]:
        return None

    def _service(self):
        return getattr(self._svc, self.service_attr, None)

    def _unavailable(self, what: str) -> CommandResult:
        return CommandResult(
            self.intent_name, False,
            message=f"I couldn't check {what} right now.",
            error="service_unavailable",
        )


class BatteryStatusHandler(_StatusHandler):
    """GET_BATTERY_STATUS — report GetSystemPowerStatus data."""

    intent_name = "GET_BATTERY_STATUS"
    service_attr = "power"

    def execute(self, entities: dict) -> CommandResult:
        power = self._service()
        if power is None:
            return self._unavailable("battery status")
        try:
            st = power.status()
        except Exception:  # noqa: BLE001
            logger.exception("Power status probe failed")
            st = None
        if st is None:
            return CommandResult(
                self.intent_name, False,
                message="I couldn't read the power status.",
                error="probe_failed",
            )

        if st.has_battery is False:
            message = "This computer doesn't have a battery."
            if st.ac_line:
                message += " It's plugged into AC power."
        elif st.percent is None:
            message = "I can't read the battery level right now."
        else:
            pct = f"{st.percent} percent"
            if st.charging:
                message = f"Your battery is at {pct} and charging."
            elif st.discharging or st.ac_line is False:
                message = f"You're running on battery at {pct}."
                if st.life_seconds:
                    message += (
                        f" Estimated time remaining: "
                        f"{_format_duration(st.life_seconds)}."
                    )
            elif st.ac_line:
                message = f"Your battery is at {pct}. AC power is connected."
            else:
                message = f"Your battery is at {pct}."

        return CommandResult(
            self.intent_name, True, message=message,
            data={
                "percent": st.percent,
                "charging": st.charging,
                "ac_line": st.ac_line,
                "has_battery": st.has_battery,
            },
            executed=True,
        )


class _NetworkHandler(_StatusHandler):
    """Base for internet/Wi-Fi/Ethernet status handlers."""

    service_attr = "network"

    def _status(self):
        net = self._service()
        if net is None:
            return None
        try:
            return net.status()
        except Exception:  # noqa: BLE001
            logger.exception("Network status probe failed")
            return None


class InternetStatusHandler(_NetworkHandler):
    intent_name = "GET_INTERNET_STATUS"

    def execute(self, entities: dict) -> CommandResult:
        st = self._status()
        if st is None:
            return self._unavailable("internet status")
        if st.internet is None:
            message = "I can't determine your internet connection."
        elif st.internet:
            message = "You appear to be connected to the internet."
        else:
            message = "I don't detect an internet connection."
        return CommandResult(
            self.intent_name, True, message=message,
            data={"internet": st.internet}, executed=True,
        )


class WifiStatusHandler(_NetworkHandler):
    intent_name = "GET_WIFI_STATUS"

    def execute(self, entities: dict) -> CommandResult:
        st = self._status()
        if st is None:
            return self._unavailable("Wi-Fi status")
        wifi = st.wifi
        if wifi.available is None or wifi.connected is None:
            message = "I can't determine your Wi-Fi status."
        elif wifi.available is False:
            message = "I couldn't find a Wi-Fi adapter on this computer."
        elif wifi.connected:
            ssid = wifi.ssid or "an unknown network"
            message = f"You're connected to Wi-Fi network {ssid}."
            if wifi.signal is not None:
                message += f" Signal strength is {wifi.signal} percent."
        else:
            message = "Wi-Fi is not connected."
        return CommandResult(
            self.intent_name, True, message=message,
            data={
                "available": wifi.available,
                "connected": wifi.connected,
                "ssid": wifi.ssid,
                "signal": wifi.signal,
            },
            executed=True,
        )


class EthernetStatusHandler(_NetworkHandler):
    intent_name = "GET_ETHERNET_STATUS"

    def execute(self, entities: dict) -> CommandResult:
        st = self._status()
        if st is None:
            return self._unavailable("Ethernet status")
        eth = st.ethernet
        if eth is None:
            message = "I can't determine your Ethernet status."
        elif eth:
            message = "You're connected through Ethernet."
        else:
            message = "Ethernet is not connected."
        return CommandResult(
            self.intent_name, True, message=message,
            data={"ethernet": eth}, executed=True,
        )


class ScreenshotHandler(_StatusHandler):
    """TAKE_SCREENSHOT — capture the virtual desktop to PNG."""

    intent_name = "TAKE_SCREENSHOT"
    service_attr = "screenshot"

    def execute(self, entities: dict) -> CommandResult:
        shot = self._service()
        if shot is None:
            return self._unavailable("the screenshot service")
        ok, detail = shot.take_screenshot()
        if not ok:
            return CommandResult(
                self.intent_name, False,
                message="I couldn't take the screenshot.",
                error=str(detail),
            )
        return CommandResult(
            self.intent_name, True,
            message=f"Screenshot saved to {detail}.",
            data={"path": detail},
            executed=True,
        )


class SendNotificationHandler(_StatusHandler):
    """SEND_NOTIFICATION — user-requested toast notification."""

    intent_name = "SEND_NOTIFICATION"
    service_attr = "notifier"

    def validate(self, entities: dict) -> Optional[str]:
        message = entities.get("message")
        if not message or not str(message).strip():
            return "missing_message"
        return None

    def execute(self, entities: dict) -> CommandResult:
        notifier = self._service()
        if notifier is None:
            return self._unavailable("the notification service")
        message = str(entities.get("message") or "").strip()
        try:
            ok = bool(notifier.notify("Jarvis", message))
        except Exception:  # noqa: BLE001
            logger.exception("Notification failed")
            ok = False
        if not ok:
            return CommandResult(
                self.intent_name, False,
                message="I couldn't show the notification.",
                error="delivery_failed",
            )
        return CommandResult(
            self.intent_name, True,
            message="Done.",
            data={"message": message},
            executed=True,
        )
