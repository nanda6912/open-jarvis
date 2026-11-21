"""Microphone device enumeration and selection."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Sentinel device id meaning "follow the Windows default input device".
AUTO_DEVICE_ID = "auto"

# Sample rates probed (in preference order) when deciding how to open a
# capture stream. 16 kHz is preferred so the wake-word model gets native
# rate audio; drivers that reject it (e.g. some WASAPI endpoints) fall
# back to a supported rate plus software resampling.
RATE_CANDIDATES = (
    16000.0, 48000.0, 44100.0, 32000.0, 24000.0,
    22050.0, 11025.0, 8000.0, 96000.0, 192000.0,
)


class AudioError(Exception):
    """Raised for microphone/audio subsystem failures."""


class PermissionDeniedError(AudioError):
    """Raised when microphone access is denied by the OS."""


class DeviceUnavailableError(AudioError):
    """Raised when a requested audio device does not exist."""


@dataclass(frozen=True)
class AudioDevice:
    """Description of an input audio device."""

    index: int
    name: str
    max_input_channels: int
    default_samplerate: float
    hostapi: int = -1

    @property
    def device_id(self) -> str:
        """Stable-ish identifier used for persistence."""
        return f"{self.hostapi}:{self.index}:{self.name}"

    @property
    def stable_id(self) -> str:
        """Index-free identity: ``hostapi:normalized-name``.

        PortAudio device indices reshuffle between enumerations
        (observed: the same Intel array reported as ``0:1`` at startup
        and ``0:2`` in a later session banner). Diagnostics should
        prefer this over ``device_id`` when comparing endpoints.
        """
        norm = " ".join((self.name or "").lower().split())
        return f"{self.hostapi}:{norm}"


class AudioDeviceManager:
    """Enumerate and select microphone input devices.

    The ``sounddevice`` module is injectable for testing.
    """

    def __init__(self, sd: Any = None) -> None:
        if sd is None:
            import sounddevice as sd  # type: ignore
        self._sd = sd

    def list_input_devices(self) -> list[AudioDevice]:
        """Return all devices that support audio input."""
        devices: list[AudioDevice] = []
        try:
            raw = self._sd.query_devices()
        except Exception as exc:  # noqa: BLE001
            raise AudioError(f"Cannot query audio devices: {exc}") from exc

        for index, info in enumerate(raw):
            try:
                max_in = int(info.get("max_input_channels", 0))
            except (TypeError, ValueError):
                max_in = 0
            if max_in <= 0:
                continue
            name = str(info.get("name", f"device-{index}"))
            try:
                rate = float(info.get("default_samplerate", 16000.0))
            except (TypeError, ValueError):
                rate = 16000.0
            devices.append(
                AudioDevice(
                    index=index,
                    name=name,
                    max_input_channels=max_in,
                    default_samplerate=rate,
                    hostapi=int(info.get("hostapi", -1)),
                )
            )
        return devices

    def default_input_device(self) -> Optional[AudioDevice]:
        """Return the system default input device, or None if absent."""
        try:
            default_idx = int(self._sd.default.device[0])
        except Exception:  # noqa: BLE001
            default_idx = -1
        for device in self.list_input_devices():
            if device.index == default_idx:
                return device
        devices = self.list_input_devices()
        return devices[0] if devices else None

    @staticmethod
    def _device_key(device: AudioDevice) -> str:
        """Return the stable part of a device id: ``hostapi:index``.

        Windows truncates device names (~32 chars), so the name portion
        of ``device_id`` can differ between enumerations. The hostapi +
        index pair is the most reliable identifier sounddevice exposes.
        """
        return f"{device.hostapi}:{device.index}"

    @staticmethod
    def _key_from_device_id(device_id: str) -> str:
        """Extract the ``hostapi:index`` key from a persisted device id."""
        parts = device_id.split(":")
        return ":".join(parts[:2]) if len(parts) >= 2 else device_id

    @staticmethod
    def _norm_name(name: str) -> str:
        """Normalize a device name for tolerant comparison.

        Strips trademark/copyright markers (``®``, ``™``, ``(R)``,
        ``(TM)``) and all non-alphanumeric characters so the same
        physical endpoint compares equal across host APIs that render
        its name differently (e.g. MME truncation vs WASAPI full name).
        """
        t = (name or "").lower()
        for marker in ("®", "™", "(r)", "(tm)"):
            t = t.replace(marker, " ")
        t = re.sub(r"[^a-z0-9]+", " ", t)
        return " ".join(t.split())

    @classmethod
    def _names_match(cls, wanted: str, actual: str) -> bool:
        """Return True when two device names identify the same endpoint.

        Exact normalized equality, or a word-boundary prefix match to
        tolerate Windows' ~32-character name truncation (e.g. an MME
        name persisted as ``microphone array intel smart`` vs the fuller
        ``microphone array intel smart sound technology ...``). The
        shorter side must be at least 16 normalized characters and end
        on a word boundary so ``microphone array 1`` cannot match
        ``microphone array 10``.
        """
        w = cls._norm_name(wanted)
        a = cls._norm_name(actual)
        if not w or not a:
            return False
        if w == a:
            return True
        short, long_ = (w, a) if len(w) <= len(a) else (a, w)
        return (
            len(short) >= 16
            and long_.startswith(short)
            and long_[len(short):].startswith(" ")
        )

    @staticmethod
    def same_physical_device(a: Optional[AudioDevice], b: Optional[AudioDevice]) -> bool:
        """Return True when two device descriptions refer to the same input.

        PortAudio indexes are not stable across enumerations, so identity
        is host API + normalized full name (+ channel count as a tiebreak).
        """
        if a is None or b is None:
            return a is b
        if a.hostapi != b.hostapi:
            return False
        if not AudioDeviceManager._names_match(a.name, b.name):
            return False
        return a.max_input_channels == b.max_input_channels

    def supported_input_rates(
        self, device: AudioDevice, preferred: tuple[float, ...] = ()
    ) -> list[float]:
        """Return sample rates the device accepts for 1-channel float32 input.

        Rates are probed with ``sounddevice.check_input_settings`` in
        preference order (``preferred`` first, then ``RATE_CANDIDATES``).
        Returns an empty list when the probe is unavailable or every rate
        is rejected — callers must then try opening the stream directly.
        """
        ordered = list(dict.fromkeys([*preferred, *RATE_CANDIDATES]))
        rates: list[float] = []
        check = getattr(self._sd, "check_input_settings", None)
        if check is None:
            return rates
        for rate in ordered:
            try:
                check(
                    device=device.index,
                    channels=1,
                    samplerate=float(rate),
                )
            except Exception:  # noqa: BLE001 - unsupported rate/endpoint
                continue
            rates.append(float(rate))
        return rates

    def _match_by_id(
        self, device_id: str, devices: list[AudioDevice]
    ) -> tuple[Optional[AudioDevice], str, int]:
        """Match a persisted device id against the current inventory.

        Returns ``(device, strategy, name_candidates)`` where strategy is
        one of ``"exact"``, ``"name"``, ``"ambiguous"``, ``"index"`` or
        ``"none"``. PortAudio indexes are runtime enumeration positions —
        they are only a last resort for legacy ids that carry **no** name
        at all. An id that names a device but whose name matches nothing
        must NOT bind whatever device now occupies that index (hot-unplug
        index reuse would silently select a different physical mic), so
        it resolves to ``"none"`` and the caller uses the explicit
        default-device fallback instead.
        """
        for device in devices:
            if device.device_id == device_id:
                return device, "exact", 1
        wanted_key = self._key_from_device_id(device_id)
        wanted_hapi, _, _ = wanted_key.partition(":")
        try:
            wanted_hapi_int = int(wanted_hapi)
        except ValueError:
            wanted_hapi_int = -1
        parts = device_id.split(":", 2)
        wanted_name = parts[2].strip() if len(parts) == 3 else ""
        name_candidates = 0
        if wanted_name:
            name_hits = [
                d for d in devices
                if d.hostapi == wanted_hapi_int
                and self._names_match(wanted_name, d.name)
            ]
            name_candidates = len(name_hits)
            if len(name_hits) == 1:
                return name_hits[0], "name", name_candidates
            if len(name_hits) > 1:
                # Duplicate names on one host API: picking by index
                # could silently bind a different physical microphone.
                return None, "ambiguous", name_candidates
            # Named id, zero name matches: the stored name is the only
            # trustworthy identity we have and it matched nothing. Do
            # NOT guess via hostapi:index — after hot-unplug index reuse
            # that would silently bind a different physical microphone.
            return None, "none", name_candidates
        # Nameless legacy id (e.g. "2:1"): hostapi:index is the only
        # identity information available.
        for device in devices:
            if self._device_key(device) == wanted_key:
                return device, "index", name_candidates
        return None, "none", name_candidates

    @staticmethod
    def _log_device(label: str, device: AudioDevice) -> str:
        return (
            f"{label} name={device.name!r} hostapi={device.hostapi} "
            f"index={device.index} channels={device.max_input_channels} "
            f"rate={device.default_samplerate:.0f}"
        )

    def find_device(self, device_id: Optional[str]) -> Optional[AudioDevice]:
        """Resolve a persisted device id without falling back to default.

        Uses the same matching order as :meth:`resolve_device` (exact id,
        then hostapi + normalized name, then hostapi + index as a last
        resort for nameless/stale-name ids) but returns ``None`` when
        nothing matches — or when the name match is ambiguous — instead
        of silently binding a different microphone.
        """
        if not device_id or device_id == AUTO_DEVICE_ID:
            return None
        device, strategy, ncand = self._match_by_id(
            device_id, self.list_input_devices()
        )
        if strategy == "index":
            logger.warning(
                "Device id %r matched by unstable hostapi:index -> %s",
                device_id, self._log_device("", device),
            )
        elif strategy == "ambiguous":
            logger.warning(
                "Device id %r is ambiguous: %d name candidates on "
                "hostapi; refusing to guess",
                device_id, ncand,
            )
        return device

    def resolve_device(self, device_id: Optional[str]) -> AudioDevice:
        """Resolve a persisted device id to a current AudioDevice.

        Match order: exact id -> hostapi + normalized name (the stable
        identity; survives index reshuffles) -> hostapi + index (only
        for nameless legacy ids that carry no better identity) ->
        default with an explicit warning. Named ids whose name matches
        nothing are never resolved by index guess — after hot-unplug
        index reuse that would silently bind a different physical
        device — and ambiguous duplicate names go to default too.
        """
        devices = self.list_input_devices()
        if device_id and device_id != AUTO_DEVICE_ID:
            device, strategy, ncand = self._match_by_id(device_id, devices)
            if device is not None:
                if strategy == "name":
                    logger.info(
                        "Configured microphone %r resolved by name "
                        "(%d candidate) -> %s",
                        device_id, ncand, self._log_device("", device),
                    )
                elif strategy == "exact":
                    logger.debug(
                        "Configured microphone %r resolved by exact id "
                        "-> %s",
                        device_id, self._log_device("", device),
                    )
                elif strategy == "index":
                    logger.warning(
                        "Configured microphone %r resolved by unstable "
                        "hostapi:index -> %s (nameless legacy id; "
                        "index may have been reused by a different "
                        "device)",
                        device_id, self._log_device("", device),
                    )
                return device
            if strategy == "ambiguous":
                logger.warning(
                    "Configured microphone %r ambiguous: %d devices share "
                    "the name on its host API; using default instead of "
                    "guessing",
                    device_id, ncand,
                )
            else:
                logger.warning(
                    "Configured microphone %r not found "
                    "(%d name candidate(s)); using default",
                    device_id, ncand,
                )
        default = self.default_input_device()
        if default is None:
            raise DeviceUnavailableError("No microphone input device available")
        logger.info(
            "Microphone selection: default input device -> %s",
            self._log_device("", default),
        )
        return default

    def matches_stable_id(
        self, device: AudioDevice, device_id: str
    ) -> bool:
        """Return whether ``device`` is the endpoint a persisted id names.

        Stable identity = host API + normalized name (index-free), so a
        device resolved by name after an index reshuffle still counts as
        the configured microphone. Ids without a name portion fall back
        to the legacy hostapi:index comparison.
        """
        if not device_id or device_id == AUTO_DEVICE_ID:
            return False
        if device.device_id == device_id:
            return True
        parts = device_id.split(":", 2)
        name = parts[2] if len(parts) == 3 else ""
        if name:
            try:
                hapi = int(parts[0])
            except (ValueError, IndexError):
                return False
            return (
                device.hostapi == hapi
                and self._names_match(name, device.name)
            )
        return self._device_key(device) == self._key_from_device_id(device_id)

    def matches_device_id(self, device: AudioDevice, device_id: str) -> bool:
        """Return whether ``device`` corresponds to a persisted device id."""
        if device.device_id == device_id:
            return True
        return self._device_key(device) == self._key_from_device_id(device_id)

    def has_input_devices(self) -> bool:
        """Return True if at least one input device exists."""
        try:
            return len(self.list_input_devices()) > 0
        except AudioError:
            return False
