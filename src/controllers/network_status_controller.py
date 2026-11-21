"""Read-only network / Wi-Fi status for VoiceOS.

Three independent probes, each degrading to ``None`` (unknown) on
failure so the voice layer can say "I can't tell" instead of guessing:

- Internet connectivity: ``wininet!InternetGetConnectedState``.
- Wi-Fi: ``wlanapi`` (WlanOpenHandle/WlanEnumInterfaces/
  WlanQueryInterface ``wlan_intf_opcode_current_connection``) — gives
  SSID and signal quality of the current association. Never returns
  passwords or keys.
- Ethernet: ``iphlpapi!GetIfTable`` — any IEEE802.3 interface with
  oper-status up counts as an active wired link.

All Win32 access is lazy and isolated so tests can inject fakes and
non-Windows hosts simply report "unknown".
"""

from __future__ import annotations

import ctypes
import logging
import re
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

WLAN_OPCODE_CURRENT_CONNECTION = 7
WLAN_INTERFACE_STATE_CONNECTED = 1

IF_TYPE_ETHERNET_CSMACD = 6
IF_OPER_STATUS_UP = 1
ERROR_INSUFFICIENT_BUFFER = 122

_WIFI_NAME_RE = re.compile(r"wi-?fi|wireless|802\.11|wlan", re.I)


@dataclass(frozen=True)
class WifiStatus:
    """Wi-Fi adapter/connection snapshot (None = unknown)."""

    available: Optional[bool]   # a Wi-Fi (802.11) adapter exists
    connected: Optional[bool]
    ssid: Optional[str]
    signal: Optional[int]       # 0-100 percent, None when unknown


@dataclass(frozen=True)
class NetworkStatus:
    internet: Optional[bool]
    wifi: WifiStatus
    ethernet: Optional[bool]


# ----------------------------------------------------------------------
# wlanapi ctypes wrapper
# ----------------------------------------------------------------------
class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


class _WLAN_INTERFACE_INFO(ctypes.Structure):
    _fields_ = [
        ("InterfaceGuid", _GUID),
        ("strInterfaceDescription", wintypes.WCHAR * 256),
        ("isState", wintypes.DWORD),
    ]


class _DOT11_SSID(ctypes.Structure):
    _fields_ = [
        ("ucSSID", ctypes.c_ubyte * 32),
        ("uSSIDLength", wintypes.ULONG),
    ]


class _WLAN_ASSOCIATION_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("dot11Ssid", _DOT11_SSID),
        ("dot11BssType", ctypes.c_int),
        ("dot11Bssid", ctypes.c_ubyte * 6),
        ("dot11PhyType", ctypes.c_int),
        ("uDot11PhyIndex", wintypes.ULONG),
        ("wlanSignalQuality", wintypes.ULONG),
        ("ulRxRate", wintypes.ULONG),
        ("ulTxRate", wintypes.ULONG),
    ]


class _WLAN_CONNECTION_ATTRIBUTES(ctypes.Structure):
    """Prefix of the real struct — only fields before the security
    attributes are read, so the tail is intentionally omitted."""

    _fields_ = [
        ("isState", ctypes.c_int),
        ("wlanConnectionMode", ctypes.c_int),
        ("strProfileName", wintypes.WCHAR * 256),
        ("wlanAssociationAttributes", _WLAN_ASSOCIATION_ATTRIBUTES),
    ]


class _WlanApi:
    """Minimal wlanapi wrapper; raises on non-Windows."""

    def __init__(self) -> None:
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("wlanapi unavailable")
        self._w = ctypes.windll.wlanapi
        w = self._w
        w.WlanOpenHandle.argtypes = [
            wintypes.DWORD, wintypes.LPVOID,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.HANDLE),
        ]
        w.WlanOpenHandle.restype = wintypes.DWORD
        w.WlanCloseHandle.argtypes = [wintypes.HANDLE, wintypes.LPVOID]
        w.WlanEnumInterfaces.argtypes = [
            wintypes.HANDLE, wintypes.LPVOID,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        w.WlanEnumInterfaces.restype = wintypes.DWORD
        w.WlanQueryInterface.argtypes = [
            wintypes.HANDLE, ctypes.POINTER(_GUID), wintypes.DWORD,
            wintypes.LPVOID, ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(ctypes.c_void_p), wintypes.LPVOID,
        ]
        w.WlanQueryInterface.restype = wintypes.DWORD
        w.WlanFreeMemory.argtypes = [ctypes.c_void_p]

    def open(self):
        negotiated = wintypes.DWORD(0)
        handle = wintypes.HANDLE()
        hr = self._w.WlanOpenHandle(2, None, ctypes.byref(negotiated),
                                    ctypes.byref(handle))
        if hr != 0:
            raise OSError(f"WlanOpenHandle failed: {hr}")
        return int(handle.value)

    def close(self, handle: int) -> None:
        try:
            self._w.WlanCloseHandle(handle, None)
        except Exception:  # noqa: BLE001
            pass

    def interfaces(self, handle: int) -> list:
        """Return [(guid_bytes, description, is_state)]."""
        ptr = ctypes.c_void_p()
        hr = self._w.WlanEnumInterfaces(handle, None, ctypes.byref(ptr))
        if hr != 0 or not ptr.value:
            if hr != 0:
                raise OSError(f"WlanEnumInterfaces failed: {hr}")
            return []
        try:
            count = ctypes.cast(ptr, ctypes.POINTER(wintypes.DWORD)).contents
            n = int(count.value)
            # Layout: DWORD count, DWORD index, then WLAN_INTERFACE_INFO[].
            base = ptr.value + ctypes.sizeof(wintypes.DWORD) * 2
            arr = (_WLAN_INTERFACE_INFO * n).from_address(base)
            out = []
            for info in arr:
                desc = info.strInterfaceDescription
                raw = bytes(
                    ctypes.string_at(ctypes.addressof(info.InterfaceGuid), 16)
                )
                out.append((raw, desc, int(info.isState)))
            return out
        finally:
            self._w.WlanFreeMemory(ptr)

    def query_connection(self, handle: int, guid_bytes: bytes):
        """Return parsed _WLAN_CONNECTION_ATTRIBUTES or None."""
        guid = _GUID.from_buffer_copy(guid_bytes)
        size = wintypes.DWORD(0)
        data = ctypes.c_void_p()
        op = wintypes.DWORD(0)
        hr = self._w.WlanQueryInterface(
            handle, ctypes.byref(guid), WLAN_OPCODE_CURRENT_CONNECTION,
            None, ctypes.byref(size), ctypes.byref(data), ctypes.byref(op),
        )
        if hr != 0 or not data.value:
            return None
        try:
            attrs = _WLAN_CONNECTION_ATTRIBUTES.from_address(data.value)
            return attrs
        finally:
            self._w.WlanFreeMemory(data)

    @staticmethod
    def _is_wifi_name(desc: str) -> bool:
        return bool(_WIFI_NAME_RE.search(desc or ""))

    @staticmethod
    def _decode_ssid(attrs) -> Optional[str]:
        assoc = attrs.wlanAssociationAttributes
        ssid = assoc.dot11Ssid
        length = min(int(ssid.uSSIDLength), 32)
        if length <= 0:
            return None
        raw = bytes(ssid.ucSSID[:length])
        try:
            return raw.decode("utf-8").strip("\x00") or None
        except UnicodeDecodeError:
            return raw.decode("utf-8", "replace").strip("\x00") or None

    def wifi_status(self) -> WifiStatus:
        """Probe all interfaces; prefer a connected Wi-Fi adapter."""
        handle = self.open()
        try:
            infos = self.interfaces(handle)
        finally:
            self.close(handle)
        wifi_infos = [i for i in infos if self._is_wifi_name(i[1])]
        if not wifi_infos:
            return WifiStatus(
                available=False, connected=False, ssid=None, signal=None
            )

        handle = self.open()
        try:
            # Pass 1: connected adapter wins.
            for guid_bytes, _desc, _state in wifi_infos:
                attrs = self.query_connection(handle, guid_bytes)
                if attrs is None:
                    continue
                if int(attrs.isState) == WLAN_INTERFACE_STATE_CONNECTED:
                    ssid = self._decode_ssid(attrs)
                    if not ssid:
                        # Some drivers leave dot11Ssid empty; the
                        # profile name is the connected network's name.
                        ssid = (attrs.strProfileName or "").strip(
                            "\x00"
                        ) or None
                    return WifiStatus(
                        available=True,
                        connected=True,
                        ssid=ssid,
                        signal=self._signal(attrs),
                    )
            # Pass 2: adapter present but not associated.
            return WifiStatus(
                available=True, connected=False, ssid=None, signal=None
            )
        finally:
            self.close(handle)

    @staticmethod
    def _signal(attrs) -> Optional[int]:
        try:
            q = int(attrs.wlanAssociationAttributes.wlanSignalQuality)
        except Exception:  # noqa: BLE001
            return None
        return q if 0 <= q <= 100 else None


# ----------------------------------------------------------------------
# iphlpapi GetIfTable wrapper (ethernet link state)
# ----------------------------------------------------------------------
class _MIB_IFROW(ctypes.Structure):
    _fields_ = [
        ("dwIndex", wintypes.DWORD),
        ("dwType", wintypes.DWORD),
        ("dwMtu", wintypes.DWORD),
        ("dwSpeed", wintypes.DWORD),
        ("dwPhysAddrLen", wintypes.DWORD),
        ("bPhysAddr", ctypes.c_ubyte * 8),
        ("dwAdminStatus", wintypes.DWORD),
        ("dwOperStatus", wintypes.DWORD),
        ("dwLastChange", wintypes.DWORD),
        ("dwInOctets", wintypes.DWORD),
        ("dwInUcastPkts", wintypes.DWORD),
        ("dwInNUcastPkts", wintypes.DWORD),
        ("dwInDiscards", wintypes.DWORD),
        ("dwInErrors", wintypes.DWORD),
        ("dwInUnknownProtos", wintypes.DWORD),
        ("dwOutOctets", wintypes.DWORD),
        ("dwOutUcastPkts", wintypes.DWORD),
        ("dwOutNUcastPkts", wintypes.DWORD),
        ("dwOutDiscards", wintypes.DWORD),
        ("dwOutErrors", wintypes.DWORD),
        ("dwOutQLen", wintypes.DWORD),
        ("dwDescrLen", wintypes.DWORD),
        ("bDescr", ctypes.c_ubyte * 256),
    ]


class _MIB_IFTABLE(ctypes.Structure):
    _fields_ = [
        ("dwNumEntries", wintypes.DWORD),
        ("table", _MIB_IFROW * 1),
    ]


class _IfTableApi:
    """GetIfTable wrapper; raises on non-Windows."""

    def __init__(self) -> None:
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("iphlpapi unavailable")
        self._ip = ctypes.windll.iphlpapi
        self._ip.GetIfTable.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL,
        ]
        self._ip.GetIfTable.restype = wintypes.DWORD

    def ethernet_up(self) -> bool:
        """True when an Ethernet-type interface has oper-status up."""
        size = wintypes.DWORD(0)
        hr = self._ip.GetIfTable(None, ctypes.byref(size), False)
        if hr != ERROR_INSUFFICIENT_BUFFER or size.value == 0:
            raise OSError(f"GetIfTable sizing failed: {hr}")
        buf = ctypes.create_string_buffer(size.value)
        hr = self._ip.GetIfTable(buf, ctypes.byref(size), False)
        if hr != 0:
            raise OSError(f"GetIfTable failed: {hr}")
        table = _MIB_IFTABLE.from_buffer(buf)
        n = int(table.dwNumEntries)
        row_size = ctypes.sizeof(_MIB_IFROW)
        base = ctypes.addressof(buf) + ctypes.sizeof(wintypes.DWORD)
        for i in range(n):
            row = _MIB_IFROW.from_address(base + i * row_size)
            if (
                int(row.dwType) == IF_TYPE_ETHERNET_CSMACD
                and int(row.dwOperStatus) == IF_OPER_STATUS_UP
            ):
                return True
        return False


# ----------------------------------------------------------------------
# Controller
# ----------------------------------------------------------------------
def _internet_connected() -> bool:
    """wininet InternetGetConnectedState; raises on failure."""
    if not hasattr(ctypes, "windll"):
        raise RuntimeError("wininet unavailable")
    flags = wintypes.DWORD(0)
    ok = ctypes.windll.wininet.InternetGetConnectedState(
        ctypes.byref(flags), 0
    )
    if not ok:
        raise OSError("InternetGetConnectedState failed")
    # Any established connection mode counts; flag bit 0 = connected.
    return bool(flags.value)


class NetworkStatusController:
    """Aggregate connectivity probes; every field may be None."""

    def __init__(
        self,
        wlan=None,
        iftable=None,
        internet_fn=None,
    ) -> None:
        self._wlan_factory = wlan if wlan is not None else _WlanApi
        self._iftable_factory = (
            iftable if iftable is not None else _IfTableApi
        )
        self._internet_fn = internet_fn or _internet_connected
        # Instances are created lazily so import never touches Win32.
        self._wlan = None
        self._iftable = None
        self._wlan_failed = False
        self._iftable_failed = False

    @staticmethod
    def _resolve(factory, method: str):
        """Accept a class, zero-arg factory or ready instance."""
        if isinstance(factory, type):
            return factory()
        if callable(factory) and not hasattr(factory, method):
            return factory()
        return factory

    def _get_wlan(self):
        if self._wlan_failed:
            return None
        if self._wlan is None:
            try:
                self._wlan = self._resolve(self._wlan_factory, "wifi_status")
            except Exception:  # noqa: BLE001
                logger.debug("WlanApi init failed", exc_info=True)
                self._wlan_failed = True
                return None
        return self._wlan

    def _get_iftable(self):
        if self._iftable_failed:
            return None
        if self._iftable is None:
            try:
                self._iftable = self._resolve(
                    self._iftable_factory, "ethernet_up"
                )
            except Exception:  # noqa: BLE001
                logger.debug("IfTableApi init failed", exc_info=True)
                self._iftable_failed = True
                return None
        return self._iftable

    def internet(self) -> Optional[bool]:
        try:
            return bool(self._internet_fn())
        except Exception:  # noqa: BLE001
            logger.debug("Internet probe failed", exc_info=True)
            return None

    def wifi(self) -> WifiStatus:
        api = self._get_wlan()
        if api is None:
            return WifiStatus(
                available=None, connected=None, ssid=None, signal=None
            )
        try:
            return api.wifi_status()
        except Exception:  # noqa: BLE001
            logger.debug("Wlan probe failed", exc_info=True)
            return WifiStatus(
                available=None, connected=None, ssid=None, signal=None
            )

    def ethernet(self) -> Optional[bool]:
        api = self._get_iftable()
        if api is None:
            return None
        try:
            return bool(api.ethernet_up())
        except Exception:  # noqa: BLE001
            logger.debug("IfTable probe failed", exc_info=True)
            return None

    def status(self) -> NetworkStatus:
        return NetworkStatus(
            internet=self.internet(),
            wifi=self.wifi(),
            ethernet=self.ethernet(),
        )
