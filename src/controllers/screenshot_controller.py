"""Full virtual-desktop screenshot capture for VoiceOS.

Captures the entire virtual screen (all monitors, including negative
origin layouts) with GDI BitBlt, then encodes PNG through GDI+ — no
third-party imaging dependency. Files are written only under
``%LOCALAPPDATA%\\VoiceOS\\screenshots`` with unique timestamped names.

Nothing is uploaded or sent anywhere; capture happens only when the
voice handler explicitly asks for it.
"""

from __future__ import annotations

import ctypes
import logging
import os
import time
import uuid
from ctypes import wintypes
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

SRCCOPY = 0x00CC0020
CAPTUREBLT = 0x40000000

_gdiplus_token = None


class _GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", wintypes.DWORD),
        ("Data2", wintypes.WORD),
        ("Data3", wintypes.WORD),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid_from_uuid(text: str) -> _GUID:
    import uuid as _uuid

    g = _GUID()
    b = _uuid.UUID(text).bytes_le
    g.Data1 = int.from_bytes(b[0:4], "little")
    g.Data2 = int.from_bytes(b[4:6], "little")
    g.Data3 = int.from_bytes(b[6:8], "little")
    for i in range(8):
        g.Data4[i] = b[8 + i]
    return g


# PNG encoder CLSID {557CF406-1A04-11D3-9A73-0000F81EF32E}
_PNG_CLSID = _guid_from_uuid("557CF406-1A04-11D3-9A73-0000F81EF32E")


class _Gdiplus:
    """Lazy GDI+ startup + PNG file save."""

    def __init__(self) -> None:
        global _gdiplus_token
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("GDI+ unavailable")
        self._g = ctypes.windll.gdiplus

        class GdiplusStartupInput(ctypes.Structure):
            _fields_ = [
                ("GdiplusVersion", wintypes.UINT),
                ("DebugEventCallback", ctypes.c_void_p),
                ("SuppressBackgroundThread", wintypes.INT),
                ("SuppressExternalCodecs", wintypes.INT),
            ]

        inp = GdiplusStartupInput()
        inp.GdiplusVersion = 1
        token = ctypes.c_size_t(0)
        if _gdiplus_token is None:
            hr = self._g.GdiplusStartup(
                ctypes.byref(token), ctypes.byref(inp), None
            )
            if hr != 0:
                raise OSError(f"GdiplusStartup failed: {hr}")
            _gdiplus_token = token

    def save_png_from_hbitmap(self, hbitmap: int, path: str) -> bool:
        """Encode ``hbitmap`` to PNG at ``path`` via GDI+."""
        g = self._g
        bitmap = ctypes.c_void_p()
        hr = g.GdipCreateBitmapFromHBITMAP(
            wintypes.HBITMAP(hbitmap), None, ctypes.byref(bitmap)
        )
        if hr != 0 or not bitmap.value:
            logger.warning("GdipCreateBitmapFromHBITMAP failed: %s", hr)
            return False
        try:
            clsid = _GUID()
            clsid.Data1 = _PNG_CLSID.Data1
            clsid.Data2 = _PNG_CLSID.Data2
            clsid.Data3 = _PNG_CLSID.Data3
            for i in range(8):
                clsid.Data4[i] = _PNG_CLSID.Data4[i]
            hr = g.GdipSaveImageToFile(
                bitmap, str(path), ctypes.byref(clsid), None
            )
            if hr != 0:
                logger.warning("GdipSaveImageToFile failed: %s", hr)
                return False
            return True
        finally:
            g.GdipDisposeImage(bitmap)


class _Win32Capture:
    """GDI screen capture; all Win32 calls are lazy ctypes."""

    def __init__(self) -> None:
        if not hasattr(ctypes, "windll"):
            raise RuntimeError("Win32 unavailable")
        self._u = ctypes.windll.user32
        self._g = ctypes.windll.gdi32
        self._set_dpi_awareness()

    @staticmethod
    def _set_dpi_awareness() -> None:
        """Best-effort per-monitor DPI awareness so captures are in
        physical pixels. Failure is harmless."""
        try:
            u = ctypes.windll.user32
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 = -4
            if hasattr(u, "SetProcessDpiAwarenessContext"):
                u.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))
            else:
                u.SetProcessDPIAware()
        except Exception:  # noqa: BLE001
            pass

    def virtual_screen(self):
        """(x, y, width, height) of the whole virtual desktop."""
        u = self._u
        x = int(u.GetSystemMetrics(SM_XVIRTUALSCREEN))
        y = int(u.GetSystemMetrics(SM_YVIRTUALSCREEN))
        w = int(u.GetSystemMetrics(SM_CXVIRTUALSCREEN))
        h = int(u.GetSystemMetrics(SM_CYVIRTUALSCREEN))
        return x, y, w, h

    def capture(self):
        """BitBlt the virtual screen; return (hbitmap, w, h) or None."""
        x, y, w, h = self.virtual_screen()
        if w <= 0 or h <= 0:
            logger.warning("Invalid virtual screen %sx%s", w, h)
            return None
        u, g = self._u, self._g
        hdc = u.GetDC(0)
        if not hdc:
            return None
        mem = None
        bmp = None
        try:
            mem = g.CreateCompatibleDC(hdc)
            if not mem:
                return None
            bmp = g.CreateCompatibleBitmap(hdc, w, h)
            if not bmp:
                return None
            old = g.SelectObject(mem, bmp)
            ok = g.BitBlt(
                mem, 0, 0, w, h, hdc, x, y, SRCCOPY | CAPTUREBLT
            )
            g.SelectObject(mem, old)
            if not ok:
                logger.warning(
                    "BitBlt failed for %sx%s @%s,%s", w, h, x, y
                )
                g.DeleteObject(bmp)
                return None
            return int(bmp), w, h
        except Exception:  # noqa: BLE001
            logger.exception("Screen capture failed")
            if bmp:
                try:
                    g.DeleteObject(bmp)
                except Exception:  # noqa: BLE001
                    pass
            return None
        finally:
            if mem:
                g.DeleteDC(mem)
            u.ReleaseDC(0, hdc)

    def release(self, capture) -> None:
        hbitmap = capture[0] if capture else 0
        if hbitmap:
            try:
                self._g.DeleteObject(hbitmap)
            except Exception:  # noqa: BLE001
                pass


class ScreenshotEngine:
    """Engine protocol: capture()->(hbitmap,w,h)|None,
    save_png(cap,path)->bool, release(cap)."""

    def __init__(self, capture_fn=None, save_fn=None, release_fn=None):
        self.capture_fn = capture_fn
        self.save_fn = save_fn
        self.release_fn = release_fn


class ScreenshotController:
    """Capture the virtual desktop to a unique PNG under LOCALAPPDATA."""

    def __init__(
        self,
        out_dir: Optional[str] = None,
        engine=None,
    ) -> None:
        if engine is not None:
            self._engine = engine
        else:
            cap = _Win32Capture()
            gdip = _Gdiplus()
            self._engine = ScreenshotEngine(
                capture_fn=cap.capture,
                save_fn=lambda c, p: gdip.save_png_from_hbitmap(c[0], p),
                release_fn=cap.release,
            )
        if out_dir is not None:
            self._dir = out_dir
        else:
            from src.core.paths import runtime_dir

            self._dir = str(runtime_dir() / "screenshots")

    @property
    def directory(self) -> str:
        return self._dir

    def _unique_path(self) -> str:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        for _ in range(100):
            name = f"screenshot_{stamp}_{uuid.uuid4().hex[:6]}.png"
            path = os.path.join(self._dir, name)
            if not os.path.exists(path):
                return path
        return os.path.join(
            self._dir, f"screenshot_{stamp}_{uuid.uuid4().hex}.png"
        )

    def take_screenshot(self):
        """Capture + save; returns (ok, detail) — the absolute PNG path
        on success, an error string on failure."""
        cap = None
        try:
            cap = self._engine.capture_fn()
        except Exception:  # noqa: BLE001
            logger.exception("Screenshot capture raised")
            cap = None
        if not cap:
            return False, "capture_failed"
        try:
            os.makedirs(self._dir, exist_ok=True)
            path = self._unique_path()
            ok = bool(self._engine.save_fn(cap, path))
            if not ok:
                return False, "save_failed"
            # Verify the file actually landed before claiming success.
            if not (os.path.isfile(path) and os.path.getsize(path) > 0):
                return False, "save_failed"
            logger.info("Screenshot saved: %s", path)
            return True, path
        except OSError as exc:
            logger.exception("Screenshot save failed")
            return False, f"save_failed: {exc}"
        finally:
            try:
                self._engine.release_fn(cap)
            except Exception:  # noqa: BLE001
                pass
