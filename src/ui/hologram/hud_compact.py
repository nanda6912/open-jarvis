"""CompactHologramHUD — V10.8 single floating hologram overlay.

A small (240–420 logical px, always square) frameless, transparent,
always-on-top window that renders the *same* :class:`OrbGLWidget` 3D
scene used by the full HUD, plus the live waveform strip and a compact
status line. The dashboard panels of :class:`HologramHUD` are replaced
by the floating-hologram presentation; the bridge-facing setter API is
preserved so :class:`HudBridge` needs no changes.

Window behaviour:

* ``Qt.Tool`` — no taskbar button, no Alt+Tab entry.
* ``WindowDoesNotAcceptFocus`` + ``WA_ShowWithoutActivating`` — the HUD
  can never steal keyboard focus.
* ``WA_TranslucentBackground`` — fully transparent background; only the
  hologram pixels are drawn.
* Position lock — when locked the native window gets ``WS_EX_TRANSPARENT``
  (click-through: mouse input reaches the windows underneath). When
  unlocked the extended style is cleared and the window can be dragged
  with the mouse. ``WS_EX_NOACTIVATE``/``WS_EX_TOOLWINDOW`` are applied
  once so clicks never activate the window even while unlocked.

All Win32 access is lazy and guarded; on non-Windows platforms the
widget falls back to ``WA_TransparentForMouseEvents`` for click-through.
"""

from __future__ import annotations

import logging
import sys

from PySide6.QtCore import QEvent, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from src.ui.hologram.hud_widgets import (
    HUD_STYLESHEET,
    HudStatusBar,
    WaveformWidget,
)
from src.ui.hologram.orb_renderer import OrbGLWidget

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Win32 extended-window-style helpers (click-through / non-activation)
# ---------------------------------------------------------------------------

_GWL_EXSTYLE = -20
_WS_EX_TRANSPARENT = 0x00000020   # hit-test passes through to windows below
_WS_EX_TOOLWINDOW = 0x00000080    # no taskbar button / Alt+Tab entry
_WS_EX_NOACTIVATE = 0x08000000    # clicks never activate the window
_WS_EX_TOPMOST = 0x00000008       # topmost Z-order band

_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOZORDER = 0x0004
_SWP_NOACTIVATE = 0x0010
_SWP_FRAMECHANGED = 0x0020

# Win32 pseudo-handle for SetWindowPos hWndInsertAfter.
_HWND_TOPMOST = -1                # (HWND)-1: insert into the topmost band


def _user32():
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        long_ptr = ctypes.c_ssize_t  # pointer-sized LONG_PTR (Win64 safe)
        u = ctypes.windll.user32
        u.SetWindowLongPtrW.argtypes = [
            wintypes.HWND, ctypes.c_int, long_ptr]
        u.SetWindowLongPtrW.restype = long_ptr
        u.GetWindowLongPtrW.argtypes = [
            wintypes.HWND, ctypes.c_int]
        u.GetWindowLongPtrW.restype = long_ptr
        u.SetWindowPos.argtypes = [
            wintypes.HWND, wintypes.HWND,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            wintypes.UINT]
        u.SetWindowPos.restype = wintypes.BOOL
        return u
    except Exception:  # noqa: BLE001
        logger.debug("Win32 user32 unavailable", exc_info=True)
        return None


def _exstyle_get(hwnd: int) -> int:
    """Current GWL_EXSTYLE as an unsigned 32-bit value, or -1."""
    u = _user32()
    if u is None or not hwnd:
        return -1
    try:
        return int(u.GetWindowLongPtrW(hwnd, _GWL_EXSTYLE)) & 0xFFFFFFFF
    except Exception:  # noqa: BLE001
        logger.debug("GetWindowLongPtrW failed for hwnd %s", hwnd,
                     exc_info=True)
        return -1


def _exstyle_update(hwnd: int, add: int = 0, remove: int = 0) -> bool:
    """Add/remove bits in the window's GWL_EXSTYLE. Returns success."""
    u = _user32()
    if u is None or not hwnd:
        return False
    try:
        cur = _exstyle_get(hwnd)
        if cur < 0:
            return False
        new = (cur | add) & ~remove & 0xFFFFFFFF
        if new == cur:
            return True
        # SetWindowLongPtr* returns the *previous* value and 0 is
        # ambiguous (failure vs. previous==0), so verify by reading back.
        u.SetWindowLongPtrW(hwnd, _GWL_EXSTYLE, new)
        if _exstyle_get(hwnd) != new:
            return False
        # SWP_FRAMECHANGED makes the new exstyle take effect. NOZORDER
        # preserves the current Z-order band; topmost is (re)asserted
        # separately via ensure_window_topmost().
        u.SetWindowPos(
            hwnd, 0, 0, 0, 0, 0,
            _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOZORDER
            | _SWP_NOACTIVATE | _SWP_FRAMECHANGED,
        )
        return True
    except Exception:  # noqa: BLE001
        logger.debug("exstyle update failed for hwnd %s", hwnd,
                     exc_info=True)
        return False


def set_click_through(hwnd: int, enabled: bool) -> bool:
    """Toggle WS_EX_TRANSPARENT on a top-level HWND."""
    return _exstyle_update(
        hwnd,
        add=_WS_EX_TRANSPARENT if enabled else 0,
        remove=0 if enabled else _WS_EX_TRANSPARENT,
    )


def apply_overlay_chrome(hwnd: int, topmost: bool = True) -> bool:
    """Apply NOACTIVATE|TOOLWINDOW (+TOPMOST) once.

    WS_EX_TOPMOST is part of the one-time chrome so the exstyle bit is
    present; the actual topmost *band* membership is established by
    :func:`ensure_window_topmost` (SetWindowLongPtr alone does not move
    a window into the topmost Z-order band). ``topmost=False`` strips
    the bit for users who disabled always-on-top.
    """
    return _exstyle_update(
        hwnd,
        add=(_WS_EX_NOACTIVATE | _WS_EX_TOOLWINDOW
             | (_WS_EX_TOPMOST if topmost else 0)),
        remove=0 if topmost else _WS_EX_TOPMOST,
    )


def ensure_window_topmost(hwnd: int, reason: str = "") -> bool:
    """Place ``hwnd`` in the topmost Z-order band via SetWindowPos.

    Uses ``HWND_TOPMOST`` with NOMOVE|NOSIZE|NOACTIVATE so position is
    untouched and the window is never activated. This is the only
    reliable way to (re)assert topmost status after HWND creation,
    style changes, moves, resizes or HWND recreation — the Qt
    ``WindowStaysOnTopHint`` flag alone is not sufficient once the
    native window has been manipulated directly.
    """
    u = _user32()
    if u is None or not hwnd:
        return False
    try:
        ok = bool(u.SetWindowPos(
            hwnd, _HWND_TOPMOST, 0, 0, 0, 0,
            _SWP_NOMOVE | _SWP_NOSIZE | _SWP_NOACTIVATE,
        ))
        if ok:
            logger.debug(
                "HUD SetWindowPos TOPMOST ok hwnd=%s reason=%s",
                hwnd, reason or "-")
        else:
            logger.warning(
                "HUD SetWindowPos TOPMOST failed hwnd=%s reason=%s",
                hwnd, reason or "-")
        return ok
    except Exception:  # noqa: BLE001
        logger.debug("HUD topmost reapply failed hwnd=%s reason=%s",
                     hwnd, reason or "-", exc_info=True)
        return False


def describe_window_state(hwnd: int) -> str:
    """Human-readable GWL_EXSTYLE summary for diagnostics."""
    ex = _exstyle_get(hwnd)
    if ex < 0:
        return f"hwnd={hwnd} exstyle=<unavailable>"
    return (
        f"hwnd={hwnd} exstyle=0x{ex:08x} "
        f"topmost={bool(ex & _WS_EX_TOPMOST)} "
        f"transparent={bool(ex & _WS_EX_TRANSPARENT)} "
        f"noactivate={bool(ex & _WS_EX_NOACTIVATE)} "
        f"toolwindow={bool(ex & _WS_EX_TOOLWINDOW)}"
    )


# ---------------------------------------------------------------------------
# Compact HUD window
# ---------------------------------------------------------------------------

class CompactHologramHUD(QMainWindow):
    """Single compact floating hologram.

    Signals:
        hideRequested: user asked to hide the HUD.
        positionChanged: a drag-move finished (controller persists it).
        topmostChanged(bool): always-on-top preference changed.
        fullscreenChanged(bool): legacy fullscreen preference changed.
    """

    hideRequested = Signal()
    positionChanged = Signal()
    topmostChanged = Signal(bool)
    fullscreenChanged = Signal(bool)

    MIN_SIDE = 240
    MAX_SIDE = 420
    DEFAULT_SIDE = 320
    WAVEFORM_HEIGHT = 44

    def __init__(self, parent: QWidget | None = None,
                 target_fps: int = 30,
                 side: int = DEFAULT_SIDE) -> None:
        super().__init__(parent)
        self.setWindowTitle("VoiceOS HUD")
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setStyleSheet(HUD_STYLESHEET)

        side = self._clamp_side(side)
        self.setMinimumSize(self.MIN_SIDE, self.MIN_SIDE)
        self.setMaximumSize(self.MAX_SIDE, self.MAX_SIDE)
        self.resize(side, side)

        central = QWidget(self)
        central.setObjectName("hudRoot")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 10)
        root.setSpacing(6)

        self.status_label = QLabel(HudStatusBar._TEXT["standby"])
        self.status_label.setObjectName("hudCompactStatus")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(
            "color: #57d7ff; font-size: 11px; font-weight: bold;"
            " letter-spacing: 2px;")
        root.addWidget(self.status_label)

        self.orb = OrbGLWidget(self, target_fps=target_fps)
        # The orb never needs mouse input — leave every click for the
        # window-level drag handler (unlocked) or pass it through
        # natively (locked).
        self.orb.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        root.addWidget(self.orb, 1)

        self.waveform = WaveformWidget()
        self.waveform.setFixedHeight(self.WAVEFORM_HEIGHT)
        self.waveform.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        root.addWidget(self.waveform)

        self._position_locked = True
        self._drag_offset: QPoint | None = None
        self._chrome_applied = False
        self._fullscreen_pref = True
        self._screen_change_connected = False

    # ------------------------------------------------------------------
    @staticmethod
    def _clamp_side(side) -> int:
        try:
            side = int(side)
        except (TypeError, ValueError):
            side = CompactHologramHUD.DEFAULT_SIDE
        return max(CompactHologramHUD.MIN_SIDE,
                   min(CompactHologramHUD.MAX_SIDE, side))

    # ------------------------------------------------------------------
    # Position lock / click-through
    # ------------------------------------------------------------------
    @property
    def position_locked(self) -> bool:
        return self._position_locked

    def set_position_locked(self, locked: bool) -> None:
        """Locked: click-through. Unlocked: mouse-opaque + draggable."""
        locked = bool(locked)
        if locked == self._position_locked and self._chrome_applied:
            return
        self._position_locked = locked
        self._apply_click_through(locked)

    def _hwnd(self) -> int:
        try:
            return int(self.winId())
        except Exception:  # noqa: BLE001
            return 0

    def _apply_click_through(self, locked: bool) -> None:
        """WS_EX_TRANSPARENT on Windows; WA attr elsewhere/fallback."""
        hwnd = self._hwnd()
        native_ok = False
        if hwnd:
            try:
                native_ok = set_click_through(hwnd, locked)
            except Exception:  # noqa: BLE001
                native_ok = False
        if not native_ok:
            # Fallback (non-Windows or unusual Qt platform): widget-level
            # mouse transparency. Children inherit; the orb/waveform are
            # already transparent for input in both modes.
            self.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, locked)
        else:
            # The FRAMECHANGED SetWindowPos used NOZORDER, which keeps
            # the current band; re-assert topmost anyway so a lost
            # WS_EX_TOPMOST can never survive a lock/unlock toggle.
            self.reassert_topmost("click-through")
        self._chrome_applied = True

    def _apply_native_chrome(self, reason: str = "chrome") -> None:
        """One-time chrome + topmost band + current click-through state.

        Order matters: the exstyle bits are written first, then
        ``SetWindowPos(HWND_TOPMOST)`` moves the (existing) HWND into
        the topmost Z-order band, then TRANSPARENT is toggled with
        NOZORDER so the band is preserved.
        """
        want_topmost = bool(
            self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
        hwnd = self._hwnd()
        if hwnd:
            try:
                apply_overlay_chrome(hwnd, topmost=want_topmost)
            except Exception:  # noqa: BLE001
                logger.debug("HUD overlay chrome failed", exc_info=True)
            self.reassert_topmost(reason)
        self._apply_click_through(self._position_locked)
        self._chrome_applied = True
        self._log_window_state(reason)

    def reassert_topmost(self, reason: str = "reassert") -> bool:
        """Re-apply ``HWND_TOPMOST`` to the current HWND.

        Safe to call after any operation that could disturb Z-order
        (show, style change, drag end, resize, screen change, HWND
        recreation). Returns False when no HWND/user32 is available or
        when the always-on-top preference is disabled.
        """
        if not (self.windowFlags()
                & Qt.WindowType.WindowStaysOnTopHint):
            return False
        hwnd = self._hwnd()
        if not hwnd:
            return False
        return ensure_window_topmost(hwnd, reason)

    def _log_window_state(self, reason: str) -> None:
        """INFO-level dump of the native window style state."""
        hwnd = self._hwnd()
        if not hwnd:
            logger.info("HUD native state (%s): no HWND yet", reason)
            return
        logger.info("HUD native state (%s): %s",
                    reason, describe_window_state(hwnd))

    # ------------------------------------------------------------------
    # Drag handling (unlocked mode only)
    # ------------------------------------------------------------------
    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (not self._position_locked
                and event.button() == Qt.MouseButton.LeftButton):
            self._drag_offset = event.globalPosition().toPoint() - self.pos()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_offset is not None:
            self.move(event.globalPosition().toPoint() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_offset is not None:
            self._drag_offset = None
            # A drag can leave the window outside the topmost band;
            # re-assert once at drag end (never per move event).
            self.reassert_topmost("drag-release")
            self.positionChanged.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    # ------------------------------------------------------------------
    # Sizing — always square
    # ------------------------------------------------------------------
    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        # Resizes are rare (programmatic placement / square enforcement);
        # re-assert topmost once per resize, never per animation frame.
        self.reassert_topmost("resize")
        s = self.width()
        if s != self.height():
            # Keep the window perfectly square; the controller is the
            # only intended resizer and always requests square geometry,
            # this is a safety net. Deferred so it cannot recurse inside
            # the resize event itself.
            QTimer.singleShot(0, lambda: self._enforce_square(s))

    def _enforce_square(self, side: int) -> None:
        side = self._clamp_side(min(self.width(), side))
        if self.width() != side or self.height() != side:
            self.resize(side, side)

    # ------------------------------------------------------------------
    # Bridge-facing API (mirrors HologramHUD)
    # ------------------------------------------------------------------
    def set_status(self, state: str, detail: str = "") -> None:
        key = state if state in HudStatusBar._TEXT else "standby"
        self.status_label.setText(HudStatusBar._TEXT[key])
        color = HudStatusBar._COLOR[key].name()
        self.status_label.setStyleSheet(
            f"color: {color}; font-size: 11px; font-weight: bold;"
            " letter-spacing: 2px;")

    def set_orb_mode(self, mode: str) -> None:
        self.orb.set_mode(mode)

    def pulse_orb(self) -> None:
        self.orb.trigger_pulse()

    def set_orb_level(self, level: float) -> None:
        self.orb.set_level(level)
        self.waveform.push(level)

    def set_orb_speaking(self, speaking: bool) -> None:
        self.orb.set_speaking(speaking)

    def set_appearance(self, **params) -> None:
        """Push appearance params into orb uniforms + waveform."""
        orb_params = {k: v for k, v in params.items()
                      if k != "waveform"}
        self.orb.set_appearance(**orb_params)
        wave = params.get("waveform")
        if wave is not None:
            setter = getattr(self.waveform, "set_intensity", None)
            if callable(setter):
                setter(wave)

    # Panel data has no place on the compact overlay; the bridge calls
    # these unconditionally, so they are accepted and ignored.
    def set_system_rows(self, *args) -> None:
        pass

    def set_desktop_rows(self, *args) -> None:
        pass

    def set_activity(self, *args) -> None:
        pass

    def set_history(self, lines) -> None:
        pass

    def set_topmost(self, on: bool, block: bool = True) -> None:
        flags = self.windowFlags()
        if on:
            flags |= Qt.WindowType.WindowStaysOnTopHint
        else:
            flags &= ~Qt.WindowType.WindowStaysOnTopHint
        was_visible = self.isVisible()
        self.setWindowFlags(flags)
        if was_visible:
            self.show()
            self._apply_native_chrome(reason="set_topmost")
        if on:
            # setWindowFlags may recreate the HWND; make sure the
            # topmost band is (re)asserted on the live window.
            self.reassert_topmost("set_topmost:on")

    def set_fullscreen(self, on: bool, block: bool = True) -> None:
        # Legacy preference hook — the compact HUD has a fixed square
        # size; the flag is recorded for API compatibility only.
        self._fullscreen_pref = bool(on)

    # ------------------------------------------------------------------
    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._connect_screen_changed()
        self._apply_native_chrome(reason="show")
        self.orb.set_rendering_active(True)

    def _connect_screen_changed(self) -> None:
        """Re-assert topmost when the window moves to another screen."""
        if self._screen_change_connected:
            return
        try:
            wh = self.windowHandle()
            sig = getattr(wh, "screenChanged", None)
            if sig is not None:
                sig.connect(self._on_screen_changed)
                self._screen_change_connected = True
        except Exception:  # noqa: BLE001
            logger.debug("HUD screenChanged connect failed",
                         exc_info=True)

    def _on_screen_changed(self, screen) -> None:  # noqa: ARG002
        """Display change: Z-order band membership must be re-asserted."""
        self.reassert_topmost("screen-changed")
        self._log_window_state("screen-changed")

    def event(self, event) -> bool:  # noqa: N802
        # WinIdChange: Qt recreated the native HWND (e.g. after a
        # setWindowFlags call). Native styles/topmost must be re-applied
        # to the *new* handle once it exists.
        if event.type() == QEvent.Type.WinIdChange:
            QTimer.singleShot(
                0, lambda: self._apply_native_chrome(reason="hwnd-recreate"))
        return super().event(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        super().hideEvent(event)
        self.orb.set_rendering_active(False)

    def closeEvent(self, event) -> None:  # noqa: N802
        """Closing the HUD never shuts down VoiceOS — hide instead."""
        logger.info("HUD close requested; hiding (voice engine unaffected)")
        self.hide()
        event.accept()


__all__ = [
    "CompactHologramHUD",
    "apply_overlay_chrome",
    "describe_window_state",
    "ensure_window_topmost",
    "set_click_through",
]
