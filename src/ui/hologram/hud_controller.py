"""HUD lifecycle controller: creation, monitor placement, persistence.

Owns the compact :class:`CompactHologramHUD` window and the
:class:`HudBridge`. V10.8 turns the HUD into a single floating
hologram overlay: a square window (240–420 logical px, default 320)
placed near the bottom-left of a target screen's ``availableGeometry``.

Monitor selection resolves a configured *identity* (monitor model /
EDID name / serial / ``DISPLAYn`` device / logical index / role word)
against live ``QScreen`` objects first, then falls back to the Windows
``\\\\.\\DISPLAY2`` device identity, then to the leftmost non-primary
screen, then the primary screen. Per-monitor DPI is handled by working
exclusively in Qt logical pixels and only converting to physical pixels
when matching ``MonitorManager`` bounds.

Failure isolation: every public method is guarded; a HUD/GL failure is
logged and degrades to hidden, never affecting voice.
"""

from __future__ import annotations

import logging
import re
import time
import warnings
from typing import Callable, Optional

from PySide6.QtCore import QTimer, QRect
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QWidget

from src.ui.hologram.hud_appearance import HudAppearance
from src.ui.hologram.hud_bridge import HudBridge
from src.ui.hologram.hud_compact import CompactHologramHUD

logger = logging.getLogger(__name__)

# settings_repository keys
KEY_ENABLED = "hud.enabled"
KEY_MONITOR = "hud.monitor"          # identity, index, "displayN" or "auto"
KEY_TOPMOST = "hud.always_on_top"
KEY_FULLSCREEN = "hud.fullscreen"    # legacy; recorded, not geometric
KEY_FPS = "hud.fps"
KEY_SIZE = "hud.size"                # square side, logical px
KEY_POSITION = "hud.position"        # "x,y" logical px on KEY_POSITION_SCREEN
KEY_POSITION_SCREEN = "hud.position_screen"  # identity the pos belongs to
KEY_LOCKED = "hud.locked"            # click-through when true
KEY_LAST_SCREEN = "hud.last_screen"  # informational: resolved identity

HUD_SIZE_MIN = 240
HUD_SIZE_MAX = 420
HUD_SIZE_DEFAULT = 320
EDGE_MARGIN = 28                     # px from screen edges (spec: 24–32)

_RECHECK_MS = 5000
# V10.7.5: display-change notifications are debounced so a burst of
# WM_DISPLAYCHANGE events produces a single placement evaluation.
_DISPLAY_DEBOUNCE_MS = 400
# Per display-signature budget for setGeometry attempts: if the
# requested geometry cannot be honoured (e.g. the OS clamps the
# window), we stop re-applying instead of repositioning every 5 s.
_MAX_PLACE_ATTEMPTS = 3
# While geometry stays unconverged, retry on a slow cadence so a
# transient failure (window not yet mapped, mode switch) can still
# recover without hot-looping.
_PLACE_RETRY_SECONDS = 120.0


class HudController:
    """Creates, places and shows/hides the holographic HUD.

    Args:
        settings_get / settings_set: persistence callbacks (usually
            ``app_core.settings_repository.get/set``).
        monitor_manager: ``MonitorManager`` for display discovery.
        hud_factory: Optional override for tests (returns a widget with
            the HologramHUD API instead of a real GL window).
        bridge_factory: Optional override for the event bridge.
    """

    def __init__(
        self,
        settings_get: Callable,
        settings_set: Callable,
        monitor_manager=None,
        config_get: Optional[Callable] = None,
        hud_factory: Optional[Callable] = None,
        bridge_factory: Optional[Callable] = None,
    ) -> None:
        self._get = settings_get
        self._set = settings_set
        self._config_get = config_get
        self._monitors = monitor_manager
        self._hud_factory = hud_factory or CompactHologramHUD
        self._bridge_factory = bridge_factory or HudBridge

        self.hud = None
        self.bridge = None
        self._failed = False
        self._shown = False
        self._current_index: Optional[int] = None
        # V10.7.5 placement state: a placement is only re-evaluated
        # when the display signature (screens, geometry, target,
        # size, lock state) actually changes, or when the window
        # geometry drifted from what we last applied.
        self._last_display_signature = None
        self._placement_attempts = 0
        self._placement_confirmed = False
        self._place_retry_after = 0.0
        self._place_stuck_logged = False
        self._debounce_timer: Optional[QTimer] = None
        self._inventory_logged = False
        # V10.8 appearance settings (presets + persisted values). The
        # manager owns persistence; the controller only pushes values
        # into the live HUD via apply_to().
        self.appearance = HudAppearance(
            settings_get, settings_set, config_get)
        self.appearance.on_change = self._apply_appearance

    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return not self._failed and self.hud is not None

    @property
    def is_shown(self) -> bool:
        return self._shown and self.hud is not None and self.hud.isVisible()

    # ------------------------------------------------------------------
    def _pref_raw(self, key: str) -> Optional[str]:
        """User setting first, then JSON config default."""
        try:
            raw = self._get(key)
            if raw is not None and str(raw) != "":
                return str(raw)
        except Exception:  # noqa: BLE001
            pass
        if self._config_get is not None:
            try:
                cfg = self._config_get(f"hud.{key.split('.', 1)[1]}")
                if cfg is not None:
                    return str(cfg)
            except Exception:  # noqa: BLE001
                pass
        return None

    def _pref_bool(self, key: str, default: bool) -> bool:
        raw = self._pref_raw(key)
        if raw is None:
            return default
        return raw.strip().lower() in ("1", "true", "yes", "on")

    def _pref_str(self, key: str, default: str) -> str:
        raw = self._pref_raw(key)
        return raw if raw is not None else default

    def _pref_int(self, key: str, default: int) -> int:
        try:
            return int(float(self._pref_str(key, str(default))))
        except Exception:  # noqa: BLE001
            return default

    # ------------------------------------------------------------------
    # Public size / lock API (tray menu)
    # ------------------------------------------------------------------
    def hud_size(self) -> int:
        """Current square side length in logical pixels."""
        side = self._pref_int(KEY_SIZE, HUD_SIZE_DEFAULT)
        return max(HUD_SIZE_MIN, min(HUD_SIZE_MAX, side))

    def is_position_locked(self) -> bool:
        """Locked = click-through; unlocked = draggable."""
        return self._pref_bool(KEY_LOCKED, True)

    def set_position_locked(self, locked: bool) -> None:
        """Toggle click-through/drag mode; persist the preference.

        Locking also persists the current position so it survives a
        restart.
        """
        locked = bool(locked)
        try:
            self._set(KEY_LOCKED, "true" if locked else "false")
        except Exception:  # noqa: BLE001
            logger.debug("HUD lock persist failed", exc_info=True)
        setter = getattr(self.hud, "set_position_locked", None)
        if callable(setter):
            try:
                setter(locked)
            except Exception:  # noqa: BLE001
                logger.debug("HUD set_position_locked failed",
                             exc_info=True)
        if locked:
            self._persist_current_position()
        logger.info("HUD position %s", "locked" if locked else "unlocked")

    def reset_position(self) -> None:
        """Forget the saved position and return to bottom-left."""
        for key in (KEY_POSITION, KEY_POSITION_SCREEN):
            try:
                self._set(key, "")
            except Exception:  # noqa: BLE001
                pass
        self._reposition_now()
        logger.info("HUD position reset to default corner")

    def set_size(self, side: int) -> None:
        """Set the square side length (240–420 logical px)."""
        try:
            side = int(side)
        except (TypeError, ValueError):
            return
        side = max(HUD_SIZE_MIN, min(HUD_SIZE_MAX, side))
        try:
            self._set(KEY_SIZE, str(side))
        except Exception:  # noqa: BLE001
            logger.debug("HUD size persist failed", exc_info=True)
        self._reposition_now()
        self._persist_current_position()
        logger.info("HUD size set to %d", side)

    # ------------------------------------------------------------------
    def enabled(self) -> bool:
        return self._pref_bool(KEY_ENABLED, True)

    # ------------------------------------------------------------------
    def start(self, **bridge_kwargs) -> bool:
        """Create the HUD window + bridge. Returns True on success.

        ``bridge_kwargs`` are forwarded to :class:`HudBridge`
        (voice_signals, state_manager, services, probes).
        """
        if self._failed:
            return False
        if not self.enabled():
            logger.info("HUD disabled via %s", KEY_ENABLED)
            return False
        try:
            fps = int(float(self._pref_str(KEY_FPS, "30")))
            fps = max(10, min(60, fps))
        except Exception:  # noqa: BLE001
            fps = 30
        try:
            self.hud = self._hud_factory(target_fps=fps)
        except Exception:  # noqa: BLE001
            logger.exception("HUD creation failed; continuing without HUD")
            self._failed = True
            return False

        try:
            self.bridge = self._bridge_factory(self.hud, **bridge_kwargs)
        except Exception:  # noqa: BLE001
            logger.exception("HUD bridge failed; HUD panels only")
            try:
                self.bridge = None
            except Exception:  # noqa: BLE001
                pass

        # Wire HUD controls back to preferences.
        try:
            self.hud.hideRequested.connect(self.hide)
            self.hud.topmostChanged.connect(
                lambda on: self._persist(KEY_TOPMOST, on))
            self.hud.fullscreenChanged.connect(
                lambda on: self._apply_fullscreen(on))
            pos_sig = getattr(self.hud, "positionChanged", None)
            if pos_sig is not None:
                pos_sig.connect(self._on_hud_moved)
        except Exception:  # noqa: BLE001
            logger.debug("HUD control wiring failed", exc_info=True)

        try:
            self._apply_preferences()
            self._apply_appearance()
            self._log_hud_settings()
            self._log_display_inventory()
            self._place_on_target_monitor()
            self.show()
        except Exception:  # noqa: BLE001
            logger.exception("HUD placement failed; using fallback")
            try:
                self.hud.show()
            except Exception:  # noqa: BLE001
                pass
        return True

    # ------------------------------------------------------------------
    def _persist(self, key: str, value) -> None:
        try:
            self._set(key, "true" if value else "false")
        except Exception:  # noqa: BLE001
            logger.debug("HUD pref persist failed for %s", key,
                         exc_info=True)

    def _apply_preferences(self) -> None:
        topmost = self._pref_bool(KEY_TOPMOST, True)
        fullscreen = self._pref_bool(KEY_FULLSCREEN, True)
        locked = self.is_position_locked()
        self.hud.set_topmost(topmost)
        setter = getattr(self.hud, "set_position_locked", None)
        if callable(setter):
            setter(locked)
        self._fullscreen_pref = fullscreen
        self._apply_fullscreen(fullscreen)

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------
    def _hud_side(self) -> int:
        return self.hud_size()

    @staticmethod
    def _edge_margin(ag) -> int:
        try:
            smallest = min(ag.width(), ag.height())
        except Exception:  # noqa: BLE001
            smallest = 0
        return max(8, min(EDGE_MARGIN, smallest // 25)) if smallest else EDGE_MARGIN

    def _geometry_for(self, screen, side: int, pos) -> QRect:
        """Square ``side``x``side`` rect on ``screen`` at ``pos`` (or the
        default bottom-left corner), clamped inside availableGeometry."""
        ag = screen.availableGeometry()
        side = max(1, int(side))
        if pos is None:
            margin = self._edge_margin(ag)
            x = ag.left() + margin
            y = ag.top() + ag.height() - side - margin
        else:
            x, y = int(pos[0]), int(pos[1])
        if ag.width() >= side:
            x = min(max(x, ag.left()), ag.right() - side + 1)
        else:
            x = ag.left()
        if ag.height() >= side:
            y = min(max(y, ag.top()), ag.bottom() - side + 1)
        else:
            y = ag.top()
        return QRect(x, y, side, side)

    def _desired_geometry(self, screen) -> QRect:
        """The geometry the HUD should occupy on ``screen``."""
        side = self._hud_side()
        ag = screen.availableGeometry()
        try:
            max_side = min(ag.width(), ag.height())
        except Exception:  # noqa: BLE001
            max_side = 0
        if max_side > 0:
            side = min(side, max_side)
        return self._geometry_for(screen, side, self._saved_position(screen))

    def _saved_position(self, screen):
        """Return the persisted (x, y) if it belongs to ``screen``."""
        raw = self._pref_raw(KEY_POSITION)
        if raw is None:
            return None
        m = re.match(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*$", str(raw))
        if not m:
            return None
        x, y = int(m.group(1)), int(m.group(2))
        pscreen = self._pref_raw(KEY_POSITION_SCREEN)
        if pscreen:
            # Position is only valid on the screen it was saved on.
            if not self._screen_matches_identity(screen, pscreen):
                return None
        else:
            # No recorded screen: accept only if it lands on this one.
            g = screen.geometry()
            if not g.contains(x + self._hud_side() // 2,
                              y + self._hud_side() // 2):
                return None
        return (x, y)

    def _persist_current_position(self) -> None:
        """Save the HUD's current position + owning screen identity."""
        if self.hud is None:
            return
        try:
            screen = self.hud.screen()
            if screen is None:
                screens = self._screens()
                screen = screens[0] if screens else None
            if screen is None:
                return
            g = self.hud.geometry()
            self._set(KEY_POSITION, f"{g.x()},{g.y()}")
            self._set(KEY_POSITION_SCREEN, self._screen_identity(screen))
        except Exception:  # noqa: BLE001
            logger.debug("HUD position persist failed", exc_info=True)

    def _on_hud_moved(self) -> None:
        """Drag finished: persist position; re-apply lock state is the
        user's choice via the tray menu."""
        self._persist_current_position()

    def _reposition_now(self) -> None:
        """Immediately re-apply desired geometry (size/lock/reset)."""
        if self.hud is None or not self._shown:
            return
        try:
            screen = self._target_screen()
            if screen is not None:
                self.hud.setGeometry(self._desired_geometry(screen))
                self._reassert_hud_topmost("reposition")
        except Exception:  # noqa: BLE001
            logger.debug("HUD reposition failed", exc_info=True)

    def _reassert_hud_topmost(self, reason: str) -> None:
        """Ask the HUD to re-enter the topmost Z-order band.

        ``setGeometry``/``setWindowFlags`` can drop a window out of the
        topmost band; re-assert after every placement. Guarded so fake
        HUDs and non-Windows builds are unaffected.
        """
        fn = getattr(self.hud, "reassert_topmost", None)
        if callable(fn):
            try:
                fn(reason)
            except Exception:  # noqa: BLE001
                logger.debug("HUD topmost reassert failed",
                             exc_info=True)

    def _apply_appearance(self) -> None:
        """Push persisted appearance values into the live HUD.

        Called at startup and after every appearance mutation. Guarded:
        appearance must never break HUD or voice operation.
        """
        if self.hud is None:
            return
        try:
            self.appearance.apply_to(self.hud)
        except Exception:  # noqa: BLE001
            logger.debug("HUD appearance apply failed", exc_info=True)

    def _apply_fullscreen(self, on: bool) -> None:
        # V10.8: the compact HUD has a fixed square size; the legacy
        # fullscreen preference is recorded for compatibility but no
        # longer drives geometry.
        self._fullscreen_pref = bool(on)

    # ------------------------------------------------------------------
    # Monitor preference resolution
    # ------------------------------------------------------------------
    @staticmethod
    def _is_legacy_index_pref(value: str) -> bool:
        """True for bare numeric values like ``"1"``/``"2"``.

        Older HUD versions auto-persisted the resolved MonitorManager
        index into ``hud.monitor``; no current UI writes that key, so a
        bare number in *user* settings is treated as a legacy index,
        not a deliberate identity choice.
        """
        return bool(re.fullmatch(r"\s*\d+\s*", str(value or "")))

    def _config_pref(self, key: str) -> Optional[str]:
        """JSON config default for ``key`` (``hud.<name>``), or None."""
        if self._config_get is None:
            return None
        try:
            cfg = self._config_get(f"hud.{key.split('.', 1)[1]}")
            if cfg is not None and str(cfg) != "":
                return str(cfg)
        except Exception:  # noqa: BLE001
            pass
        return None

    def _user_pref(self, key: str) -> Optional[str]:
        """Raw persisted user setting for ``key``, or None."""
        try:
            raw = self._get(key)
            if raw is not None and str(raw) != "":
                return str(raw)
        except Exception:  # noqa: BLE001
            pass
        return None

    def _monitor_pref_candidates(self):
        """Ordered ``(pref, source)`` candidates for ``hud.monitor``.

        Precedence (V10.8 fix):

        1. A deliberate *user* identity (model / EDID name / serial /
           ``displayN`` / ``primary`` / role word) — non-numeric.
        2. The configured default (``config/default_settings.json``
           ``hud.monitor``, currently ``"dell"``).
        3. A legacy bare-numeric *user* value — auto-persisted monitor
           index from pre-V10.8 builds. It must not silently override
           an explicit configured identity, but still applies when it
           is the only signal available.
        """
        user = self._user_pref(KEY_MONITOR)
        cfg = self._config_pref(KEY_MONITOR)
        out = []
        if user and not self._is_legacy_index_pref(user):
            out.append((user, "user"))
        if cfg:
            out.append((cfg, "config"))
        if user and self._is_legacy_index_pref(user):
            out.append((user, "user:legacy-index"))
        return out

    # ------------------------------------------------------------------
    # Screen identity helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _screen_attr(screen, attr: str) -> str:
        try:
            v = getattr(screen, attr, None)
            if callable(v):
                v = v()
            return "" if v is None else str(v).strip()
        except Exception:  # noqa: BLE001
            return ""

    def _screen_texts(self, screen) -> str:
        """All human/machine identity strings Qt exposes for a screen."""
        parts = [
            self._screen_attr(screen, "name"),
            self._screen_attr(screen, "model"),
            self._screen_attr(screen, "manufacturer"),
            self._screen_attr(screen, "serialNumber"),
        ]
        return " ".join(p for p in parts if p)

    def _screen_identity(self, screen) -> str:
        """Best stable identity string for persistence/logging."""
        for attr in ("name", "serialNumber", "model"):
            v = self._screen_attr(screen, attr)
            if v:
                return v
        try:
            idx = self._screens().index(screen)
        except Exception:  # noqa: BLE001
            idx = -1
        return f"screen:{idx}"

    @staticmethod
    def _norm(text: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())

    def _screen_matches_identity(self, screen, identity) -> bool:
        """Does ``screen`` satisfy the configured identity string?

        Accepts: EDID/monitor names ("DELL 1908FP"), manufacturers,
        serials, ``displayN`` device indices, logical monitor indices,
        and the role word ``primary``.
        """
        ident = self._norm(str(identity or ""))
        if not ident or ident in ("auto", "default"):
            return False
        if ident == "primary":
            try:
                return QGuiApplication.primaryScreen() is screen
            except Exception:  # noqa: BLE001
                return False
        # "display2" / "\\\\.\\display2" / "display:2"
        m = re.search(r"display\s*#?\s*(\d+)$", ident)
        if m:
            return self._display_index_for_screen(screen) == int(m.group(1))
        # Bare number: logical MonitorManager index (1 = primary).
        if ident.isdigit():
            mon = self._monitor_for_index(int(ident))
            return (
                mon is not None
                and self._screen_for_bounds(getattr(mon, "bounds", None))
                is screen
            )
        # Token-subset match against all Qt-reported identities.
        tokens = set(ident.split())
        hay = set(self._norm(self._screen_texts(screen)).split())
        return bool(tokens) and tokens <= hay

    # ------------------------------------------------------------------
    # MonitorManager <-> QScreen mapping (DPI aware)
    # ------------------------------------------------------------------
    def _monitor_list(self):
        if self._monitors is None:
            return []
        try:
            return list(self._monitors.monitors())
        except Exception:  # noqa: BLE001
            logger.debug("Monitor enumeration failed", exc_info=True)
            return []

    @staticmethod
    def _bounds_tuple(bounds):
        """Normalise MonitorRect/QRect/tuple to (l, t, r, b)."""
        if bounds is None:
            return None
        try:
            # MonitorRect exposes plain attributes.
            left = bounds.left
            if not callable(left):
                return (int(left), int(bounds.top),
                        int(bounds.right), int(bounds.bottom))
        except Exception:  # noqa: BLE001
            pass
        try:
            # QRect-style accessors.
            return (int(bounds.left()), int(bounds.top()),
                    int(bounds.right()), int(bounds.bottom()))
        except Exception:  # noqa: BLE001
            pass
        try:
            l, t, r, b = bounds
            return (int(l), int(t), int(r), int(b))
        except Exception:  # noqa: BLE001
            return None

    def _screen_variants(self, screen):
        """Candidate physical/logical rects for ``screen``.

        MonitorManager bounds come from user32: physical pixels when the
        process is per-monitor DPI aware (Qt makes it so), logical
        pixels otherwise. A QScreen reports logical geometry; scaling by
        ``devicePixelRatio`` approximates the physical rect. Both are
        yielded so either awareness mode matches.
        """
        g = screen.geometry()
        yield (g.x(), g.y(), g.x() + g.width(), g.y() + g.height())
        try:
            d = float(screen.devicePixelRatio() or 1.0)
        except Exception:  # noqa: BLE001
            d = 1.0
        if abs(d - 1.0) > 1e-6:
            yield (
                round(g.x() * d), round(g.y() * d),
                round((g.x() + g.width()) * d),
                round((g.y() + g.height()) * d),
            )

    @staticmethod
    def _rects_close(a, b, tol: int = 3) -> bool:
        return all(abs(x - y) <= tol for x, y in zip(a, b))

    def _screen_for_bounds(self, bounds):
        """Match MonitorManager/user32 bounds to a QScreen."""
        want = self._bounds_tuple(bounds)
        if want is None:
            return None
        screens = self._screens()
        # Exact/near-exact match first.
        for s in screens:
            for v in self._screen_variants(s):
                if self._rects_close(v, want):
                    return s
        # Fallback: monitor centre inside the screen rect.
        cx, cy = (want[0] + want[2]) // 2, (want[1] + want[3]) // 2
        for s in screens:
            for v in self._screen_variants(s):
                if v[0] <= cx <= v[2] and v[1] <= cy <= v[3]:
                    return s
        return None

    def _display_index_for_screen(self, screen) -> Optional[int]:
        """Windows ``\\\\.\\DISPLAYn`` number for ``screen``, if resolvable."""
        for m in self._monitor_list():
            device = getattr(m, "device", "") or ""
            dm = re.search(r"display(\d+)$", device.lower())
            if not dm:
                continue
            if self._screen_for_bounds(getattr(m, "bounds", None)) is screen:
                return int(dm.group(1))
        return None

    def _screen_for_display_index(self, index: int):
        """QScreen backing Windows display device ``\\\\.\\DISPLAYn``."""
        for m in self._monitor_list():
            device = getattr(m, "device", "") or ""
            dm = re.search(r"display(\d+)$", device.lower())
            if dm and int(dm.group(1)) == int(index):
                s = self._screen_for_bounds(getattr(m, "bounds", None))
                if s is not None:
                    return s
        # Fallback: Qt screen name may itself be a device path.
        needle = f"display{int(index)}"
        for s in self._screens():
            if needle in (self._screen_attr(s, "name") or "").lower():
                return s
        return None

    def target_monitor_index(self) -> Optional[int]:
        """Pick the HUD display's logical MonitorManager index.

        Legacy informational API kept for compatibility: explicit
        numeric / ``displayN`` prefs resolve directly; otherwise the
        resolved QScreen is mapped back through monitor bounds.
        """
        mons = self._monitor_list()
        if not mons:
            return None
        for pref, _source in self._monitor_pref_candidates():
            p = (pref or "").strip().lower()
            if not p or p in ("auto", "default"):
                continue
            # "displayN" Windows device identity.
            m = re.search(r"display\s*#?\s*(\d+)$", self._norm(p))
            if m:
                n = int(m.group(1))
                for mon in mons:
                    dm = re.search(
                        r"display(\d+)$", (mon.device or "").lower())
                    if dm and int(dm.group(1)) == n:
                        return mon.index
            # Bare digits: logical index first, device index second.
            if p.isdigit():
                idx = int(p)
                if self._monitor_for_index(idx) is not None:
                    return idx
                for mon in mons:
                    dm = re.search(
                        r"display(\d+)$", (mon.device or "").lower())
                    if dm and int(dm.group(1)) == idx:
                        return mon.index
            # Identity tokens (model/EDID/serial/manufacturer): resolve
            # through the same screen path as _target_screen.
            screen = self._screen_for_pref(pref)
            if screen is not None:
                for mon in mons:
                    if (self._screen_for_bounds(
                            getattr(mon, "bounds", None)) is screen):
                        return mon.index
                break
        screen = self._target_screen()
        if screen is not None:
            for m in mons:
                if (self._screen_for_bounds(getattr(m, "bounds", None))
                        is screen):
                    return m.index
        # Convention fallback: Monitor 2 — LEFT hosts the HUD.
        m2 = next((m for m in mons if m.index == 2), None)
        if m2 is not None:
            return m2.index
        non_primary = [m for m in mons if not m.primary]
        if non_primary:
            leftmost = min(non_primary,
                           key=lambda m: (m.bounds.left, m.bounds.top))
            return leftmost.index
        return mons[0].index if mons else None

    def _monitor_for_index(self, index):
        for m in self._monitor_list():
            if m.index == index:
                return m
        return None

    def _screens(self):
        try:
            return list(QGuiApplication.screens())
        except Exception:  # noqa: BLE001
            return []

    def _primary_screen(self):
        try:
            return QGuiApplication.primaryScreen()
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------------------------------------------
    # Target screen resolution
    # ------------------------------------------------------------------
    def _screen_for_pref(self, pref: str):
        """Resolve an explicit configured target to a QScreen."""
        p = (pref or "").strip().lower()
        if not p or p in ("auto", "default"):
            return None
        if p == "primary":
            return self._primary_screen()
        # "displayN" device identity.
        m = re.search(r"display\s*#?\s*(\d+)$", self._norm(p))
        if m:
            return self._screen_for_display_index(int(m.group(1)))
        # Bare digits: logical MonitorManager index first, then the
        # Windows display-device index.
        if p.isdigit():
            idx = int(p)
            mon = self._monitor_for_index(idx)
            if mon is not None:
                s = self._screen_for_bounds(getattr(mon, "bounds", None))
                if s is not None:
                    return s
            return self._screen_for_display_index(idx)
        # Identity tokens (model / EDID name / serial / manufacturer).
        for s in self._screens():
            if self._screen_matches_identity(s, pref):
                return s
        return None

    @staticmethod
    def _pref_looks_dell(pref: str) -> bool:
        p = (pref or "").lower()
        return "dell" in p or "1908fp" in p

    def _auto_screen(self, screens):
        """Default target when nothing explicit resolves.

        Order: Dell-like external identity -> Windows DISPLAY2 device ->
        leftmost non-primary -> primary.
        """
        primary = self._primary_screen()
        for s in screens:
            try:
                if self._screen_matches_identity(s, "dell"):
                    return s
            except Exception:  # noqa: BLE001
                continue
        s = self._screen_for_display_index(2)
        if s is not None:
            return s
        others = [s for s in screens if s is not primary]
        if others:
            return min(others,
                       key=lambda s: (s.geometry().x(), s.geometry().y()))
        return primary if primary is not None else screens[0]

    def _resolve_target_screen(self):
        """Return ``(screen, reason)`` for the HUD target.

        ``reason`` is a short provenance string used in diagnostics:
        which preference candidate selected the screen, or which
        fallback stage produced it.
        """
        screens = self._screens()
        if not screens:
            return None, "no-screens"
        for pref, source in self._monitor_pref_candidates():
            p = (pref or "").strip().lower()
            if not p or p in ("auto", "default"):
                continue
            screen = self._screen_for_pref(pref)
            if screen is not None:
                return screen, f"pref={pref!r} source={source}"
            # The configured target is the Dell but its identity cannot
            # be resolved right now: DISPLAY2 is the documented
            # fallback before trying weaker candidates.
            if self._pref_looks_dell(pref):
                screen = self._screen_for_display_index(2)
                if screen is not None:
                    return (screen,
                            f"pref={pref!r} source={source} "
                            "via=display2-fallback")
        screen = self._auto_screen(screens)
        return screen, "auto-fallback"

    def _target_screen(self):
        screen, _reason = self._resolve_target_screen()
        return screen

    @staticmethod
    def _monitor_index_for_screen(screen) -> Optional[int]:
        return None  # resolved dynamically; informational only

    def _place_on_target_monitor(self) -> None:
        screen, reason = self._resolve_target_screen()
        if screen is None or self.hud is None:
            return
        self.hud.setGeometry(self._desired_geometry(screen))
        self._reassert_hud_topmost("place")
        idx = self.target_monitor_index()
        self._current_index = idx
        try:
            self._set(KEY_LAST_SCREEN, self._screen_identity(screen))
        except Exception:  # noqa: BLE001
            pass
        logger.info(
            "HUD target screen: %r identity=%r geometry=%s "
            "available=%s dpr=%.3f selected_by=%s",
            self._screen_attr(screen, "name"),
            self._screen_identity(screen),
            screen.geometry(), screen.availableGeometry(),
            float(screen.devicePixelRatio() or 1.0),
            reason,
        )

    def _log_hud_settings(self) -> None:
        """Dump every HUD setting with its source (user DB vs config).

        Makes monitor-selection provenance auditable from the log: a
        stale persisted ``hud.monitor`` is visible next to the config
        default it was superseded by.
        """
        for key in (
            KEY_ENABLED, KEY_MONITOR, KEY_TOPMOST, KEY_FULLSCREEN,
            KEY_FPS, KEY_SIZE, KEY_POSITION, KEY_POSITION_SCREEN,
            KEY_LOCKED,
        ):
            try:
                user = self._get(key)
            except Exception:  # noqa: BLE001
                user = "<error>"
            cfg = self._config_pref(key)
            logger.info(
                "HUD setting %s: user=%r config=%r", key, user, cfg)
        try:
            logger.info(
                "HUD monitor pref candidates: %r",
                self._monitor_pref_candidates())
        except Exception:  # noqa: BLE001
            logger.debug("HUD pref candidates failed", exc_info=True)

    # ------------------------------------------------------------------
    # Display inventory logging (once per run + on demand)
    # ------------------------------------------------------------------
    def _log_display_inventory(self, force: bool = False) -> None:
        if self._inventory_logged and not force:
            return
        self._inventory_logged = True
        try:
            screens = self._screens()
            logger.info("HUD display inventory: %d screen(s)", len(screens))
            for i, s in enumerate(screens):
                logger.info(
                    "  screen[%d] name=%r model=%r manufacturer=%r "
                    "serial=%r geometry=%s available=%s dpr=%.3f",
                    i,
                    self._screen_attr(s, "name"),
                    self._screen_attr(s, "model"),
                    self._screen_attr(s, "manufacturer"),
                    self._screen_attr(s, "serialNumber"),
                    s.geometry(), s.availableGeometry(),
                    float(s.devicePixelRatio() or 1.0),
                )
            for m in self._monitor_list():
                logger.info(
                    "  monitor device=%s index=%d bounds=%s primary=%s",
                    getattr(m, "device", "?"), getattr(m, "index", "?"),
                    getattr(m, "bounds", "?"), getattr(m, "primary", "?"),
                )
        except Exception:  # noqa: BLE001
            logger.debug("HUD display inventory failed", exc_info=True)

    # ------------------------------------------------------------------
    # V10.7.5 display signature + event-driven placement
    # ------------------------------------------------------------------
    def _display_signature(self):
        """Identity of the current display configuration.

        Any change — monitor added/removed, resolution or geometry
        change, target-display change, size or lock change — produces a
        different signature and re-arms placement.
        """
        parts = []
        for s in self._screens():
            try:
                g = s.geometry()
                ag = s.availableGeometry()
                parts.append((
                    self._screen_attr(s, "name"),
                    self._screen_identity(s),
                    int(g.x()), int(g.y()), int(g.width()), int(g.height()),
                    int(ag.x()), int(ag.y()),
                    int(ag.width()), int(ag.height()),
                    round(float(s.devicePixelRatio()), 3),
                ))
            except Exception:  # noqa: BLE001
                parts.append(())
        try:
            screen = self._target_screen()
            target = self._screen_identity(screen) if screen is not None else None
        except Exception:  # noqa: BLE001
            target = None
        return (
            tuple(parts),
            target,
            self._hud_side(),
            self.is_position_locked(),
        )

    def _connect_display_signals(self) -> None:
        """React to real display changes; the 5 s timer stays as a
        fallback poll (it is now signature-gated, so it is cheap)."""
        try:
            app = QGuiApplication.instance()
        except Exception:  # noqa: BLE001
            return
        if app is None:
            return
        connected = []
        for sig_name in (
            "screenAdded", "screenRemoved", "primaryScreenChanged",
        ):
            try:
                sig = getattr(app, sig_name, None)
                if sig is not None:
                    sig.connect(self._on_display_change)
                    connected.append(sig_name)
            except Exception:  # noqa: BLE001
                logger.debug("HUD display signal %s unavailable",
                             sig_name, exc_info=True)
        if connected:
            self._display_signals_connected = connected

    def _on_display_change(self, *_args) -> None:
        """Debounced entry point for Qt display notifications."""
        try:
            timer = getattr(self, "_debounce_timer", None)
            if timer is None:
                timer = QTimer(self.hud)
                timer.setSingleShot(True)
                timer.setInterval(_DISPLAY_DEBOUNCE_MS)
                timer.timeout.connect(self._recheck_placement)
                self._debounce_timer = timer
            timer.start()
        except Exception:  # noqa: BLE001
            logger.debug("HUD display-change debounce failed",
                         exc_info=True)

    def _start_recheck_timer(self) -> None:
        if getattr(self, "_recheck_timer", None) is not None:
            return
        self._connect_display_signals()
        self._recheck_timer = QTimer(self.hud)
        self._recheck_timer.setInterval(_RECHECK_MS)
        self._recheck_timer.timeout.connect(self._recheck_placement)
        self._recheck_timer.start()

    def _recheck_placement(self) -> None:
        """Re-place the HUD only when the display actually changed.

        V10.7.5: previously this compared ``hud.geometry()`` against
        the desired rect every 5 s and re-applied ``setGeometry``
        whenever they differed — on the user's single-monitor desktop
        the comparison never converged, so the HUD was re-placed (and
        logged) every tick forever. Now placement is gated on a
        display *signature*: nothing is touched while display
        identity, geometry, target and mode are unchanged, and a
        bounded per-signature attempt budget prevents a
        non-converging geometry from hot-looping.

        V10.8: while the position is *unlocked* and the display
        signature is unchanged, the recheck is a no-op so the user can
        drag the HUD without the timer snapping it back.
        """
        if self.hud is None or not self._shown:
            return
        try:
            screen, _reason = self._resolve_target_screen()
            if screen is None:
                return
            signature = self._display_signature()
            signature_changed = signature != self._last_display_signature
            if signature_changed:
                self._last_display_signature = signature
                self._placement_attempts = 0
                self._placement_confirmed = False
                self._place_retry_after = 0.0
                self._place_stuck_logged = False
                self._inventory_logged = False
                self._log_display_inventory()
                logger.debug(
                    "HUD display signature changed; re-evaluating "
                    "placement"
                )
            elif not self.is_position_locked():
                # Unlocked + stable display: the user owns the position.
                return
            want = self._desired_geometry(screen)
            actual = self.hud.geometry()
            if actual == want:
                if not self._placement_confirmed:
                    self._placement_confirmed = True
                    self._placement_attempts = 0
                    logger.debug("HUD placement confirmed: %s", want)
                return
            if self._placement_confirmed:
                # Geometry was right before and drifted — an external
                # move; allow one fresh placement attempt.
                self._placement_confirmed = False
                self._placement_attempts = 0
            now = time.monotonic()
            if self._placement_attempts >= _MAX_PLACE_ATTEMPTS:
                if now < self._place_retry_after:
                    return
                # Slow retry: the mismatch may be transient.
                self._place_retry_after = now + _PLACE_RETRY_SECONDS
                self._placement_attempts = 0
            self._placement_attempts += 1
            self.hud.setGeometry(want)
            self._reassert_hud_topmost("placement")
            new_actual = self.hud.geometry()
            if new_actual == want:
                self._placement_confirmed = True
                self._placement_attempts = 0
                self._place_stuck_logged = False
                logger.info("HUD placed on display: %s", want)
            else:
                # Once non-convergence is established the slow retries
                # stay quiet — only a geometry *change* or a new display
                # signature produces INFO/WARNING output again.
                log = logger.debug if self._place_stuck_logged else (
                    logger.info
                )
                log(
                    "HUD re-placed on display (attempt %d): "
                    "want=%s actual=%s",
                    self._placement_attempts, want, new_actual,
                )
                if (
                    self._placement_attempts >= _MAX_PLACE_ATTEMPTS
                    and not self._place_stuck_logged
                ):
                    self._place_stuck_logged = True
                    logger.warning(
                        "HUD geometry does not converge to %s "
                        "(actual=%s); pausing re-placement for %.0fs",
                        want, new_actual, _PLACE_RETRY_SECONDS,
                    )
        except Exception:  # noqa: BLE001
            logger.debug("HUD placement recheck failed", exc_info=True)

    # ------------------------------------------------------------------
    def show(self) -> None:
        if self.hud is None:
            return
        try:
            self._shown = True
            self.hud.show()
            # Never activateWindow(): the HUD must not steal keyboard
            # focus. WindowDoesNotAcceptFocus + WS_EX_NOACTIVATE keep it
            # non-activating even when raised.
            self.hud.raise_()
            self._reassert_hud_topmost("show")
            self._start_recheck_timer()
            logger.info("HUD shown")
        except Exception:  # noqa: BLE001
            logger.exception("HUD show failed")

    def hide(self) -> None:
        """Hide the HUD; the voice engine continues unaffected."""
        self._shown = False
        if self.hud is not None:
            try:
                self.hud.hide()
            except Exception:  # noqa: BLE001
                pass
        logger.info("HUD hidden (voice engine unaffected)")

    def toggle(self) -> None:
        if self.is_shown:
            self.hide()
        else:
            self.show()

    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        """Tear down HUD + bridge. Voice engine shutdown is separate."""
        try:
            if self.bridge is not None:
                self.bridge.shutdown()
        except Exception:  # noqa: BLE001
            pass
        try:
            timer = getattr(self, "_recheck_timer", None)
            if timer is not None:
                timer.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            debounce = getattr(self, "_debounce_timer", None)
            if debounce is not None:
                debounce.stop()
        except Exception:  # noqa: BLE001
            pass
        try:
            app = QGuiApplication.instance()
            connected = getattr(self, "_display_signals_connected", None)
            if app is not None and connected:
                for sig_name in connected:
                    sig = getattr(app, sig_name, None)
                    if sig is not None:
                        with warnings.catch_warnings():
                            # PySide6 warns (not raises) when a
                            # connection is already gone.
                            warnings.simplefilter("ignore")
                            try:
                                sig.disconnect(self._on_display_change)
                            except Exception:  # noqa: BLE001
                                pass
                self._display_signals_connected = None
        except Exception:  # noqa: BLE001
            pass
        try:
            if self.hud is not None:
                self.hud.close()
                self.hud.deleteLater()
        except Exception:  # noqa: BLE001
            pass
        self.hud = None
        self.bridge = None
