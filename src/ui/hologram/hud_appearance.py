"""HUD appearance presets and persistence (V10.8 balanced hologram).

Owns the seven user-facing appearance parameters and maps them onto
the existing :class:`OrbGLWidget` uniforms and the waveform widget.
Nothing here touches geometry, OpenGL objects, audio or monitor
selection — it only feeds renderer uniforms.

Persistence uses the same two-tier scheme as the rest of the HUD:
persisted ``user_settings`` keys (``hud.appearance.*``) win over the
JSON config section ``hud.appearance``, which in turn falls back to
the built-in Balanced preset. Values are stored as integer
percentages (0–100) and clamped on read *and* write.

State-dependent visual modulation lives in the renderer (per-mode
gain multipliers); it is render-time only and never overwrites the
values managed here.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Renderer-facing parameter names (0..1 fractions internally).
PARAMS = (
    "brightness",   # overall luminosity
    "glow",         # sphere emissive / fresnel rim
    "opacity",      # hologram artwork alpha (NOT window opacity)
    "ring",         # orbital ring illumination
    "particle",     # particle shell illumination
    "waveform",     # cyan/purple waveform intensity
    "speed",        # animation speed multiplier base (1.0 = 100%)
)

# Persisted setting keys.
KEY_PRESET = "hud.appearance.preset"


def _key(param: str) -> str:
    return f"hud.appearance.{param}"


# Presets as 0..1 fractions. Balanced is the mandated default.
BALANCED = {
    "brightness": 0.65,
    "glow": 0.55,
    "opacity": 0.95,
    "ring": 0.75,
    "particle": 0.60,
    "waveform": 0.70,
    "speed": 1.00,
}
PRESETS = {
    "subtle": {
        "brightness": 0.45,
        "glow": 0.35,
        "opacity": 0.85,
        "ring": 0.50,
        "particle": 0.40,
        "waveform": 0.50,
        "speed": 0.75,
    },
    "balanced": dict(BALANCED),
    "intense": {
        "brightness": 0.85,
        "glow": 0.80,
        "opacity": 1.00,
        "ring": 0.95,
        "particle": 0.85,
        "waveform": 1.00,
        "speed": 1.00,
    },
}
DEFAULT_PRESET = "balanced"


def _clamp01(value) -> Optional[float]:
    """Parse ``value`` as a 0..1 fraction; None when unparseable."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v:  # NaN
        return None
    return max(0.0, min(1.0, v))


class HudAppearance:
    """Loads, edits and persists HUD appearance settings.

    Args:
        settings_get / settings_set: persistence callbacks
            (``settings_repository.get/set``).
        config_get: ``ConfigurationManager.get`` for JSON defaults.
        on_change: optional callable invoked after every mutation so
            the controller can re-apply values to the live HUD.
    """

    def __init__(
        self,
        settings_get: Callable,
        settings_set: Callable,
        config_get: Optional[Callable] = None,
        on_change: Optional[Callable] = None,
    ) -> None:
        self._get = settings_get
        self._set = settings_set
        self._config_get = config_get
        self.on_change: Optional[Callable] = None
        self._external_on_change = on_change
        self._values = self._load()
        self._preset = self._load_preset()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def _user_value(self, key: str) -> Optional[str]:
        try:
            raw = self._get(key)
            if raw is not None and str(raw) != "":
                return str(raw)
        except Exception:  # noqa: BLE001
            pass
        return None

    def _config_value(self, dotted: str) -> Optional[str]:
        if self._config_get is None:
            return None
        try:
            cfg = self._config_get(dotted)
            if cfg is not None and str(cfg) != "":
                return str(cfg)
        except Exception:  # noqa: BLE001
            pass
        return None

    def _load_preset(self) -> str:
        raw = (self._user_value(KEY_PRESET)
               or self._config_value("hud.appearance.preset")
               or DEFAULT_PRESET)
        p = str(raw).strip().lower()
        return p if p in PRESETS or p == "custom" else DEFAULT_PRESET

    def _load(self) -> dict:
        """Resolve every parameter: user -> config -> Balanced."""
        values = {}
        for param in PARAMS:
            fallback = BALANCED[param]
            raw = (self._user_value(_key(param))
                   or self._config_value(f"hud.appearance.{param}"))
            if raw is None:
                values[param] = fallback
                continue
            # Values may be stored as 0-100 percentages or 0..1
            # fractions; normalise to a clamped 0..1 fraction.
            try:
                v = float(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "HUD appearance %s=%r invalid; using %.2f",
                    param, raw, fallback)
                values[param] = fallback
                continue
            if v > 1.0:
                v /= 100.0
            values[param] = max(0.0, min(1.0, v))
        return values

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def preset(self) -> str:
        return self._preset

    @property
    def values(self) -> dict:
        return dict(self._values)

    def value(self, param: str) -> float:
        return self._values.get(param, BALANCED.get(param, 0.0))

    # ------------------------------------------------------------------
    # Mutation + persistence
    # ------------------------------------------------------------------
    def _persist_value(self, param: str) -> None:
        try:
            # Store integer percentages — compact and unambiguous.
            self._set(_key(param),
                      str(int(round(self._values[param] * 100))))
        except Exception:  # noqa: BLE001
            logger.debug("HUD appearance persist failed for %s",
                         param, exc_info=True)

    def _persist_preset(self) -> None:
        try:
            self._set(KEY_PRESET, self._preset)
        except Exception:  # noqa: BLE001
            logger.debug("HUD appearance preset persist failed",
                         exc_info=True)

    def _notify(self) -> None:
        for cb in (self.on_change, self._external_on_change):
            if cb is None:
                continue
            try:
                cb()
            except Exception:  # noqa: BLE001
                logger.debug("HUD appearance on_change failed",
                             exc_info=True)

    def set_preset(self, name: str) -> bool:
        """Apply a named preset (subtle/balanced/intense).

        Persists the preset name *and* all seven values so a later
        config change cannot silently alter a saved customisation.
        """
        key = str(name or "").strip().lower()
        if key not in PRESETS:
            return False
        self._preset = key
        self._values = dict(PRESETS[key])
        for param in PARAMS:
            self._persist_value(param)
        self._persist_preset()
        logger.info("HUD appearance preset -> %s", key)
        self._notify()
        return True

    def set_value(self, param: str, fraction) -> bool:
        """Set one parameter (0..1 or 0-100). Marks preset ``custom``."""
        if param not in PARAMS:
            return False
        try:
            raw = float(fraction)
        except (TypeError, ValueError):
            return False
        if raw != raw:  # NaN
            return False
        # Accept both 0..1 fractions and 0-100 percentages.
        v = raw / 100.0 if raw > 1.0 else raw
        self._values[param] = max(0.0, min(1.0, v))
        if self._preset != "custom":
            self._preset = "custom"
            self._persist_preset()
        self._persist_value(param)
        self._notify()
        return True

    def set_value_percent(self, param: str, percent) -> bool:
        """Slider-friendly entry point: 0–100."""
        try:
            p = float(percent)
        except (TypeError, ValueError):
            return False
        return self.set_value(param, p / 100.0)

    def reset(self) -> bool:
        """Reset to the Balanced preset."""
        return self.set_preset(DEFAULT_PRESET)

    # ------------------------------------------------------------------
    # Applying to the live HUD
    # ------------------------------------------------------------------
    def renderer_params(self) -> dict:
        """0..1 fractions keyed by renderer-facing names."""
        return {p: self._values[p] for p in PARAMS}

    def apply_to(self, hud) -> bool:
        """Push current values into the HUD's orb + waveform.

        Missing widgets (test fakes, GL failure fallback) are skipped
        silently — appearance must never break the HUD.
        """
        try:
            orb = getattr(hud, "orb", None)
            wave = getattr(hud, "waveform", None)
            setter = getattr(orb, "set_appearance", None)
            if callable(setter):
                v = self._values
                setter(
                    brightness=v["brightness"],
                    glow=v["glow"],
                    opacity=v["opacity"],
                    ring=v["ring"],
                    particle=v["particle"],
                    speed=v["speed"],
                )
            wset = getattr(wave, "set_intensity", None)
            if callable(wset):
                wset(self._values["waveform"])
            return True
        except Exception:  # noqa: BLE001
            logger.debug("HUD appearance apply failed", exc_info=True)
            return False


__all__ = [
    "BALANCED",
    "DEFAULT_PRESET",
    "PARAMS",
    "PRESETS",
    "HudAppearance",
]
