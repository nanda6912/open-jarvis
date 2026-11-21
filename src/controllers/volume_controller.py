"""Windows master output-volume control via Core Audio (pycaw).

Targets the DEFAULT AUDIO RENDER endpoint only (``GetSpeakers``) —
never capture/microphone endpoints. All COM access is lazy and
initialized per worker thread; handlers call these methods from the
STT/command worker thread, never the Qt GUI thread.

The backend is injectable so tests never touch real audio hardware.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)


class _PycawBackend:
    """Adapt an pycaw endpoint to the scalar/mute backend protocol."""

    def __init__(self, endpoint) -> None:
        self._endpoint = endpoint

    @property
    def _vol(self):
        # pycaw exposes IAudioEndpointVolume as ``EndpointVolume``.
        return self._endpoint.EndpointVolume

    def get_scalar(self) -> float:
        return float(self._vol.GetMasterVolumeLevelScalar())

    def set_scalar(self, value: float) -> None:
        self._vol.SetMasterVolumeLevelScalar(float(value), None)

    def get_mute(self) -> bool:
        return bool(self._vol.GetMute())

    def set_mute(self, muted: bool) -> None:
        self._vol.SetMute(bool(muted), None)


def _default_backend_factory():
    """Build a backend for the default render endpoint, or None."""
    import ctypes

    if not hasattr(ctypes, "windll"):
        return None
    try:
        import comtypes  # noqa: F401  (ensures COM support is present)
        from pycaw.pycaw import AudioUtilities
    except ImportError:
        logger.warning("pycaw/comtypes unavailable; volume disabled")
        return None
    endpoint = AudioUtilities.GetSpeakers()
    if endpoint is None:
        return None
    return _PycawBackend(endpoint)


class VolumeController:
    """Get/set/mute the Windows master output volume (0-100 percent).

    Methods return ``None`` when no audio endpoint is available or the
    operation fails, so handlers can report honestly instead of
    claiming success.
    """

    def __init__(self, backend=None, backend_factory=None) -> None:
        # ``backend`` may be a ready backend or a zero-arg factory used
        # once per thread (COM objects are thread-affine).
        self._backend = backend
        self._factory = backend_factory or _default_backend_factory
        self._tls = threading.local()

    # -- backend resolution -------------------------------------------------
    def _get_backend(self):
        if self._backend is not None and not callable(self._backend):
            return self._backend
        sentinel = object()
        backend = getattr(self._tls, "backend", sentinel)
        if backend is not sentinel:
            return backend  # may be None: endpoint already probed
        factory = self._backend if callable(self._backend) else self._factory
        try:
            self._ensure_com()
            backend = factory()
        except Exception:  # noqa: BLE001
            logger.exception("Audio endpoint initialization failed")
            backend = None
        self._tls.backend = backend
        return backend

    @staticmethod
    def _ensure_com() -> None:
        """Initialize COM on this thread (idempotent per thread)."""
        try:
            import comtypes
        except ImportError:
            return
        tls = getattr(VolumeController, "_com_tls", None)
        if tls is None:
            VolumeController._com_tls = tls = threading.local()
        if getattr(tls, "initialized", False):
            return
        try:
            comtypes.CoInitialize()
        except Exception:  # noqa: BLE001 - S_FALSE/already-init is fine
            logger.debug("CoInitialize returned non-zero", exc_info=True)
        tls.initialized = True

    # -- public API ----------------------------------------------------------
    def get_percent(self) -> Optional[int]:
        """Current master volume 0-100, or None without an endpoint."""
        backend = self._get_backend()
        if backend is None:
            return None
        try:
            return int(round(float(backend.get_scalar()) * 100))
        except Exception:  # noqa: BLE001
            logger.exception("GetMasterVolumeLevelScalar failed")
            return None

    def set_percent(self, percent: int) -> Optional[int]:
        """Set master volume to ``percent`` (clamped 0-100)."""
        backend = self._get_backend()
        if backend is None:
            return None
        target = max(0, min(int(percent), 100))
        try:
            backend.set_scalar(target / 100.0)
        except Exception:  # noqa: BLE001
            logger.exception("SetMasterVolumeLevelScalar failed")
            return None
        return target

    def adjust(self, delta: int) -> Optional[int]:
        """Change volume by ``delta`` percent; returns new level."""
        backend = self._get_backend()
        if backend is None:
            return None
        try:
            current = float(backend.get_scalar())
            target = max(0.0, min(current + int(delta) / 100.0, 1.0))
            backend.set_scalar(target)
        except Exception:  # noqa: BLE001
            logger.exception("Volume adjust failed")
            return None
        return int(round(target * 100))

    def is_muted(self) -> Optional[bool]:
        backend = self._get_backend()
        if backend is None:
            return None
        try:
            return bool(backend.get_mute())
        except Exception:  # noqa: BLE001
            logger.exception("GetMute failed")
            return None

    def set_mute(self, muted: bool) -> bool:
        backend = self._get_backend()
        if backend is None:
            return False
        try:
            backend.set_mute(bool(muted))
        except Exception:  # noqa: BLE001
            logger.exception("SetMute failed")
            return False
        return True
