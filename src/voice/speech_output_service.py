"""Queued text-to-speech playback worker.

``speak()`` enqueues text; a dedicated worker thread synthesizes each
item and plays the resulting waveform through ``sounddevice``. An
optional ``gate`` object (e.g. the microphone capture service) is paused
for the whole synthesize+playback window so Jarvis never hears itself.

All callbacks run on the worker thread — receivers must marshal to the
GUI thread themselves.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Optional

import numpy as np

from src.voice.tts_engine import SpeechSynthesizer
from src.voice.tts_text import normalize_for_speech

logger = logging.getLogger(__name__)

QUEUE_MAX_ITEMS = 16


def resample_audio(
    samples: np.ndarray, src_rate: float, dst_rate: float
) -> np.ndarray:
    """Resample mono float32 audio using linear interpolation."""
    if src_rate == dst_rate or len(samples) == 0:
        return samples
    duration = len(samples) / src_rate
    n_out = max(1, int(round(duration * dst_rate)))
    x_old = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


class SoundDevicePlayer:
    """Default audio player: blocking playback via sounddevice."""

    name = "sounddevice"

    def __init__(self, sd: Optional[object] = None) -> None:
        if sd is None:
            import sounddevice as sd  # type: ignore
        self._sd = sd
        self.last_device_rate = 0.0

    def default_output_rate(self) -> float:
        """Return the system default output sample rate."""
        try:
            idx = self._sd.default.device[1]
            info = self._sd.query_devices(idx)
            rate = float(info.get("default_samplerate", 0.0))
        except Exception:  # noqa: BLE001
            rate = 0.0
        self.last_device_rate = rate
        return rate

    def play(self, samples: np.ndarray, sample_rate: int) -> None:
        """Play ``samples`` (blocking). Resamples to the device rate."""
        if samples.size == 0:
            return
        device_rate = self.default_output_rate()
        audio = samples
        rate = float(sample_rate)
        if device_rate and abs(device_rate - rate) > 1.0:
            audio = resample_audio(samples, rate, device_rate)
            rate = device_rate
        self._sd.play(audio, samplerate=int(rate))
        self._sd.wait()


class SpeechOutputService:
    """Worker that speaks queued text through a synthesizer + player.

    Args:
        synthesizer: TTS backend.
        player: Playback backend; defaults to :class:`SoundDevicePlayer`.
        gate: Optional object with ``pause()``/``resume()`` (the capture
            service) invoked around each playback to prevent echo.
    """

    def __init__(
        self,
        synthesizer: SpeechSynthesizer,
        player: Optional[SoundDevicePlayer] = None,
        gate: Optional[object] = None,
        queue_size: int = QUEUE_MAX_ITEMS,
        on_started: Optional[Callable[[str], None]] = None,
        on_finished: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._synthesizer = synthesizer
        self._player = player if player is not None else SoundDevicePlayer()
        self._gate = gate
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue(
            maxsize=queue_size
        )
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._speaking = threading.Event()
        self._current: Optional[str] = None
        self._on_started = on_started
        self._on_finished = on_finished
        self._on_error = on_error
        self._level_callback: Optional[
            Callable[[list[float], float], None]
        ] = None
        self.synthesized_count = 0
        self.playback_errors = 0

    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def is_speaking(self) -> bool:
        return self._speaking.is_set()

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    @property
    def engine_name(self) -> str:
        return getattr(self._synthesizer, "name", "unknown")

    @property
    def model_dir(self) -> Optional[str]:
        md = getattr(self._synthesizer, "model_dir", None)
        return str(md) if md is not None else None

    def set_callbacks(
        self,
        on_started: Optional[Callable[[str], None]] = None,
        on_finished: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Register or replace event callbacks."""
        self._on_started = on_started
        self._on_finished = on_finished
        self._on_error = on_error

    def set_level_callback(
        self,
        callback: Optional[Callable[[list[float], float], None]],
    ) -> None:
        """Register an optional amplitude-envelope callback (V10.7 HUD).

        Called on the playback worker thread right before audio plays,
        with ``(levels, chunk_seconds)``: per-chunk RMS values
        normalized to 0..1 and the duration of each chunk. This is real
        measured TTS waveform amplitude — receivers must marshal to the
        GUI thread themselves. Independent of ``set_callbacks`` so it is
        never reset by session wiring.
        """
        self._level_callback = callback

    def stats(self) -> dict:
        return {
            "synthesizer": self.engine_name,
            "speaking": self.is_speaking,
            "pending": self.pending_count,
            "synthesized": self.synthesized_count,
            "playback_errors": self.playback_errors,
        }

    # ------------------------------------------------------------------
    def speak(self, text: str) -> bool:
        """Enqueue ``text`` for speech. Returns False if dropped.

        The queue stores ``(display_text, synth_text)`` pairs: the
        synthesizer receives the TTS-normalized form (numbers as words,
        CamelCase split, paths reduced) while callbacks/UI keep the
        original display text.
        """
        if not text or not text.strip():
            return False
        display = text.strip()
        synth = normalize_for_speech(display)
        try:
            self._queue.put_nowait((display, synth))
            return True
        except queue.Full:
            logger.warning("Speech queue full; dropped: %r", text[:40])
            return False

    def start(self) -> None:
        """Start the playback worker (idempotent)."""
        if self.is_running:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="speech-output", daemon=True
        )
        self._thread.start()
        logger.info("Speech output started (engine=%s)", self.engine_name)

    def stop(self) -> None:
        """Stop the worker and discard queued speech."""
        self._stop.set()
        self._clear_queue()
        thread, self._thread = self._thread, None
        self._speaking.clear()
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=2.0)
        self._resume_gate()
        logger.info("Speech output stopped")

    def shutdown(self) -> None:
        """Stop playback and release the synthesizer."""
        self.stop()
        try:
            self._synthesizer.close()
        except Exception:  # noqa: BLE001
            pass

    def prewarm(self, text: str = "Ready.") -> float:
        """Synthesize ``text`` once and discard the audio.

        Warms the TTS runtime (ONNX session, lexicon) at startup so the
        first real utterance does not pay cold-start cost. Returns the
        elapsed seconds, or -1.0 on failure. Call before ``speak()``.
        """
        t0 = time.perf_counter()
        try:
            samples, _rate = self._synthesizer.synthesize(
                normalize_for_speech(text)
            )
            _ = samples  # discarded — warmup only
        except Exception:  # noqa: BLE001
            logger.debug("TTS prewarm failed", exc_info=True)
            return -1.0
        return time.perf_counter() - t0

    def _clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    # ------------------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self._speak_one(*item)

    def _speak_one(self, display_text: str, synth_text: str) -> None:
        text = display_text
        self._current = text
        self._speaking.set()
        self._emit_started(text)
        self._pause_gate()
        try:
            samples, rate = self._synthesizer.synthesize(synth_text)
            self.synthesized_count += 1
            if samples.size:
                self._emit_level_envelope(samples, rate)
                self._player.play(samples, rate)
        except Exception as exc:  # noqa: BLE001
            self.playback_errors += 1
            logger.exception("Speech synthesis/playback error")
            self._emit_error(f"Speech output failed: {exc}")
        finally:
            self._resume_gate()
            self._speaking.clear()
            self._current = None
            self._emit_finished(text)

    # ------------------------------------------------------------------
    def _pause_gate(self) -> None:
        if self._gate is None:
            return
        try:
            logger.info("TTS gate: capture paused (playback start)")
            self._gate.pause()
        except Exception:  # noqa: BLE001
            logger.debug("Audio gate pause failed", exc_info=True)

    def _resume_gate(self) -> None:
        if self._gate is None:
            return
        try:
            self._gate.resume()
            logger.info("TTS gate: capture resumed (playback end)")
        except Exception:  # noqa: BLE001
            logger.debug("Audio gate resume failed", exc_info=True)

    def _emit_level_envelope(
        self, samples: np.ndarray, rate: int
    ) -> None:
        """Compute a ~10 Hz RMS envelope of the waveform about to play."""
        cb = self._level_callback
        if cb is None or samples.size == 0:
            return
        try:
            chunk = max(1, int(rate) // 10)
            levels = [
                float(np.sqrt(np.mean(samples[i:i + chunk] ** 2)))
                for i in range(0, samples.size, chunk)
            ]
            peak = max(levels) or 1.0
            levels = [min(1.0, v / peak * 1.4) for v in levels]
            cb(levels, chunk / max(1.0, float(rate)))
        except Exception:  # noqa: BLE001 - HUD data is best-effort
            logger.debug("TTS level envelope failed", exc_info=True)

    def _emit_started(self, text: str) -> None:
        if self._on_started is not None:
            try:
                self._on_started(text)
            except Exception:  # noqa: BLE001
                pass

    def _emit_finished(self, text: str) -> None:
        if self._on_finished is not None:
            try:
                self._on_finished(text)
            except Exception:  # noqa: BLE001
                pass

    def _emit_error(self, message: str) -> None:
        if self._on_error is not None:
            try:
                self._on_error(message)
            except Exception:  # noqa: BLE001
                pass
