"""Continuous microphone capture feeding a bounded audio queue.

The sounddevice callback only copies/resamples frames into a bounded
queue — it never performs inference or blocks. Resampling to the
engine rate (16 kHz) happens on the audio callback thread but is a
cheap numpy operation.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections import deque
from typing import Any, Callable, Optional

import numpy as np

from src.voice.audio_device_manager import (
    AudioDevice,
    AudioError,
    DeviceUnavailableError,
    PermissionDeniedError,
)

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
QUEUE_MAX_FRAMES = 250  # ~15 s of 100 ms frames; bounds memory


def resample_to_16k(samples: np.ndarray, src_rate: float) -> np.ndarray:
    """Resample mono float32 audio to 16 kHz using linear interpolation."""
    if src_rate == TARGET_SAMPLE_RATE:
        return samples
    if len(samples) == 0:
        return samples
    duration = len(samples) / src_rate
    n_out = max(1, int(round(duration * TARGET_SAMPLE_RATE)))
    x_old = np.linspace(0.0, duration, num=len(samples), endpoint=False)
    x_new = np.linspace(0.0, duration, num=n_out, endpoint=False)
    return np.interp(x_new, x_old, samples).astype(np.float32)


class AudioCaptureService:
    """Manages a sounddevice input stream and a bounded frame queue."""

    def __init__(
        self,
        sd: Any = None,
        device_manager: Any = None,
        target_rate: int = TARGET_SAMPLE_RATE,
        queue_size: int = QUEUE_MAX_FRAMES,
    ) -> None:
        if sd is None:
            import sounddevice as sd  # type: ignore
        self._sd = sd
        self._device_manager = device_manager
        self._target_rate = target_rate
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_size)
        self._stream: Any = None
        self._device: Optional[AudioDevice] = None
        self._sample_rate = 0.0
        self._lock = threading.Lock()
        self._paused = False
        self._on_error: Optional[Callable[[Exception], None]] = None
        self._on_disconnect: Optional[Callable[[], None]] = None
        self.frames_dropped = 0
        # Diagnostic counters (no raw audio is stored or logged).
        self.frames_captured = 0
        self.frames_queued = 0
        self.frames_paused = 0
        self.frames_flushed = 0
        self.callback_seq = 0
        self.last_callback_monotonic = 0.0
        self.last_callback_channels = 0
        self.last_callback_frames = 0
        self.last_indata_rms = 0.0
        self.last_indata_peak = 0.0
        self.channel_rms: list[float] = []
        self.channel_peaks: list[float] = []
        self.max_channel_rms = 0.0
        self.last_rms = 0.0
        self.last_peak = 0.0
        self.last_frame_monotonic = 0.0
        self.max_rms = 0.0
        # Callback-liveness tracking: inter-callback interval (EMA) and
        # stream start time let the standby watchdog distinguish a dead
        # PortAudio stream (active=True but no callbacks) from a healthy
        # one. Monotonic clock only — never wall time.
        self._stream_started_monotonic = 0.0
        self._prev_callback_monotonic = 0.0
        self.first_callback_monotonic = 0.0
        self.callback_interval_seconds = 0.0
        self.signal_frames = 0      # frames with rms >= SIGNAL_RMS
        self.clipped_frames = 0     # frames peaking at/near full scale
        # Gaps between consecutive callbacks that were far longer than
        # the established interval — early evidence of endpoint stalls
        # before the stall threshold is reached.
        self.missed_callback_intervals = 0
        self.consecutive_missed_callbacks = 0
        # V10.6: PortAudio status events. An input overflow means the
        # driver dropped samples BEFORE our callback — invisible to
        # frames_dropped/queue_depth and a plausible cause of missed
        # wake words. Bounded event log (no audio, status text only).
        self.status_warnings = 0
        self.overflow_callbacks = 0
        self._status_events: deque = deque(maxlen=50)
        # Multi-channel capture: when the endpoint exposes several
        # inputs (e.g. the 4-channel Intel SST array) the stream opens
        # with up to 4 channels and the callback feeds the
        # highest-energy channel — never a blind average, which can
        # cancel out-of-phase sources and halve the speech level.
        self._stream_channels = 0
        self._selected_channel = 0
        self._channel_ema: list[float] = []
        self._channel_switch_count = 0
        # Incremented on every start()/stop() so a callback invoked by a
        # replaced or closed PortAudio stream can be recognised as
        # stale and dropped instead of feeding the shared queue.
        self._stream_generation = 0

    # RMS above which a frame counts as carrying signal (not silence).
    SIGNAL_RMS = 0.005
    # Peak at/above which a frame is treated as clipped.
    CLIP_PEAK = 0.98

    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        """Return whether the capture stream is active."""
        return self._stream is not None and self._stream.active

    @property
    def device(self) -> Optional[AudioDevice]:
        """Return the device currently being captured."""
        return self._device

    @property
    def sample_rate(self) -> int:
        """Return the stream sample rate (device native rate)."""
        return int(self._sample_rate)

    @property
    def paused(self) -> bool:
        """Return whether frames are being discarded."""
        return self._paused

    @property
    def input_channels(self) -> int:
        """Channels the open stream actually delivers (0 if closed)."""
        return self._stream_channels

    # ------------------------------------------------------------------
    def start(
        self,
        device_id: Optional[str] = None,
        blocksize: int = 1600,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None,
    ) -> AudioDevice:
        """Start capturing from the resolved device.

        Raises:
            DeviceUnavailableError: No input device available.
            PermissionDeniedError: OS denied microphone access.
            AudioError: Other audio subsystem failure.
        """
        with self._lock:
            if self.is_running:
                return self._device  # type: ignore[return-value]

            # New stream generation: any callback still in flight from
            # a previous stream (PortAudio can invoke it concurrently
            # with close()) carries an older generation and is dropped.
            # Bump BEFORE releasing the stale stream so its tail
            # callbacks cannot enqueue into the new stream's queue.
            self._stream_generation += 1
            stream_gen = self._stream_generation

            def _stream_callback(
                indata: np.ndarray,
                frames: int,
                time_info: Any,
                status: Any,
            ) -> None:
                if stream_gen != self._stream_generation:
                    return
                self._audio_callback(indata, frames, time_info, status)

            # A previous stream may have died without close() (device
            # removal, driver abort). Release it before opening a new one
            # so we never leak or double-open PortAudio streams.
            stale, self._stream = self._stream, None
            if stale is not None:
                try:
                    stale.stop()
                    stale.close()
                except Exception:  # noqa: BLE001
                    logger.debug("Error closing stale audio stream",
                                 exc_info=True)

            self._on_error = on_error
            self._on_disconnect = on_disconnect

            if self._device_manager is not None:
                device = self._device_manager.resolve_device(device_id)
            else:
                device = AudioDevice(0, "default", 1, 16000.0)

            self._sample_rate = float(device.default_samplerate)
            # Prefer capturing directly at the engine rate (16 kHz):
            # Windows shared-mode audio transparently resamples, which
            # avoids our crude linear-interpolation downsampler entirely.
            # Drivers that reject 16 kHz (the WASAPI "Invalid sample
            # rate" failure) are handled by probing supported rates and
            # falling back to the device-native rate + software
            # resampling.
            preferred = (float(self._target_rate), self._sample_rate)
            supported = []
            if self._device_manager is not None:
                try:
                    supported = self._device_manager.supported_input_rates(
                        device, preferred=preferred
                    )
                except Exception:  # noqa: BLE001 - probe is best-effort
                    logger.debug("Sample-rate probe failed", exc_info=True)
            if supported:
                candidate_rates = supported
            else:
                candidate_rates = list(dict.fromkeys(preferred))
            # Channel strategy: multi-channel endpoints (mic arrays,
            # stereo headsets) are opened with up to 4 channels so the
            # callback can pick the strongest input; mono endpoints
            # and drivers that reject multichannel fall back to 1.
            max_ch = int(
                getattr(device, "max_input_channels", 1) or 1
            )
            channel_candidates = (
                [min(max_ch, 4), 1] if max_ch > 1 else [1]
            )
            stream = None
            opened_channels = 1
            last_exc: Optional[Exception] = None
            for rate in candidate_rates:
                for channels in channel_candidates:
                    try:
                        candidate = self._sd.InputStream(
                            device=device.index,
                            channels=channels,
                            samplerate=rate,
                            blocksize=blocksize,
                            dtype="float32",
                            callback=_stream_callback,
                        )
                        candidate.start()
                        stream = candidate
                        opened_channels = channels
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_exc = exc
                        logger.info(
                            "Capture at %.0f Hz / %d ch unavailable "
                            "(%s); trying next",
                            rate, channels, exc,
                        )
                if stream is not None:
                    break
            if stream is None:
                self._map_error(last_exc or AudioError("no usable rate"))

            # Trust the rate the stream actually opened at.
            actual = getattr(stream, "samplerate", None)
            self._sample_rate = (
                float(actual) if actual else float(rate)
            )
            self._stream = stream
            self._device = device
            self._stream_channels = opened_channels
            self._selected_channel = 0
            self._channel_ema = []
            self._channel_switch_count = 0
            self._stream_started_monotonic = time.monotonic()
            self._prev_callback_monotonic = 0.0
            # Reset per-stream liveness baselines: a stale
            # last_callback_monotonic from the previous stream would
            # either mask a fresh stall or fake an instant one.
            self.last_callback_monotonic = 0.0
            self.last_frame_monotonic = 0.0
            logger.info(
                "Audio capture started: %s @ %.0f Hz (%d ch)",
                device.name,
                self._sample_rate,
                opened_channels,
            )
            return device

    def stop(self) -> None:
        """Stop capture and close the stream."""
        with self._lock:
            stream, self._stream = self._stream, None
            self._device = None
            # Invalidate any callback still in flight on the closing
            # stream so it cannot enqueue frames after the drain below.
            self._stream_generation += 1
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception:  # noqa: BLE001
                logger.exception("Error closing audio stream")
            logger.info("Audio capture stopped")
        self._drain_queue()

    def pause(self) -> None:
        """Discard incoming frames without closing the stream."""
        self._paused = True

    def resume(self) -> None:
        """Resume processing incoming frames."""
        self._paused = False

    # ------------------------------------------------------------------
    def get_frame(self, timeout: float = 0.5) -> Optional[np.ndarray]:
        """Return the next 16 kHz float32 frame, or None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stats(self) -> dict[str, Any]:
        """Return diagnostic counters for the capture pipeline."""
        now = time.monotonic()
        return {
            "frames_captured": self.frames_captured,
            "frames_queued": self.frames_queued,
            "frames_dropped": self.frames_dropped,
            "frames_paused": self.frames_paused,
            "frames_flushed": self.frames_flushed,
            "callback_seq": self.callback_seq,
            "callback_channels": self.last_callback_channels,
            "callback_frames": self.last_callback_frames,
            "indata_rms": self.last_indata_rms,
            "indata_peak": self.last_indata_peak,
            "channel_rms": list(self.channel_rms),
            "channel_peaks": list(self.channel_peaks),
            "channel_ema": [round(v, 6) for v in self._channel_ema],
            "max_channel_rms": self.max_channel_rms,
            # Current backlog depth (frames_queued is the cumulative
            # total ever enqueued — queue_depth is the live depth).
            "queue_depth": self._queue.qsize(),
            "rms": self.last_rms,
            "peak": self.last_peak,
            "max_rms": self.max_rms,
            "signal_frames": self.signal_frames,
            "clipped_frames": self.clipped_frames,
            "sample_rate": self.sample_rate,
            "device": self._device.name if self._device else None,
            "running": self.is_running,
            "frame_age_seconds": (
                now - self.last_frame_monotonic
                if self.last_frame_monotonic
                else None
            ),
            # Standby-watchdog diagnostics (monotonic).
            "callback_age_seconds": self.callback_stall_seconds(),
            "callback_interval_seconds": (
                self.callback_interval_seconds or None
            ),
            "stream_uptime_seconds": (
                now - self._stream_started_monotonic
                if self._stream_started_monotonic
                else None
            ),
            "missed_callback_intervals": self.missed_callback_intervals,
            "consecutive_missed_callbacks":
                self.consecutive_missed_callbacks,
            "status_warnings": self.status_warnings,
            "overflow_callbacks": self.overflow_callbacks,
            "recent_status_events": [
                {
                    "age_seconds": round(now - t, 3),
                    "status": s,
                }
                for t, s in list(self._status_events)[-5:]
            ],
            "input_channels": self._stream_channels or None,
            "selected_channel": (
                self._selected_channel
                if self._stream_channels > 1 else None
            ),
            "channel_switch_count": self._channel_switch_count,
            "device_stable_id": (
                self._device.stable_id if self._device else None
            ),
            "paused": self._paused,
        }

    def _update_channel_selection(self, ch_rms) -> None:
        """Track per-channel energy and pick the strongest input.

        Mic arrays often expose speech on one channel plus
        noise-reference channels; picking the highest-energy channel
        preserves speech level without averaging (which can cancel
        anti-phase content). Hysteresis prevents rapid flapping.
        """
        n = len(ch_rms)
        if len(self._channel_ema) != n:
            self._channel_ema = [0.0] * n
        alpha = 0.08
        for i, v in enumerate(ch_rms):
            v = float(v)
            self._channel_ema[i] = (
                v if self._channel_ema[i] == 0.0
                else (1.0 - alpha) * self._channel_ema[i] + alpha * v
            )
        best = max(range(n), key=lambda i: self._channel_ema[i])
        cur = self._selected_channel
        if (
            best != cur
            and n > cur
            and self._channel_ema[best]
            > self._channel_ema[cur] * 1.25 + 1e-4
        ):
            self._selected_channel = best
            self._channel_switch_count += 1
            logger.info(
                "Capture channel switch %d -> %d (ema=%s)",
                cur, best,
                [round(x, 5) for x in self._channel_ema],
            )

    def callback_stall_seconds(self) -> Optional[float]:
        """Seconds since the last audio callback, or None.

        Returns None when no stream is running or no timing baseline
        exists. A paused stream still receives callbacks (frames are
        discarded inside the callback), so intentional TTS/STT capture
        pauses do NOT register as stalls.
        """
        if not self.is_running:
            return None
        baseline = (
            self.last_callback_monotonic or self._stream_started_monotonic
        )
        if not baseline:
            return None
        return time.monotonic() - baseline

    def flush(self) -> None:
        """Discard all queued frames (e.g. stale audio before resume)."""
        before = self._queue.qsize()
        self._drain_queue()
        self.frames_flushed += before

    def _drain_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return

    # ------------------------------------------------------------------
    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: Any,
    ) -> None:
        """sounddevice callback — must be fast and non-blocking."""
        try:
            if status:
                status_text = str(status)
                self.status_warnings += 1
                now_status = time.monotonic()
                self._status_events.append((now_status, status_text))
                if "overflow" in status_text.lower():
                    self.overflow_callbacks += 1
                    logger.warning(
                        "Audio input overflow (PortAudio dropped "
                        "samples before callback): %s", status_text
                    )
                else:
                    logger.warning("Audio stream status: %s", status)
            self.frames_captured += 1
            self.callback_seq += 1
            now = time.monotonic()
            if self._prev_callback_monotonic:
                interval = now - self._prev_callback_monotonic
                expected = self.callback_interval_seconds
                if expected and interval > max(0.5, expected * 4):
                    self.missed_callback_intervals += 1
                    self.consecutive_missed_callbacks += 1
                else:
                    self.consecutive_missed_callbacks = 0
                self.callback_interval_seconds = (
                    interval
                    if not self.callback_interval_seconds
                    else 0.8 * self.callback_interval_seconds
                    + 0.2 * interval
                )
            self._prev_callback_monotonic = now
            if not self.first_callback_monotonic:
                self.first_callback_monotonic = now
            self.last_callback_monotonic = now
            arr = np.asarray(indata)
            # Measure the RAW input before any downmix/resample: shape
            # is (frames, channels). Per-channel RMS exposes muted or
            # mis-mapped channels on multi-channel arrays.
            self.last_callback_frames = int(frames)
            self.last_callback_channels = int(arr.shape[1]) if arr.ndim > 1 else 1
            if arr.size:
                f32 = arr.astype(np.float32, copy=False)
                if f32.ndim > 1 and f32.shape[1] > 1:
                    ch_rms = np.sqrt(np.mean(f32 * f32, axis=0))
                    self.channel_rms = [float(v) for v in ch_rms]
                    self.channel_peaks = [
                        float(v) for v in np.max(np.abs(f32), axis=0)
                    ]
                    # V10.3.3: never adapt channel selection from paused
                    # frames. The pause gate exists precisely because the
                    # microphone hears Jarvis's own TTS playback; letting
                    # the EMA learn from speaker bleed can pin
                    # _selected_channel to a playback-dominated input and
                    # starve the KWS of the user's voice after standby.
                    if not self._paused:
                        self._update_channel_selection(ch_rms)
                    self.max_channel_rms = max(
                        self.max_channel_rms, float(ch_rms.max())
                    )
                    self.last_indata_rms = float(
                        np.sqrt(np.mean(f32.reshape(-1) ** 2))
                    )
                else:
                    flat = f32.reshape(-1)
                    self.channel_rms = [
                        float(np.sqrt(np.mean(flat * flat)))
                    ]
                    self.channel_peaks = [float(np.max(np.abs(flat)))]
                    self.last_indata_rms = self.channel_rms[0]
                    self.max_channel_rms = max(
                        self.max_channel_rms, self.last_indata_rms
                    )
                self.last_indata_peak = float(np.max(np.abs(f32)))
            if self._paused:
                self.frames_paused += 1
                return
            if arr.ndim > 1 and arr.shape[1] > 1:
                # Multi-channel stream: feed the selected (strongest)
                # channel only — never a blind average.
                sel = min(self._selected_channel, arr.shape[1] - 1)
                mono = arr[:, sel].astype(np.float32).reshape(-1)
            else:
                mono = np.asarray(indata, dtype=np.float32).reshape(-1)
            if self._sample_rate != self._target_rate:
                mono = resample_to_16k(mono, self._sample_rate)
            # ``indata`` is a view into PortAudio's recycled input ring
            # buffer: later callbacks overwrite the same memory. A queued
            # frame MUST own its samples — otherwise a consumer that
            # falls behind (e.g. STT blocked inside transcribe()) reads
            # corrupted/reused audio instead of what was captured.
            # resample_to_16k already returns an owning array.
            if not mono.flags.owndata:
                mono = mono.copy()
            if mono.size:
                rms = float(np.sqrt(np.mean(mono * mono)))
                peak = float(np.max(np.abs(mono)))
                self.last_rms = rms
                self.last_peak = peak
                if rms > self.max_rms:
                    self.max_rms = rms
                if rms >= self.SIGNAL_RMS:
                    self.signal_frames += 1
                if peak >= self.CLIP_PEAK:
                    self.clipped_frames += 1
            self.last_frame_monotonic = time.monotonic()
            try:
                self._queue.put_nowait(mono)
                self.frames_queued += 1
            except queue.Full:
                # Drop oldest frame to keep latency bounded.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(mono)
                except queue.Full:
                    pass
                self.frames_dropped += 1
        except Exception as exc:  # noqa: BLE001
            logger.exception("Audio callback error")
            if self._on_error:
                try:
                    self._on_error(exc)
                except Exception:  # noqa: BLE001
                    pass

    @staticmethod
    def _map_error(exc: Exception) -> None:
        """Translate sounddevice errors into AudioError subclasses."""
        name = type(exc).__name__.lower()
        msg = str(exc).lower()
        if "permission" in name or "permission" in msg or "access" in msg:
            raise PermissionDeniedError(f"Microphone access denied: {exc}") from exc
        if "device" in msg or "no device" in msg:
            raise DeviceUnavailableError(f"Audio device unavailable: {exc}") from exc
        raise AudioError(f"Audio initialization failed: {exc}") from exc
