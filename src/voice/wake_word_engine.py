






"""Wake-word engine abstraction and sherpa-onnx implementation.

The engine detects the exact phrase "Hello Jarvis" using sherpa-onnx
keyword spotting (open-vocabulary custom keywords, fully offline).
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from src.core.paths import kws_model_dir
from src.voice import kws_parity_diagnostic

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
DEFAULT_MODEL_DIR = kws_model_dir()
WAKE_PHRASE = "hello jarvis"


class WakeWordError(Exception):
    """Base error for wake-word engine failures."""


class ModelUnavailableError(WakeWordError):
    """Raised when the wake-word model files are missing."""


class WakeWordEngine(Protocol):
    """Interface that any wake-word backend must implement."""

    @property
    def name(self) -> str: ...

    def process(self, samples_16k: np.ndarray) -> float:
        """Process one 16 kHz mono float32 frame; return confidence 0..1."""
        ...

    def rescore(
        self,
        samples_16k: np.ndarray,
        lead_seconds: tuple = (0.5,),
        trail_seconds: float = 0.5,
        trace: Optional[Any] = None,
    ):
        """Re-decode captured audio on fresh streams; return keyword or None."""
        ...

    def keyword_detected(self) -> Optional[str]:
        """Return the detected keyword since last call, or None."""
        ...

    def reset(self) -> None: ...

    def close(self) -> None: ...


class NullWakeWordEngine:
    """No-op engine used when no model is available (tests/fallback)."""

    name = "null"

    @property
    def keyword_count(self) -> int:
        return 0

    @property
    def keywords_sha256(self) -> Optional[str]:
        return None

    @property
    def inference_ms_ema(self) -> float:
        return 0.0

    def process(self, samples_16k: np.ndarray) -> float:
        return 0.0

    def rescore(
        self,
        samples_16k: np.ndarray,
        lead_seconds: tuple = (0.5,),
        trail_seconds: float = 0.5,
        trace: Optional[Any] = None,
    ):
        return None

    def keyword_detected(self) -> Optional[str]:
        return None

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


class SherpaOnnxKeywordSpotter:
    """sherpa-onnx keyword spotter for the exact phrase "Hello Jarvis".

    Expects model files in ``model_dir``: ``encoder*.onnx``,
    ``decoder*.onnx``, ``joiner*.onnx``, ``tokens.txt`` and
    ``keywords.txt`` (BPE-tokenized ``hello jarvis`` entry).
    """

    name = "sherpa-onnx-kws"

    def __init__(self, model_dir: Path | str, num_threads: int = 2) -> None:
        import sherpa_onnx  # noqa: F401 - imported lazily for tests

        self._model_dir = Path(model_dir)
        files = {
            "encoder": self._find("encoder*.onnx"),
            "decoder": self._find("decoder*.onnx"),
            "joiner": self._find("joiner*.onnx"),
            "tokens": self._find("tokens.txt"),
            "keywords": (
                self._find("keywords_jarvis.txt") or self._find("keywords.txt")
            ),
        }
        missing = [k for k, v in files.items() if v is None]
        empty = [k for k, v in files.items() if v is not None and v.stat().st_size == 0]
        if missing or empty:
            problems = missing + [f"{k} (empty file)" for k in empty]
            raise ModelUnavailableError(
                f"Wake-word model files missing/invalid in {self._model_dir}: "
                f"{', '.join(problems)}. See README for download instructions."
            )

        self._sherpa = sherpa_onnx
        try:
            self._spotter = sherpa_onnx.KeywordSpotter(
            tokens=str(files["tokens"]),
            encoder=str(files["encoder"]),
            decoder=str(files["decoder"]),
            joiner=str(files["joiner"]),
                keywords_file=str(files["keywords"]),
                num_threads=num_threads,
                provider="cpu",
            )
        except ModelUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(
                f"Wake-word model failed to load from {self._model_dir}: {exc}"
            ) from exc
        self._stream = self._spotter.create_stream()
        # V10.7.5: serialize every operation that touches the spotter or
        # its streams. The KWS worker calls process() continuously while
        # controller threads may call reset() during standby handover;
        # without this lock a reset could interleave with an in-flight
        # decode and corrupt stream state.
        self._lock = threading.RLock()
        # V10.7.4 diagnostic: monotonic id of the current KWS stream so
        # parity reports can prove which stream generation each frame
        # was decoded on. Inert when the diagnostic env flag is unset.
        self._stream_generation = 0
        self._pending_keyword: Optional[str] = None
        self._keywords_file = files["keywords"]
        self.inferences = 0
        self.last_inference_monotonic = 0.0
        self.last_inference_seconds = 0.0
        self._inference_ms_ema = 0.0
        # Keyword-set provenance: count + content hash so a runtime log
        # proves exactly which pronunciation variants were loaded.
        self._keyword_count, self._keywords_sha256 = self._inspect_keywords(
            self._keywords_file
        )
        logger.info(
            "Wake-word engine loaded from %s (keywords: %s, "
            "variants=%d, sha256=%s)",
            self._model_dir,
            self._keywords_file,
            self._keyword_count,
            self._keywords_sha256,
        )

    @property
    def model_dir(self) -> Path:
        """Return the directory containing the model files."""
        return self._model_dir

    @property
    def keywords_file(self) -> Optional[Path]:
        """Return the keyword file in use, if known."""
        return self._keywords_file

    @property
    def keyword_count(self) -> int:
        """Number of keyword entries loaded from the keywords file."""
        return self._keyword_count

    @property
    def keywords_sha256(self) -> Optional[str]:
        """SHA-256 of the keywords file content (provenance)."""
        return self._keywords_sha256

    @property
    def inference_ms_ema(self) -> float:
        """Exponential moving average of per-frame inference time (ms)."""
        return self._inference_ms_ema

    @staticmethod
    def _inspect_keywords(path: Optional[Path]) -> tuple[int, Optional[str]]:
        """Return (entry_count, sha256) for a keywords file."""
        if path is None:
            return 0, None
        try:
            data = Path(path).read_bytes()
        except OSError:
            return 0, None
        digest = hashlib.sha256(data).hexdigest()
        count = sum(
            1 for line in data.decode("utf-8", "replace").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
        return count, digest

    def _find(self, pattern: str) -> Optional[Path]:
        matches = sorted(self._model_dir.rglob(pattern))
        return matches[0] if matches else None

    def process(self, samples_16k: np.ndarray) -> float:
        """Feed a 16 kHz float32 frame; return 1.0 on keyword trigger."""
        with self._lock:
            return self._process_locked(samples_16k)

    def _process_locked(self, samples_16k: np.ndarray) -> float:
        self._stream.accept_waveform(
            TARGET_SAMPLE_RATE, samples_16k.astype(np.float32)
        )
        t0 = time.perf_counter()
        decode_calls = 0
        while self._spotter.is_ready(self._stream):
            self._spotter.decode_stream(self._stream)
            decode_calls += 1
        elapsed = time.perf_counter() - t0
        # sherpa-onnx get_result() returns the detected keyword as a
        # plain string ("" when nothing was detected) — not an object.
        result = self._spotter.get_result(self._stream)
        kws_parity_diagnostic.engine_result(
            self, raw=result, decode_calls=decode_calls,
            elapsed=elapsed,
        )
        self.inferences += 1
        self.last_inference_monotonic = time.monotonic()
        self.last_inference_seconds = elapsed
        ms = elapsed * 1000.0
        self._inference_ms_ema = (
            ms if not self._inference_ms_ema
            else 0.8 * self._inference_ms_ema + 0.2 * ms
        )
        keyword = result.strip() if isinstance(result, str) else ""
        if keyword:
            self._pending_keyword = keyword
            try:
                tokens = self._spotter.tokens(self._stream)
            except Exception:  # noqa: BLE001
                tokens = []
            # V10.3.3: discard the stream entirely after a detection.
            # sherpa-onnx reset_stream() leaves the spotter in a state
            # where audio fed before the next utterance suppresses the
            # following detection (alternating detect/miss). A fresh
            # stream is cheap and provably clean.
            self._fresh_stream("detection")
            logger.info("KWS keyword result: %r tokens=%s", keyword, tokens)
            return 1.0
        return 0.0

    def rescore(
        self,
        samples_16k: np.ndarray,
        lead_seconds: tuple = (0.5,),
        trail_seconds: float = 0.5,
        trace: Optional[Any] = None,
        should_abort: Optional[Any] = None,
    ):
        """Re-decode captured audio through fresh KWS streams.

        V10.7.5: the production stream is position-sensitive — a
        borderline utterance can score below threshold at one stream
        anchor yet detect cleanly at another (V10.7.3 §3). When an
        activity window closes without a detection, the detection
        service replays that window's audio here at several fresh
        anchors. Each lead value produces an independent
        ``create_stream()`` generation; decoding stops at the first
        keyword emission.

        This NEVER touches ``self._stream``, `_pending_keyword``, or
        the inference counters — it is a pure side computation. The
        engine lock serializes it against process()/reset() so the
        shared spotter is never used concurrently.

        V10.8 Task 3: the engine lock is now acquired per 100 ms decode
        chunk instead of for the whole call, so live ``process()``
        frames interleave between chunks — a multi-second rescore can
        no longer starve live KWS. ``should_abort`` is an optional
        ``callable -> bool`` checked between anchors and chunks; when
        it returns True the rescore stops early (bounded shutdown and
        per-job budgets). Mutual exclusion is unchanged: every spotter
        call still completes while holding the lock.

        ``trace`` is an optional callable ``trace(event: str, **fields)``
        invoked with ``"anchor"`` once per evaluated anchor,
        ``"skipped"`` when an anchor cannot be evaluated, and
        ``"complete"`` once at the end — used by the V10.7.5.1
        production wake trace. Trace failures never affect rescoring.

        Returns ``(keyword, lead_seconds)`` on the first hit, else
        ``None``.
        """
        audio = np.asarray(samples_16k, dtype=np.float32).reshape(-1)
        audio_seconds = audio.size / float(TARGET_SAMPLE_RATE)
        requested = len(lead_seconds) if lead_seconds else 0
        if audio.size == 0 or not lead_seconds:
            self._rescore_trace(trace, "skipped",
                                reason="empty_audio_or_no_anchors",
                                stage="input")
            self._rescore_trace(
                trace, "complete", anchors_requested=requested,
                anchors_evaluated=0, hit=False, keyword=None,
                lead_seconds=None, elapsed_ms=0.0, aborted=False)
            return None
        chunk = 1600  # 100 ms @ 16 kHz — same cadence as live frames
        trail_n = max(0, int(trail_seconds * TARGET_SAMPLE_RATE))
        trail = np.zeros(trail_n, dtype=np.float32)
        evaluated = 0
        t_all = time.monotonic()
        hit_keyword = None
        hit_lead = None
        aborted = False

        def _abort() -> bool:
            if should_abort is None:
                return False
            try:
                return bool(should_abort())
            except Exception:  # noqa: BLE001
                return False

        for lead in lead_seconds:
            if _abort():
                aborted = True
                break
            lead_n = max(0, int(lead * TARGET_SAMPLE_RATE))
            padded = np.concatenate(
                [np.zeros(lead_n, dtype=np.float32), audio, trail]
            )
            padded_seconds = padded.size / float(TARGET_SAMPLE_RATE)
            try:
                with self._lock:
                    stream = self._spotter.create_stream()
            except Exception as exc:  # noqa: BLE001
                logger.debug("KWS rescore stream creation failed",
                             exc_info=True)
                evaluated += 1
                self._rescore_trace(
                    trace, "anchor", lead_seconds=float(lead),
                    padded_seconds=round(padded_seconds, 3),
                    audio_seconds=round(audio_seconds, 3),
                    hit=False, keyword=None, elapsed_ms=0.0,
                    error=repr(exc))
                continue
            keyword = None
            error = None
            t0 = time.monotonic()
            try:
                for off in range(0, padded.size, chunk):
                    if _abort():
                        aborted = True
                        break
                    seg = padded[off:off + chunk]
                    with self._lock:
                        stream.accept_waveform(
                            TARGET_SAMPLE_RATE,
                            seg.astype(np.float32, copy=False),
                        )
                        while self._spotter.is_ready(stream):
                            self._spotter.decode_stream(stream)
                        result = self._spotter.get_result(stream)
                    # NB: no parity-diagnostic hook here — rescore
                    # runs outside the per-frame record lifecycle
                    # and must not overwrite live-frame fields.
                    if isinstance(result, str) and result.strip():
                        keyword = result.strip()
                        break
            except Exception as exc:  # noqa: BLE001
                logger.debug("KWS rescore decode failed",
                             exc_info=True)
                keyword = None
                error = repr(exc)
            evaluated += 1
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self._rescore_trace(
                trace, "anchor", lead_seconds=float(lead),
                padded_seconds=round(padded_seconds, 3),
                audio_seconds=round(audio_seconds, 3),
                hit=bool(keyword), keyword=keyword,
                elapsed_ms=round(elapsed_ms, 2), error=error)
            if keyword:
                hit_keyword = keyword
                hit_lead = float(lead)
                break
        elapsed_ms = (time.monotonic() - t_all) * 1000.0
        self._rescore_trace(
            trace, "complete", anchors_requested=requested,
            anchors_evaluated=evaluated, hit=bool(hit_keyword),
            keyword=hit_keyword, lead_seconds=hit_lead,
            elapsed_ms=round(elapsed_ms, 2), aborted=aborted)
        if hit_keyword:
            logger.info(
                "KWS rescore hit: %r (lead=%.2fs)", hit_keyword, hit_lead
            )
            return hit_keyword, hit_lead
        return None

    @staticmethod
    def _rescore_trace(trace: Optional[Any], event: str, **fields) -> None:
        """Invoke the optional rescore trace callback; never raise."""
        if trace is None:
            return
        try:
            trace(event, **fields)
        except Exception:  # noqa: BLE001
            logger.debug("KWS rescore trace callback failed",
                         exc_info=True)

    def keyword_detected(self) -> Optional[str]:
        with self._lock:
            kw, self._pending_keyword = self._pending_keyword, None
            return kw

    def _fresh_stream(self, reason: str = "fresh") -> None:
        """Replace the KWS stream with a clean one.

        ``reset_stream()`` was found (V10.3.3) to leave the sherpa-onnx
        keyword spotter in a state where audio arriving before the next
        utterance suppresses that utterance's detection — the exact
        post-"Goodbye Jarvis" failure. ``create_stream()`` allocates a
        new stream on the shared model; detections are rare so the cost
        is negligible. Callers must hold ``self._lock`` (or go through
        ``reset()``/``process()`` which take it).
        """
        try:
            self._stream = self._spotter.create_stream()
        except Exception:  # noqa: BLE001
            # Fall back to the supported reset if stream creation fails.
            logger.debug("KWS create_stream failed; using reset_stream",
                         exc_info=True)
            self._spotter.reset_stream(self._stream)
        self._stream_generation += 1
        kws_parity_diagnostic.on_stream_event(self, reason)

    def reset(self) -> None:
        with self._lock:
            self._fresh_stream("reset")
            self._pending_keyword = None

    def close(self) -> None:
        pass


def create_engine(model_dir: Path | str | None = None) -> WakeWordEngine:
    """Create the configured wake-word engine.

    Falls back to NullWakeWordEngine when model files are absent so the
    application still runs (detection simply never triggers).
    """
    directory = Path(model_dir) if model_dir else kws_model_dir()
    try:
        return SherpaOnnxKeywordSpotter(directory)
    except ModelUnavailableError as exc:
        logger.warning("%s Wake-word detection disabled.", exc)
        return NullWakeWordEngine()
    except Exception:  # noqa: BLE001
        logger.exception("Wake-word engine initialization failed")
        return NullWakeWordEngine()
