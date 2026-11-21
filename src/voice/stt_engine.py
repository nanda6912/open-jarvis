"""Offline speech-to-text engine abstraction and sherpa-onnx backend.

The recognizer transcribes a complete 16 kHz mono float32 utterance to
text. All inference happens on caller-managed worker threads — never on
the GUI thread.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np

from src.core.paths import stt_model_dir

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
DEFAULT_STT_MODEL_DIR = stt_model_dir()


class SpeechRecognitionError(Exception):
    """Raised for STT engine failures."""


def prepare_utterance(samples: np.ndarray) -> np.ndarray:
    """Preprocess one utterance for the decoder.

    Gently boosts quiet utterances toward a usable level with a hard
    gain cap, so low-level speech from laptop-array microphones is not
    decoded at the noise floor. Loud audio passes through unchanged.

    Edge trimming was evaluated and rejected: the int8 Moonshine
    decoder proved chaotically sensitive to utterance length on the
    recorded-command corpus (micro-changes flipped whole words both
    ways), so only level normalization is applied. VAD decisions
    upstream are unaffected.
    """
    audio = np.asarray(samples, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return audio
    peak = float(np.max(np.abs(audio)))
    if 1e-6 < peak < 0.7:
        audio = audio * min(0.7 / peak, 6.0)
    return np.ascontiguousarray(audio, dtype=np.float32)


class ModelUnavailableError(SpeechRecognitionError):
    """Raised when the STT model files are missing."""


class SpeechRecognizer(Protocol):
    """Interface that any speech-to-text backend must implement."""

    @property
    def name(self) -> str: ...

    def transcribe(self, samples_16k: np.ndarray) -> str:
        """Transcribe one 16 kHz mono float32 utterance; return text."""
        ...

    def close(self) -> None: ...


class NullSpeechRecognizer:
    """No-op recognizer used when no model is available (tests/fallback)."""

    name = "null"

    def transcribe(self, samples_16k: np.ndarray) -> str:
        return ""

    def close(self) -> None:
        pass


class SherpaOnnxRecognizer:
    """sherpa-onnx offline recognizer supporting several model families.

    ``model_type`` selects the loader: ``moonshine`` (default, tiny
    English model optimised for short commands), ``transducer``
    (zipformer encoder/decoder/joiner), ``whisper`` or ``nemo_ctc``.
    """

    name = "sherpa-onnx"

    def __init__(
        self,
        model_dir: Path | str,
        model_type: str = "auto",
        num_threads: int = 2,
        language: str = "en",
        provider: str = "cpu",
    ) -> None:
        import sherpa_onnx  # noqa: F401 - imported lazily for tests

        self._model_dir = Path(model_dir)
        self._num_threads = num_threads
        self._language = language
        self._provider = provider
        self._model_type = self._resolve_type(model_type)
        self.inferences = 0
        self.last_inference_monotonic = 0.0
        self._load()

    # ------------------------------------------------------------------
    @property
    def model_dir(self) -> Path:
        return self._model_dir

    @property
    def model_type(self) -> str:
        return self._model_type

    def _resolve_type(self, model_type: str) -> str:
        if model_type != "auto":
            return model_type
        d = self._model_dir
        if self._find(d, "preprocess.onnx") and self._find(d, "encode*.onnx"):
            return "moonshine"
        if (
            self._find(d, "encoder*.onnx")
            and self._find(d, "decoder*.onnx")
            and self._find(d, "joiner*.onnx")
        ):
            return "transducer"
        if self._find(d, "model*.onnx") and self._find(d, "tokens.txt"):
            return "nemo_ctc"
        if self._find(d, "*encoder*.onnx") and self._find(d, "*decoder*.onnx"):
            return "whisper"
        raise ModelUnavailableError(
            f"Cannot determine STT model type in {self._model_dir}; "
            "set voice.stt_engine explicitly."
        )

    @staticmethod
    def _find(directory: Path, pattern: str) -> Optional[Path]:
        matches = sorted(directory.rglob(pattern))
        return matches[0] if matches else None

    @staticmethod
    def _pick(directory: Path, pattern: str) -> Optional[Path]:
        """Pick a model file, preferring int8 quantized variants."""
        matches = sorted(directory.rglob(pattern))
        if not matches:
            return None
        int8 = [m for m in matches if "int8" in m.name]
        return int8[0] if int8 else matches[0]

    def _require(self, **kwargs: str) -> dict[str, str]:
        missing = [k for k, v in kwargs.items() if not v]
        if missing:
            raise ModelUnavailableError(
                f"STT model files missing in {self._model_dir}: "
                f"{', '.join(missing)}. See README for download instructions."
            )
        return kwargs

    def _load(self) -> None:
        import sherpa_onnx

        d = self._model_dir
        try:
            if self._model_type == "moonshine":
                files = self._require(
                    preprocessor=str(self._find(d, "preprocess.onnx") or ""),
                    encoder=str(self._pick(d, "encode*.onnx") or ""),
                    uncached_decoder=str(
                        self._pick(d, "uncached_decode*.onnx") or ""
                    ),
                    cached_decoder=str(
                        self._pick(d, "cached_decode*.onnx") or ""
                    ),
                    tokens=str(self._find(d, "tokens.txt") or ""),
                )
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_moonshine(
                    preprocessor=files["preprocessor"],
                    encoder=files["encoder"],
                    uncached_decoder=files["uncached_decoder"],
                    cached_decoder=files["cached_decoder"],
                    tokens=files["tokens"],
                    num_threads=self._num_threads,
                    provider=self._provider,
                )
            elif self._model_type == "transducer":
                files = self._require(
                    encoder=str(self._pick(d, "encoder*.onnx") or ""),
                    decoder=str(self._pick(d, "decoder*.onnx") or ""),
                    joiner=str(self._pick(d, "joiner*.onnx") or ""),
                    tokens=str(self._find(d, "tokens.txt") or ""),
                )
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=files["encoder"],
                    decoder=files["decoder"],
                    joiner=files["joiner"],
                    tokens=files["tokens"],
                    num_threads=self._num_threads,
                    provider=self._provider,
                )
            elif self._model_type == "whisper":
                files = self._require(
                    encoder=str(self._pick(d, "*encoder*.onnx") or ""),
                    decoder=str(self._pick(d, "*decoder*.onnx") or ""),
                    tokens=str(self._find(d, "*tokens*.txt") or ""),
                )
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
                    encoder=files["encoder"],
                    decoder=files["decoder"],
                    tokens=files["tokens"],
                    language=self._language,
                    num_threads=self._num_threads,
                    provider=self._provider,
                )
            elif self._model_type == "nemo_ctc":
                files = self._require(
                    model=str(self._pick(d, "model*.onnx") or ""),
                    tokens=str(self._find(d, "tokens.txt") or ""),
                )
                self._recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
                    model=files["model"],
                    tokens=files["tokens"],
                    num_threads=self._num_threads,
                    provider=self._provider,
                )
            else:
                raise ModelUnavailableError(
                    f"Unsupported STT model type: {self._model_type}"
                )
        except ModelUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(
                f"STT model failed to load from {self._model_dir}: {exc}"
            ) from exc
        logger.info(
            "STT engine loaded: %s (%s) from %s",
            self.name,
            self._model_type,
            self._model_dir,
        )

    def transcribe(self, samples_16k: np.ndarray) -> str:
        """Transcribe a complete utterance (16 kHz mono float32)."""
        stream = self._recognizer.create_stream()
        stream.accept_waveform(
            TARGET_SAMPLE_RATE,
            prepare_utterance(samples_16k),
        )
        self._recognizer.decode_stream(stream)
        result = stream.result
        text = getattr(result, "text", result)
        self.inferences += 1
        self.last_inference_monotonic = time.monotonic()
        return str(text).strip() if text else ""

    def close(self) -> None:
        pass


def _model_dirs(base: Path | str) -> list[Path]:
    """Return candidate model directories under ``base`` (recursive)."""
    base_path = Path(base)
    if not base_path.exists():
        return []
    dirs = [base_path] + [p for p in base_path.rglob("*") if p.is_dir()]
    # Prefer directories that look like a model (contain .onnx/.txt).
    scored = []
    for d in dirs:
        score = sum(1 for _ in d.glob("*.onnx")) + sum(
            1 for _ in d.glob("*.txt")
        )
        if score:
            scored.append((-score, str(d)))
    scored.sort()
    return [Path(s) for _, s in scored] or [base_path]


def create_recognizer(
    model_dir: Optional[Path | str] = None,
    model_type: str = "auto",
    num_threads: int = 2,
    language: str = "en",
) -> SpeechRecognizer:
    """Create the configured STT recognizer.

    ``model_dir`` may point at the models root (auto-discovers the first
    model subdirectory) or directly at a model directory. Falls back to
    :class:`NullSpeechRecognizer` when no usable model is found so the
    application still runs (recognition simply returns empty text).
    """
    directory = Path(model_dir) if model_dir else stt_model_dir()
    candidates = _model_dirs(directory)
    last_error: Optional[Exception] = None
    for candidate in candidates:
        try:
            return SherpaOnnxRecognizer(
                candidate,
                model_type=model_type,
                num_threads=num_threads,
                language=language,
            )
        except ModelUnavailableError as exc:
            last_error = exc
            logger.debug("STT candidate %s unavailable: %s", candidate, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.exception("STT engine init failed for %s", candidate)
            continue
    if last_error is not None:
        logger.warning("STT disabled: %s", last_error)
    else:
        logger.warning(
            "STT disabled: no model files found under %s", directory
        )
    return NullSpeechRecognizer()
