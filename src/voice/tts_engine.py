"""Offline text-to-speech engine abstraction and sherpa-onnx backend.

The synthesizer converts text to a 1-D float32 waveform at the model's
native sample rate. Generation runs on caller-managed worker threads —
never on the GUI thread.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Protocol

import numpy as np

from src.core.paths import tts_model_dir

logger = logging.getLogger(__name__)

DEFAULT_TTS_MODEL_DIR = tts_model_dir()
DEFAULT_LEXICON_NAME = "lexicon-en-us.txt"


class SpeechSynthesisError(Exception):
    """Raised for TTS engine failures."""


class ModelUnavailableError(SpeechSynthesisError):
    """Raised when the TTS model files are missing."""


class SpeechSynthesizer(Protocol):
    """Interface that any text-to-speech backend must implement."""

    @property
    def name(self) -> str: ...

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return ``(samples_float32, sample_rate)`` for ``text``."""
        ...

    def close(self) -> None: ...


class NullSpeechSynthesizer:
    """No-op synthesizer used when no model is available (tests/fallback).

    Produces a short silent buffer so playback plumbing can be exercised
    without a model.
    """

    name = "null"

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        return np.zeros(2205, dtype=np.float32), 22050

    def close(self) -> None:
        pass


class SherpaOnnxSynthesizer:
    """sherpa-onnx VITS (Piper) offline TTS backend.

    Expects a model directory containing ``*.onnx``, ``tokens.txt`` and
    (for Piper voices) an English lexicon — either ``lexicon.txt`` inside
    the model dir or a ``lexicon-en-us.txt`` in a parent directory.
    """

    name = "sherpa-onnx-vits"

    def __init__(
        self,
        model_dir: Path | str,
        num_threads: int = 2,
        speed: float = 1.0,
        sid: int = 0,
        provider: str = "cpu",
    ) -> None:
        import sherpa_onnx  # noqa: F401 - imported lazily for tests

        self._model_dir = Path(model_dir)
        self._num_threads = num_threads
        self._speed = speed
        self._sid = sid
        self._provider = provider
        self.syntheses = 0
        self._load()

    # ------------------------------------------------------------------
    @property
    def model_dir(self) -> Path:
        return self._model_dir

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

    def _find_lexicon(self) -> Optional[Path]:
        for candidate in (
            self._model_dir / "lexicon.txt",
            self._model_dir / DEFAULT_LEXICON_NAME,
            self._model_dir.parent / DEFAULT_LEXICON_NAME,
            self._model_dir.parent.parent / DEFAULT_LEXICON_NAME,
        ):
            if candidate.is_file():
                return candidate
        return None

    def _load(self) -> None:
        import sherpa_onnx

        d = self._model_dir
        model = self._pick(d, "*.onnx")
        tokens = self._find(d, "tokens.txt")
        lexicon = self._find_lexicon()
        missing = []
        if model is None:
            missing.append("*.onnx")
        if tokens is None:
            missing.append("tokens.txt")
        if lexicon is None:
            missing.append("lexicon (lexicon.txt or lexicon-en-us.txt)")
        if missing:
            raise ModelUnavailableError(
                f"TTS model files missing in {self._model_dir}: "
                f"{', '.join(missing)}. See README for download instructions."
            )

        try:
            self._tts = sherpa_onnx.OfflineTts(
                sherpa_onnx.OfflineTtsConfig(
                    model=sherpa_onnx.OfflineTtsModelConfig(
                        num_threads=self._num_threads,
                        vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                            model=str(model),
                            lexicon=str(lexicon),
                            tokens=str(tokens),
                        ),
                    ),
                    max_num_sentences=1,
                )
            )
        except ModelUnavailableError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(
                f"TTS model failed to load from {self._model_dir}: {exc}"
            ) from exc
        logger.info(
            "TTS engine loaded: %s (%s) from %s",
            self.name,
            model.name,
            self._model_dir,
        )

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize ``text`` to a float32 waveform."""
        if not text or not text.strip():
            return np.zeros(0, dtype=np.float32), 22050
        audio = self._tts.generate(
            text.strip(), sid=self._sid, speed=self._speed
        )
        samples = np.asarray(audio.samples, dtype=np.float32)
        self.syntheses += 1
        return samples, int(audio.sample_rate)

    def close(self) -> None:
        pass


def _model_dirs(base: Path | str) -> list[Path]:
    """Return candidate model directories under ``base`` (recursive)."""
    base_path = Path(base)
    if not base_path.exists():
        return []
    dirs = [base_path] + [p for p in base_path.rglob("*") if p.is_dir()]
    scored = []
    for d in dirs:
        score = sum(1 for _ in d.glob("*.onnx")) + sum(
            1 for _ in d.glob("*.txt")
        )
        if score:
            scored.append((-score, str(d)))
    scored.sort()
    return [Path(s) for _, s in scored] or [base_path]


def create_synthesizer(
    model_dir: Optional[Path | str] = None,
    num_threads: int = 2,
    speed: float = 1.0,
    sid: int = 0,
) -> SpeechSynthesizer:
    """Create the configured TTS synthesizer.

    ``model_dir`` may point at the models root (auto-discovers the first
    model subdirectory) or directly at a model directory. Falls back to
    :class:`NullSpeechSynthesizer` when no usable model is found.
    """
    directory = Path(model_dir) if model_dir else tts_model_dir()
    candidates = _model_dirs(directory)
    last_error: Optional[Exception] = None
    for candidate in candidates:
        # Skip the bare models root itself when it has no model files;
        # subdirectories are tried in score order.
        try:
            return SherpaOnnxSynthesizer(
                candidate,
                num_threads=num_threads,
                speed=speed,
                sid=sid,
            )
        except ModelUnavailableError as exc:
            last_error = exc
            logger.debug("TTS candidate %s unavailable: %s", candidate, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            logger.exception("TTS engine init failed for %s", candidate)
            continue
    if last_error is not None:
        logger.warning("TTS disabled: %s", last_error)
    else:
        logger.warning(
            "TTS disabled: no model files found under %s", directory
        )
    return NullSpeechSynthesizer()
