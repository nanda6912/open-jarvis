"""Canonical speech-recognition outcome taxonomy (V10.8 Task 4).

Every recognition attempt must be classifiable by the stage where it
ended. This module defines the shared category names and the mappings
from the service-level ``on_no_speech`` reasons and the controller-level
recognition ``outcome`` strings onto those categories, so diagnostics,
the recognition log and tests all speak the same vocabulary.

Categories describe WHERE an attempt terminated — they do not measure
transcript correctness (that requires a reference transcript and is
Task 5 scope).
"""

from __future__ import annotations

from typing import Optional

# --- Canonical categories -------------------------------------------------
CAPTURE_FAILED = "capture_failed"
NO_AUDIO_RECEIVED = "no_audio_received"
VAD_REJECTED = "vad_rejected"
VAD_ACCEPTED = "vad_accepted"
STT_EMPTY_TRANSCRIPT = "stt_empty_transcript"
STT_TRANSCRIPT_PRODUCED = "stt_transcript_produced"
STT_EXCEPTION = "stt_exception"
INTENT_UNKNOWN = "intent_unknown"
ENTITY_MISSING = "entity_missing"
ENTITY_AMBIGUOUS = "entity_ambiguous"
HANDLER_FAILED = "handler_failed"
COMMAND_EXECUTED = "command_executed"

#: All category names, for validation and documentation.
CATEGORIES = frozenset({
    CAPTURE_FAILED,
    NO_AUDIO_RECEIVED,
    VAD_REJECTED,
    VAD_ACCEPTED,
    STT_EMPTY_TRANSCRIPT,
    STT_TRANSCRIPT_PRODUCED,
    STT_EXCEPTION,
    INTENT_UNKNOWN,
    ENTITY_MISSING,
    ENTITY_AMBIGUOUS,
    HANDLER_FAILED,
    COMMAND_EXECUTED,
})

#: ``on_no_speech`` reason -> category. Reasons are emitted by
#: ``SpeechRecognitionService``; unknown reasons map to None so callers
#: can keep the raw reason as the classification.
_REASON_CATEGORY = {
    # Emitted by _check_starvation when the capture stream is dead.
    "capture_failed": CAPTURE_FAILED,
    # Emitted by _check_starvation when the queue simply delivers no
    # frames while the stream reports healthy.
    "no_audio_frames": NO_AUDIO_RECEIVED,
    # Emitted once per sustained below-threshold stretch.
    "audio_below_vad_threshold": VAD_REJECTED,
    # Emitted when VAD accepted an utterance but STT decoded "".
    "vad_accepted_stt_empty": STT_EMPTY_TRANSCRIPT,
}

#: Controller recognition ``outcome`` -> category. Only terminal
#: outcomes that answer "where did this attempt end" are mapped;
#: intermediate outcomes (wake_repeat, duplicate_ignored, ...) keep
#: ``category=None`` — their ``outcome`` already carries the detail.
_OUTCOME_CATEGORY = {
    "executed": COMMAND_EXECUTED,
    "handler_failed": HANDLER_FAILED,
    "entity_missing": ENTITY_MISSING,
    "entity_ambiguous": ENTITY_AMBIGUOUS,
    "intent_unknown": INTENT_UNKNOWN,
    "stt_exception": STT_EXCEPTION,
    "no_speech": None,  # resolved via the reason -> see below
}


def category_for_reason(reason: Optional[str]) -> Optional[str]:
    """Map an ``on_no_speech`` reason string to a canonical category."""
    if not reason:
        return None
    return _REASON_CATEGORY.get(str(reason))


def category_for_outcome(
    outcome: Optional[str],
    detail: Optional[str] = None,
) -> Optional[str]:
    """Map a controller recognition outcome to a canonical category.

    ``detail`` carries the no-speech reason for ``outcome == "no_speech"``
    entries so the VAD/empty-STT distinction survives aggregation.
    """
    if not outcome:
        return None
    if outcome == "no_speech":
        return category_for_reason(detail)
    return _OUTCOME_CATEGORY.get(outcome)


def is_category(name: Optional[str]) -> bool:
    """True when ``name`` is a canonical category."""
    return name in CATEGORIES


__all__ = [
    "CAPTURE_FAILED",
    "NO_AUDIO_RECEIVED",
    "VAD_REJECTED",
    "VAD_ACCEPTED",
    "STT_EMPTY_TRANSCRIPT",
    "STT_TRANSCRIPT_PRODUCED",
    "STT_EXCEPTION",
    "INTENT_UNKNOWN",
    "ENTITY_MISSING",
    "ENTITY_AMBIGUOUS",
    "HANDLER_FAILED",
    "COMMAND_EXECUTED",
    "CATEGORIES",
    "category_for_reason",
    "category_for_outcome",
    "is_category",
]