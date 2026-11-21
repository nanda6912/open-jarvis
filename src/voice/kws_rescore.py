"""Shared wake-word rescore orchestration (V10.8 Task 3).

This module holds the production rescore candidate preparation and
candidate execution logic so BOTH the live detection service and the
offline benchmark run the identical code path — no simplified
benchmark-only variant.

Candidate order (production, preserved):

1. ``full`` — the activity-window audio (tail-capped).
2. ``speech_trimmed`` — window audio starting near its first
   clear-speech frame.
3. ``recent_context`` — speech-trimmed tail of the ~5 s diagnostic
   ring, which retains pre-window context the activity gate clipped.

Everything here is pure numpy / engine calls: no microphone, no I/O,
no audio persistence.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

# Production defaults — identical to the values historically hard-coded
# in WakeWordDetectionService (V10.7.5 / V10.7.5.1).
DEFAULT_LEAD_SECONDS = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.2, 1.5, 1.8)
DEFAULT_TRAIL_SECONDS = 0.5
DEFAULT_MIN_SECONDS = 0.3
DEFAULT_MAX_SECONDS = 8.0
DEFAULT_SPEECH_GATE = 0.02
DEFAULT_TRIM_MARGIN_SECONDS = 0.4
DEFAULT_TRIM_MIN_DROP_SECONDS = 0.3
# The diagnostic ring must hold this much MORE audio than the window
# buffer before recent_context is considered non-duplicate.
DEFAULT_DIAG_EXTRA_SECONDS = 0.3
# Scheduling policy defaults (Phase E).
DEFAULT_JOB_BUDGET_SECONDS = 6.0
DEFAULT_QUEUE_MAXSIZE = 4
DEFAULT_JOB_TTL_SECONDS = 15.0


@dataclass(frozen=True)
class RescorePolicy:
    """Tunables for candidate preparation and job scheduling."""

    lead_seconds: tuple = DEFAULT_LEAD_SECONDS
    trail_seconds: float = DEFAULT_TRAIL_SECONDS
    min_seconds: float = DEFAULT_MIN_SECONDS
    max_seconds: float = DEFAULT_MAX_SECONDS
    speech_gate: float = DEFAULT_SPEECH_GATE
    trim_margin_seconds: float = DEFAULT_TRIM_MARGIN_SECONDS
    trim_min_drop_seconds: float = DEFAULT_TRIM_MIN_DROP_SECONDS
    diag_extra_seconds: float = DEFAULT_DIAG_EXTRA_SECONDS
    job_budget_seconds: float = DEFAULT_JOB_BUDGET_SECONDS
    queue_maxsize: int = DEFAULT_QUEUE_MAXSIZE
    job_ttl_seconds: float = DEFAULT_JOB_TTL_SECONDS


def concat_float32(frames) -> np.ndarray:
    """Concatenate float32 frame arrays into one 1-D array."""
    return np.concatenate(
        [np.asarray(f, dtype=np.float32) for f in frames]
    )


def speech_trimmed(
    audio: np.ndarray,
    sample_rate: int,
    speech_gate: float = DEFAULT_SPEECH_GATE,
    trim_margin_seconds: float = DEFAULT_TRIM_MARGIN_SECONDS,
    trim_min_drop_seconds: float = DEFAULT_TRIM_MIN_DROP_SECONDS,
) -> Optional[np.ndarray]:
    """Return audio starting near the first clear-speech frame.

    Only 100 ms hops at/above ``speech_gate`` count as speech onset; a
    margin of pre-speech audio is kept so a quiet phrase onset is never
    cut. Returns None when the trim would drop less than
    ``trim_min_drop_seconds`` — i.e. the candidate would be effectively
    identical to the full window.
    """
    hop = int(sample_rate * 0.1)
    first = None
    for off in range(0, audio.size - hop + 1, hop):
        seg = audio[off : off + hop]
        rms = float(np.sqrt(np.mean(seg * seg)))
        if rms >= speech_gate:
            first = off
            break
    if first is None:
        return None
    margin = int(trim_margin_seconds * sample_rate)
    start = max(0, first - margin)
    min_drop = int(trim_min_drop_seconds * sample_rate)
    if start < min_drop:
        return None
    return audio[start:]


def recent_context_trimmed(
    diag_audio: Optional[np.ndarray],
    window_reference_samples: int,
    sample_rate: int,
    min_seconds: float = DEFAULT_MIN_SECONDS,
    speech_gate: float = DEFAULT_SPEECH_GATE,
    trim_margin_seconds: float = DEFAULT_TRIM_MARGIN_SECONDS,
    trim_min_drop_seconds: float = DEFAULT_TRIM_MIN_DROP_SECONDS,
    diag_extra_seconds: float = DEFAULT_DIAG_EXTRA_SECONDS,
) -> Optional[np.ndarray]:
    """Speech-trimmed diagnostic-ring audio, when it adds context.

    Returns None unless the ring holds meaningfully MORE audio than the
    window buffer (``diag_extra_seconds``) — i.e. it contains
    pre-window context the window clip lost — and a clear-speech onset
    exists to trim to.
    """
    if diag_audio is None or diag_audio.size == 0:
        return None
    min_n = int(min_seconds * sample_rate)
    if diag_audio.size <= (
        window_reference_samples + int(diag_extra_seconds * sample_rate)
    ):
        return None
    trimmed = speech_trimmed(
        diag_audio, sample_rate, speech_gate,
        trim_margin_seconds, trim_min_drop_seconds,
    )
    if trimmed is None or trimmed.size < min_n:
        return None
    return trimmed


def plan_candidates(
    window_audio: np.ndarray,
    diag_audio: Optional[np.ndarray],
    policy: RescorePolicy,
    sample_rate: int,
    window_reference_samples: Optional[int] = None,
) -> tuple[list, Optional[str]]:
    """Build the ordered rescore candidate list.

    ``window_audio`` is the concatenated activity-window KWS input.
    ``diag_audio`` is the concatenated diagnostic ring (may be None).
    ``window_reference_samples`` is the uncapped window sample count
    used for the recent_context duplicate check (defaults to the
    tail-capped length).

    Returns ``(candidates, error)`` where ``candidates`` is a list of
    ``(name, audio)`` pairs in production order, or ``([], reason)``
    when the window cannot be rescored (``empty``, ``audio_too_short``).
    """
    audio = np.asarray(window_audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return [], "empty_buffer"
    max_n = int(policy.max_seconds * sample_rate)
    if audio.size > max_n:
        # Keep the tail: the wake phrase is at the end of the activity
        # window, not its beginning.
        audio = audio[-max_n:]
    min_n = int(policy.min_seconds * sample_rate)
    if audio.size < min_n:
        return [], "audio_too_short"
    if window_reference_samples is None:
        window_reference_samples = audio.size

    candidates = [("full", audio)]
    trimmed = speech_trimmed(
        audio, sample_rate, policy.speech_gate,
        policy.trim_margin_seconds, policy.trim_min_drop_seconds,
    )
    if trimmed is not None:
        candidates.append(("speech_trimmed", trimmed))
    recent = recent_context_trimmed(
        diag_audio, window_reference_samples, sample_rate,
        policy.min_seconds, policy.speech_gate,
        policy.trim_margin_seconds, policy.trim_min_drop_seconds,
        policy.diag_extra_seconds,
    )
    if recent is not None:
        recent_seconds = recent.size / float(sample_rate)
        duplicate = any(
            abs(recent_seconds - c.size / float(sample_rate)) < 0.3
            for _, c in candidates
        )
        if not duplicate:
            candidates.append(("recent_context", recent))
    return candidates, None


def _call_engine_rescore(
    engine: Any,
    audio: np.ndarray,
    lead_seconds: tuple,
    trail_seconds: float,
    trace: Optional[Callable],
    should_abort: Optional[Callable[[], bool]],
):
    """Invoke ``engine.rescore`` with graceful signature fallback.

    Tries the fullest signature first (``trace`` + ``should_abort``),
    then drops ``should_abort``, then drops ``trace`` — matching the
    historical TypeError fallback for engines predating those hooks.
    """
    try:
        return engine.rescore(
            audio,
            lead_seconds=lead_seconds,
            trail_seconds=trail_seconds,
            trace=trace,
            should_abort=should_abort,
        )
    except TypeError:
        pass
    try:
        return engine.rescore(
            audio,
            lead_seconds=lead_seconds,
            trail_seconds=trail_seconds,
            trace=trace,
        )
    except TypeError:
        pass
    return engine.rescore(
        audio,
        lead_seconds=lead_seconds,
        trail_seconds=trail_seconds,
    )


def run_candidates(
    engine: Any,
    candidates: list,
    policy: RescorePolicy,
    *,
    event: Optional[Callable[[str, dict], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
    sample_rate: int = 16000,
) -> dict:
    """Run rescore candidates in order; first hit wins.

    ``event(kind, fields)`` receives lifecycle notifications:

    - ``"candidate_start"`` — fields: candidate, audio_seconds,
      samples, sample_rate, anchors_requested, trail_seconds
    - ``"anchor"`` — engine anchor trace fields + candidate
    - ``"skipped"`` — engine skipped trace fields + candidate
    - ``"candidate_complete"`` — engine complete fields + candidate
    - ``"candidate_failed"`` — candidate, error, elapsed_ms

    ``should_abort`` is checked between candidates and (when the engine
    supports it) between anchors inside ``engine.rescore``.

    Returns an aggregate report dict; ``report["result"]`` is the
    ``(keyword, lead_seconds)`` hit or None.
    """
    report: dict[str, Any] = {
        "attempted": True,
        "succeeded": False,
        "result": None,
        "candidates_attempted": 0,
        "candidates_completed": 0,
        "anchors_requested": 0,
        "anchors_evaluated": 0,
        "anchors_evaluated_complete": True,
        "elapsed_ms": None,
        "error": None,
        "hit_candidate": None,
        "budget_exceeded": False,
        "aborted": False,
        "candidate_stats": [],
    }
    t_all = time.monotonic()

    def _emit(kind: str, **fields) -> None:
        if event is None:
            return
        try:
            event(kind, fields)
        except Exception:  # noqa: BLE001 - observers must not break rescore
            pass

    for cand_name, cand_audio in candidates:
        if should_abort is not None and should_abort():
            report["aborted"] = True
            report["budget_exceeded"] = True
            break
        report["candidates_attempted"] += 1
        cand_stats = {
            "candidate": cand_name,
            "anchors_requested": len(policy.lead_seconds),
            "anchors_evaluated": None,
            "anchor_events": 0,
            "hit": False,
            "elapsed_ms": None,
            "error": None,
        }
        cand_audio_seconds = cand_audio.size / float(sample_rate)
        _emit(
            "candidate_start",
            candidate=cand_name,
            audio_seconds=round(cand_audio_seconds, 3),
            samples=int(cand_audio.size),
            sample_rate=sample_rate,
            anchors_requested=len(policy.lead_seconds),
            trail_seconds=policy.trail_seconds,
        )

        def _trace(ev: str, _cand=cand_name, _stats=cand_stats, **fields):
            if ev == "anchor":
                _stats["anchor_events"] += 1
                _emit("anchor", candidate=_cand, **fields)
            elif ev == "skipped":
                _emit("skipped", candidate=_cand, **fields)
            elif ev == "complete":
                req = fields.get("anchors_requested")
                evn = fields.get("anchors_evaluated")
                if req is not None:
                    _stats["anchors_requested"] = int(req)
                if evn is not None:
                    _stats["anchors_evaluated"] = int(evn)
                _stats["hit"] = bool(fields.get("hit"))
                _emit("candidate_complete", candidate=_cand, **fields)

        t0 = time.monotonic()
        try:
            result = _call_engine_rescore(
                engine, cand_audio, policy.lead_seconds,
                policy.trail_seconds, _trace, should_abort,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)
            cand_stats["elapsed_ms"] = elapsed_ms
            cand_stats["error"] = repr(exc)
            cand_stats["hit"] = False
            if cand_stats["anchors_evaluated"] is None:
                cand_stats["anchors_evaluated"] = (
                    cand_stats["anchor_events"] or 0
                )
            _emit(
                "candidate_failed", candidate=cand_name,
                error=repr(exc), elapsed_ms=elapsed_ms,
            )
            report["candidate_stats"].append(cand_stats)
            report["candidates_completed"] += 1
            if report["error"] is None:
                report["error"] = repr(exc)
            continue
        elapsed_ms = round((time.monotonic() - t0) * 1000.0, 2)
        cand_stats["elapsed_ms"] = elapsed_ms
        cand_stats["hit"] = bool(result)
        if cand_stats["anchors_evaluated"] is None:
            if cand_stats["anchor_events"]:
                cand_stats["anchors_evaluated"] = cand_stats[
                    "anchor_events"
                ]
            elif result is None:
                # A full miss on a conforming engine evaluated every
                # requested anchor; a hit's exact count is unknown
                # without the trace callback, so leave it None.
                cand_stats["anchors_evaluated"] = cand_stats[
                    "anchors_requested"
                ]
        report["candidate_stats"].append(cand_stats)
        report["candidates_completed"] += 1
        req = cand_stats.get("anchors_requested")
        if req is not None:
            report["anchors_requested"] += int(req)
        evn = cand_stats.get("anchors_evaluated")
        if evn is None:
            report["anchors_evaluated_complete"] = False
        else:
            report["anchors_evaluated"] += int(evn)
        if cand_stats.get("error") and report["error"] is None:
            report["error"] = cand_stats["error"]
        if result:
            report["succeeded"] = True
            report["result"] = result
            report["hit_candidate"] = cand_name
            break
    report["elapsed_ms"] = round((time.monotonic() - t_all) * 1000.0, 2)
    return report


__all__ = [
    "RescorePolicy",
    "DEFAULT_LEAD_SECONDS",
    "DEFAULT_TRAIL_SECONDS",
    "DEFAULT_MIN_SECONDS",
    "DEFAULT_MAX_SECONDS",
    "DEFAULT_SPEECH_GATE",
    "DEFAULT_TRIM_MARGIN_SECONDS",
    "DEFAULT_TRIM_MIN_DROP_SECONDS",
    "DEFAULT_DIAG_EXTRA_SECONDS",
    "DEFAULT_JOB_BUDGET_SECONDS",
    "DEFAULT_QUEUE_MAXSIZE",
    "DEFAULT_JOB_TTL_SECONDS",
    "concat_float32",
    "speech_trimmed",
    "recent_context_trimmed",
    "plan_candidates",
    "run_candidates",
]
